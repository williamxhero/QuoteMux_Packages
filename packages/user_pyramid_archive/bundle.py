"""Build the immutable, evidence-first Pyramid post-adjusted futures archive.

This package does not publish rows.  In particular it is not a provider package:
the caller must later classify the fact-normalized candidates against QuoteMux.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

PACKAGE_ID = "user_pyramid_archive"
PACKAGE_VERSION = "2026.8.26"
GENERATION_ID = "futures_user_pyramid_archive_bundle_v1"
SOURCE_KEY = "pyramid_back_adjusted_20260714"
FACT_NORMALIZATION_VERSION = "quotemux_fact_source_key_v1"
SOURCE_THREAD_ID = "01a031ef-006c-7de0-a585-68eecdf769c7"

# This map is intentionally not inferred from filenames.  TL0 is Treasury T;
# TLL0 is a different product and must never silently become T or TL.
PRODUCTS: dict[str, tuple[str, str]] = {
    "agL0.txt": ("ag", "SHFE"), "alL0.txt": ("al", "SHFE"),
    "APL0.txt": ("AP", "CZCE"), "CFL0.txt": ("CF", "CZCE"),
    "cuL0.txt": ("cu", "SHFE"), "hcL0.txt": ("hc", "SHFE"),
    "iL0.txt": ("i", "DCE"), "jL0.txt": ("j", "DCE"),
    "mL0.txt": ("m", "DCE"), "MAL0.txt": ("MA", "CZCE"),
    "niL0.txt": ("ni", "SHFE"), "pL0.txt": ("p", "DCE"),
    "ruL0.txt": ("ru", "SHFE"), "scL0.txt": ("sc", "INE"),
    "TL0.txt": ("T", "CFFEX"), "TAL0.txt": ("TA", "CZCE"),
    "TFL0.txt": ("TF", "CFFEX"), "vL0.txt": ("v", "DCE"),
    "yL0.txt": ("y", "DCE"), "lhL0.txt": ("lh", "DCE"),
    "SAL0.txt": ("SA", "CZCE"), "aoL0.txt": ("ao", "SHFE"),
    "siL0.txt": ("si", "GFEX"),
}

NORMALIZED_SCHEMA = pa.schema([
    ("product_code", pa.string()), ("exchange", pa.string()),
    ("bar_time", pa.string()), ("open", pa.float64()), ("high", pa.float64()),
    ("low", pa.float64()), ("close", pa.float64()), ("volume", pa.float64()),
    ("open_interest", pa.float64()), ("adjustment_offset", pa.float64()),
    ("source_key", pa.string()),
])
STAGED_SCHEMA = pa.schema([
    ("product_code", pa.string()), ("exchange", pa.string()), ("raw_path", pa.string()),
    ("source_line", pa.int64()), ("bar_time", pa.string()), ("open", pa.float64()),
    ("high", pa.float64()), ("low", pa.float64()), ("close", pa.float64()),
    ("volume", pa.float64()), ("adjustment_offset", pa.float64()),
    ("timestamp_group", pa.string()), ("status", pa.string()), ("reason", pa.string()),
])


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _row_bytes(row: Mapping[str, object]) -> bytes:
    return _canonical({name: row.get(name) for name in NORMALIZED_SCHEMA.names}) + b"\n"


def _staged_bytes(row: Mapping[str, object]) -> bytes:
    return _canonical({name: row.get(name) for name in STAGED_SCHEMA.names}) + b"\n"


def _hash_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_hashed(source: Path, target: Path) -> tuple[int, str]:
    target.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    digest = hashlib.sha256()
    with source.open("rb") as inp, target.open("xb") as out:
        for chunk in iter(lambda: inp.read(1024 * 1024), b""):
            out.write(chunk)
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _parse(fields: list[str], line_number: int, source: Path) -> dict[str, object]:
    if len(fields) != 7:
        raise ValueError(f"{source}:{line_number}: expected exactly 7 TSV fields")
    try:
        value = datetime.strptime(fields[0].strip(), "%Y/%m/%d-%H:%M")
        numbers = [float(part.strip()) for part in fields[1:]]
    except ValueError as exc:
        raise ValueError(f"{source}:{line_number}: invalid timestamp or numeric field") from exc
    open_, high, low, close, volume, offset = numbers
    valid = all(number == number and abs(number) != float("inf") for number in numbers)
    valid = valid and volume >= 0 and high >= max(open_, close, low) and low <= min(open_, close, high)
    return {"bar_time": value, "open": open_, "high": high, "low": low, "close": close,
            "volume": volume, "adjustment_offset": offset, "_valid": valid}


def _rows(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="gbk", newline="") as source:
        for line_number, fields in enumerate(csv.reader(source, delimiter="\t"), 1):
            row = _parse(fields, line_number, path)
            row["_source_line"] = line_number
            yield row


def _timestamp_groups(path: Path) -> Iterable[tuple[datetime, list[dict[str, object]]]]:
    """Yield adjacent timestamp groups without materialising a multi-million-row file."""
    current_time: datetime | None = None
    group: list[dict[str, object]] = []
    for row in _rows(path):
        timestamp = row["bar_time"]
        if current_time is not None and timestamp < current_time:
            raise ValueError(f"{path}:{row['_source_line']}: timestamps are not sorted")
        if current_time is None or timestamp == current_time:
            current_time = timestamp
            group.append(row)
            continue
        yield current_time, group
        current_time, group = timestamp, [row]
    if current_time is not None:
        yield current_time, group


def _verify_source_root(source_root: Path) -> list[tuple[str, str, str, Path]]:
    data_dir = source_root / "后复权数据"
    if not data_dir.is_dir():
        raise ValueError(f"missing 后复权数据 directory: {data_dir}")
    prohibited = data_dir / "TLL0.txt"
    if not prohibited.is_file():
        raise ValueError("expected TLL0.txt exists so TL/T distinction can be verified")
    missing = [name for name in PRODUCTS if not (data_dir / name).is_file()]
    if missing:
        raise ValueError(f"missing required raw files: {', '.join(missing)}")
    return [(name, *PRODUCTS[name], data_dir / name) for name in sorted(PRODUCTS, key=lambda key: PRODUCTS[key][0])]


def validate_authorization(value: Mapping[str, object]) -> dict[str, object]:
    """Require explicit, persisted authorization rather than inferring entitlement."""
    authorization = dict(value)
    required = {
        "status": "private_research_authorized",
        "source_thread_id": SOURCE_THREAD_ID,
        "private_server_retention": True,
        "transformation": True,
        "private_research": True,
        "redistribution": False,
    }
    for key, expected in required.items():
        if authorization.get(key) != expected:
            raise ValueError(f"authorization.{key} must be {expected!r}")
    if not isinstance(authorization.get("evidence"), str) or not authorization["evidence"].strip():
        raise ValueError("authorization.evidence must be non-empty")
    return authorization


def preflight(source_root: Path, output: Path, authorization: Mapping[str, object]) -> dict[str, object]:
    if output.exists():
        raise ValueError(f"output must be new: {output}")
    sources = _verify_source_root(source_root)
    return {"status": "preflight_ok", "generation_id": GENERATION_ID, "products": [p for _, p, _, _ in sources],
            "raw_bytes": sum(path.stat().st_size for _, _, _, path in sources), "authorization": validate_authorization(authorization)}


def _flush(writer: pq.ParquetWriter, rows: list[dict[str, object]], schema: pa.Schema) -> None:
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=schema))
        rows.clear()


def _write_jsonl(stream: Any, digest: Any, value: Mapping[str, object]) -> None:
    encoded = _canonical(value) + b"\n"
    stream.write(encoded)
    digest.update(encoded)


def build_bundle(source_root: Path, output: Path, authorization: Mapping[str, object], batch_size: int = 100_000,
                 expected_raw_aggregate_sha256: str | None = None,
                 expected_source_normalized_rowset_sha256: str | None = None) -> dict[str, object]:
    """Copy, validate and normalize exactly the authorized 23 source files atomically."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    approved_authorization = validate_authorization(authorization)
    preflight(source_root, output, approved_authorization)
    sources = _verify_source_root(source_root)
    temporary = output.with_name(f"{output.name}.partial-{uuid.uuid4().hex}")
    temporary.mkdir(parents=True)
    staged_partial = temporary / "staged.parquet.partial"
    normalized_partial = temporary / "normalized.parquet.partial"
    facts_partial = temporary / "fact_normalized.parquet.partial"
    intervals_partial = temporary / "intervals.jsonl.partial"
    source_hash = hashlib.sha256()
    fact_hash = hashlib.sha256()
    staged_hash = hashlib.sha256()
    raw_entries: list[dict[str, object]] = []
    coverage: dict[str, dict[str, object]] = {}
    total_staged = total_safe = 0
    try:
        with (pq.ParquetWriter(staged_partial, STAGED_SCHEMA, compression="snappy") as staged_writer,
              pq.ParquetWriter(normalized_partial, NORMALIZED_SCHEMA, compression="snappy") as source_writer,
              pq.ParquetWriter(facts_partial, NORMALIZED_SCHEMA, compression="snappy") as fact_writer,
              intervals_partial.open("xb") as interval_stream):
            interval_hash = hashlib.sha256()
            staged_batch: list[dict[str, object]] = []
            source_batch: list[dict[str, object]] = []
            fact_batch: list[dict[str, object]] = []
            for filename, product, exchange, source in sources:
                raw_path = f"raw/{product}.txt"
                raw_size, raw_sha = _copy_hashed(source, temporary / raw_path)
                raw_entries.append({"path": raw_path, "logical_name": f"{product}_source", "product_code": product,
                                    "exchange": exchange, "encoding": "gbk", "size_bytes": raw_size, "sha256": raw_sha})
                first_observed: datetime | None = None
                last_observed: datetime | None = None
                accepted_start: datetime | None = None
                accepted_end: datetime | None = None
                accepted_count = raw_rows = conflict_keys = conflict_rows = invalid_rows = 0
                previous: datetime | None = None
                for timestamp, items in _timestamp_groups(temporary / raw_path):
                    first_observed = first_observed or timestamp
                    last_observed = timestamp
                    raw_rows += len(items)
                    reason = "" if len(items) == 1 and bool(items[0]["_valid"]) else (
                        "duplicate_conflicting_timestamp" if len(items) != 1 else "invalid_ohlcv")
                    status = "valid" if reason == "" else "excluded"
                    if len(items) != 1:
                        conflict_keys += 1; conflict_rows += len(items)
                    elif reason:
                        invalid_rows += 1
                    if previous and timestamp > previous + timedelta(minutes=1):
                        _write_jsonl(interval_stream, interval_hash, {"product_code": product, "exchange": exchange,
                            "start_time": (previous + timedelta(minutes=1)).isoformat(sep=" "),
                            "end_time": (timestamp - timedelta(minutes=1)).isoformat(sep=" "), "status": "residual",
                            "evidence_sha256": raw_sha, "detail": {"reason": "unclassified_no_observed_bar_wall_clock_gap", "may_include_out_of_session": True}})
                    previous = timestamp
                    for item in items:
                        staged = {"product_code": product, "exchange": exchange, "raw_path": raw_path,
                                  "source_line": item["_source_line"], "bar_time": timestamp.isoformat(sep=" "),
                                  **{key: item[key] for key in ("open", "high", "low", "close", "volume", "adjustment_offset")},
                                  "timestamp_group": f"{product}|{timestamp.isoformat(sep=' ')}", "status": status, "reason": reason}
                        staged_batch.append(staged); staged_hash.update(_staged_bytes(staged)); total_staged += 1
                    if reason:
                        _write_jsonl(interval_stream, interval_hash, {"product_code": product, "exchange": exchange,
                            "start_time": timestamp.isoformat(sep=" "), "end_time": timestamp.isoformat(sep=" "),
                            "status": "excluded", "evidence_sha256": raw_sha, "detail": {"reason": reason,
                            "source_lines": [item["_source_line"] for item in items]}})
                    else:
                        item = items[0]
                        source_row = {"product_code": product, "exchange": exchange, "bar_time": timestamp.isoformat(sep=" "),
                                      **{key: item[key] for key in ("open", "high", "low", "close", "volume", "adjustment_offset")},
                                      "open_interest": None, "source_key": f"pyramid:{raw_sha}"}
                        fact_row = {**source_row, "source_key": SOURCE_KEY}
                        source_batch.append(source_row); fact_batch.append(fact_row)
                        source_hash.update(_row_bytes(source_row)); fact_hash.update(_row_bytes(fact_row))
                        total_safe += 1; accepted_count += 1
                        accepted_start = accepted_start or timestamp; accepted_end = timestamp
                    if len(staged_batch) >= batch_size:
                        _flush(staged_writer, staged_batch, STAGED_SCHEMA)
                    if len(source_batch) >= batch_size:
                        _flush(source_writer, source_batch, NORMALIZED_SCHEMA); _flush(fact_writer, fact_batch, NORMALIZED_SCHEMA)
                _write_jsonl(interval_stream, interval_hash, {"product_code": product, "exchange": exchange,
                    "start_time": accepted_start.isoformat(sep=" ") if accepted_start else None,
                    "end_time": accepted_end.isoformat(sep=" ") if accepted_end else None, "status": "accepted",
                    "evidence_sha256": raw_sha, "detail": {"bar_count": accepted_count, "contiguous": False,
                    "meaning": "observed eligible rows; session gaps remain residual"}})
                if first_observed is None or last_observed is None:
                    raise ValueError(f"empty source file: {source}")
                coverage[product] = {"actual_start": first_observed.isoformat(sep=" "), "actual_end": last_observed.isoformat(sep=" "),
                    "exchange": exchange, "raw_rows": raw_rows, "valid_rows": accepted_count,
                    "conflicting_timestamp_keys": conflict_keys, "conflicting_rows_removed": conflict_rows, "invalid_ohlcv_rows": invalid_rows}
            _flush(staged_writer, staged_batch, STAGED_SCHEMA)
            _flush(source_writer, source_batch, NORMALIZED_SCHEMA)
            _flush(fact_writer, fact_batch, NORMALIZED_SCHEMA)
        staged = temporary / "staged.parquet"; normalized = temporary / "normalized.parquet"; facts = temporary / "fact_normalized.parquet"; intervals = temporary / "intervals.jsonl"
        staged_partial.replace(staged); normalized_partial.replace(normalized); facts_partial.replace(facts); intervals_partial.replace(intervals)
        raw_aggregate = hashlib.sha256(_canonical(raw_entries)).hexdigest()
        if expected_raw_aggregate_sha256 and raw_aggregate != expected_raw_aggregate_sha256:
            raise ValueError(f"raw aggregate SHA-256 mismatch: {raw_aggregate}")
        if expected_source_normalized_rowset_sha256 and source_hash.hexdigest() != expected_source_normalized_rowset_sha256:
            raise ValueError(f"source-normalized rowset SHA-256 mismatch: {source_hash.hexdigest()}")
        manifest = {"schema_version": "futures_user_pyramid_archive_bundle_v1", "package_id": PACKAGE_ID, "package_version": PACKAGE_VERSION,
                    "generation_id": GENERATION_ID, "authorization": approved_authorization, "source_lineage": {"source_class": "user_provided",
                    "source_identity": "pyramid_post_adjusted_20260714", "vendor_entitlement": "unknown_not_asserted", "ao_exchange_correction": "aoL0.txt maps to SHFE; legacy GFEX hashes are audit-only", "oi_semantics": "unavailable",
                    "fields": "OHLCV,adjustment_offset; OI unavailable", "missing_bar_semantics": "excluded/residual skipped; never interpolated"},
                    "raw_aggregate_sha256": raw_aggregate,
                    "raw_aggregate_algorithm": "sha256(canonical_json(artifact_bundle.raw_files; sort_keys,separators=(',',':'),utf8))",
                    "raw_files": raw_entries, "staged_artifact_sha256": _hash_path(staged),
                    "staged_rowset_sha256": staged_hash.hexdigest(), "normalized_artifact_sha256": _hash_path(normalized),
                    "source_normalized_rowset_sha256": source_hash.hexdigest(), "fact_normalization": {"version": FACT_NORMALIZATION_VERSION,
                    "description": "replace only source_key with pyramid_back_adjusted_20260714", "source_key": SOURCE_KEY,
                    "fact_normalized_artifact_sha256": _hash_path(facts), "fact_normalized_rowset_sha256": fact_hash.hexdigest()},
                    "normalized_row_count": total_safe, "staged_row_count": total_staged, "product_coverage": coverage,
                    "interval_artifact": {"path": "intervals.jsonl", "sha256": _hash_path(intervals), "rowset_sha256": interval_hash.hexdigest()}}
        manifest_path = temporary / "manifest.json"; manifest_path.write_bytes(_canonical(manifest))
        for path in (staged, normalized, facts, intervals, manifest_path):
            with path.open("rb") as stream:
                try:
                    os.fsync(stream.fileno())
                except OSError:
                    # Windows does not permit fsync on a read-only handle.
                    pass
        temporary.replace(output)
        return {"bundle": str(output), "manifest": str(output / "manifest.json"), "normalized_rows": total_safe,
                "source_normalized_rowset_sha256": source_hash.hexdigest(), "fact_normalized_rowset_sha256": fact_hash.hexdigest()}
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
