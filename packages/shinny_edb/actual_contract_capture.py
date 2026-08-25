"""Immutable, read-only capture seam for Shinny EDB actual futures contracts.

This is deliberately separate from the public main-continuous handler.  The
caller must supply the exact exchange and actual delivery contract selected by
an independently captured mapping; this module never rewrites the symbol to a
main-continuous instrument and never writes a fact table.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import os
from typing import Callable, Mapping
from zoneinfo import ZoneInfo

import requests

from quotemux.infra.provider_runtime.core import call_provider_api


EDB_KLINE_URL = "https://edb.shinnytech.com/md/kline"
SHINNY_EDB_PACKAGE_VERSION = "2026.8.25"
_TIMESTAMP_CONTRACT = {
    "source_timezone": "Asia/Shanghai",
    "source_timestamp_field": "datetime_nano",
    "source_bar_timestamp": "minute_start",
    "target_timestamp": "source_minute_start_plus_1m",
    "frequency": "1m",
}
_REQUIRED_CSV_FIELDS = frozenset({"datetime_nano", "open", "high", "low", "close", "volume", "close_oi"})


class ShinnyEdbCaptureError(RuntimeError):
    """An EDB capture failed and must not produce an artifact."""


class ShinnyEdbHistoryUnavailableError(ShinnyEdbCaptureError):
    """EDB explicitly rejected the requested historical range (HTTP 403)."""


class ShinnyEdbCaptureValidationError(ShinnyEdbCaptureError):
    """The response cannot prove a complete, exact requested capture window."""


@dataclass(frozen=True)
class ActualContractRawBar:
    product_code: str
    exchange: str
    actual_contract: str
    bar_time: str
    session_anchor_date: str
    trading_day: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    open_interest: Decimal


@dataclass(frozen=True)
class ImmutableActualContractCapture:
    """A self-contained artifact contract consumable by derived_core.

    ``artifact_sha256`` hashes ``canonical_artifact_bytes``: the exact UTF-8
    JSON payload without a self-referential digest field.  ``rowset_sha256``
    independently pins the ordered raw OHLCV/OI rows.
    """

    schema_version: str
    source_package: str
    source_package_version: str
    capture_id: str
    request: Mapping[str, str]
    captured_at: str
    timestamp_contract: Mapping[str, str]
    raw_fields: tuple[str, ...]
    rowset_sha256: str
    artifact_sha256: str
    rows: tuple[ActualContractRawBar, ...]

    def canonical_artifact_bytes(self) -> bytes:
        payload = {
            "schema_version": self.schema_version,
            "source_package": self.source_package,
            "source_package_version": self.source_package_version,
            "capture_id": self.capture_id,
            "request": dict(self.request),
            "captured_at": self.captured_at,
            "timestamp_contract": dict(self.timestamp_contract),
            "raw_fields": list(self.raw_fields),
            "rowset_sha256": self.rowset_sha256,
            "rows": [asdict(row) for row in self.rows],
        }
        return _canonical_json_bytes(payload)

    def artifact_envelope(self) -> dict[str, object]:
        """Return the serializable immutable artifact, including its digest."""

        return {
            "schema_version": self.schema_version,
            "source_package": self.source_package,
            "source_package_version": self.source_package_version,
            "capture_id": self.capture_id,
            "request": dict(self.request),
            "captured_at": self.captured_at,
            "timestamp_contract": dict(self.timestamp_contract),
            "raw_fields": list(self.raw_fields),
            "rowset_sha256": self.rowset_sha256,
            "artifact_sha256": self.artifact_sha256,
            "rows": [asdict(row) for row in self.rows],
        }


def _timeout_seconds() -> float:
    try:
        return max(5.0, float(os.getenv("MHK_SHINNY_EDB_TIMEOUT_SECONDS", "60")))
    except ValueError:
        return 60.0


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _minute_time(value: str, field: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ShinnyEdbCaptureValidationError(f"{field} must be an ISO datetime") from exc
    if parsed.tzinfo is not None:
        raise ShinnyEdbCaptureValidationError(f"{field} must be timezone-naive Asia/Shanghai")
    if parsed.second != 0 or parsed.microsecond != 0:
        raise ShinnyEdbCaptureValidationError(f"{field} must be aligned to a 1m boundary")
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def _date(value: str, field: str) -> str:
    try:
        return datetime.strptime(value, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError as exc:
        raise ShinnyEdbCaptureValidationError(f"{field} must be YYYY-MM-DD") from exc


def _decimal(value: object, field: str) -> Decimal:
    text = str(value or "").strip()
    if not text:
        raise ShinnyEdbCaptureValidationError(f"EDB response is missing {field}")
    try:
        result = Decimal(text)
    except InvalidOperation as exc:
        raise ShinnyEdbCaptureValidationError(f"EDB {field} is not decimal") from exc
    if not result.is_finite():
        raise ShinnyEdbCaptureValidationError(f"EDB {field} is not finite")
    return result


def _fetch_exact_csv(exchange: str, actual_contract: str, start_time: str, end_time: str) -> str:
    symbol = f"{exchange}.{actual_contract}"

    def _invoke() -> str:
        response = requests.get(
            EDB_KLINE_URL,
            params={"period": 60, "symbol": symbol, "start_time": start_time, "end_time": end_time},
            timeout=_timeout_seconds(),
        )
        if response.status_code == 403:
            raise ShinnyEdbHistoryUnavailableError(
                f"shinny_edb HTTP 403 history unavailable for {symbol} {start_time}..{end_time}"
            )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise ShinnyEdbCaptureError(
                f"shinny_edb HTTP {response.status_code} for {symbol} {start_time}..{end_time}"
            ) from exc
        return response.text

    return str(call_provider_api("shinny_edb", "futures.contract.1m.capture", _invoke))


def _parse_rows(
    *,
    payload: str,
    product_code: str,
    exchange: str,
    actual_contract: str,
    session_anchor_date: str,
    trading_day: str,
) -> tuple[ActualContractRawBar, ...]:
    reader = csv.DictReader(io.StringIO(payload))
    if reader.fieldnames is None or not _REQUIRED_CSV_FIELDS.issubset(reader.fieldnames):
        missing = sorted(_REQUIRED_CSV_FIELDS.difference(reader.fieldnames or []))
        raise ShinnyEdbCaptureValidationError(f"EDB response missing required fields {missing}")
    rows: list[ActualContractRawBar] = []
    seen: set[str] = set()
    for record in reader:
        nano_text = str(record.get("datetime_nano", "")).strip()
        if not nano_text:
            raise ShinnyEdbCaptureValidationError("EDB response has an empty datetime_nano")
        try:
            nanos = int(nano_text)
        except ValueError as exc:
            raise ShinnyEdbCaptureValidationError("EDB datetime_nano is not an integer") from exc
        source_start = datetime.fromtimestamp(nanos / 1_000_000_000, ZoneInfo("Asia/Shanghai")).replace(tzinfo=None)
        if source_start.second != 0 or source_start.microsecond != 0:
            raise ShinnyEdbCaptureValidationError("EDB datetime_nano is not aligned to one minute")
        bar_time = (source_start + timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
        if bar_time in seen:
            raise ShinnyEdbCaptureValidationError(f"EDB response has duplicate bar_time {bar_time}")
        seen.add(bar_time)
        if bar_time[:10] not in {session_anchor_date, trading_day}:
            raise ShinnyEdbCaptureValidationError(
                f"EDB bar_time {bar_time} is outside declared session_anchor_date/trading_day"
            )
        open_ = _decimal(record.get("open"), "open")
        high = _decimal(record.get("high"), "high")
        low = _decimal(record.get("low"), "low")
        close = _decimal(record.get("close"), "close")
        volume = _decimal(record.get("volume"), "volume")
        open_interest = _decimal(record.get("close_oi"), "close_oi")
        if high < max(open_, close) or low > min(open_, close) or high < low:
            raise ShinnyEdbCaptureValidationError(f"EDB response has invalid OHLC at {bar_time}")
        if volume < 0 or open_interest < 0:
            raise ShinnyEdbCaptureValidationError(f"EDB response has negative volume/open_interest at {bar_time}")
        rows.append(
            ActualContractRawBar(
                product_code=product_code,
                exchange=exchange,
                actual_contract=actual_contract,
                bar_time=bar_time,
                session_anchor_date=session_anchor_date,
                trading_day=trading_day,
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=volume,
                open_interest=open_interest,
            )
        )
    if not rows:
        raise ShinnyEdbCaptureValidationError("EDB response is empty")
    if [row.bar_time for row in rows] != sorted(row.bar_time for row in rows):
        raise ShinnyEdbCaptureValidationError("EDB response is not ordered by bar_time")
    return tuple(rows)


def capture_future_actual_contract_1m(
    *,
    product_code: str,
    exchange: str,
    actual_contract: str,
    start_time: str,
    end_time: str,
    session_anchor_date: str,
    trading_day: str,
    captured_at: datetime | None = None,
    fetch_csv: Callable[[str, str, str, str], str] | None = None,
) -> ImmutableActualContractCapture:
    """Capture one exact actual-contract range and return an immutable artifact.

    HTTP 403 is classified as a historical-entitlement failure.  Empty results,
    response truncation (first/last bar differs from the requested bounds), bad
    fields, duplicate/out-of-order minutes, or invalid OHLCV/OI all fail closed.
    ``session_anchor_date`` and ``trading_day`` are explicit rather than
    inferred from wall-clock dates: a 21:xx night bar normally belongs to the
    following trading day.  ``fetch_csv`` exists only for provider-mock tests
    and controlled adapters.
    """

    product_code = product_code.strip()
    exchange = exchange.strip()
    actual_contract = actual_contract.strip()
    if not product_code or not exchange or not actual_contract:
        raise ShinnyEdbCaptureValidationError("product_code, exchange, and actual_contract must be non-empty")
    requested_start = _minute_time(start_time, "start_time")
    requested_end = _minute_time(end_time, "end_time")
    session_anchor_date = _date(session_anchor_date, "session_anchor_date")
    trading_day = _date(trading_day, "trading_day")
    if requested_end < requested_start:
        raise ShinnyEdbCaptureValidationError("end_time precedes start_time")
    fetch = fetch_csv or _fetch_exact_csv
    rows = _parse_rows(
        payload=fetch(exchange, actual_contract, requested_start, requested_end),
        product_code=product_code,
        exchange=exchange,
        actual_contract=actual_contract,
        session_anchor_date=session_anchor_date,
        trading_day=trading_day,
    )
    if rows[0].bar_time != requested_start or rows[-1].bar_time != requested_end:
        raise ShinnyEdbCaptureValidationError(
            f"EDB response is truncated: got {rows[0].bar_time}..{rows[-1].bar_time}, expected {requested_start}..{requested_end}"
        )
    capture_time = captured_at or datetime.now(timezone.utc)
    if capture_time.tzinfo is None:
        raise ShinnyEdbCaptureValidationError("captured_at must be timezone-aware")
    captured_at_text = capture_time.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    rowset_sha256 = _sha256(_canonical_json_bytes([asdict(row) for row in rows]))
    request = {
        "product_code": product_code,
        "exchange": exchange,
        "actual_contract": actual_contract,
        "symbol": f"{exchange}.{actual_contract}",
        "start_time": requested_start,
        "end_time": requested_end,
        "session_anchor_date": session_anchor_date,
        "trading_day": trading_day,
    }
    artifact_seed = {
        "schema_version": "shinny_edb_actual_contract_1m_capture_v1",
        "source_package": "shinny_edb",
        "source_package_version": SHINNY_EDB_PACKAGE_VERSION,
        "request": request,
        "captured_at": captured_at_text,
        "timestamp_contract": _TIMESTAMP_CONTRACT,
        "raw_fields": sorted({"product_code", "exchange", "actual_contract", "bar_time", "session_anchor_date", "trading_day", "open", "high", "low", "close", "volume", "open_interest"}),
        "rowset_sha256": rowset_sha256,
    }
    capture_id = f"shinny_edb-{_sha256(_canonical_json_bytes(artifact_seed))[:24]}"
    preliminary = ImmutableActualContractCapture(
        schema_version="shinny_edb_actual_contract_1m_capture_v1",
        source_package="shinny_edb",
        source_package_version=SHINNY_EDB_PACKAGE_VERSION,
        capture_id=capture_id,
        request=request,
        captured_at=captured_at_text,
        timestamp_contract=_TIMESTAMP_CONTRACT,
        raw_fields=tuple(sorted(artifact_seed["raw_fields"])),
        rowset_sha256=rowset_sha256,
        artifact_sha256="",
        rows=rows,
    )
    return ImmutableActualContractCapture(
        schema_version=preliminary.schema_version,
        source_package=preliminary.source_package,
        source_package_version=preliminary.source_package_version,
        capture_id=preliminary.capture_id,
        request=preliminary.request,
        captured_at=preliminary.captured_at,
        timestamp_contract=preliminary.timestamp_contract,
        raw_fields=preliminary.raw_fields,
        rowset_sha256=preliminary.rowset_sha256,
        artifact_sha256=_sha256(preliminary.canonical_artifact_bytes()),
        rows=preliminary.rows,
    )
