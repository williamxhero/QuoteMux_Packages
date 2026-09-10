"""Pure, fail-closed derivation for repairing futures back-adjusted 1m gaps.

This module deliberately has no QuoteMux client or database dependency.  A
repair orchestrator must first capture immutable source artifacts and read the
frozen target rows, then hand those facts to :func:`derive_back_adjusted_1m`.
The returned rows are *staged* only; the owner of the publication transaction
is responsible for writing them after independently checking the frozen target
dataset version.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re
from typing import Iterable, Mapping


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SERIES_TYPE = "back_adjusted_continuous"
_REQUIRED_RAW_FIELDS = frozenset(
    {
        "product_code",
        "actual_contract",
        "bar_time",
        "session_anchor_date",
        "trading_day",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "open_interest",
    }
)
_REQUIRED_MAPPING_FIELDS = frozenset({"product_code", "mapping_effective_date", "actual_contract"})


class FuturesRepairValidationError(ValueError):
    """Raised when a source fact cannot prove a safe missing-row derivation."""


@dataclass(frozen=True)
class ImmutableArtifact:
    """Identity of a capture or mapping artifact, never a mutable provider query."""

    source: str
    capture_id: str
    version: str
    request_ranges: tuple[Mapping[str, object], ...]
    artifact_sha256: str
    rowset_sha256: str
    timestamp_contract: Mapping[str, object]
    record_fields: tuple[str, ...]


@dataclass(frozen=True)
class ActualContractMapping:
    product_code: str
    mapping_effective_date: str
    actual_contract: str


@dataclass(frozen=True)
class OffsetSegment:
    """One frozen product/day/offset interval that may not cross a roll."""

    product_code: str
    session_anchor_date: str
    trading_day: str
    mapping_effective_date: str
    start_time: str
    end_time: str
    actual_contract: str
    adjustment_offset: Decimal
    tick_size: Decimal


@dataclass(frozen=True)
class ExactGapRange:
    """A precomputed exact set of expected missing minute keys.

    Session construction is intentionally outside this module.  Passing the
    expected timestamps explicitly prevents a repair from silently filling a
    break, a holiday, or a pre-listing period.
    """

    product_code: str
    start_time: str
    end_time: str
    expected_bar_times: tuple[str, ...]


@dataclass(frozen=True)
class RawContractBar:
    product_code: str
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
class FrozenBackAdjustedBar:
    product_code: str
    bar_time: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    open_interest: Decimal | None
    adjustment_offset: Decimal


@dataclass(frozen=True)
class StagedBackAdjustedBar:
    product_code: str
    series_type: str
    bar_time: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    open_interest: Decimal
    adjustment_offset: Decimal
    source_key: str


@dataclass(frozen=True)
class DerivationResult:
    """Pure result for a later, single publication transaction."""

    staged_rows: tuple[StagedBackAdjustedBar, ...]
    derivation_manifest: Mapping[str, object]
    staged_artifact: Mapping[str, object]
    staged_artifact_bytes: bytes


def _text(value: object, field: str) -> str:
    result = str(value).strip()
    if not result:
        raise FuturesRepairValidationError(f"{field} must be non-empty")
    return result


def _sha256(value: object, field: str) -> str:
    result = _text(value, field)
    if not _SHA256_RE.fullmatch(result):
        raise FuturesRepairValidationError(f"{field} must be a lowercase SHA-256 hex digest")
    return result


def _decimal(value: object, field: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise FuturesRepairValidationError(f"{field} must be decimal") from exc
    if not result.is_finite():
        raise FuturesRepairValidationError(f"{field} must be finite")
    return result


def _minute_time(value: object, field: str) -> str:
    text = _text(value, field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise FuturesRepairValidationError(f"{field} must be an ISO datetime") from exc
    if parsed.tzinfo is not None:
        raise FuturesRepairValidationError(f"{field} must be timezone-naive in its declared timestamp contract")
    if parsed.second != 0 or parsed.microsecond != 0:
        raise FuturesRepairValidationError(f"{field} must be aligned to a one-minute boundary")
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def _date_from_time(bar_time: str) -> str:
    return bar_time[:10]


def _date(value: object, field: str) -> str:
    text = _text(value, field)
    try:
        return datetime.strptime(text, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError as exc:
        raise FuturesRepairValidationError(f"{field} must be YYYY-MM-DD") from exc


def _canonical_sha256(rows: Iterable[Mapping[str, object]]) -> str:
    return hashlib.sha256(_canonical_json_bytes(list(rows))).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _validate_artifact(artifact: ImmutableArtifact, label: str, required_fields: frozenset[str]) -> None:
    _text(artifact.source, f"{label}.source")
    _text(artifact.capture_id, f"{label}.capture_id")
    _text(artifact.version, f"{label}.version")
    if not artifact.request_ranges:
        raise FuturesRepairValidationError(f"{label}.request_ranges must be non-empty")
    _sha256(artifact.artifact_sha256, f"{label}.artifact_sha256")
    _sha256(artifact.rowset_sha256, f"{label}.rowset_sha256")
    if not artifact.timestamp_contract:
        raise FuturesRepairValidationError(f"{label}.timestamp_contract must be non-empty")
    if not required_fields.issubset(artifact.record_fields):
        missing = sorted(required_fields.difference(artifact.record_fields))
        raise FuturesRepairValidationError(f"{label}.record_fields missing {missing}")


def _validate_ohlc(open_: Decimal, high: Decimal, low: Decimal, close: Decimal, prefix: str) -> None:
    if high < max(open_, close) or low > min(open_, close) or high < low:
        raise FuturesRepairValidationError(f"{prefix} has invalid OHLC ordering")


def _assert_tick(value: Decimal, tick_size: Decimal, field: str) -> None:
    if tick_size <= 0:
        raise FuturesRepairValidationError("tick_size must be positive")
    if value % tick_size != 0:
        raise FuturesRepairValidationError(f"{field} is not aligned to tick size {tick_size}")


def _as_raw_bar(value: RawContractBar | Mapping[str, object]) -> RawContractBar:
    if isinstance(value, RawContractBar):
        value = asdict(value)
    return RawContractBar(
        product_code=_text(value.get("product_code"), "raw.product_code"),
        actual_contract=_text(value.get("actual_contract"), "raw.actual_contract"),
        bar_time=_minute_time(value.get("bar_time"), "raw.bar_time"),
        session_anchor_date=_date(value.get("session_anchor_date"), "raw.session_anchor_date"),
        trading_day=_date(value.get("trading_day"), "raw.trading_day"),
        open=_decimal(value.get("open"), "raw.open"),
        high=_decimal(value.get("high"), "raw.high"),
        low=_decimal(value.get("low"), "raw.low"),
        close=_decimal(value.get("close"), "raw.close"),
        volume=_decimal(value.get("volume"), "raw.volume"),
        open_interest=_decimal(value.get("open_interest"), "raw.open_interest"),
    )


def _as_frozen_bar(value: FrozenBackAdjustedBar | Mapping[str, object]) -> FrozenBackAdjustedBar:
    if isinstance(value, FrozenBackAdjustedBar):
        value = asdict(value)
    raw_oi = value.get("open_interest")
    return FrozenBackAdjustedBar(
        product_code=_text(value.get("product_code"), "frozen.product_code"),
        bar_time=_minute_time(value.get("bar_time"), "frozen.bar_time"),
        open=_decimal(value.get("open"), "frozen.open"),
        high=_decimal(value.get("high"), "frozen.high"),
        low=_decimal(value.get("low"), "frozen.low"),
        close=_decimal(value.get("close"), "frozen.close"),
        volume=_decimal(value.get("volume"), "frozen.volume"),
        open_interest=None if raw_oi is None else _decimal(raw_oi, "frozen.open_interest"),
        adjustment_offset=_decimal(value.get("adjustment_offset"), "frozen.adjustment_offset"),
    )


def _as_mapping(value: ActualContractMapping | Mapping[str, object]) -> ActualContractMapping:
    if isinstance(value, ActualContractMapping):
        value = asdict(value)
    return ActualContractMapping(
        product_code=_text(value.get("product_code"), "mapping.product_code"),
        mapping_effective_date=_date(value.get("mapping_effective_date"), "mapping.mapping_effective_date"),
        actual_contract=_text(value.get("actual_contract"), "mapping.actual_contract"),
    )


def _as_segment(value: OffsetSegment | Mapping[str, object]) -> OffsetSegment:
    if isinstance(value, OffsetSegment):
        value = asdict(value)
    return OffsetSegment(
        product_code=_text(value.get("product_code"), "segment.product_code"),
        session_anchor_date=_date(value.get("session_anchor_date"), "segment.session_anchor_date"),
        trading_day=_date(value.get("trading_day"), "segment.trading_day"),
        mapping_effective_date=_date(value.get("mapping_effective_date"), "segment.mapping_effective_date"),
        start_time=_minute_time(value.get("start_time"), "segment.start_time"),
        end_time=_minute_time(value.get("end_time"), "segment.end_time"),
        actual_contract=_text(value.get("actual_contract"), "segment.actual_contract"),
        adjustment_offset=_decimal(value.get("adjustment_offset"), "segment.adjustment_offset"),
        tick_size=_decimal(value.get("tick_size"), "segment.tick_size"),
    )


def _as_gap(value: ExactGapRange | Mapping[str, object]) -> ExactGapRange:
    if isinstance(value, ExactGapRange):
        value = asdict(value)
    expected = value.get("expected_bar_times")
    if not isinstance(expected, (list, tuple)):
        raise FuturesRepairValidationError("gap.expected_bar_times must be a non-empty list")
    return ExactGapRange(
        product_code=_text(value.get("product_code"), "gap.product_code"),
        start_time=_minute_time(value.get("start_time"), "gap.start_time"),
        end_time=_minute_time(value.get("end_time"), "gap.end_time"),
        expected_bar_times=tuple(_minute_time(item, "gap.expected_bar_times") for item in expected),
    )


def derive_back_adjusted_1m(
    *,
    source_capture: ImmutableArtifact,
    contract_mapping_capture: ImmutableArtifact,
    frozen_dataset_version: str,
    gap_ranges_artifact_sha256: str,
    ruleset_sha256: str,
    raw_contract_bars: Iterable[RawContractBar | Mapping[str, object]],
    frozen_back_adjusted_bars: Iterable[FrozenBackAdjustedBar | Mapping[str, object]],
    actual_contract_mappings: Iterable[ActualContractMapping | Mapping[str, object]],
    offset_segments: Iterable[OffsetSegment | Mapping[str, object]],
    exact_gap_ranges: Iterable[ExactGapRange | Mapping[str, object]],
) -> DerivationResult:
    """Derive only proven missing rows from immutable, source-native artifacts.

    Every output key must be listed by an explicit gap range and contained in a
    single mapped contract/offset segment.  All existing target rows in that
    segment must be present in the raw capture and prove
    ``raw OHLC == adjusted OHLC + adjustment_offset`` at the product tick.
    The function never queries a provider or writes a database.
    """

    _validate_artifact(source_capture, "source_capture", _REQUIRED_RAW_FIELDS)
    _validate_artifact(contract_mapping_capture, "contract_mapping_capture", _REQUIRED_MAPPING_FIELDS)
    frozen_dataset_version = _text(frozen_dataset_version, "frozen_dataset_version")
    gap_ranges_artifact_sha256 = _sha256(gap_ranges_artifact_sha256, "gap_ranges_artifact_sha256")
    ruleset_sha256 = _sha256(ruleset_sha256, "ruleset_sha256")

    raw = tuple(_as_raw_bar(item) for item in raw_contract_bars)
    frozen = tuple(_as_frozen_bar(item) for item in frozen_back_adjusted_bars)
    mappings = tuple(_as_mapping(item) for item in actual_contract_mappings)
    segments = tuple(_as_segment(item) for item in offset_segments)
    gaps = tuple(_as_gap(item) for item in exact_gap_ranges)
    if not raw or not frozen or not mappings or not segments or not gaps:
        raise FuturesRepairValidationError("raw bars, frozen bars, mappings, segments, and gaps must all be non-empty")

    raw_payload = [asdict(item) for item in raw]
    if _canonical_sha256(raw_payload) != source_capture.rowset_sha256:
        raise FuturesRepairValidationError("source_capture.rowset_sha256 does not identify the supplied raw rows")
    mapping_payload = [asdict(item) for item in mappings]
    if _canonical_sha256(mapping_payload) != contract_mapping_capture.rowset_sha256:
        raise FuturesRepairValidationError("contract_mapping_capture.rowset_sha256 does not identify the supplied mappings")

    raw_by_key: dict[tuple[str, str], RawContractBar] = {}
    for item in raw:
        key = (item.product_code, item.bar_time)
        if key in raw_by_key:
            raise FuturesRepairValidationError(f"duplicate raw bar {key}")
        if item.volume < 0 or item.open_interest < 0:
            raise FuturesRepairValidationError(f"raw bar {key} has negative volume/open_interest")
        _validate_ohlc(item.open, item.high, item.low, item.close, f"raw bar {key}")
        raw_by_key[key] = item

    frozen_by_key: dict[tuple[str, str], FrozenBackAdjustedBar] = {}
    for item in frozen:
        key = (item.product_code, item.bar_time)
        if key in frozen_by_key:
            raise FuturesRepairValidationError(f"duplicate frozen bar {key}")
        if item.volume < 0 or (item.open_interest is not None and item.open_interest < 0):
            raise FuturesRepairValidationError(f"frozen bar {key} has negative volume/open_interest")
        _validate_ohlc(item.open, item.high, item.low, item.close, f"frozen bar {key}")
        frozen_by_key[key] = item

    mapping_by_effective_date: dict[tuple[str, str], ActualContractMapping] = {}
    for item in mappings:
        key = (item.product_code, item.mapping_effective_date)
        if key in mapping_by_effective_date:
            raise FuturesRepairValidationError(f"duplicate actual-contract mapping {key}")
        mapping_by_effective_date[key] = item

    segments_by_day: dict[tuple[str, str], list[OffsetSegment]] = {}
    for item in segments:
        if _date_from_time(item.start_time) not in {item.session_anchor_date, item.trading_day} or _date_from_time(item.end_time) not in {item.session_anchor_date, item.trading_day}:
            raise FuturesRepairValidationError("offset segment is outside its explicit session_anchor_date/trading_day")
        if item.end_time < item.start_time:
            raise FuturesRepairValidationError("offset segment end_time precedes start_time")
        if item.tick_size <= 0:
            raise FuturesRepairValidationError("offset segment tick_size must be positive")
        if item.trading_day != item.mapping_effective_date:
            raise FuturesRepairValidationError("offset segment trading_day must equal its mapping_effective_date")
        mapping = mapping_by_effective_date.get((item.product_code, item.mapping_effective_date))
        if mapping is None or mapping.actual_contract != item.actual_contract:
            raise FuturesRepairValidationError(
                f"offset segment {(item.product_code, item.mapping_effective_date)} lacks the same exact actual-contract mapping"
            )
        segments_by_day.setdefault((item.product_code, item.mapping_effective_date), []).append(item)
    for key, day_segments in segments_by_day.items():
        previous_end: str | None = None
        for item in sorted(day_segments, key=lambda candidate: candidate.start_time):
            if previous_end is not None and item.start_time <= previous_end:
                raise FuturesRepairValidationError(f"overlapping offset segments for {key}")
            previous_end = item.end_time

    def segment_for(product_code: str, bar_time: str) -> OffsetSegment:
        matches = [
            item
            for item in segments
            if item.product_code == product_code
            and item.start_time <= bar_time <= item.end_time
        ]
        if len(matches) != 1:
            raise FuturesRepairValidationError(
                f"bar {(product_code, bar_time)} is not in exactly one frozen offset segment"
            )
        return matches[0]

    expected_gap_keys: set[tuple[str, str]] = set()
    for gap in gaps:
        if not gap.expected_bar_times:
            raise FuturesRepairValidationError("exact gap range must list expected minute keys")
        if tuple(sorted(gap.expected_bar_times)) != gap.expected_bar_times:
            raise FuturesRepairValidationError("exact gap range expected_bar_times must be sorted")
        if len(set(gap.expected_bar_times)) != len(gap.expected_bar_times):
            raise FuturesRepairValidationError("exact gap range expected_bar_times contains duplicates")
        if gap.expected_bar_times[0] != gap.start_time or gap.expected_bar_times[-1] != gap.end_time:
            raise FuturesRepairValidationError("exact gap range bounds must equal its first/last expected minute")
        for bar_time in gap.expected_bar_times:
            if not gap.start_time <= bar_time <= gap.end_time:
                raise FuturesRepairValidationError("gap minute lies outside its declared bounds")
            key = (gap.product_code, bar_time)
            if key in expected_gap_keys:
                raise FuturesRepairValidationError(f"duplicate expected gap key {key}")
            if key in frozen_by_key:
                raise FuturesRepairValidationError(f"expected gap key already exists in frozen target {key}")
            expected_gap_keys.add(key)
            segment_for(*key)

    # Prove every frozen overlap in each segment against the same raw contract.
    overlap_volume_oi_mismatches: list[dict[str, object]] = []
    proof_count = 0
    for key, target in frozen_by_key.items():
        segment = segment_for(*key)
        source = raw_by_key.get(key)
        if source is None:
            raise FuturesRepairValidationError(f"missing raw overlap needed to prove segment {key}")
        if source.actual_contract != segment.actual_contract:
            raise FuturesRepairValidationError(f"raw actual contract does not match mapping at {key}")
        if source.session_anchor_date != segment.session_anchor_date or source.trading_day != segment.trading_day:
            raise FuturesRepairValidationError(f"raw session_anchor_date/trading_day does not match frozen segment at {key}")
        for name in ("open", "high", "low", "close"):
            raw_price = getattr(source, name)
            target_price = getattr(target, name)
            _assert_tick(raw_price, segment.tick_size, f"raw {name} {key}")
            _assert_tick(target_price, segment.tick_size, f"target {name} {key}")
            if raw_price - target_price != segment.adjustment_offset:
                raise FuturesRepairValidationError(
                    f"raw {name} does not equal target {name} plus frozen adjustment_offset at {key}"
                )
        if source.volume != target.volume or (target.open_interest is not None and source.open_interest != target.open_interest):
            # Source revisions must never overwrite frozen rows.  They are
            # reported in the manifest while the staged output contains gap keys
            # only and preserves source-native volume/OI for those new keys.
            overlap_volume_oi_mismatches.append({"product_code": key[0], "bar_time": key[1]})
        proof_count += 1

    staged: list[StagedBackAdjustedBar] = []
    for key in sorted(expected_gap_keys):
        source = raw_by_key.get(key)
        if source is None:
            raise FuturesRepairValidationError(f"source capture does not contain expected gap key {key}")
        segment = segment_for(*key)
        mapping = mapping_by_effective_date[(key[0], segment.mapping_effective_date)]
        if source.actual_contract != mapping.actual_contract or source.actual_contract != segment.actual_contract:
            raise FuturesRepairValidationError(f"source actual contract does not match frozen mapping at {key}")
        if source.session_anchor_date != segment.session_anchor_date or source.trading_day != segment.trading_day:
            raise FuturesRepairValidationError(f"source session_anchor_date/trading_day does not match frozen segment at {key}")
        adjusted = tuple(getattr(source, name) - segment.adjustment_offset for name in ("open", "high", "low", "close"))
        for name, price in zip(("open", "high", "low", "close"), adjusted, strict=True):
            _assert_tick(price, segment.tick_size, f"derived {name} {key}")
        _validate_ohlc(*adjusted, f"derived bar {key}")
        staged.append(
            StagedBackAdjustedBar(
                product_code=key[0],
                series_type=_SERIES_TYPE,
                bar_time=key[1],
                open=adjusted[0],
                high=adjusted[1],
                low=adjusted[2],
                close=adjusted[3],
                volume=source.volume,
                open_interest=source.open_interest,
                adjustment_offset=segment.adjustment_offset,
                source_key=f"derived_core:{source_capture.source}:{source_capture.capture_id}:{source.actual_contract}",
            )
        )
    if not staged:
        raise FuturesRepairValidationError("derivation produced zero staged rows")

    exact_missing_keys = [
        {"product_code": product_code, "bar_time": bar_time}
        for product_code, bar_time in sorted(expected_gap_keys)
    ]
    staged_artifact = {
        "schema_version": "futures_back_adjusted_1m_staged_artifact_v1",
        "series_type": _SERIES_TYPE,
        "frozen_dataset_version": frozen_dataset_version,
        "ruleset_sha256": ruleset_sha256,
        "gap_ranges_artifact_sha256": gap_ranges_artifact_sha256,
        "source_capture": asdict(source_capture),
        "contract_mapping_capture": asdict(contract_mapping_capture),
        "exact_missing_keys": exact_missing_keys,
        "rows": [asdict(item) for item in staged],
    }
    staged_artifact_bytes = _canonical_json_bytes(staged_artifact)
    staged_artifact_sha256 = hashlib.sha256(staged_artifact_bytes).hexdigest()
    manifest = {
        "schema_version": "futures_back_adjusted_1m_derivation_v1",
        "series_type": _SERIES_TYPE,
        "frozen_dataset_version": frozen_dataset_version,
        "ruleset_sha256": ruleset_sha256,
        "gap_ranges_artifact_sha256": gap_ranges_artifact_sha256,
        "source_capture": asdict(source_capture),
        "contract_mapping_capture": asdict(contract_mapping_capture),
        "row_count": len(staged),
        "staged_rowset_sha256": _canonical_sha256(asdict(item) for item in staged),
        "staged_artifact_sha256": staged_artifact_sha256,
        "exact_missing_keys": exact_missing_keys,
        "overlap_proof_row_count": proof_count,
        "overlap_volume_oi_mismatch_count": len(overlap_volume_oi_mismatches),
        "overlap_volume_oi_mismatches": overlap_volume_oi_mismatches,
        "writes_database": False,
    }
    return DerivationResult(
        staged_rows=tuple(staged),
        derivation_manifest=manifest,
        staged_artifact=staged_artifact,
        staged_artifact_bytes=staged_artifact_bytes,
    )
