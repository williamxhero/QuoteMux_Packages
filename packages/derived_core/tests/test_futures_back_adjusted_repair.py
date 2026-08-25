from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal
import hashlib
import json

import pytest

from quotemux_packages.derived_core.futures_back_adjusted_repair import (
    ActualContractMapping,
    ExactGapRange,
    FrozenBackAdjustedBar,
    FuturesRepairValidationError,
    ImmutableArtifact,
    OffsetSegment,
    RawContractBar,
    derive_back_adjusted_1m,
)


def _digest(rows: object) -> str:
    text = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _artifact(*, source: str, capture_id: str, rows: list[object], record_fields: tuple[str, ...]) -> ImmutableArtifact:
    return ImmutableArtifact(
        source=source,
        capture_id=capture_id,
        version="2026-08-25",
        request_ranges=({"start_time": "2026-02-02 09:01:00", "end_time": "2026-02-02 09:03:00"},),
        artifact_sha256="a" * 64,
        rowset_sha256=_digest(rows),
        timestamp_contract={"timezone": "Asia/Shanghai", "frequency": "1m", "bar_timestamp": "minute_start"},
        record_fields=record_fields,
    )


def _raw(bar_time: str, *, contract: str = "AG2604.SHF", close: str = "6000") -> RawContractBar:
    close_value = Decimal(close)
    return RawContractBar(
        product_code="ag",
        actual_contract=contract,
        bar_time=bar_time,
        open=close_value - Decimal("1"),
        high=close_value + Decimal("2"),
        low=close_value - Decimal("3"),
        close=close_value,
        volume=Decimal("12"),
        open_interest=Decimal("34"),
    )


def _frozen(bar_time: str, *, close: str = "5000") -> FrozenBackAdjustedBar:
    close_value = Decimal(close)
    return FrozenBackAdjustedBar(
        product_code="ag",
        bar_time=bar_time,
        open=close_value - Decimal("1"),
        high=close_value + Decimal("2"),
        low=close_value - Decimal("3"),
        close=close_value,
        volume=Decimal("12"),
        open_interest=Decimal("34"),
        adjustment_offset=Decimal("1000"),
    )


def _derive(raw: list[RawContractBar], frozen: list[FrozenBackAdjustedBar] | None = None, gaps: tuple[str, ...] = ("2026-02-02 09:02:00",)):
    frozen = frozen or [_frozen("2026-02-02 09:01:00"), _frozen("2026-02-02 09:03:00")]
    mappings = [ActualContractMapping("ag", "2026-02-02", "AG2604.SHF")]
    return derive_back_adjusted_1m(
        source_capture=_artifact(
            source="formal_source",
            capture_id="capture-001",
            rows=[asdict(item) for item in raw],
            record_fields=("product_code", "actual_contract", "bar_time", "open", "high", "low", "close", "volume", "open_interest"),
        ),
        contract_mapping_capture=_artifact(
            source="tushare",
            capture_id="mapping-001",
            rows=[asdict(item) for item in mappings],
            record_fields=("product_code", "trade_date", "actual_contract"),
        ),
        frozen_dataset_version="mhd-v1-frozen",
        gap_ranges_artifact_sha256="b" * 64,
        ruleset_sha256="c" * 64,
        raw_contract_bars=raw,
        frozen_back_adjusted_bars=frozen,
        actual_contract_mappings=mappings,
        offset_segments=[
            OffsetSegment(
                product_code="ag",
                trade_date="2026-02-02",
                start_time="2026-02-02 09:01:00",
                end_time="2026-02-02 09:03:00",
                actual_contract="AG2604.SHF",
                adjustment_offset=Decimal("1000"),
                tick_size=Decimal("1"),
            )
        ],
        exact_gap_ranges=[
            ExactGapRange(
                product_code="ag",
                start_time=gaps[0],
                end_time=gaps[-1],
                expected_bar_times=gaps,
            )
        ],
    )


def test_derives_only_explicit_gap_keys_with_immutable_lineage() -> None:
    result = _derive([_raw("2026-02-02 09:01:00"), _raw("2026-02-02 09:02:00"), _raw("2026-02-02 09:03:00")])

    assert [(item.product_code, item.bar_time, item.close, item.volume, item.open_interest) for item in result.staged_rows] == [
        ("ag", "2026-02-02 09:02:00", Decimal("5000"), Decimal("12"), Decimal("34"))
    ]
    assert result.staged_rows[0].series_type == "back_adjusted_continuous"
    assert result.staged_rows[0].source_key == "derived_core:formal_source:capture-001:AG2604.SHF"
    assert result.derivation_manifest["frozen_dataset_version"] == "mhd-v1-frozen"
    assert result.derivation_manifest["row_count"] == 1
    assert result.derivation_manifest["overlap_proof_row_count"] == 2
    assert result.derivation_manifest["writes_database"] is False


def test_fails_closed_when_any_frozen_overlap_is_not_in_the_capture() -> None:
    with pytest.raises(FuturesRepairValidationError, match="missing raw overlap"):
        _derive([_raw("2026-02-02 09:01:00"), _raw("2026-02-02 09:02:00")])


def test_fails_closed_when_raw_contract_does_not_match_daily_mapping() -> None:
    raw = [_raw("2026-02-02 09:01:00"), _raw("2026-02-02 09:02:00", contract="AG2605.SHF"), _raw("2026-02-02 09:03:00")]
    with pytest.raises(FuturesRepairValidationError, match="actual contract"):
        _derive(raw)


def test_fails_closed_when_offset_proof_is_not_exact_at_tick_size() -> None:
    raw = [_raw("2026-02-02 09:01:00"), _raw("2026-02-02 09:02:00"), _raw("2026-02-02 09:03:00", close="6001")]
    with pytest.raises(FuturesRepairValidationError, match="adjustment_offset"):
        _derive(raw)


def test_fails_closed_when_gap_contains_duplicate_expected_minutes() -> None:
    raw = [_raw("2026-02-02 09:01:00"), _raw("2026-02-02 09:02:00"), _raw("2026-02-02 09:03:00")]
    with pytest.raises(FuturesRepairValidationError, match="duplicates"):
        _derive(raw, gaps=("2026-02-02 09:02:00", "2026-02-02 09:02:00"))


def test_never_overwrites_overlap_when_source_volume_has_a_revision() -> None:
    raw = [_raw("2026-02-02 09:01:00"), _raw("2026-02-02 09:02:00"), _raw("2026-02-02 09:03:00")]
    raw[0] = RawContractBar(**{**asdict(raw[0]), "volume": Decimal("999")})
    result = _derive(raw)

    assert [item.bar_time for item in result.staged_rows] == ["2026-02-02 09:02:00"]
    assert result.derivation_manifest["overlap_volume_oi_mismatch_count"] == 1
