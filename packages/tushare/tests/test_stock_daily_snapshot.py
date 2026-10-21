from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from quotemux_packages.tushare import source


def _row(ts_code: str = "600000.SH", trade_date: str = "20261019", **overrides):
    values = {
        "ts_code": ts_code,
        "trade_date": trade_date,
        "open": 10.0,
        "high": 11.0,
        "low": 9.0,
        "close": 10.5,
        "pre_close": 10.0,
        "change": 0.5,
        "pct_chg": 5.0,
        "vol": 12345.0,
        "amount": 67890.0,
    }
    values.update(overrides)
    return values


def _frame(rows) -> pd.DataFrame:
    return pd.DataFrame(rows)


@pytest.fixture
def cache_path(monkeypatch, tmp_path):
    path = tmp_path / "daily-snapshot.parquet"
    monkeypatch.setattr(source, "build_cache_path", lambda *_args, **_kwargs: path)
    return path


@pytest.fixture
def provider(monkeypatch):
    class Provider:
        def daily(self, **_kwargs):
            raise AssertionError("call_tushare_api seam must be used")

    monkeypatch.setattr(source, "get_ts_pro", lambda: Provider())


def _capture_daily(monkeypatch, frame: pd.DataFrame):
    calls = []

    def fake_call(api_name, func, **kwargs):
        calls.append((api_name, func, kwargs))
        return frame

    monkeypatch.setattr(source, "call_tushare_api", fake_call)
    return calls


def test_full_snapshot_requests_documented_fields_and_converts_units(
    monkeypatch, provider, cache_path
):
    frame = _frame(
        [
            _row("600000.SH", open=10, high=11, low=9, close=10.5, pre_close=10, change=0.5, pct_chg=5, vol=12345, amount=67890),
            _row("000001.SZ", open=12, high=13, low=11, close=12.5, pre_close=12, change=0.5, pct_chg=4.1667, vol=2, amount=3),
            _row("920000.BJ", open=20, high=21, low=19, close=20.5, pre_close=20, change=0.5, pct_chg=2.5, vol=4, amount=5),
        ]
    )
    calls = _capture_daily(monkeypatch, frame)

    items = source.get_stock_daily_snapshot_full("2026-10-19")

    assert len(calls) == 1
    api_name, _func, kwargs = calls[0]
    assert api_name == "daily"
    assert kwargs == {
        "trade_date": "20261019",
        "fields": "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount",
    }
    assert [item.code for item in items] == ["000001", "600000", "920000"]
    by_code = {item.code: item for item in items}
    assert by_code["600000"].volume == 12345 * 100
    assert by_code["600000"].amount == 67890 * 1000
    assert by_code["600000"].trade_time == "2026-10-19"
    assert by_code["600000"].freq == "1d"
    assert by_code["600000"].adjust == "none"
    assert by_code["600000"].pre_close == 10.0
    assert by_code["600000"].pct_chg == 5.0
    assert by_code["600000"].is_suspended is False
    assert by_code["600000"].is_st is False
    assert all(item.trade_time == "2026-10-19" for item in items)


def test_compact_date_is_sent_as_yyyymmdd(monkeypatch, provider, cache_path):
    frame = _frame([_row()])
    calls = _capture_daily(monkeypatch, frame)

    items = source.get_stock_daily_snapshot_full("20261019")

    assert calls[0][2]["trade_date"] == "20261019"
    assert items[0].trade_time == "2026-10-19"


def test_date_object_is_accepted(monkeypatch, provider, cache_path):
    calls = _capture_daily(monkeypatch, _frame([_row()]))

    items = source.get_stock_daily_snapshot_full(date(2026, 10, 19))

    assert calls[0][2]["trade_date"] == "20261019"
    assert items[0].trade_time == "2026-10-19"


@pytest.mark.parametrize(
    "trade_date",
    ["", "2026/10/19", "2026-02-30", "2026-2-3", "2026101", None, "2026-10-19T00:00:00"],
)
def test_invalid_or_loose_dates_are_rejected_before_provider_io(monkeypatch, trade_date):
    monkeypatch.setattr(
        source,
        "build_cache_path",
        lambda *_args, **_kwargs: pytest.fail("must reject invalid date before cache/provider IO"),
    )
    monkeypatch.setattr(
        source,
        "get_ts_pro",
        lambda: pytest.fail("must reject invalid date before provider IO"),
    )

    with pytest.raises(ValueError, match="exact"):
        source.get_stock_daily_snapshot_full(trade_date)


def test_response_row_for_another_date_is_rejected(monkeypatch, provider, cache_path):
    _capture_daily(monkeypatch, _frame([_row(trade_date="20261018")]))

    with pytest.raises(RuntimeError, match="returned 2026-10-18"):
        source.get_stock_daily_snapshot_full("2026-10-19")
    assert not cache_path.exists()


def test_identity_market_mismatch_and_index_collisions_are_not_published(
    monkeypatch, provider, cache_path
):
    frame = _frame(
        [
            # SSE Composite uses the same six digits as Ping An Bank.
            _row("000001.SH"),
            _row("000001.SZ"),
            _row("399001.SZ"),
            _row("899050.BJ"),
        ]
    )
    _capture_daily(monkeypatch, frame)

    items = source.get_stock_daily_snapshot_full("2026-10-19")

    assert [item.code for item in items] == ["000001"]


def test_duplicate_identity_is_rejected(monkeypatch, provider, cache_path):
    _capture_daily(monkeypatch, _frame([_row("600000.SH"), _row("600000.SH")]))

    with pytest.raises(RuntimeError, match="duplicate identity"):
        source.get_stock_daily_snapshot_full("2026-10-19")


def test_missing_required_column_is_rejected(monkeypatch, provider, cache_path):
    row = _row()
    row.pop("amount")
    _capture_daily(monkeypatch, _frame([row]))

    with pytest.raises(RuntimeError, match="missing columns: amount"):
        source.get_stock_daily_snapshot_full("2026-10-19")


@pytest.mark.parametrize("field_name", ["open", "close", "pre_close", "pct_chg", "vol", "amount"])
def test_missing_or_nonfinite_required_value_is_rejected(
    monkeypatch, provider, cache_path, field_name
):
    _capture_daily(
        monkeypatch,
        _frame([_row(**{field_name: None})]),
    )

    with pytest.raises(RuntimeError, match=field_name):
        source.get_stock_daily_snapshot_full("2026-10-19")


def test_zero_volume_row_is_real_and_not_marked_suspended(monkeypatch, provider, cache_path):
    _capture_daily(
        monkeypatch,
        _frame(
            [
                _row(
                    open=10,
                    high=10,
                    low=10,
                    close=10,
                    pre_close=10,
                    change=0,
                    pct_chg=0,
                    vol=0,
                    amount=0,
                )
            ]
        ),
    )

    items = source.get_stock_daily_snapshot_full("2026-10-19")

    assert len(items) == 1
    assert items[0].volume == 0
    assert items[0].amount == 0
    assert items[0].is_suspended is False


def test_absent_codes_do_not_get_synthetic_suspension_rows(monkeypatch, provider, cache_path):
    _capture_daily(monkeypatch, _frame([_row("600000.SH")]))

    items = source.get_stock_daily_snapshot_full("2026-10-19")

    assert [item.code for item in items] == ["600000"]
    assert all(item.is_suspended is False for item in items)


def test_cache_is_reused_and_invalid_cache_is_refetched(monkeypatch, provider, cache_path):
    calls = _capture_daily(monkeypatch, _frame([_row()]))

    first = source.get_stock_daily_snapshot_full("2026-10-19")
    second = source.get_stock_daily_snapshot_full("2026-10-19")
    assert [item.code for item in first] == ["600000"]
    assert [item.code for item in second] == ["600000"]
    assert len(calls) == 1

    _frame([_row(trade_date="20261018")]).to_parquet(cache_path, index=False)
    third = source.get_stock_daily_snapshot_full("2026-10-19")

    assert [item.code for item in third] == ["600000"]
    assert len(calls) == 2


def test_truncation_at_query_cap_is_rejected(monkeypatch, provider, cache_path):
    monkeypatch.setattr(source, "TUSHARE_DAILY_SNAPSHOT_MAX_ROWS", 2)
    _capture_daily(monkeypatch, _frame([_row("600000.SH"), _row("000001.SZ")]))

    with pytest.raises(RuntimeError, match="query cap"):
        source.get_stock_daily_snapshot_full("2026-10-19")


def test_provider_unavailable_returns_empty(monkeypatch, cache_path):
    monkeypatch.setattr(source, "get_ts_pro", lambda: None)

    assert source.get_stock_daily_snapshot_full("2026-10-19") == []


def test_api_failure_returns_empty_for_fallback(monkeypatch, provider, cache_path):
    def fail(*_args, **_kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(source, "call_tushare_api", fail)

    assert source.get_stock_daily_snapshot_full("2026-10-19") == []


def test_timeout_is_not_swallowed(monkeypatch, provider, cache_path):
    def timeout(*_args, **_kwargs):
        raise TimeoutError("quota wait exhausted")

    monkeypatch.setattr(source, "call_tushare_api", timeout)

    with pytest.raises(TimeoutError, match="quota wait"):
        source.get_stock_daily_snapshot_full("2026-10-19")


def test_manifest_advertises_exact_date_snapshot_with_units():
    manifest = json.loads(
        (Path(__file__).parents[1] / "quotemux_package.json").read_text(encoding="utf-8-sig")
    )
    snapshot = [
        cap
        for cap in manifest["capabilities"]
        if cap["capability_id"] == "stocks.quotes.daily_snapshot"
    ]
    assert len(snapshot) == 1
    assert snapshot[0]["handler_name"] == "get_stock_daily_snapshot_full"
    assert snapshot[0]["support_level"] == "native"
    assert manifest["handler_targets"]["get_stock_daily_snapshot_full"] == (
        "quotemux_packages.tushare.source:get_stock_daily_snapshot_full"
    )
    notes = snapshot[0]["notes"]
    assert "vol*100" in notes
    assert "amount*1000" in notes
    assert "not synthesized" in notes
    assert "adj_factor" in notes

def test_malformed_ts_code_is_rejected(monkeypatch, provider, cache_path):
    _capture_daily(monkeypatch, _frame([_row("600000")]))

    with pytest.raises(RuntimeError, match="invalid ts_code"):
        source.get_stock_daily_snapshot_full("2026-10-19")


def test_legacy_alias_uses_the_same_strict_full_snapshot(monkeypatch, provider, cache_path):
    calls = _capture_daily(monkeypatch, _frame([_row()]))

    items = source.get_stock_daily_snapshot("2026-10-19")

    assert [item.code for item in items] == ["600000"]
    assert calls[0][2]["trade_date"] == "20261019"
