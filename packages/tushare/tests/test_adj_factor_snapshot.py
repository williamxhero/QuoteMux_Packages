from __future__ import annotations

import pandas as pd

from quotemux_packages.tushare import source


def test_adj_factor_snapshot_keeps_provider_factors(monkeypatch) -> None:
    class Provider:
        def adj_factor(self, **_kwargs):
            raise AssertionError("call_tushare_api seam must be used")

    monkeypatch.setattr(source, "get_ts_pro", lambda: Provider())
    monkeypatch.setattr(
        source,
        "call_tushare_api",
        lambda api_name, func, **kwargs: pd.DataFrame(
            [
                {"ts_code": "600000.SH", "trade_date": "20260721", "adj_factor": 17.2},
                {"ts_code": "000001.SZ", "trade_date": "20260721", "adj_factor": 123.4},
            ]
        ),
    )

    items = source.get_adj_factor_snapshot("2026-07-21")

    assert [(item.code, item.trade_date, item.adj_factor) for item in items] == [
        ("000001", "20260721", 123.4),
        ("600000", "20260721", 17.2),
    ]


def test_ths_daily_snapshot_keeps_provider_ohlcva_without_derivation(monkeypatch) -> None:
    class Provider:
        def ths_daily(self, **_kwargs):
            raise AssertionError("call_tushare_api seam must be used")

    monkeypatch.setattr(source, "get_ts_pro", lambda: Provider())
    monkeypatch.setattr(
        source,
        "call_tushare_api",
        lambda api_name, func, **kwargs: pd.DataFrame(
            [
                {
                    "ts_code": "885311.TI", "trade_date": "20260721",
                    "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5,
                    "pre_close": 9.8, "change": 0.7, "pct_change": 7.14,
                    "vol": 100.0, "amount": None,
                }
            ]
        ),
    )

    items = source.get_ths_daily_snapshot("2026-07-21")

    assert len(items) == 1
    assert items[0].board_code == "885311"
    assert items[0].trade_time == "2026-07-21"
    assert items[0].open == 10.0
    assert items[0].volume == 100.0
    assert items[0].amount is None
