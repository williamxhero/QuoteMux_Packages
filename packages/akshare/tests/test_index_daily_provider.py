from __future__ import annotations

from datetime import datetime

import pandas as pd

from quotemux_packages.akshare import source


def test_index_daily_prefers_tencent_route_and_normalizes_chinese_columns(monkeypatch) -> None:
    calls: list[str] = []

    class FakeAk:
        def stock_zh_index_daily_tx(self, **kwargs):
            calls.append(f"tx:{kwargs['symbol']}")
            return pd.DataFrame(
                [
                    {"日期": "2026-10-28", "开盘": 5141.8, "最高": 5197.26, "最低": 5110.81, "收盘": 5177.51, "成交额": 934627653.0},
                ]
            )

        def stock_zh_index_daily(self, **kwargs):
            calls.append(f"sina:{kwargs['symbol']}")
            raise AssertionError("legacy endpoint must not be used when TX returns data")

    monkeypatch.setattr(source, "ak", FakeAk())
    monkeypatch.setattr(source, "_call_ak", lambda api_name, func, *args, **kwargs: func(*args, **kwargs))
    frame = source._fetch_index_daily_frame("000985.CSI", "1d", datetime(2026, 10, 28), datetime(2026, 10, 28))

    assert calls == ["tx:sh000985"]
    assert frame[["index_code", "trade_time", "open", "high", "low", "close", "amount"]].to_dict("records") == [
        {
            "index_code": "000985",
            "trade_time": pd.Timestamp("2026-10-28"),
            "open": 5141.8,
            "high": 5197.26,
            "low": 5110.81,
            "close": 5177.51,
            "amount": 934627653.0,
        }
    ]


def test_index_daily_falls_back_to_legacy_endpoint_when_tx_is_empty(monkeypatch) -> None:
    calls: list[str] = []

    class FakeAk:
        def stock_zh_index_daily_tx(self, **kwargs):
            calls.append("tx")
            return pd.DataFrame()

        def stock_zh_index_daily(self, **kwargs):
            calls.append("sina")
            return pd.DataFrame(
                [{"date": "2016-01-04", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10}]
            )

    monkeypatch.setattr(source, "ak", FakeAk())
    monkeypatch.setattr(source, "_call_ak", lambda api_name, func, *args, **kwargs: func(*args, **kwargs))
    frame = source._fetch_index_daily_frame("000001.SH", "1d", datetime(2016, 1, 4), datetime(2016, 1, 4))

    assert calls == ["tx", "sina"]
    assert len(frame) == 1
    assert frame.loc[0, "close"] == 1.5
