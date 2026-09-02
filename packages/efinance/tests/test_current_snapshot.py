from __future__ import annotations

import pandas as pd

from quotemux_packages.efinance import source


def test_current_snapshot_reads_realtime_price_without_using_minute_history(monkeypatch) -> None:
    realtime = pd.DataFrame([{"代码": "600519", "最新价": 1400.7, "更新时间": "2026-09-02 13:30:09"}])

    class _Stock:
        @staticmethod
        def get_realtime_quotes(arg):
            assert arg is None
            return realtime

    monkeypatch.setattr(source, "ef", type("EF", (), {"stock": _Stock})())
    monkeypatch.setattr(source, "call_provider_api", lambda provider, api_name, func, *args, **kwargs: func(*args, **kwargs))

    result = source.get_current_stock_price_snapshots(["600519"], "2026-09-02T13:30:10+08:00")

    assert [(item.code, item.price, item.source_time) for item in result] == [("600519", 1400.7, "2026-09-02T13:30:09+08:00")]
