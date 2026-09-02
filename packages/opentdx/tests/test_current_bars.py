from __future__ import annotations

from types import SimpleNamespace

from quotemux_packages.opentdx import source


def test_current_bar_initializes_a_fresh_client_before_first_kline_request(monkeypatch) -> None:
    events: list[str] = []

    class _Client:
        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, *args):
            events.append("exit")

        def stock_kline(self, market, code, period, *, start, count, times, adjust):
            del market, code, period, start, count, times, adjust
            assert events == ["enter"]
            events.append("kline")
            return [
                {
                    "date_time": "2026-09-02 13:30:00",
                    "open": 1400.0, "high": 1401.0, "low": 1399.0, "close": 1400.5,
                    "vol": 1200, "amount": 1_680_600.0,
                }
            ]

    monkeypatch.setattr(source, "TdxClient", _Client)
    monkeypatch.setattr(source, "MARKET", SimpleNamespace(BJ="BJ", SH="SH", SZ="SZ"))
    monkeypatch.setattr(source, "PERIOD", SimpleNamespace(MINS="MINS", DAILY="DAILY", WEEKLY="WEEKLY", MONTHLY="MONTHLY"))
    monkeypatch.setattr(source, "ADJUST", SimpleNamespace(NONE="NONE", QFQ="QFQ", HFQ="HFQ"))
    monkeypatch.setattr(source, "_CLIENT_STATE", __import__("threading").local())
    source._client_factory.cache_clear()
    monkeypatch.setattr(source, "call_provider_api", lambda provider, api_name, invoke: invoke())

    result = source.get_current_stock_bars(["600519"], "2026-09-02T13:30:08+08:00")

    assert events == ["enter", "kline"]
    assert [(bar.code, bar.close, bar.volume, bar.amount) for bar in result.bars] == [("600519", 1400.5, 1200, 1_680_600.0)]
    assert result.attempts[0].outcome == "ok"
