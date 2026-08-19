from __future__ import annotations

from datetime import datetime

import pandas as pd

from quotemux_packages.mootdx import source


def _bar(at: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "datetime": at,
                "open": 10.0,
                "high": 10.2,
                "low": 9.9,
                "close": 10.1,
                "vol": 100,
                "amount": 1010.0,
            }
        ]
    )


class _CappedBarsClient:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def bars(self, *, symbol: str, frequency: int, start: int = 0, offset: int = 800) -> pd.DataFrame:
        del symbol, frequency
        effective_offset = min(offset, 800)
        self.calls.append((start, effective_offset))
        rows = {
            0: _bar("2026-07-21 15:00:00"),
            800: _bar("2026-07-20 15:00:00"),
            1600: _bar("2026-07-19 15:00:00"),
        }
        return rows.get(start, pd.DataFrame())

    def index(self, *, symbol: str, frequency: int, start: int = 0, offset: int = 800) -> pd.DataFrame:
        return self.bars(symbol=symbol, frequency=frequency, start=start, offset=offset)


class _EmptyBarsClient:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def bars(self, *, symbol: str, frequency: int, start: int = 0, offset: int = 800) -> pd.DataFrame:
        del symbol, frequency
        self.calls.append((start, offset))
        return pd.DataFrame()


class _SinglePageClient:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.calls: list[tuple[int, int]] = []

    def bars(self, *, symbol: str, frequency: int, start: int = 0, offset: int = 800) -> pd.DataFrame:
        del symbol, frequency
        self.calls.append((start, offset))
        return self.frame if start == 0 else pd.DataFrame()


def test_fetch_count_includes_distance_from_today_for_exact_historical_day(monkeypatch) -> None:
    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            del tz
            return cls(2026, 8, 17, 12, 0, 0)

    monkeypatch.setattr(source, "datetime", _FrozenDatetime)

    assert source._estimate_fetch_count(
        "1m",
        _FrozenDatetime(2026, 7, 21),
        _FrozenDatetime(2026, 7, 21, 23, 59, 59),
    ) == 28 * 242


def test_stock_history_pages_with_bounded_start_instead_of_oversized_offset(monkeypatch) -> None:
    client = _CappedBarsClient()
    monkeypatch.setattr(source, "_estimate_fetch_count", lambda *args: 1_801)
    monkeypatch.setattr(source, "_call_mootdx", lambda api_name, callback: callback(client))

    result = source._fetch_stock_history_frame(
        "600000",
        "1m",
        datetime(2026, 7, 19),
        datetime(2026, 7, 21, 23, 59, 59),
    )

    assert client.calls == [(0, 800), (800, 800), (1600, 201)]
    assert result["trade_time"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist() == [
        "2026-07-19 15:00:00",
        "2026-07-20 15:00:00",
        "2026-07-21 15:00:00",
    ]


def test_index_history_uses_the_same_bounded_start_pagination(monkeypatch) -> None:
    client = _CappedBarsClient()
    monkeypatch.setattr(source, "_estimate_fetch_count", lambda *args: 1_801)
    monkeypatch.setattr(source, "_call_mootdx", lambda api_name, callback: callback(client))

    result = source._fetch_index_history_frame(
        "000001",
        "1m",
        datetime(2026, 7, 19),
        datetime(2026, 7, 21, 23, 59, 59),
    )

    assert client.calls == [(0, 800), (800, 800), (1600, 201)]
    assert result["index_code"].unique().tolist() == ["000001"]
    assert len(result) == 3


def test_empty_server_is_polled_before_using_next_server(monkeypatch) -> None:
    empty_client = _EmptyBarsClient()
    populated_client = _CappedBarsClient()
    clients = {
        ("empty", 1): empty_client,
        ("populated", 2): populated_client,
    }

    class _Quotes:
        @staticmethod
        def factory(*, market: str, server: tuple[str, int], bestip: bool, timeout: int):
            del market, bestip, timeout
            return clients[server]

    monkeypatch.setattr(source, "Quotes", _Quotes)
    monkeypatch.setattr(source, "_resolve_servers", lambda: list(clients))
    monkeypatch.setattr(source, "call_provider_api", lambda provider, api_name, invoke: invoke())

    result = source._call_mootdx(
        "quotes.bars",
        lambda client: source._fetch_paged_bars(
            client,
            "bars",
            symbol="600000",
            frequency=8,
            fetch_count=1_801,
        ),
    )

    assert empty_client.calls == [(0, 800)]
    assert populated_client.calls == [(0, 800), (800, 800), (1600, 201)]
    assert len(result) == 3


def test_server_without_requested_date_is_polled_before_using_next_server(monkeypatch) -> None:
    shallow_client = _SinglePageClient(_bar("2026-07-21 15:00:00"))
    deep_client = _SinglePageClient(_bar("2026-07-19 15:00:00"))
    clients = {
        ("shallow", 1): shallow_client,
        ("deep", 2): deep_client,
    }

    class _Quotes:
        @staticmethod
        def factory(*, market: str, server: tuple[str, int], bestip: bool, timeout: int):
            del market, bestip, timeout
            return clients[server]

    monkeypatch.setattr(source, "Quotes", _Quotes)
    monkeypatch.setattr(source, "_resolve_servers", lambda: list(clients))
    monkeypatch.setattr(source, "call_provider_api", lambda provider, api_name, invoke: invoke())
    monkeypatch.setattr(source, "_estimate_fetch_count", lambda *args: 242)

    result = source._fetch_stock_history_frame(
        "600000",
        "1m",
        datetime(2026, 7, 19),
        datetime(2026, 7, 19, 23, 59, 59),
    )

    assert shallow_client.calls == [(0, 242)]
    assert deep_client.calls == [(0, 242)]
    assert result["trade_time"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist() == ["2026-07-19 15:00:00"]
