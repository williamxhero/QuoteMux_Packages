from __future__ import annotations

from datetime import datetime

import pytest

from quotemux_packages.derived_core import source


def test_derive_current_30m_bar_requires_every_elapsed_minute_and_returns_one_explicit_bar() -> None:
    expected = [f"2026-09-02T13:{minute:02d}:00+08:00" for minute in range(30, 33)]
    bars = [
        {"interval_start": expected[0], "open": 1400.0, "high": 1401.0, "low": 1399.0, "close": 1400.5, "volume": 100, "amount": 140050.0},
        {"interval_start": expected[1], "open": 1400.5, "high": 1402.0, "low": 1400.0, "close": 1401.5, "volume": 120, "amount": 168180.0},
        {"interval_start": expected[2], "open": 1401.5, "high": 1403.0, "low": 1401.0, "close": 1402.5, "volume": 130, "amount": 182325.0},
    ]

    result = source.derive_current_stock_bar_30m("600519", expected, bars)

    assert result == {
        "code": "600519", "interval_start": "2026-09-02T13:30:00+08:00", "open": 1400.0, "high": 1403.0,
        "low": 1399.0, "close": 1402.5, "volume": 350, "amount": 490555.0, "source_semantics": "derived",
    }


def test_derive_current_30m_bar_refuses_an_unexplained_elapsed_minute() -> None:
    expected = ["2026-09-02T13:30:00+08:00", "2026-09-02T13:31:00+08:00"]

    with pytest.raises(source.CurrentPeriodDataIncomplete, match="13:31"):
        source.derive_current_stock_bar_30m("600519", expected, [{"interval_start": expected[0], "open": 1400.0, "high": 1401.0, "low": 1399.0, "close": 1400.5, "volume": 100, "amount": 140050.0}])
