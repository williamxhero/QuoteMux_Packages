from __future__ import annotations

import pandas as pd


def test_strategy_factor_window_exposes_explicit_no_price_limit_status(monkeypatch) -> None:
    """A source-backed no-limit fact must not be conflated with missing prices."""
    from quotemux.infra.db import client as db_client
    from quotemux_packages.derived_core import source

    calls: list[str] = []

    def fake_query_dataframe(query: str, params: tuple[object, ...]) -> pd.DataFrame:
        calls.append(query)
        return pd.DataFrame(
            [
                {
                    "trade_date": "2025-04-28",
                    "code": "300630",
                    "close": 12.34,
                    "is_st": False,
                    "is_suspended": False,
                    "upper_limit": None,
                    "lower_limit": None,
                    "price_band_state": "no_price_limit",
                }
            ]
        )

    monkeypatch.setattr(db_client, "query_dataframe", fake_query_dataframe)

    items = source.get_strategy_factor_window("2025-04-28", "2025-04-28", "SZ.300630")

    assert items[0].upper_limit is None
    assert items[0].lower_limit is None
    assert items[0].price_band_state == "no_price_limit"
    assert "(to_jsonb(price_band) ->> 'price_band_status') = 'no_price_limit'" in calls[0]
    assert "and price_band.upper_limit is null" in calls[0]
    assert "and price_band.lower_limit is null then 'no_price_limit'" in calls[0]
