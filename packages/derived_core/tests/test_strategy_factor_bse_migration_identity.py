from __future__ import annotations

import pandas as pd


def test_strategy_factor_window_admits_only_exact_active_bse_migrations(monkeypatch) -> None:
    """A same-day old BSE code needs one exact mapping to an active 920 identity."""
    from quotemux.infra.db import client as db_client
    from quotemux_packages.derived_core import source

    calls: list[tuple[str, tuple[object, ...]]] = []

    def fake_query_dataframe(query: str, params: tuple[object, ...]) -> pd.DataFrame:
        calls.append((query, params))
        # The database regression fixture has three 2024-01-05 old-code rows:
        # 873690 has one active 920690 successor and is returned; 873691 points
        # to an unlisted successor; 873692 has two successor identities.  Only
        # the first may reach this result frame.
        return pd.DataFrame([{"trade_date": "2024-01-05", "code": "873690"}])

    monkeypatch.setattr(db_client, "query_dataframe", fake_query_dataframe)

    items = source.get_strategy_factor_window("2024-01-05", "2024-01-05", "873690,873691,873692")

    assert [item.code for item in items] == ["873690"]
    query, params = calls[0]
    assert params[:4] == ("2024-01-05", "2024-01-05", "873690,873691,873692", ["873690", "873691", "873692"])
    assert "join ref.stock identity_rows" in query
    assert "identity_rows.delisted_date is null or identity_rows.delisted_date > day_rows.trade_date" in query
    assert "join ref.stock_code_migration migration" in query
    assert "migration.old_market = day_rows.market" in query
    assert "migration.old_code = day_rows.code" in query
    assert "migration.trade_date = day_rows.trade_date" in query
    assert "migration.old_market = 'BJSE'" in query
    assert "migration.new_market = 'BJSE'" in query
    assert "successor_rows.listed_date <= day_rows.trade_date" in query
    assert "successor_rows.delisted_date is null or successor_rows.delisted_date > day_rows.trade_date" in query
    assert "competing_migration.new_market, competing_migration.new_code" in query
    assert "is distinct from (migration.new_market, migration.new_code)" in query
