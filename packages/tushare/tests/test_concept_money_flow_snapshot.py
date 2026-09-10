from __future__ import annotations

import pandas as pd

from quotemux_packages.tushare import source


def test_concept_money_flow_snapshot_fetches_trade_day_once_and_keeps_yi_units(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class Provider:
        def moneyflow_cnt_ths(self, **_kwargs):
            raise AssertionError("call_tushare_api seam must be used")

    monkeypatch.setattr(source, "get_ts_pro", lambda: Provider())

    def fake_call(api_name: str, _func, **kwargs: object) -> pd.DataFrame:
        calls.append((api_name, kwargs))
        return pd.DataFrame(
            [
                {
                    "trade_date": "20260908",
                    "ts_code": "885001.TI",
                    "net_buy_amount": 21.0,
                    "net_sell_amount": 19.0,
                    "net_amount": 2.0,
                },
                {
                    "trade_date": "20260908",
                    "ts_code": "885002.TI",
                    "net_buy_amount": 8.5,
                    "net_sell_amount": 9.0,
                    "net_amount": -0.5,
                },
            ]
        )

    monkeypatch.setattr(source, "call_tushare_api", fake_call)

    items = source.get_concept_daily_money_flow_snapshot("2026-09-08", "concept", 10000, 0)

    assert calls == [("moneyflow_cnt_ths", {"trade_date": "20260908"})]
    assert [(item.board_code, item.trade_date, item.scope, item.inflow, item.outflow, item.net_inflow) for item in items] == [
        ("885001", "2026-09-08", "concept", 21.0, 19.0, 2.0),
        ("885002", "2026-09-08", "concept", 8.5, 9.0, -0.5),
    ]
