from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SOURCE_PATH = Path(__file__).resolve().parents[1] / "source.py"
SPEC = importlib.util.spec_from_file_location("calendar_test_akshare_source", SOURCE_PATH)
assert SPEC is not None and SPEC.loader is not None
source = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(source)


def test_calendar_derives_closed_facts_from_complete_open_day_history(monkeypatch) -> None:
    monkeypatch.setattr(source, "build_cache_path", lambda *_args, **_kwargs: "unused")
    monkeypatch.setattr(
        source,
        "read_cache_frame",
        lambda _path: pd.DataFrame([{"trade_date": "2012-01-04"}]),
    )

    items = source.get_trading_calendar("SSE", "2012-01-01", "2012-01-04", None)

    assert [(item.trade_date, item.is_open) for item in items] == [
        ("2012-01-01", False),
        ("2012-01-02", False),
        ("2012-01-03", False),
        ("2012-01-04", True),
    ]
