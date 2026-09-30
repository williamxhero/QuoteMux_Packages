from datetime import datetime, timedelta
from types import SimpleNamespace

import pandas as pd

from quotemux_packages.tushare import rate_limit, source


def test_stk_mins_rate_limit_matches_provider_quota(monkeypatch) -> None:
    monkeypatch.delenv("MHK_TUSHARE_STK_MINS_MAX_CALLS_PER_MINUTE", raising=False)
    rate_limit.get_tushare_api_rate_limiter.cache_clear()
    try:
        limiter = rate_limit.get_tushare_api_rate_limiter("stk_mins")
        assert limiter._max_calls_per_minute == 1
        assert limiter._period_seconds == 60.0
    finally:
        rate_limit.get_tushare_api_rate_limiter.cache_clear()


def _minute_frame() -> pd.DataFrame:
    times = [datetime(2026, 9, 29, 9, 30) + timedelta(minutes=index) for index in range(241)]
    times = [value for value in times if value.time() <= datetime(2026, 9, 29, 11, 30).time()]
    times.extend(datetime(2026, 9, 29, 13, 1) + timedelta(minutes=index) for index in range(120))
    return pd.DataFrame(
        {
            "trade_time": times,
            "open": [10.0] * len(times),
            "high": [10.5] * len(times),
            "low": [9.5] * len(times),
            "close": [10.0] * len(times),
            "vol": [100.0] * len(times),
            "amount": [1000.0] * len(times),
        }
    )


def test_tushare_intraday_uses_stk_mins_and_normalizes_opening_auction(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(source, "get_provider_api_key", lambda: "token")
    monkeypatch.setattr(source, "get_ts_pro", lambda: SimpleNamespace(stk_mins=lambda **kwargs: None))
    monkeypatch.setattr(source, "ts", SimpleNamespace(set_token=lambda token: None))

    def fake_call(api_name, func, *args, **kwargs):
        calls.append((api_name, kwargs))
        return _minute_frame()

    monkeypatch.setattr(source, "call_tushare_api", fake_call)

    result = source._fetch_stock_quotes_frame(
        "000001",
        "1m",
        datetime(2026, 9, 29),
        datetime(2026, 9, 29, 23, 59, 59),
        "none",
    )

    assert calls == [
        (
            "stk_mins",
            {
                "ts_code": "000001.SZ",
                "start_date": "2026-09-29 00:00:00",
                "end_date": "2026-09-29 23:59:59",
                "freq": "1min",
                "offset": 0,
                "limit": 8000,
            },
        )
    ]
    assert len(result) == 240
    assert result["trade_time"].dt.strftime("%H:%M:%S").tolist()[0] == "09:31:00"
    assert result["trade_time"].dt.strftime("%H:%M:%S").tolist()[-1] == "15:00:00"
    assert result.iloc[0]["open"] == 10.0
    assert result.iloc[0]["volume2"] == 200.0
    assert result.iloc[0]["amount"] == 2000.0
