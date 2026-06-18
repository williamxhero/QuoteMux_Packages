from __future__ import annotations

import pandas as pd

from platform_models import HLSignalItem, ShareholderChangeItem, ShareholderCountItem, StockQuoteItem, TechnicalFactorItem, TradingCalendarItem
from quotemux.infra.common import normalize_stock_code


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - 100 / (1 + rs)


def _quote_mux():
    from quotemux import QuoteMux

    return QuoteMux()


def _trading_calendar_request(exchange: str, start_date: str, end_date: str, is_open: bool | None):
    from quotemux import TradingCalendarRequest

    return TradingCalendarRequest(exchange=exchange, start_date=start_date, end_date=end_date, is_open=is_open)


def _offset_date_text(trade_date: str, days: int) -> str:
    from datetime import timedelta

    from quotemux.infra.common import parse_date_text

    trade_day = parse_date_text(trade_date)
    if trade_day is None:
        return ""
    return (trade_day + timedelta(days=days)).strftime("%Y-%m-%d")


def _stock_quotes_request(
    code: str,
    freq: str,
    trade_date: str,
    start_date: str,
    end_date: str,
    adjust: str,
):
    from quotemux import StockQuotesRequest

    return StockQuotesRequest(
        codes=[code],
        freq=freq,
        trade_date=trade_date,
        start_date=start_date,
        end_date=end_date,
        adjust=adjust,
        limit=5000,
    )


def _build_technical_factor_items(quote_items: list[StockQuoteItem], adjust: str) -> list[TechnicalFactorItem]:
    if quote_items == []:
        return []
    frame = pd.DataFrame([item.model_dump() for item in quote_items])
    frame["trade_date"] = frame["trade_time"].astype(str)
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["high"] = pd.to_numeric(frame["high"], errors="coerce")
    frame["low"] = pd.to_numeric(frame["low"], errors="coerce")
    frame = frame.sort_values("trade_date").reset_index(drop=True)
    frame["ma5"] = frame["close"].rolling(5, min_periods=5).mean()
    frame["ma10"] = frame["close"].rolling(10, min_periods=10).mean()
    frame["ma20"] = frame["close"].rolling(20, min_periods=20).mean()
    frame["ma60"] = frame["close"].rolling(60, min_periods=60).mean()
    frame["ema12"] = frame["close"].ewm(span=12, adjust=False).mean()
    frame["ema26"] = frame["close"].ewm(span=26, adjust=False).mean()
    frame["dif"] = frame["ema12"] - frame["ema26"]
    frame["dea"] = frame["dif"].ewm(span=9, adjust=False).mean()
    frame["macd"] = (frame["dif"] - frame["dea"]) * 2
    frame["rsi6"] = _rsi(frame["close"], 6)
    frame["rsi12"] = _rsi(frame["close"], 12)
    frame["rsi24"] = _rsi(frame["close"], 24)
    low_n = frame["low"].rolling(9, min_periods=9).min()
    high_n = frame["high"].rolling(9, min_periods=9).max()
    rsv = (frame["close"] - low_n) / (high_n - low_n).replace(0, pd.NA) * 100
    frame["kdj_k"] = rsv.ewm(com=2, adjust=False).mean()
    frame["kdj_d"] = frame["kdj_k"].ewm(com=2, adjust=False).mean()
    frame["kdj_j"] = 3 * frame["kdj_k"] - 2 * frame["kdj_d"]
    boll_mid = frame["close"].rolling(20, min_periods=20).mean()
    boll_std = frame["close"].rolling(20, min_periods=20).std()
    frame["boll_upper"] = boll_mid + 2 * boll_std
    frame["boll_mid"] = boll_mid
    frame["boll_lower"] = boll_mid - 2 * boll_std
    return [
        TechnicalFactorItem(
            code=str(row["code"]),
            trade_date=str(row["trade_date"]),
            adjust=adjust,
            ma5=float(row["ma5"]) if pd.notna(row["ma5"]) else None,
            ma10=float(row["ma10"]) if pd.notna(row["ma10"]) else None,
            ma20=float(row["ma20"]) if pd.notna(row["ma20"]) else None,
            ma60=float(row["ma60"]) if pd.notna(row["ma60"]) else None,
            ema12=float(row["ema12"]) if pd.notna(row["ema12"]) else None,
            ema26=float(row["ema26"]) if pd.notna(row["ema26"]) else None,
            dif=float(row["dif"]) if pd.notna(row["dif"]) else None,
            dea=float(row["dea"]) if pd.notna(row["dea"]) else None,
            macd=float(row["macd"]) if pd.notna(row["macd"]) else None,
            rsi6=float(row["rsi6"]) if pd.notna(row["rsi6"]) else None,
            rsi12=float(row["rsi12"]) if pd.notna(row["rsi12"]) else None,
            rsi24=float(row["rsi24"]) if pd.notna(row["rsi24"]) else None,
            kdj_k=float(row["kdj_k"]) if pd.notna(row["kdj_k"]) else None,
            kdj_d=float(row["kdj_d"]) if pd.notna(row["kdj_d"]) else None,
            kdj_j=float(row["kdj_j"]) if pd.notna(row["kdj_j"]) else None,
            boll_upper=float(row["boll_upper"]) if pd.notna(row["boll_upper"]) else None,
            boll_mid=float(row["boll_mid"]) if pd.notna(row["boll_mid"]) else None,
            boll_lower=float(row["boll_lower"]) if pd.notna(row["boll_lower"]) else None,
        )
        for _, row in frame.iterrows()
    ]


def _build_shareholder_change_items(items: list[ShareholderCountItem]) -> list[ShareholderChangeItem]:
    rows: list[ShareholderChangeItem] = []
    previous_count: int | None = None
    for item in sorted(items, key=lambda value: value.trade_date):
        change_count = item.holder_count - previous_count if item.holder_count is not None and previous_count is not None else None
        change_pct = None
        if change_count is not None and previous_count not in {None, 0}:
            change_pct = change_count / previous_count * 100
        rows.append(
            ShareholderChangeItem(
                code=item.code,
                trade_date=item.trade_date,
                holder_count=item.holder_count,
                change_count=change_count,
                change_pct=change_pct,
            )
        )
        previous_count = item.holder_count
    return rows


def _build_hl_signal_items(code: str, quote_items: list[StockQuoteItem]) -> list[HLSignalItem]:
    if quote_items == []:
        return []
    frame = pd.DataFrame([item.model_dump() for item in quote_items])
    frame["trade_time_dt"] = pd.to_datetime(frame["trade_time"], errors="coerce")
    frame["high"] = pd.to_numeric(frame["high"], errors="coerce")
    frame["low"] = pd.to_numeric(frame["low"], errors="coerce")
    frame = frame.dropna(subset=["trade_time_dt", "high", "low"])
    if frame.empty:
        return []
    frame["trade_date"] = frame["trade_time_dt"].dt.strftime("%Y-%m-%d")
    items: list[HLSignalItem] = []
    for trade_date, group in frame.groupby("trade_date", sort=True):
        max_high = group["high"].max()
        min_low = group["low"].min()
        high_rows = group[group["high"] == max_high].sort_values("trade_time_dt")
        low_rows = group[group["low"] == min_low].sort_values("trade_time_dt")
        high_dt = high_rows.iloc[0]["trade_time_dt"] if not high_rows.empty else None
        low_dt = low_rows.iloc[0]["trade_time_dt"] if not low_rows.empty else None
        high_time = high_dt.strftime("%H:%M:%S") if pd.notna(high_dt) else ""
        low_time = low_dt.strftime("%H:%M:%S") if pd.notna(low_dt) else ""
        if high_time and low_time and high_dt < low_dt:
            first_extreme = "high"
            signal = "high_first"
        elif high_time and low_time and low_dt < high_dt:
            first_extreme = "low"
            signal = "low_first"
        else:
            first_extreme = ""
            signal = "same_time"
        items.append(HLSignalItem(code=code, trade_date=str(trade_date), first_extreme=first_extreme, high_time=high_time, low_time=low_time, signal=signal))
    return items


def get_technical_factors(code: str, trade_date: str, start_date: str, end_date: str, adjust: str) -> list[TechnicalFactorItem]:
    normalized = normalize_stock_code(code)
    if normalized == "":
        return []
    quote_items = _quote_mux().stocks.get_quotes(_stock_quotes_request(normalized, "1d", trade_date, start_date, end_date, adjust))
    return _build_technical_factor_items(quote_items, adjust)


def get_shareholder_changes(code: str, trade_date: str, start_date: str, end_date: str) -> list[ShareholderChangeItem]:
    normalized = normalize_stock_code(code)
    if normalized == "":
        return []
    count_items = _quote_mux().stocks.get_shareholder_count(normalized, trade_date, start_date, end_date)
    return _build_shareholder_change_items(count_items)


def get_hl_signal(code: str, trade_date: str, start_date: str, end_date: str) -> list[HLSignalItem]:
    normalized = normalize_stock_code(code)
    if normalized == "":
        return []
    quote_items = _quote_mux().stocks.get_quotes(_stock_quotes_request(normalized, "1m", trade_date, start_date, end_date, "none"))
    return _build_hl_signal_items(normalized, quote_items)


def get_previous_trading_days(exchange: str, trade_date: str, n: int) -> list[TradingCalendarItem]:
    start_date = _offset_date_text(trade_date, -max(n * 8, 30))
    items = _quote_mux().markets.get_trading_calendar(_trading_calendar_request(exchange, start_date, trade_date, True))
    return [item for item in items if item.trade_date < trade_date][-n:]


def get_next_trading_days(exchange: str, trade_date: str, n: int) -> list[TradingCalendarItem]:
    from datetime import date

    from quotemux.infra.common import parse_date_text

    trade_day = parse_date_text(trade_date)
    end_date = ""
    if trade_day is not None:
        try:
            next_year_day = trade_day.replace(year=trade_day.year + 1)
        except ValueError:
            next_year_day = date(trade_day.year + 1, 2, 28)
        end_date = next_year_day.strftime("%Y-%m-%d")
    items = _quote_mux().markets.get_trading_calendar(_trading_calendar_request(exchange, trade_date, end_date, True))
    return [item for item in items if item.trade_date > trade_date][:n]


def get_yearly_trading_calendar(exchange: str, start_year: int, end_year: int) -> list[TradingCalendarItem]:
    return _quote_mux().markets.get_trading_calendar(_trading_calendar_request(exchange, f"{start_year}-01-01", f"{end_year}-12-31", None))
