from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from platform_models import EtfDailyQuoteItem
from quotemux.infra.cache.store import build_cache_path, filter_frame_by_date_range, merge_cache_frame, read_cache_frame, write_cache_frame
from quotemux.infra.common import format_date_value

from .source import _call_ef, ef


def _normalize_ts_code(value: object) -> str:
    text = str(value or "").strip().upper()
    if len(text) == 9 and text[6] == "." and text[:6].isdigit() and text[7:] in {"SH", "SZ"}:
        return text
    return ""


def _date_range(trade_date: str, start_date: str, end_date: str) -> tuple[str, str]:
    start_value = format_date_value(trade_date or start_date)
    end_value = format_date_value(trade_date or end_date)
    if start_value == "" and end_value == "":
        end_value = datetime.now().strftime("%Y-%m-%d")
        start_value = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    elif start_value == "":
        start_value = end_value
    elif end_value == "":
        end_value = start_value
    return start_value, end_value


def _number_column(frame: pd.DataFrame, column_name: str) -> pd.Series:
    if column_name not in frame.columns:
        return pd.Series(pd.NA, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column_name], errors="coerce")


def _fetch_daily_frame(ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    if ef is None:
        return pd.DataFrame()
    try:
        frame = _call_ef(
            "stock.get_quote_history",
            ef.stock.get_quote_history,
            ts_code[:6],
            start_date.replace("-", ""),
            end_date.replace("-", ""),
            101,
            0,
            suppress_error=True,
        )
    except Exception:
        return pd.DataFrame()
    if frame is None or frame.empty:
        return pd.DataFrame()
    work = frame.copy()
    work["ts_code"] = ts_code
    work["trade_date"] = work["日期"].map(format_date_value) if "日期" in work.columns else ""
    work["open"] = _number_column(work, "开盘")
    work["high"] = _number_column(work, "最高")
    work["low"] = _number_column(work, "最低")
    work["close"] = _number_column(work, "收盘")
    work["change"] = _number_column(work, "涨跌额")
    work["pct_chg"] = _number_column(work, "涨跌幅")
    work["pre_close"] = work["close"] - work["change"]
    missing_pre_close = work["pre_close"].isna() & work["close"].notna() & work["pct_chg"].notna() & (work["pct_chg"] != -100)
    work.loc[missing_pre_close, "pre_close"] = work.loc[missing_pre_close, "close"] / (1 + work.loc[missing_pre_close, "pct_chg"] / 100)
    work["volume"] = _number_column(work, "成交量")
    work["amount"] = _number_column(work, "成交额") / 1000
    work = work[work["trade_date"] != ""]
    return work[["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "volume", "amount"]].drop_duplicates(subset=["ts_code", "trade_date"], keep="last")


def _to_items(frame: pd.DataFrame) -> list[EtfDailyQuoteItem]:
    if frame.empty:
        return []
    return [
        EtfDailyQuoteItem(
            ts_code=str(row["ts_code"]),
            trade_date=format_date_value(row["trade_date"]),
            open=float(row["open"]) if pd.notna(row["open"]) else None,
            high=float(row["high"]) if pd.notna(row["high"]) else None,
            low=float(row["low"]) if pd.notna(row["low"]) else None,
            close=float(row["close"]) if pd.notna(row["close"]) else None,
            pre_close=float(row["pre_close"]) if pd.notna(row["pre_close"]) else None,
            change=float(row["change"]) if pd.notna(row["change"]) else None,
            pct_chg=float(row["pct_chg"]) if pd.notna(row["pct_chg"]) else None,
            volume=float(row["volume"]) if pd.notna(row["volume"]) else None,
            amount=float(row["amount"]) if pd.notna(row["amount"]) else None,
        )
        for _, row in frame.sort_values(["ts_code", "trade_date"]).iterrows()
    ]


def get_etf_daily_quotes(ts_codes: list[str], trade_date: str, start_date: str, end_date: str) -> list[EtfDailyQuoteItem]:
    actual_start, actual_end = _date_range(trade_date, start_date, end_date)
    items: list[EtfDailyQuoteItem] = []
    for ts_code in dict.fromkeys(_normalize_ts_code(item) for item in ts_codes):
        if ts_code == "":
            continue
        cache_path = build_cache_path("efinance", ["funds", "etf", "quotes", "daily"], {"ts_code": ts_code})
        cache_frame = read_cache_frame(cache_path)
        cached_window = filter_frame_by_date_range(cache_frame, "trade_date", actual_start, actual_end)
        if cached_window.empty:
            fetched_frame = _fetch_daily_frame(ts_code, actual_start, actual_end)
            if not fetched_frame.empty:
                cache_frame = merge_cache_frame(cache_frame, fetched_frame, ["ts_code", "trade_date"], ["trade_date"])
                write_cache_frame(cache_path, cache_frame)
                cached_window = filter_frame_by_date_range(cache_frame, "trade_date", actual_start, actual_end)
        items.extend(_to_items(cached_window))
    return items
