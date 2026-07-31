from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from platform_models import EtfCatalogItem, EtfDailyQuoteItem
from quotemux.infra.cache.store import build_cache_path, filter_frame_by_date_range, merge_cache_frame, read_cache_frame, write_cache_frame
from quotemux.infra.common import format_date_value

from .rate_limit import call_tushare_api
from .source import get_ts_pro


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


def _as_text(frame: pd.DataFrame, column_name: str) -> pd.Series:
    if column_name not in frame.columns:
        return pd.Series("", index=frame.index, dtype="object")
    return frame[column_name].fillna("").astype(str)


def _fetch_catalog_frame() -> pd.DataFrame:
    pro = get_ts_pro()
    if pro is None:
        return pd.DataFrame()
    try:
        frame = call_tushare_api(
            "fund_basic",
            pro.fund_basic,
            market="E",
            fields="ts_code,name,management,custodian,fund_type,list_date,delist_date",
        )
    except Exception:
        return pd.DataFrame()
    if frame is None or frame.empty:
        return pd.DataFrame()
    work = frame.copy()
    work["ts_code"] = _as_text(work, "ts_code").map(_normalize_ts_code)
    work = work[work["ts_code"] != ""]
    work["code"] = work["ts_code"].str[:6]
    work["market"] = work["ts_code"].str[-2:].map({"SH": "SHSE", "SZ": "SZSE"}).fillna("")
    for column_name in ("name", "management", "custodian", "fund_type"):
        work[column_name] = _as_text(work, column_name)
    for column_name in ("list_date", "delist_date"):
        work[column_name] = _as_text(work, column_name).map(format_date_value)
    return work[["ts_code", "code", "market", "name", "fund_type", "management", "custodian", "list_date", "delist_date"]].drop_duplicates(subset=["ts_code"], keep="last")


def get_etf_catalog() -> list[EtfCatalogItem]:
    cache_path = build_cache_path("tushare", ["funds", "etf", "catalog"], {"market": "E"})
    frame = read_cache_frame(cache_path)
    if frame.empty:
        frame = _fetch_catalog_frame()
        if not frame.empty:
            write_cache_frame(cache_path, frame)
    if frame.empty:
        return []
    return [
        EtfCatalogItem(
            ts_code=str(row["ts_code"]),
            code=str(row["code"]),
            market=str(row["market"]),
            name=str(row["name"]),
            fund_type=str(row["fund_type"]),
            management=str(row["management"]),
            custodian=str(row["custodian"]),
            list_date=format_date_value(row["list_date"]),
            delist_date=format_date_value(row["delist_date"]),
        )
        for _, row in frame.sort_values("ts_code").iterrows()
    ]


def _fetch_daily_frame(ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    pro = get_ts_pro()
    if pro is None:
        return pd.DataFrame()
    try:
        frame = call_tushare_api(
            "fund_daily",
            pro.fund_daily,
            ts_code=ts_code,
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
        )
    except Exception:
        return pd.DataFrame()
    if frame is None or frame.empty:
        return pd.DataFrame()
    work = frame.copy()
    work["ts_code"] = _normalize_ts_code(ts_code)
    work["trade_date"] = _as_text(work, "trade_date").map(format_date_value)
    work = work[work["trade_date"] != ""]
    for column_name in ("open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"):
        if column_name not in work.columns:
            work[column_name] = None
    work["volume"] = pd.to_numeric(work["vol"], errors="coerce")
    work["amount"] = pd.to_numeric(work["amount"], errors="coerce")
    return work[["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "volume", "amount"]]


def _to_items(frame: pd.DataFrame) -> list[EtfDailyQuoteItem]:
    if frame.empty:
        return []
    items: list[EtfDailyQuoteItem] = []
    for _, row in frame.sort_values(["ts_code", "trade_date"]).iterrows():
        items.append(
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
        )
    return items


def get_etf_daily_quotes(ts_codes: list[str], trade_date: str, start_date: str, end_date: str) -> list[EtfDailyQuoteItem]:
    actual_start, actual_end = _date_range(trade_date, start_date, end_date)
    items: list[EtfDailyQuoteItem] = []
    for ts_code in dict.fromkeys(_normalize_ts_code(item) for item in ts_codes):
        if ts_code == "":
            continue
        cache_path = build_cache_path("tushare", ["funds", "etf", "quotes", "daily"], {"ts_code": ts_code})
        cache_frame = read_cache_frame(cache_path)
        cached_window = filter_frame_by_date_range(cache_frame, "trade_date", actual_start, actual_end)
        cached_dates = set(cached_window["trade_date"].astype(str)) if not cached_window.empty and "trade_date" in cached_window.columns else set()
        if cached_window.empty or len(cached_dates) == 0:
            fetched_frame = _fetch_daily_frame(ts_code, actual_start, actual_end)
            if not fetched_frame.empty:
                cache_frame = merge_cache_frame(cache_frame, fetched_frame, ["ts_code", "trade_date"], ["trade_date"])
                write_cache_frame(cache_path, cache_frame)
                cached_window = filter_frame_by_date_range(cache_frame, "trade_date", actual_start, actual_end)
        items.extend(_to_items(cached_window))
    return items
