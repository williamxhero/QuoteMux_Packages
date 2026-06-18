from __future__ import annotations

import sys
_saved_paths = [path for path in sys.path if "quotemux_packages" in path or ("packages" in path and "site-packages" not in path and "dist-packages" not in path)]
for path in _saved_paths:
    try:
        sys.path.remove(path)
    except ValueError:
        pass

try:
    import akshare as ak
except Exception:
    ak = None
finally:
    for path in reversed(_saved_paths):
        sys.path.insert(0, path)

try:
    from curl_cffi import requests as curl_requests
except Exception:
    curl_requests = None

from datetime import datetime, timedelta
from functools import lru_cache
import hashlib
from io import StringIO

import pandas as pd
import re
import requests.adapters

_original_request = requests.Session.request

def _patched_request(self, method, url, **kwargs):
    if url:
        url = re.sub(r'(https?://)\d+\.push2\.eastmoney\.com', r'\1push2.eastmoney.com', url)
    return _original_request(self, method, url, **kwargs)

requests.Session.request = _patched_request

from quotemux.infra.cache.store import build_cache_path, filter_frame_by_date_range, filter_frame_by_datetime_range, latest_n_rows, merge_cache_frame, read_cache_frame, write_cache_frame
from quotemux.infra.common import add_quote_metrics, aggregate_ohlc, build_time_bounds, format_date_value, format_datetime_value, normalize_index_code, normalize_stock_code, split_csv
from quotemux.runtime_core.quality import build_akshare_index_symbol, calibrate_quote_units
from quotemux.infra.provider_runtime.core import call_provider_api
from platform_models import AuctionItem, BlockTradeItem, BoardCatalogItem, BoardCategoryItem, BoardMemberItem, BoardMoneyFlowItem, BoardQuoteItem, ConnectCapitalFlowItem, DisclosureDateItem, DividendItem, DragonTigerInstitutionItem, DragonTigerItem, ExpressItem, ForecastItem, HKConnectHoldingItem, HotMoneyDetailItem, IndexMemberItem, IndexQuoteItem, MainBusinessItem, MarketCapitalFlowItem, NewsEventItem, PledgeDetailItem, PledgeStatItem, RepurchaseItem, ResearchReportItem, RightsIssueItem, ShareChangeItem, ShareholderChangeItem, ShareholderCountItem, ShareholderTop10Item, StockFinanceIndicatorItem, StockFinancialStatementItem, StockMoneyFlowItem, StockProfileItem, StockQuoteItem, SurveyItem, TradingCalendarItem, UnlockScheduleItem


DEFAULT_LOOKBACK_DAYS = 30
AKSHARE_PERIOD_MAP = {
    "1d": "daily",
    "1w": "weekly",
    "1mo": "monthly",
}
AKSHARE_MINUTE_PERIOD_MAP = {
    "1m": "1",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "60m": "60",
}
BOARD_CATEGORIES = ("concept", "industry")
BOARD_FREQ_MAP = {"1d": ("daily", "日k"), "1w": ("weekly", "周k"), "1mo": ("monthly", "月k")}


def _is_available() -> bool:
    return ak is not None


def _require_available() -> None:
    if not _is_available():
        raise RuntimeError("akshare 不可用")


def _call_ak(api_name: str, func, *args, **kwargs):
    _require_available()
    return call_provider_api("akshare", api_name, func, *args, **kwargs)


def _float_value(value: object) -> float | None:
    number = pd.to_numeric(value, errors="coerce")
    return float(number) if pd.notna(number) else None


def _money_text_to_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).replace(",", "").replace("%", "").strip()
    if text in {"", "-", "--", "None", "nan"}:
        return None
    multiplier = 1.0
    if text.endswith("亿"):
        multiplier = 100000000.0
        text = text[:-1]
    elif text.endswith("万"):
        multiplier = 10000.0
        text = text[:-1]
    number = pd.to_numeric(text, errors="coerce")
    return float(number) * multiplier if pd.notna(number) else None


def _int_value(value: object) -> int | None:
    number = pd.to_numeric(value, errors="coerce")
    return int(number) if pd.notna(number) else None


def _text_value(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _quote_bool_flag(row: pd.Series | dict[str, object], *names: str) -> bool:
    for name in names:
        if isinstance(row, dict):
            value = row.get(name)
        else:
            value = row.get(name) if name in row else None
        if value is None or pd.isna(value):
            continue
        text_value = str(value).strip().lower()
        if text_value in {"1", "true", "y", "yes", "t", "st", "suspended", "halt"}:
            return True
        if text_value in {"0", "false", "n", "no", "f", "normal", "trading", "active"}:
            return False
        number = pd.to_numeric(value, errors="coerce")
        if pd.notna(number):
            return bool(int(number))
    return False


def _name_indicates_st(name: str) -> bool:
    upper_name = name.upper().replace(" ", "")
    return upper_name.startswith("ST") or upper_name.startswith("*ST")

def _date_window(trade_date: str, start_date: str, end_date: str, lookback_days: int) -> tuple[str, str]:
    actual_trade_date = format_date_value(trade_date)
    if actual_trade_date:
        return actual_trade_date, actual_trade_date
    actual_start = format_date_value(start_date)
    actual_end = format_date_value(end_date)
    if actual_start == "" and actual_end == "":
        end_dt = datetime.now()
        return (end_dt - timedelta(days=lookback_days)).strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")
    if actual_start == "":
        actual_start = actual_end
    if actual_end == "":
        actual_end = actual_start
    return actual_start, actual_end


def _date_range_from_request(trade_date: str, start_date: str, end_date: str, lookback_days: int) -> tuple[str, str]:
    return _date_window(trade_date, start_date, end_date, lookback_days)


def _date_in_window(value: str, start_value: str, end_value: str) -> bool:
    actual_value = format_date_value(value)
    actual_start = format_date_value(start_value)
    actual_end = format_date_value(end_value)
    if actual_value == "":
        return False
    if actual_start and actual_value < actual_start:
        return False
    if actual_end and actual_value > actual_end:
        return False
    return True


def _stock_market(code: str) -> str:
    normalized_code = normalize_stock_code(code)
    if normalized_code.startswith("6"):
        return "sh"
    if normalized_code.startswith(("4", "8", "9")):
        return "bj"
    return "sz"


def _resolve_time_window(
    trade_date: str,
    start_date: str,
    end_date: str,
    start_time: str,
    end_time: str,
    count: int | None,
    intraday: bool,
) -> tuple[datetime, datetime]:
    start_dt, end_dt = build_time_bounds(trade_date, start_date, end_date, start_time, end_time, count, intraday)
    if start_dt is None and end_dt is None:
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=DEFAULT_LOOKBACK_DAYS if intraday else 400)
    elif start_dt is None:
        start_dt = end_dt - timedelta(days=DEFAULT_LOOKBACK_DAYS if intraday else 400)
    elif end_dt is None:
        end_dt = datetime.now()
    return start_dt, end_dt


def _fetch_stock_daily_frame(code: str, freq: str, start_dt: datetime, end_dt: datetime, adjust: str) -> pd.DataFrame:
    try:
        result = _call_ak(
            "stock_zh_a_hist",
            ak.stock_zh_a_hist,
            symbol=normalize_stock_code(code),
            period=AKSHARE_PERIOD_MAP[freq],
            start_date=start_dt.strftime("%Y%m%d"),
            end_date=end_dt.strftime("%Y%m%d"),
            adjust="" if adjust == "none" else adjust,
        )
    except Exception:
        return pd.DataFrame()
    if result is None or result.empty:
        return pd.DataFrame()
    work = result.copy()
    work["code"] = work["股票代码"].astype(str).str.zfill(6)
    work["trade_time"] = pd.to_datetime(work["日期"], errors="coerce")
    work["freq"] = freq
    work["open"] = pd.to_numeric(work["开盘"], errors="coerce")
    work["high"] = pd.to_numeric(work["最高"], errors="coerce")
    work["low"] = pd.to_numeric(work["最低"], errors="coerce")
    work["close"] = pd.to_numeric(work["收盘"], errors="coerce")
    work["volume"] = pd.to_numeric(work["成交量"], errors="coerce")
    work["amount"] = pd.to_numeric(work["成交额"], errors="coerce")
    work["change"] = pd.to_numeric(work["涨跌额"], errors="coerce") if "涨跌额" in work.columns else pd.NA
    work["pct_chg"] = pd.to_numeric(work["涨跌幅"], errors="coerce") if "涨跌幅" in work.columns else pd.NA
    work["pre_close"] = work["close"] - work["change"]
    missing_pre_close = work["pre_close"].isna() & work["close"].notna() & work["pct_chg"].notna() & (work["pct_chg"] != -100)
    work.loc[missing_pre_close, "pre_close"] = work.loc[missing_pre_close, "close"] / (1 + work.loc[missing_pre_close, "pct_chg"] / 100)
    work = work[["code", "trade_time", "freq", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "volume", "amount"]]
    work = work.dropna(subset=["trade_time"])
    work, _ = calibrate_quote_units(work)
    return work.drop_duplicates(subset=["code", "trade_time", "freq"], keep="last").sort_values("trade_time").reset_index(drop=True)


def _fetch_stock_intraday_frame(code: str, freq: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    result = _call_ak(
        "stock_zh_a_hist_min_em",
        ak.stock_zh_a_hist_min_em,
        symbol=normalize_stock_code(code),
        start_date=start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        end_date=end_dt.strftime("%Y-%m-%d %H:%M:%S"),
        period=AKSHARE_MINUTE_PERIOD_MAP[freq],
        adjust="",
    )
    if result is None or result.empty:
        return pd.DataFrame()
    work = result.copy()
    work["code"] = normalize_stock_code(code)
    work["trade_time"] = pd.to_datetime(work["时间"], errors="coerce")
    work["freq"] = freq
    work["open"] = pd.to_numeric(work["开盘"], errors="coerce")
    work["high"] = pd.to_numeric(work["最高"], errors="coerce")
    work["low"] = pd.to_numeric(work["最低"], errors="coerce")
    work["close"] = pd.to_numeric(work["收盘"], errors="coerce")
    work["volume"] = pd.to_numeric(work["成交量"], errors="coerce")
    work["amount"] = pd.to_numeric(work["成交额"], errors="coerce")
    work = work[["code", "trade_time", "freq", "open", "high", "low", "close", "volume", "amount"]]
    work = work.dropna(subset=["trade_time"])
    work, _ = calibrate_quote_units(work)
    return work.drop_duplicates(subset=["code", "trade_time", "freq"], keep="last").sort_values("trade_time").reset_index(drop=True)


def _fetch_index_daily_frame(index_code: str, freq: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    result = _call_ak(
        "stock_zh_index_daily",
        ak.stock_zh_index_daily,
        symbol=build_akshare_index_symbol(index_code),
    )
    if result is None or result.empty:
        return pd.DataFrame()
    work = result.copy()
    work["index_code"] = normalize_index_code(index_code)
    work["trade_time"] = pd.to_datetime(work["date"], errors="coerce")
    work["freq"] = freq
    work["open"] = pd.to_numeric(work["open"], errors="coerce")
    work["high"] = pd.to_numeric(work["high"], errors="coerce")
    work["low"] = pd.to_numeric(work["low"], errors="coerce")
    work["close"] = pd.to_numeric(work["close"], errors="coerce")
    if "volume" not in work.columns:
        work["volume"] = pd.NA
    work["volume"] = pd.to_numeric(work["volume"], errors="coerce")
    work["amount"] = pd.NA
    work = work[["index_code", "trade_time", "freq", "open", "high", "low", "close", "volume", "amount"]]
    work = work.dropna(subset=["trade_time"])
    request_start_dt = start_dt
    if _is_single_day_window(request_start_dt, end_dt):
        request_start_dt = request_start_dt - timedelta(days=10)
    work = work[(work["trade_time"] >= request_start_dt) & (work["trade_time"] <= end_dt)]
    work, _ = calibrate_quote_units(work)
    if freq == "1w":
        work = work.set_index("trade_time").resample("W-FRI", label="left", closed="left").agg({"index_code": "last", "freq": "last", "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "amount": "sum"}).reset_index()
        work = work[work["close"].notna()]
    elif freq == "1mo":
        work = work.set_index("trade_time").resample("ME", label="left", closed="left").agg({"index_code": "last", "freq": "last", "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "amount": "sum"}).reset_index()
        work = work[work["close"].notna()]
    return work.drop_duplicates(subset=["index_code", "trade_time", "freq"], keep="last").sort_values("trade_time").reset_index(drop=True)


def _frame_to_stock_quotes(df: pd.DataFrame, freq: str, adjust: str) -> list[StockQuoteItem]:
    items: list[StockQuoteItem] = []
    if df.empty:
        return items
    work = df.sort_values("trade_time").copy()
    if "pre_close" not in work.columns:
        work["pre_close"] = pd.NA
    if "change" not in work.columns:
        work["change"] = pd.NA
    if "pct_chg" not in work.columns:
        work["pct_chg"] = pd.NA
    shifted_pre_close = work["close"].shift(1)
    work["pre_close"] = work["pre_close"].fillna(shifted_pre_close)
    work["change"] = work["change"].fillna(work["close"] - work["pre_close"])
    work["pct_chg"] = work["pct_chg"].fillna(work["change"] / work["pre_close"] * 100)
    for _, row in work.iterrows():
        items.append(
            StockQuoteItem(
                code=str(row["code"]),
                trade_time=format_datetime_value(row["trade_time"], freq),
                freq=freq,
                open=float(row["open"]) if pd.notna(row["open"]) else None,
                high=float(row["high"]) if pd.notna(row["high"]) else None,
                low=float(row["low"]) if pd.notna(row["low"]) else None,
                close=float(row["close"]) if pd.notna(row["close"]) else None,
                pre_close=float(row["pre_close"]) if pd.notna(row["pre_close"]) else None,
                change=float(row["change"]) if pd.notna(row["change"]) else None,
                pct_chg=float(row["pct_chg"]) if pd.notna(row["pct_chg"]) else None,
                volume=float(row["volume"]) if pd.notna(row["volume"]) else None,
                amount=float(row["amount"]) if pd.notna(row["amount"]) else None,
                adjust=adjust,
                is_suspended=bool(row["is_suspended"]) if "is_suspended" in row and pd.notna(row["is_suspended"]) else False,
                is_st=bool(row["is_st"]) if "is_st" in row and pd.notna(row["is_st"]) else False,
            )
        )
    return items


def _frame_to_index_quotes(df: pd.DataFrame, freq: str) -> list[IndexQuoteItem]:
    items: list[IndexQuoteItem] = []
    if df.empty:
        return items
    work = df.sort_values("trade_time").copy()
    work["pre_close"] = work["close"].shift(1)
    work["change"] = work["close"] - work["pre_close"]
    work["pct_chg"] = work["change"] / work["pre_close"] * 100
    for _, row in work.iterrows():
        items.append(
            IndexQuoteItem(
                index_code=str(row["index_code"]),
                trade_time=format_datetime_value(row["trade_time"], freq),
                freq=freq,
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


def _tail_request_window(start_dt: datetime, end_dt: datetime, count: int | None) -> tuple[datetime, datetime]:
    if count:
        return start_dt, end_dt
    if start_dt.date() == end_dt.date():
        return start_dt - timedelta(days=10), end_dt
    return start_dt, end_dt


def _is_single_day_window(start_dt: datetime, end_dt: datetime) -> bool:
    return start_dt.date() == end_dt.date()


def _stock_daily_cache_complete(frame: pd.DataFrame) -> bool:
    required_columns = {"close", "pre_close", "change", "pct_chg", "amount"}
    if frame.empty or not required_columns.issubset(set(frame.columns)):
        return False
    return not frame[list(required_columns)].isna().any(axis=1).any()


def get_stock_quotes(
    codes: list[str],
    freq: str,
    trade_date: str,
    start_date: str,
    end_date: str,
    start_time: str,
    end_time: str,
    count: int | None,
    adjust: str,
) -> list[StockQuoteItem]:
    if freq == "tick":
        return []
    start_dt, end_dt = _resolve_time_window(trade_date, start_date, end_date, start_time, end_time, count, freq.endswith("m"))
    items: list[StockQuoteItem] = []
    for code in codes:
        normalized_code = normalize_stock_code(code)
        cache_path = build_cache_path("akshare", ["stocks", "quotes"], {"code": normalized_code, "freq": freq, "adjust": adjust})
        cache_df = read_cache_frame(cache_path)
        filtered_cache = filter_frame_by_datetime_range(cache_df, "trade_time", start_dt, end_dt)
        cache_incomplete = not freq.endswith("m") and not _stock_daily_cache_complete(filtered_cache)
        if filtered_cache.empty or cache_incomplete or (count and len(filtered_cache) < count):
            if freq.endswith("m"):
                fetched_df = _fetch_stock_intraday_frame(normalized_code, freq, start_dt, end_dt)
            else:
                fetched_df = _fetch_stock_daily_frame(normalized_code, freq, start_dt, end_dt, adjust)
            if not fetched_df.empty:
                if cache_incomplete and {"code", "trade_time", "freq"}.issubset(set(cache_df.columns)):
                    fetched_keys = set(zip(fetched_df["code"], fetched_df["trade_time"], fetched_df["freq"]))
                    cache_keys = list(zip(cache_df["code"], cache_df["trade_time"], cache_df["freq"]))
                    cache_df = cache_df[[key not in fetched_keys for key in cache_keys]]
                cache_df = merge_cache_frame(cache_df, fetched_df, ["code", "trade_time", "freq"], ["trade_time"])
                write_cache_frame(cache_path, cache_df)
        filtered_df = filter_frame_by_datetime_range(cache_df, "trade_time", start_dt, end_dt)
        if not freq.endswith("m") and _is_single_day_window(start_dt, end_dt) and not cache_df.empty and "trade_time" in cache_df.columns:
            cache_times = pd.to_datetime(cache_df["trade_time"], errors="coerce")
            previous_df = cache_df[cache_times < start_dt]
            previous_df = latest_n_rows(previous_df, "trade_time", 1)
            if not previous_df.empty:
                filtered_df = merge_cache_frame(previous_df, filtered_df, ["code", "trade_time", "freq"], ["trade_time"])
        filtered_df = latest_n_rows(filtered_df, "trade_time", count)
        quote_items = _frame_to_stock_quotes(filtered_df, freq, adjust)
        if not freq.endswith("m") and _is_single_day_window(start_dt, end_dt):
            quote_items = [item for item in quote_items if item.trade_time == start_dt.strftime("%Y-%m-%d")]
        items.extend(quote_items)
    return items


def get_index_quotes(index_codes: list[str], freq: str, trade_date: str, start_date: str, end_date: str, count: int | None) -> list[IndexQuoteItem]:
    start_dt, end_dt = _resolve_time_window(trade_date, start_date, end_date, "", "", count, False)
    items: list[IndexQuoteItem] = []
    request_count = (count + 1) if count else (2 if _is_single_day_window(start_dt, end_dt) else None)
    fetch_start_dt, fetch_end_dt = _tail_request_window(start_dt, end_dt, count)
    for index_code in index_codes:
        normalized_code = normalize_index_code(index_code)
        cache_path = build_cache_path("akshare", ["indexes", "quotes"], {"index_code": normalized_code, "freq": freq})
        cache_df = read_cache_frame(cache_path)
        filtered_cache = filter_frame_by_datetime_range(cache_df, "trade_time", start_dt, end_dt)
        if filtered_cache.empty or (request_count and len(filtered_cache) < request_count):
            fetched_df = _fetch_index_daily_frame(normalized_code, freq, fetch_start_dt, fetch_end_dt)
            if not fetched_df.empty:
                cache_df = merge_cache_frame(cache_df, fetched_df, ["index_code", "trade_time", "freq"], ["trade_time"])
                write_cache_frame(cache_path, cache_df)
        filtered_df = filter_frame_by_datetime_range(cache_df, "trade_time", start_dt, end_dt)
        if request_count and start_dt is not None and not cache_df.empty:
            cache_times = pd.to_datetime(cache_df["trade_time"], errors="coerce")
            previous_df = cache_df[cache_times < start_dt]
            previous_df = latest_n_rows(previous_df, "trade_time", 1)
            if not previous_df.empty:
                filtered_df = merge_cache_frame(previous_df, filtered_df, ["index_code", "trade_time", "freq"], ["trade_time"])
        filtered_df = latest_n_rows(filtered_df, "trade_time", request_count)
        quote_items = _frame_to_index_quotes(filtered_df, freq)
        if _is_single_day_window(start_dt, end_dt):
            quote_items = [item for item in quote_items if item.trade_time == start_dt.strftime("%Y-%m-%d")]
        if count:
            quote_items = quote_items[-count:]
        items.extend(quote_items)
    return items


def get_index_members(index_code: str, trade_date: str) -> list[IndexMemberItem]:
    del trade_date
    result = _call_ak("index_stock_cons", ak.index_stock_cons, symbol=normalize_index_code(index_code))
    if result is None or result.empty:
        return []
    work = result.copy()
    work["code"] = work["品种代码"].astype(str).str.zfill(6)
    work["name"] = work["品种名称"].fillna("").astype(str)
    items: list[IndexMemberItem] = []
    for _, row in work.iterrows():
        items.append(
            IndexMemberItem(
                index_code=normalize_index_code(index_code),
                code=str(row["code"]),
                name=str(row["name"]),
                weight=None,
                trade_date="",
            )
        )
    return items


def get_trading_calendar(exchange: str, start_date: str, end_date: str, is_open: bool | None) -> list[TradingCalendarItem]:
    del is_open
    if exchange not in {"SSE", "SZSE", "BSE"}:
        return []
    cache_path = build_cache_path("akshare", ["markets", "calendar", "trading"], {"exchange": exchange.lower()})
    cache_df = read_cache_frame(cache_path)
    if cache_df.empty:
        fetched_df = _call_ak("tool_trade_date_hist_sina", ak.tool_trade_date_hist_sina)
        if fetched_df is not None and not fetched_df.empty:
            cache_df = fetched_df.copy()
            write_cache_frame(cache_path, cache_df)
    filtered_df = filter_frame_by_date_range(cache_df, "trade_date", start_date, end_date)
    if filtered_df.empty:
        return []
    items: list[TradingCalendarItem] = []
    for _, row in filtered_df.iterrows():
        trade_date_value = format_date_value(row["trade_date"])
        items.append(
            TradingCalendarItem(
                exchange=exchange,
                trade_date=trade_date_value,
                is_open=True,
                pretrade_date="",
            )
        )
    return items


def _board_catalog_fetchers(category: str) -> tuple[tuple[str, object], ...]:
    if category == "concept":
        return (
            ("stock_board_concept_name_em", ak.stock_board_concept_name_em),
            ("stock_board_concept_name_ths", ak.stock_board_concept_name_ths),
        )
    return (
        ("stock_board_industry_name_em", ak.stock_board_industry_name_em),
        ("stock_board_industry_name_ths", ak.stock_board_industry_name_ths),
    )


def _normalize_board_catalog_frame(fetched_df: pd.DataFrame, category: str) -> pd.DataFrame:
    if fetched_df is None or fetched_df.empty:
        return pd.DataFrame()
    code_column = ""
    name_column = ""
    for candidate in ("板块代码", "代码", "code"):
        if candidate in fetched_df.columns:
            code_column = candidate
            break
    for candidate in ("板块名称", "名称", "name"):
        if candidate in fetched_df.columns:
            name_column = candidate
            break
    if code_column == "" or name_column == "":
        return pd.DataFrame()
    work = fetched_df.copy()
    work["board_code"] = work[code_column].fillna("").astype(str).str.upper()
    work["board_name"] = work[name_column].fillna("").astype(str)
    work["category"] = category
    work["status"] = "active"
    work = work[(work["board_code"] != "") & (work["board_name"] != "")]
    return work[["board_code", "board_name", "category", "status"]]


def _fetch_board_catalog_frame(category: str) -> pd.DataFrame:
    for api_name, fetcher in _board_catalog_fetchers(category):
        try:
            frame = _normalize_board_catalog_frame(_call_ak(api_name, fetcher), category)
        except Exception:
            continue
        if not frame.empty:
            return frame
    return pd.DataFrame()


def _load_board_catalog_frame(category: str) -> pd.DataFrame:
    if category not in BOARD_CATEGORIES:
        return pd.DataFrame()
    cache_path = build_cache_path("akshare", ["boards", "catalog"], {"category": category})
    cache_df = read_cache_frame(cache_path)
    if cache_df.empty:
        fetched_df = _fetch_board_catalog_frame(category)
        if not fetched_df.empty:
            write_cache_frame(cache_path, fetched_df)
            cache_df = fetched_df
    return cache_df
    if cache_df.empty:
        if category == "concept":
            fetched_df = _call_ak("stock_board_concept_name_em", ak.stock_board_concept_name_em)
        else:
            fetched_df = _call_ak("stock_board_industry_name_em", ak.stock_board_industry_name_em)
        if fetched_df is not None and not fetched_df.empty:
            work = fetched_df.copy()
            work["board_code"] = work["板块代码"].fillna("").astype(str).str.upper()
            work["board_name"] = work["板块名称"].fillna("").astype(str)
            work["category"] = category
            work["status"] = "active"
            cache_df = work[["board_code", "board_name", "category", "status"]]
            write_cache_frame(cache_path, cache_df)
    return cache_df


def _board_row(board_code: str) -> pd.Series | None:
    normalized_code = str(board_code).upper()
    if normalized_code == "":
        return None
    for category in BOARD_CATEGORIES:
        try:
            catalog_df = _load_board_catalog_frame(category)
        except Exception:
            continue
        if catalog_df.empty:
            continue
        matched = catalog_df[catalog_df["board_code"].astype(str).str.upper() == normalized_code]
        if matched.empty:
            continue
        return matched.iloc[0]
    return None


def _board_symbol_and_category(board_code: str) -> tuple[str, str]:
    row = _board_row(board_code)
    if row is None:
        return "", ""
    board_name = _text_value(row.get("board_name"))
    category = _text_value(row.get("category"))
    return board_name, category


def get_board_catalog(category: str, market: str, status: str, limit: int, offset: int) -> list[BoardCatalogItem]:
    if market and market != "a_share":
        return []
    actual_categories = (category,) if category in BOARD_CATEGORIES else BOARD_CATEGORIES
    frames = []
    for item in actual_categories:
        try:
            frame = _load_board_catalog_frame(item)
            if frame is not None and not frame.empty:
                frames.append(frame)
        except Exception:
            pass
    if frames == []:
        return []
    work = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["board_code"], keep="last")
    if status:
        work = work[work["status"] == status]
    work = work.sort_values("board_code").iloc[offset: offset + limit]
    return [
        BoardCatalogItem(
            board_code=str(row["board_code"]),
            board_name=str(row["board_name"]),
            category=str(row["category"]),
            market="a_share",
            status=str(row["status"]),
        )
        for _, row in work.iterrows()
    ]


def get_board_profile(board_code: str) -> BoardCatalogItem | None:
    row = _board_row(board_code)
    if row is None:
        return None
    return BoardCatalogItem(
        board_code=str(row["board_code"]),
        board_name=str(row["board_name"]),
        category=str(row["category"]),
        market="a_share",
        status=str(row["status"]),
    )


def get_board_categories(parent_code: str, level: int | None) -> list[BoardCategoryItem]:
    items = [
        BoardCategoryItem(category_code="concept", category_name="概念板块", parent_code="", level=1, sort_order=1),
        BoardCategoryItem(category_code="industry", category_name="行业板块", parent_code="", level=1, sort_order=2),
    ]
    if parent_code:
        items = [item for item in items if item.parent_code == parent_code]
    if level is not None:
        items = [item for item in items if item.level == level]
    return items


def _ths_member_page_url(category: str, code: str, page: int | None) -> str:
    path = "gn" if category == "concept" else "thshy"
    if page is None:
        return f"http://q.10jqka.com.cn/{path}/detail/code/{code}/"
    return f"http://q.10jqka.com.cn/{path}/detail/code/{code}/field/199112/order/desc/page/{page}/ajax/1/"


def _ths_headers(referer: str) -> dict[str, str]:
    from akshare.datasets import get_ths_js
    import py_mini_racer

    js_code = py_mini_racer.MiniRacer()
    with open(get_ths_js("ths.js"), encoding="utf-8") as file_obj:
        js_code.eval(file_obj.read())
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36",
        "Cookie": f"v={js_code.call('v')}",
        "Referer": referer,
    }


def _read_ths_member_frame(html: str) -> pd.DataFrame:
    try:
        frames = pd.read_html(StringIO(html))
    except ValueError:
        return pd.DataFrame()
    for frame in frames:
        if "代码" in frame.columns and "名称" in frame.columns:
            return frame[["代码", "名称"]]
    return pd.DataFrame()


def _ths_page_count(html: str) -> int:
    matched = re.search(r'page_info[^>]*>\s*\d+/(\d+)', html)
    if matched is None:
        return 1
    number = pd.to_numeric(matched.group(1), errors="coerce")
    return int(number) if pd.notna(number) else 1


def _fetch_ths_board_members_frame(board_code: str, category: str) -> pd.DataFrame:
    if category not in BOARD_CATEGORIES or not board_code.isdigit():
        return pd.DataFrame()
    detail_url = _ths_member_page_url(category, board_code, None)
    try:
        first_response = requests.get(detail_url, headers=_ths_headers(detail_url), timeout=15)
    except Exception:
        return pd.DataFrame()
    first_frame = _read_ths_member_frame(first_response.text)
    frames = [first_frame] if not first_frame.empty else []
    page_count = _ths_page_count(first_response.text)
    for page in range(2, page_count + 1):
        try:
            response = requests.get(_ths_member_page_url(category, board_code, page), headers=_ths_headers(detail_url), timeout=15)
        except Exception:
            continue
        frame = _read_ths_member_frame(response.text)
        if not frame.empty:
            frames.append(frame)
    if frames == []:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["代码"], keep="last")


def get_board_members(board_code: str, trade_date: str) -> list[BoardMemberItem]:
    del trade_date
    normalized = str(board_code).upper()
    symbol, category = _board_symbol_and_category(normalized)
    member_sources = (
        (("stock_board_concept_cons_em", ak.stock_board_concept_cons_em),) if category == "concept" else
        (("stock_board_industry_cons_em", ak.stock_board_industry_cons_em),) if category == "industry" else
        (
            ("stock_board_concept_cons_em", ak.stock_board_concept_cons_em),
            ("stock_board_industry_cons_em", ak.stock_board_industry_cons_em),
        )
    )
    request_symbols = [normalized]
    if symbol != "" and symbol != normalized:
        request_symbols.append(symbol)
    result = pd.DataFrame()
    for api_name, func in member_sources:
        for request_symbol in request_symbols:
            try:
                current = _call_ak(api_name, func, symbol=request_symbol)
            except Exception:
                continue
            if current is not None and not current.empty:
                result = current
                break
        if not result.empty:
            break
    if result.empty:
        result = _fetch_ths_board_members_frame(normalized, category)
    if result.empty:
        return []
    items: list[BoardMemberItem] = []
    for _, row in result.iterrows():
        code = normalize_stock_code(str(row.get("代码", "")))
        if code == "":
            continue
        code = code.zfill(6)
        items.append(
            BoardMemberItem(
                board_code=normalized,
                code=code,
                name=str(row.get("名称", "")),
                weight=None,
                join_date="",
            )
        )
    return sorted(items, key=lambda item: item.code)


def _fetch_board_quote_frame(board_code: str, freq: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    normalized = str(board_code).upper()
    if freq not in BOARD_FREQ_MAP:
        return pd.DataFrame()
    symbol, category = _board_symbol_and_category(normalized)
    if symbol == "" or category == "":
        return pd.DataFrame()
    request_start_dt = start_dt
    if request_start_dt == end_dt:
        request_start_dt = request_start_dt - timedelta(days=10)
    result = pd.DataFrame()
    try:
        if category == "concept":
            result = _call_ak(
                "stock_board_concept_hist_em",
                ak.stock_board_concept_hist_em,
                symbol=symbol,
                period=BOARD_FREQ_MAP[freq][0],
                start_date=request_start_dt.strftime("%Y%m%d"),
                end_date=end_dt.strftime("%Y%m%d"),
                adjust="",
            )
        else:
            result = _call_ak(
                "stock_board_industry_hist_em",
                ak.stock_board_industry_hist_em,
                symbol=symbol,
                start_date=request_start_dt.strftime("%Y%m%d"),
                end_date=end_dt.strftime("%Y%m%d"),
                period=BOARD_FREQ_MAP[freq][1],
                adjust="",
            )
    except Exception:
        return pd.DataFrame()
    if result.empty:
        return pd.DataFrame()
    work = result.copy()
    work["board_code"] = normalized
    work["trade_time"] = pd.to_datetime(work["日期"], errors="coerce")
    work["freq"] = freq
    work["open"] = pd.to_numeric(work["开盘"], errors="coerce")
    work["high"] = pd.to_numeric(work["最高"], errors="coerce")
    work["low"] = pd.to_numeric(work["最低"], errors="coerce")
    work["close"] = pd.to_numeric(work["收盘"], errors="coerce")
    work["volume"] = pd.to_numeric(work["成交量"], errors="coerce")
    work["amount"] = pd.to_numeric(work["成交额"], errors="coerce")
    work = work[["board_code", "trade_time", "freq", "open", "high", "low", "close", "volume", "amount"]]
    work = work.dropna(subset=["trade_time"])
    return work.drop_duplicates(subset=["board_code", "trade_time", "freq"], keep="last").sort_values("trade_time").reset_index(drop=True)


def get_board_quotes(board_codes: list[str], freq: str, trade_date: str, start_date: str, end_date: str, start_time: str, end_time: str, count: int | None) -> list[BoardQuoteItem]:
    del start_time
    del end_time
    if freq not in BOARD_FREQ_MAP:
        return []
    start_dt, end_dt = _resolve_time_window(trade_date, start_date, end_date, "", "", count, False)
    items: list[BoardQuoteItem] = []
    request_count = (count + 1) if count else (2 if _is_single_day_window(start_dt, end_dt) else None)
    for board_code in board_codes:
        normalized = str(board_code).upper()
        cache_path = build_cache_path("akshare", ["boards", "quotes"], {"board_code": normalized, "freq": freq})
        cache_df = read_cache_frame(cache_path)
        filtered_cache = filter_frame_by_datetime_range(cache_df, "trade_time", start_dt, end_dt)
        if filtered_cache.empty or (request_count and len(filtered_cache) < request_count):
            fetched_df = _fetch_board_quote_frame(normalized, freq, start_dt, end_dt)
            if not fetched_df.empty:
                cache_df = merge_cache_frame(cache_df, fetched_df, ["board_code", "trade_time", "freq"], ["trade_time"])
                write_cache_frame(cache_path, cache_df)
        filtered_df = filter_frame_by_datetime_range(cache_df, "trade_time", start_dt, end_dt)
        if request_count and not cache_df.empty:
            cache_times = pd.to_datetime(cache_df["trade_time"], errors="coerce")
            previous_df = cache_df[cache_times < start_dt]
            previous_df = latest_n_rows(previous_df, "trade_time", 1)
            if not previous_df.empty:
                filtered_df = merge_cache_frame(previous_df, filtered_df, ["board_code", "trade_time", "freq"], ["trade_time"])
        filtered_df = latest_n_rows(filtered_df, "trade_time", request_count)
        if filtered_df.empty:
            continue
        work = filtered_df.sort_values("trade_time").copy()
        work["pre_close"] = work["close"].shift(1)
        work["change"] = work["close"] - work["pre_close"]
        work["pct_chg"] = work["change"] / work["pre_close"] * 100
        if _is_single_day_window(start_dt, end_dt):
            work = work[work["trade_time"].dt.strftime("%Y-%m-%d") == start_dt.strftime("%Y-%m-%d")]
        if count:
            work = latest_n_rows(work, "trade_time", count)
        for _, row in work.iterrows():
            items.append(
                BoardQuoteItem(
                    board_code=normalized,
                    trade_time=format_datetime_value(row["trade_time"], freq),
                    freq=freq,
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


def get_board_money_flow(board_code: str, trade_date: str, start_date: str, end_date: str, scope: str) -> list[BoardMoneyFlowItem]:
    normalized = str(board_code).upper()
    symbol, category = _board_symbol_and_category(normalized)
    if scope and category and scope != category:
        return []
    if category != "industry":
        return []
    try:
        result = _call_ak("stock_sector_fund_flow_hist", ak.stock_sector_fund_flow_hist, symbol=symbol)
    except Exception:
        result = pd.DataFrame()
    if result is None or result.empty:
        if trade_date != "":
            return [item for item in get_board_daily_money_flow_snapshot(trade_date, category, 10000, 0) if item.board_code == normalized]
        return []
    actual_trade_date = format_date_value(trade_date)
    actual_start = format_date_value(start_date)
    actual_end = format_date_value(end_date)
    items: list[BoardMoneyFlowItem] = []
    for _, row in result.iterrows():
        row_date = format_date_value(row.get("日期", ""))
        if actual_trade_date and row_date != actual_trade_date:
            continue
        if actual_start and row_date < actual_start:
            continue
        if actual_end and row_date > actual_end:
            continue
        items.append(
            BoardMoneyFlowItem(
                board_code=normalized,
                trade_date=row_date,
                scope=category,
                inflow=None,
                outflow=None,
                net_inflow=_float_value(row.get("主力净流入-净额")),
            )
        )
    return sorted(items, key=lambda item: item.trade_date)


def _snapshot_board_code_from_name(board_name: str, category: str) -> str:
    catalog_df = _load_board_catalog_frame(category)
    if catalog_df.empty:
        return ""
    matched = catalog_df[catalog_df["board_name"] == board_name]
    if matched.empty:
        return ""
    return str(matched.iloc[0]["board_code"])


def get_board_daily_money_flow_snapshot(trade_date: str, scope: str, limit: int, offset: int) -> list[BoardMoneyFlowItem]:
    actual_scope = scope if scope in BOARD_CATEGORIES else "industry"
    if actual_scope == "concept":
        result = _call_ak("stock_fund_flow_concept", ak.stock_fund_flow_concept, symbol="即时")
    else:
        result = _call_ak("stock_fund_flow_industry", ak.stock_fund_flow_industry, symbol="即时")
    if result is None or result.empty:
        return []
    actual_trade_date = format_date_value(trade_date) or datetime.now().strftime("%Y-%m-%d")
    items: list[BoardMoneyFlowItem] = []
    for _, row in result.iloc[offset: offset + limit].iterrows():
        board_name = str(row.get("行业", ""))
        board_code = _snapshot_board_code_from_name(board_name, actual_scope)
        if board_code == "":
            continue
        items.append(
            BoardMoneyFlowItem(
                board_code=board_code,
                trade_date=actual_trade_date,
                scope=actual_scope,
                inflow=_money_text_to_float(row.get("流入资金")),
                outflow=_money_text_to_float(row.get("流出资金")),
                net_inflow=_money_text_to_float(row.get("净额")),
            )
        )
    return sorted(items, key=lambda item: (item.board_code, item.trade_date))


def get_market_capital_flow(trade_date: str, start_date: str, end_date: str) -> list[MarketCapitalFlowItem]:
    actual_start, actual_end = _date_window(trade_date, start_date, end_date, 120)
    result = _call_ak("stock_market_fund_flow", ak.stock_market_fund_flow)
    if result is None or result.empty:
        return []
    items: list[MarketCapitalFlowItem] = []
    for _, row in result.iterrows():
        row_date = format_date_value(row.get("日期", ""))
        if row_date < actual_start or row_date > actual_end:
            continue
        items.append(
            MarketCapitalFlowItem(
                trade_date=row_date,
                market="all",
                main_inflow=None,
                main_outflow=None,
                net_inflow=_float_value(row.get("主力净流入-净额")),
            )
        )
    return sorted(items, key=lambda item: (item.trade_date, item.market))


def get_connect_capital_flow(trade_date: str, start_date: str, end_date: str) -> list[ConnectCapitalFlowItem]:
    actual_start, actual_end = _date_window(trade_date, start_date, end_date, 120)
    market_symbols = {
        "northbound": "北向资金",
        "sh_connect": "沪股通",
        "sz_connect": "深股通",
        "southbound": "南向资金",
        "sh_hk": "港股通沪",
        "sz_hk": "港股通深",
    }
    items: list[ConnectCapitalFlowItem] = []
    for market, symbol in market_symbols.items():
        result = _call_ak("stock_hsgt_hist_em", ak.stock_hsgt_hist_em, symbol=symbol)
        if result is None or result.empty:
            continue
        for _, row in result.iterrows():
            row_date = format_date_value(row.get("日期", ""))
            if row_date < actual_start or row_date > actual_end:
                continue
            items.append(
                ConnectCapitalFlowItem(
                    trade_date=row_date,
                    market=market,
                    buy_amount=_float_value(row.get("买入成交额")),
                    sell_amount=_float_value(row.get("卖出成交额")),
                    net_amount=_float_value(row.get("当日成交净买额")),
                )
            )
    return sorted(_with_akshare_northbound_totals(items), key=lambda item: (item.trade_date, item.market))


def _with_akshare_northbound_totals(items: list[ConnectCapitalFlowItem]) -> list[ConnectCapitalFlowItem]:
    keyed_items = {(item.trade_date, item.market): item for item in items}
    dates = sorted({item.trade_date for item in items})
    for trade_date in dates:
        existing = keyed_items.get((trade_date, "northbound"))
        if existing is not None and existing.buy_amount is not None and existing.sell_amount is not None and existing.net_amount is not None:
            continue
        sh_item = keyed_items.get((trade_date, "sh_connect"))
        sz_item = keyed_items.get((trade_date, "sz_connect"))
        if sh_item is None or sz_item is None:
            continue
        buy_amount = _sum_optional_values(sh_item.buy_amount, sz_item.buy_amount)
        sell_amount = _sum_optional_values(sh_item.sell_amount, sz_item.sell_amount)
        net_amount = _sum_optional_values(sh_item.net_amount, sz_item.net_amount)
        keyed_items[(trade_date, "northbound")] = ConnectCapitalFlowItem(
            trade_date=trade_date,
            market="northbound",
            buy_amount=buy_amount,
            sell_amount=sell_amount,
            net_amount=net_amount,
        )
    return list(keyed_items.values())


def _sum_optional_values(first: float | None, second: float | None) -> float | None:
    if first is None or second is None:
        return None
    return first + second


def get_block_trades(trade_date: str, start_date: str, end_date: str, code: str, limit: int) -> list[BlockTradeItem]:
    actual_start, actual_end = _date_window(trade_date, start_date, end_date, 30)
    result = _call_ak("stock_dzjy_mrmx", ak.stock_dzjy_mrmx, symbol="A股", start_date=actual_start.replace("-", ""), end_date=actual_end.replace("-", ""))
    if result is None or result.empty:
        return []
    normalized_code = normalize_stock_code(code)
    items: list[BlockTradeItem] = []
    for _, row in result.iterrows():
        row_code = normalize_stock_code(str(row.get("证券代码", "")))
        if normalized_code and row_code != normalized_code:
            continue
        items.append(
            BlockTradeItem(
                trade_date=format_date_value(row.get("交易日期", "")),
                code=row_code,
                name=str(row.get("证券简称", "")),
                price=_float_value(row.get("成交价")),
                volume=_float_value(row.get("成交量")),
                amount=_float_value(row.get("成交额")),
                buyer=str(row.get("买方营业部", "")),
                seller=str(row.get("卖方营业部", "")),
            )
        )
    return sorted(items, key=lambda item: (item.trade_date, item.code, item.buyer, item.seller))[:limit]


def get_dragon_tiger(trade_date: str, start_date: str, end_date: str, code: str, limit: int) -> list[DragonTigerItem]:
    actual_start, actual_end = _date_window(trade_date, start_date, end_date, 30)
    result = _call_ak("stock_lhb_detail_em", ak.stock_lhb_detail_em, start_date=actual_start.replace("-", ""), end_date=actual_end.replace("-", ""))
    if result is None or result.empty:
        return []
    normalized_code = normalize_stock_code(code)
    items: list[DragonTigerItem] = []
    for _, row in result.iterrows():
        row_code = normalize_stock_code(str(row.get("代码", "")))
        if normalized_code and row_code != normalized_code:
            continue
        items.append(
            DragonTigerItem(
                trade_date=format_date_value(row.get("上榜日", "")),
                code=row_code,
                name=str(row.get("名称", "")),
                reason=str(row.get("上榜原因", row.get("解读", ""))),
                buy_amount=_float_value(row.get("龙虎榜买入额")),
                sell_amount=_float_value(row.get("龙虎榜卖出额")),
                net_amount=_float_value(row.get("龙虎榜净买额")),
            )
        )
    return sorted(items, key=lambda item: (item.trade_date, item.code, item.reason))[:limit]


def get_dragon_tiger_institutions(trade_date: str, start_date: str, end_date: str, code: str, limit: int) -> list[DragonTigerInstitutionItem]:
    actual_start, actual_end = _date_window(trade_date, start_date, end_date, 30)
    result = _call_ak("stock_lhb_jgmmtj_em", ak.stock_lhb_jgmmtj_em, start_date=actual_start.replace("-", ""), end_date=actual_end.replace("-", ""))
    if result is None or result.empty:
        return []
    normalized_code = normalize_stock_code(code)
    items: list[DragonTigerInstitutionItem] = []
    for _, row in result.iterrows():
        row_code = normalize_stock_code(str(row.get("代码", "")))
        if normalized_code and row_code != normalized_code:
            continue
        buy_count = pd.to_numeric(row.get("买方机构数"), errors="coerce")
        sell_count = pd.to_numeric(row.get("卖方机构数"), errors="coerce")
        institution_count = None
        if pd.notna(buy_count) or pd.notna(sell_count):
            institution_count = int((0 if pd.isna(buy_count) else buy_count) + (0 if pd.isna(sell_count) else sell_count))
        items.append(
            DragonTigerInstitutionItem(
                trade_date=format_date_value(row.get("上榜日期", "")),
                code=row_code,
                name=str(row.get("名称", "")),
                buy_amount=_float_value(row.get("机构买入总额")),
                sell_amount=_float_value(row.get("机构卖出总额")),
                net_amount=_float_value(row.get("机构买入净额")),
                institution_count=institution_count,
            )
        )
    return sorted(items, key=lambda item: (item.trade_date, item.code))[:limit]


def get_hot_money_details(trade_date: str, start_date: str, end_date: str, name: str, limit: int) -> list[HotMoneyDetailItem]:
    actual_start, actual_end = _date_window(trade_date, start_date, end_date, 30)
    active_frame = _call_ak("stock_lhb_hyyyb_em", ak.stock_lhb_hyyyb_em, start_date=actual_start.replace("-", ""), end_date=actual_end.replace("-", ""))
    if active_frame is None or active_frame.empty:
        return []
    work = active_frame.copy()
    if "上榜日" in work.columns:
        work["上榜日"] = pd.to_datetime(work["上榜日"], errors="coerce")
    if "总买卖净额" in work.columns:
        work["总买卖净额"] = pd.to_numeric(work["总买卖净额"], errors="coerce")
    sort_columns = [column for column in ("上榜日", "总买卖净额") if column in work.columns]
    if sort_columns:
        work = work.sort_values(sort_columns, ascending=False)

    collected: list[HotMoneyDetailItem] = []
    seen: set[tuple[str, str, str]] = set()
    max_departments = min(max(limit, 6), 12)
    for _, active_row in work.head(max_departments).iterrows():
        department_code = str(active_row.get("营业部代码", ""))
        if department_code == "":
            continue
        detail_frame = _call_ak("stock_lhb_yyb_detail_em", ak.stock_lhb_yyb_detail_em, symbol=department_code)
        if detail_frame is None or detail_frame.empty:
            continue
        for _, row in detail_frame.iterrows():
            row_date = format_date_value(row.get("交易日期", ""))
            if not _date_in_window(row_date, actual_start, actual_end):
                continue
            department_name = str(row.get("营业部名称", ""))
            if name and name not in department_name:
                continue
            row_code = normalize_stock_code(str(row.get("股票代码", "")))
            key = (row_date, department_name, row_code)
            if key in seen:
                continue
            seen.add(key)
            collected.append(
                HotMoneyDetailItem(
                    trade_date=row_date,
                    name=department_name,
                    code=row_code,
                    stock_name=str(row.get("股票名称", "")),
                    buy_amount=_float_value(row.get("买入金额")),
                    sell_amount=_float_value(row.get("卖出金额")),
                    net_amount=_float_value(row.get("净额")),
                )
            )
            if len(collected) >= limit:
                return sorted(collected, key=lambda item: (item.trade_date, item.name, item.code), reverse=True)[:limit]
    return sorted(collected, key=lambda item: (item.trade_date, item.name, item.code), reverse=True)[:limit]


@lru_cache(maxsize=1)
def _latest_akshare_trade_date() -> str:
    result = _call_ak("tool_trade_date_hist_sina", ak.tool_trade_date_hist_sina)
    if result is None or result.empty or "trade_date" not in result.columns:
        return ""
    today_text = datetime.now().strftime("%Y-%m-%d")
    dates = sorted(format_date_value(value) for value in result["trade_date"].tolist())
    valid_dates = [value for value in dates if value != "" and value <= today_text]
    return valid_dates[-1] if valid_dates else ""


def _stock_tick_symbol(code: str) -> str:
    normalized_code = normalize_stock_code(code)
    market = _stock_market(normalized_code)
    return f"{market}{normalized_code}"


def _open_auction_from_tick(code: str, trade_date: str) -> AuctionItem | None:
    actual_code = normalize_stock_code(code)
    actual_trade_date = format_date_value(trade_date)
    if actual_code == "" or actual_trade_date == "" or actual_trade_date != _latest_akshare_trade_date():
        return None
    result = _call_ak("stock_zh_a_tick_tx_js", ak.stock_zh_a_tick_tx_js, symbol=_stock_tick_symbol(actual_code))
    if result is None or result.empty or "成交时间" not in result.columns:
        return None
    work = result.copy()
    work["成交时间"] = work["成交时间"].astype(str)
    work = work[(work["成交时间"] >= "09:25:00") & (work["成交时间"] < "09:26:00")]
    if work.empty:
        return None
    row = work.sort_values("成交时间").iloc[0]
    return AuctionItem(
        code=actual_code,
        trade_date=actual_trade_date,
        auction_time=str(row.get("成交时间", "")),
        price=_float_value(row.get("成交价格")),
        volume=_float_value(row.get("成交量")),
        amount=_float_value(row.get("成交金额")),
        session="open",
    )


def get_auctions(code: str, session: str, trade_date: str, start_date: str, end_date: str) -> list[AuctionItem]:
    actual_session = session or "open"
    if actual_session != "open":
        return []
    actual_trade_date = format_date_value(trade_date or end_date or start_date)
    item = _open_auction_from_tick(code, actual_trade_date)
    return [] if item is None else [item]


def get_market_open_auctions(codes: str, trade_date: str) -> list[AuctionItem]:
    items: list[AuctionItem] = []
    for code in split_csv(codes):
        item = _open_auction_from_tick(code, trade_date)
        if item is not None:
            items.append(item)
    return sorted(items, key=lambda item: (item.trade_date, item.code, item.auction_time))


def _news_event_id(prefix: str, *values: str) -> str:
    digest = hashlib.sha1("|".join(values).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _news_session_tag(time_text: str) -> str:
    if len(time_text) < 13:
        return ""
    try:
        hour = int(time_text[11:13])
    except ValueError:
        return ""
    if hour < 9:
        return "盘前"
    if hour < 15:
        return "盘中"
    return "盘后"


def _market_news_events(trade_date: str, limit: int, include_content_text: bool) -> list[NewsEventItem]:
    actual_trade_date = format_date_value(trade_date)
    result = _call_ak("news_cctv", ak.news_cctv, date=actual_trade_date.replace("-", ""))
    if result is None or result.empty:
        return []
    items: list[NewsEventItem] = []
    for _, row in result.iterrows():
        row_date = format_date_value(row.get("date", actual_trade_date))
        if row_date != actual_trade_date:
            continue
        title = _text_value(row.get("title", ""))
        content_text = _text_value(row.get("content", ""))
        if title == "":
            continue
        items.append(
            NewsEventItem(
                event_id=_news_event_id("akshare-cctv", row_date, title),
                trade_date=row_date,
                announcement_time=row_date,
                crawl_time="",
                session_tag="盘后",
                event_type="重要新闻",
                title=title,
                summary=content_text,
                content_text=content_text if include_content_text else "",
                importance_score=3,
                sentiment="中性",
                source_name="央视新闻",
                primary_detail_url="",
                related_stock_codes=[],
                related_stock_names=[],
                related_board_codes=[],
                related_board_names=[],
                topic_tags=[],
                mentioned_stock_codes=[],
                mentioned_stock_names=[],
                mentioned_board_names=[],
            )
        )
    return items[:limit]


def _stock_news_events(trade_date: str, stock_code: str, limit: int, include_content_text: bool) -> list[NewsEventItem]:
    actual_trade_date = format_date_value(trade_date)
    normalized_code = normalize_stock_code(stock_code)
    result = _call_ak("stock_news_em", ak.stock_news_em, symbol=normalized_code)
    if result is None or result.empty:
        return []
    items: list[NewsEventItem] = []
    for _, row in result.iterrows():
        announcement_time = format_datetime_value(row.get("发布时间", ""), "1m")
        row_date = format_date_value(announcement_time[:10])
        if row_date != actual_trade_date:
            continue
        title = _text_value(row.get("新闻标题", ""))
        content_text = _text_value(row.get("新闻内容", ""))
        detail_url = _text_value(row.get("新闻链接", ""))
        if title == "":
            continue
        items.append(
            NewsEventItem(
                event_id=_news_event_id("akshare-stock-news", normalized_code, announcement_time, title, detail_url),
                trade_date=row_date,
                announcement_time=announcement_time,
                crawl_time="",
                session_tag=_news_session_tag(announcement_time),
                event_type="个股新闻",
                title=title,
                summary=content_text,
                content_text=content_text if include_content_text else "",
                importance_score=3,
                sentiment="中性",
                source_name=_text_value(row.get("文章来源", "")),
                primary_detail_url=detail_url,
                related_stock_codes=[normalized_code],
                related_stock_names=[],
                related_board_codes=[],
                related_board_names=[],
                topic_tags=[],
                mentioned_stock_codes=[normalized_code],
                mentioned_stock_names=[],
                mentioned_board_names=[],
            )
        )
    return sorted(items, key=lambda item: (item.announcement_time, item.event_id), reverse=True)[:limit]


def get_news_events(
    trade_date: str,
    announcement_date: str,
    crawl_date: str,
    stock_code: str,
    event_type: str,
    min_importance_score: int | None,
    sort_by: str,
    limit: int,
    offset: int,
    include_sources: bool,
    include_content_text: bool,
) -> list[NewsEventItem]:
    del crawl_date, include_sources
    actual_trade_date = format_date_value(trade_date)
    if actual_trade_date == "":
        return []
    actual_announcement_date = format_date_value(announcement_date) if announcement_date != "" else ""
    if actual_announcement_date not in {"", actual_trade_date}:
        return []
    items = _stock_news_events(actual_trade_date, stock_code, limit + offset, include_content_text) if stock_code != "" else _market_news_events(actual_trade_date, limit + offset, include_content_text)
    if event_type != "":
        items = [item for item in items if item.event_type == event_type]
    if min_importance_score is not None:
        items = [item for item in items if item.importance_score >= min_importance_score]
    if sort_by == "crawl_time":
        items = sorted(items, key=lambda item: (item.crawl_time, item.announcement_time, item.event_id), reverse=True)
    else:
        items = sorted(items, key=lambda item: (item.announcement_time, item.importance_score, item.event_id), reverse=True)
    return items[offset: offset + limit]


def get_stock_money_flow(code: str, trade_date: str, start_date: str, end_date: str, view: str) -> list[StockMoneyFlowItem]:
    normalized_code = normalize_stock_code(code)
    actual_start, actual_end = _date_range_from_request(trade_date, start_date, end_date, 120)
    if curl_requests is None:
        result = _call_ak("stock_individual_fund_flow", ak.stock_individual_fund_flow, stock=normalized_code, market=_stock_market(normalized_code))
    else:
        market_map = {"sh": 1, "sz": 0, "bj": 0}
        response = curl_requests.get(
            "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get",
            params={
                "lmt": "0",
                "klt": "101",
                "secid": f"{market_map[_stock_market(normalized_code)]}.{normalized_code}",
                "fields1": "f1,f2,f3,f7",
                "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
                "ut": "b2884a393a59ad64002292a3e90d46a5",
            },
            impersonate="chrome",
        )
        data = response.json().get("data", {}) if response.status_code == 200 else {}
        klines = data.get("klines", [])
        if klines == []:
            result = pd.DataFrame()
        else:
            result = pd.DataFrame([item.split(",") for item in klines])
            result.columns = [
                "日期",
                "主力净流入-净额",
                "小单净流入-净额",
                "中单净流入-净额",
                "大单净流入-净额",
                "超大单净流入-净额",
                "主力净流入-净占比",
                "小单净流入-净占比",
                "中单净流入-净占比",
                "大单净流入-净占比",
                "超大单净流入-净占比",
                "收盘价",
                "涨跌幅",
                "_unused1",
                "_unused2",
            ]
    if result is None or result.empty:
        return []
    actual_view = view if view else "main"
    items: list[StockMoneyFlowItem] = []
    for _, row in result.iterrows():
        row_date = format_date_value(row.get("日期", ""))
        if not _date_in_window(row_date, actual_start, actual_end):
            continue
        items.append(
            StockMoneyFlowItem(
                code=normalized_code,
                trade_date=row_date,
                view=actual_view,
                main_inflow=None,
                main_outflow=None,
                net_inflow=_float_value(row.get("主力净流入-净额")),
            )
        )
    return sorted(items, key=lambda item: (item.code, item.trade_date, item.view))


def get_shareholder_count(code: str, trade_date: str, start_date: str, end_date: str) -> list[ShareholderCountItem]:
    normalized_code = normalize_stock_code(code)
    actual_start, actual_end = _date_range_from_request(trade_date, start_date, end_date, 720)
    result = _call_ak("stock_zh_a_gdhs_detail_em", ak.stock_zh_a_gdhs_detail_em, symbol=normalized_code)
    if result is None or result.empty:
        return []
    items: list[ShareholderCountItem] = []
    for _, row in result.iterrows():
        row_date = format_date_value(row.get("股东户数统计截止日", ""))
        if not _date_in_window(row_date, actual_start, actual_end):
            continue
        items.append(
            ShareholderCountItem(
                code=normalized_code,
                trade_date=row_date,
                holder_count=_int_value(row.get("股东户数-本次")),
                avg_holding=_float_value(row.get("户均持股数量")),
            )
        )
    return sorted(items, key=lambda item: (item.code, item.trade_date))


def get_shareholder_changes(code: str, trade_date: str, start_date: str, end_date: str) -> list[ShareholderChangeItem]:
    count_items = get_shareholder_count(code, trade_date, start_date, end_date)
    rows: list[ShareholderChangeItem] = []
    previous_count: int | None = None
    for item in sorted(count_items, key=lambda value: value.trade_date):
        change_count = item.holder_count - previous_count if item.holder_count is not None and previous_count is not None else None
        change_pct = None
        if change_count is not None and previous_count not in {None, 0}:
            change_pct = change_count / previous_count * 100
        rows.append(ShareholderChangeItem(code=item.code, trade_date=item.trade_date, holder_count=item.holder_count, change_count=change_count, change_pct=change_pct))
        previous_count = item.holder_count
    return rows


def get_shareholder_top10(code: str, report_period: str, start_period: str, end_period: str, float_only: bool) -> list[ShareholderTop10Item]:
    normalized_code = normalize_stock_code(code)
    periods = _period_candidates(report_period, start_period, end_period)
    if periods == []:
        return []
    items: list[ShareholderTop10Item] = []
    for period in periods:
        if float_only:
            result = _call_ak("stock_gdfx_free_top_10_em", ak.stock_gdfx_free_top_10_em, symbol=_em_prefixed_stock_code(normalized_code), date=period)
            ratio_column = "占总流通股本持股比例"
        else:
            result = _call_ak("stock_gdfx_top_10_em", ak.stock_gdfx_top_10_em, symbol=_em_prefixed_stock_code(normalized_code), date=period)
            ratio_column = "占总股本持股比例"
        if result is None or result.empty:
            continue
        for _, row in result.iterrows():
            items.append(
                ShareholderTop10Item(
                    code=normalized_code,
                    report_period=format_date_value(period),
                    rank=_int_value(row.get("名次")),
                    shareholder_name=_text_value(row.get("股东名称", "")),
                    holding_volume=_float_value(row.get("持股数")),
                    holding_ratio=_float_value(row.get(ratio_column)),
                    change_volume=_float_value(row.get("增减")),
                )
            )
    return sorted(items, key=lambda item: (item.code, item.report_period, item.rank or 0, item.shareholder_name))


def get_hk_connect_holdings(code: str, trade_date: str, start_date: str, end_date: str) -> list[HKConnectHoldingItem]:
    normalized_code = normalize_stock_code(code)
    actual_start, actual_end = _date_range_from_request(trade_date, start_date, end_date, 120)
    result = _call_ak("stock_hsgt_individual_em", ak.stock_hsgt_individual_em, symbol=normalized_code)
    if result is None or result.empty:
        return []
    items: list[HKConnectHoldingItem] = []
    for _, row in result.iterrows():
        row_date = format_date_value(row.get("持股日期", ""))
        if not _date_in_window(row_date, actual_start, actual_end):
            continue
        items.append(
            HKConnectHoldingItem(
                code=normalized_code,
                trade_date=row_date,
                holding_volume=_float_value(row.get("持股数量")),
                holding_ratio=_float_value(row.get("持股数量占A股百分比")),
                change_volume=_float_value(row.get("今日增持股数")),
            )
        )
    return sorted(items, key=lambda item: (item.code, item.trade_date))


def get_pledge_stats(code: str, trade_date: str, start_date: str, end_date: str) -> list[PledgeStatItem]:
    normalized_code = normalize_stock_code(code)
    actual_start, actual_end = _date_range_from_request(trade_date, start_date, end_date, 30)
    query_date = actual_end.replace("-", "")
    result = _call_ak("stock_gpzy_pledge_ratio_em", ak.stock_gpzy_pledge_ratio_em, date=query_date)
    if result is None or result.empty:
        return []
    items: list[PledgeStatItem] = []
    for _, row in result.iterrows():
        row_code = normalize_stock_code(str(row.get("股票代码", "")))
        row_date = format_date_value(row.get("交易日期", ""))
        if normalized_code and row_code != normalized_code:
            continue
        if not _date_in_window(row_date, actual_start, actual_end):
            continue
        items.append(
            PledgeStatItem(
                code=row_code,
                trade_date=row_date,
                pledge_volume=_float_value(row.get("质押股数")),
                pledge_ratio=_float_value(row.get("质押比例")),
                unrestricted_pledge_volume=_float_value(row.get("无限售股质押数")),
            )
        )
    return sorted(items, key=lambda item: (item.code, item.trade_date))


def get_pledge_details(code: str, start_date: str, end_date: str, status: str) -> list[PledgeDetailItem]:
    normalized_code = normalize_stock_code(code)
    actual_start, actual_end = _date_range_from_request("", start_date, end_date, 720)
    if status:
        return []
    result = _call_ak("stock_gpzy_pledge_ratio_detail_em", ak.stock_gpzy_pledge_ratio_detail_em)
    if result is None or result.empty:
        return []
    items: list[PledgeDetailItem] = []
    for _, row in result.iterrows():
        row_code = normalize_stock_code(str(row.get("股票代码", "")))
        row_start = format_date_value(row.get("质押开始日期", ""))
        if normalized_code and row_code != normalized_code:
            continue
        if not _date_in_window(row_start, actual_start, actual_end):
            continue
        items.append(
            PledgeDetailItem(
                code=row_code,
                holder_name=_text_value(row.get("股东名称", "")),
                start_date=row_start,
                end_date="",
                pledge_volume=_float_value(row.get("质押股份数量")),
                pledge_ratio=_float_value(row.get("占总股本比例")),
                status="",
            )
        )
    return sorted(items, key=lambda item: (item.code, item.start_date, item.holder_name))


def get_dividends(code: str, start_date: str, end_date: str) -> list[DividendItem]:
    normalized_code = normalize_stock_code(code)
    actual_start, actual_end = _date_range_from_request("", start_date, end_date, 720)
    result = _call_ak("stock_dividend_cninfo", ak.stock_dividend_cninfo, symbol=normalized_code)
    if result is None or result.empty:
        return []
    items: list[DividendItem] = []
    for _, row in result.iterrows():
        announce_date = format_date_value(row.get("实施方案公告日期", ""))
        if not _date_in_window(announce_date, actual_start, actual_end):
            continue
        items.append(
            DividendItem(
                code=normalized_code,
                announce_date=announce_date,
                record_date=format_date_value(row.get("股权登记日", "")),
                ex_date=format_date_value(row.get("除权日", "")),
                pay_date=format_date_value(row.get("派息日", "")),
                cash_dividend_per_share=_float_value(row.get("派息比例")) / 10 if _float_value(row.get("派息比例")) is not None else None,
                stock_dividend_per_share=_float_value(row.get("送股比例")) / 10 if _float_value(row.get("送股比例")) is not None else None,
                capital_reserve_per_share=_float_value(row.get("转增比例")) / 10 if _float_value(row.get("转增比例")) is not None else None,
            )
        )
    return sorted(items, key=lambda item: (item.code, item.announce_date, item.record_date, item.ex_date))


def get_repurchases(code: str, start_date: str, end_date: str) -> list[RepurchaseItem]:
    normalized_code = normalize_stock_code(code)
    actual_start, actual_end = _date_range_from_request("", start_date, end_date, 720)
    result = _call_ak("stock_repurchase_em", ak.stock_repurchase_em)
    if result is None or result.empty:
        return []
    items: list[RepurchaseItem] = []
    for _, row in result.iterrows():
        row_code = normalize_stock_code(str(row.get("股票代码", "")))
        if row_code != normalized_code:
            continue
        announce_date = format_date_value(row.get("最新公告日期", ""))
        if not _date_in_window(announce_date, actual_start, actual_end):
            continue
        items.append(
            RepurchaseItem(
                code=normalized_code,
                announce_date=announce_date,
                progress=_text_value(row.get("实施进度", "")),
                repurchase_volume=_float_value(row.get("已回购股份数量")),
                repurchase_amount=_float_value(row.get("已回购金额")),
                highest_price=_float_value(row.get("已回购股份价格区间-上限")),
                lowest_price=_float_value(row.get("已回购股份价格区间-下限")),
            )
        )
    return sorted(items, key=lambda item: (item.code, item.announce_date, item.progress))


def get_rights_issues(code: str, start_date: str, end_date: str) -> list[RightsIssueItem]:
    normalized_code = normalize_stock_code(code)
    actual_start, actual_end = _date_range_from_request("", start_date, end_date, 1440)
    result = _call_ak("stock_allotment_cninfo", ak.stock_allotment_cninfo, symbol=normalized_code, start_date=actual_start.replace("-", ""), end_date=actual_end.replace("-", ""))
    if result is None or result.empty:
        return []
    items: list[RightsIssueItem] = []
    for _, row in result.iterrows():
        announce_date = format_date_value(row.get("公告日期", ""))
        if not _date_in_window(announce_date, actual_start, actual_end):
            continue
        items.append(
            RightsIssueItem(
                code=normalized_code,
                announce_date=announce_date,
                rights_ratio=_float_value(row.get("配股比例")),
                rights_price=_float_value(row.get("配股价格")),
                record_date=format_date_value(row.get("股权登记日", "")),
                ex_date=format_date_value(row.get("除权基准日", "")),
            )
        )
    return sorted(items, key=lambda item: (item.code, item.announce_date, item.record_date))


def get_share_changes(code: str, trade_date: str, start_date: str, end_date: str) -> list[ShareChangeItem]:
    normalized_code = normalize_stock_code(code)
    actual_start, actual_end = _date_range_from_request(trade_date, start_date, end_date, 1440)
    result = _call_ak("stock_zh_a_gbjg_em", ak.stock_zh_a_gbjg_em, symbol=_em_secucode(normalized_code))
    if result is None or result.empty:
        return []
    items: list[ShareChangeItem] = []
    for _, row in result.iterrows():
        change_date = format_date_value(row.get("变更日期", ""))
        if not _date_in_window(change_date, actual_start, actual_end):
            continue
        items.append(
            ShareChangeItem(
                code=normalized_code,
                change_date=change_date,
                reason=_text_value(row.get("变动原因", "")),
                total_share=_float_value(row.get("总股本")),
                float_share=_float_value(row.get("已流通股份")),
                restricted_share=_float_value(row.get("流通受限股份")),
            )
        )
    return sorted(items, key=lambda item: (item.code, item.change_date, item.reason))


def get_unlock_schedules(code: str, unlock_date: str, start_date: str, end_date: str) -> list[UnlockScheduleItem]:
    normalized_code = normalize_stock_code(code)
    actual_start, actual_end = _date_range_from_request(unlock_date, start_date, end_date, 720)
    result = _call_ak("stock_restricted_release_queue_em", ak.stock_restricted_release_queue_em, symbol=normalized_code)
    if result is None or result.empty:
        return []
    items: list[UnlockScheduleItem] = []
    for _, row in result.iterrows():
        row_date = format_date_value(row.get("解禁时间", ""))
        if not _date_in_window(row_date, actual_start, actual_end):
            continue
        items.append(
            UnlockScheduleItem(
                code=normalized_code,
                unlock_date=row_date,
                holder_type="",
                unlock_volume=_float_value(row.get("解禁数量")),
                unlock_ratio=_float_value(row.get("占总市值比例")),
                share_type=_text_value(row.get("限售股类型", "")),
            )
        )
    return sorted(items, key=lambda item: (item.code, item.unlock_date, item.share_type))


def get_disclosure_dates(code: str, report_period: str, start_period: str, end_period: str) -> list[DisclosureDateItem]:
    normalized_code = normalize_stock_code(code)
    periods = _period_candidates(report_period, start_period, end_period)
    items: list[DisclosureDateItem] = []
    for period in periods:
        result = _call_ak("stock_report_disclosure", ak.stock_report_disclosure, market="沪深京", period=_ak_period_text(period))
        if result is None or result.empty:
            continue
        for _, row in result.iterrows():
            row_code = normalize_stock_code(str(row.get("股票代码", "")))
            if normalized_code and row_code != normalized_code:
                continue
            items.append(
                DisclosureDateItem(
                    code=row_code,
                    report_period=format_date_value(period),
                    plan_date=format_date_value(row.get("首次预约", "")),
                    actual_date=format_date_value(row.get("实际披露", "")),
                    change_reason="",
                )
            )
    return sorted(items, key=lambda item: (item.code, item.report_period, item.plan_date, item.actual_date))


def get_express(code: str, report_period: str, start_period: str, end_period: str) -> list[ExpressItem]:
    normalized_code = normalize_stock_code(code)
    periods = _period_candidates(report_period, start_period, end_period)
    items: list[ExpressItem] = []
    for period in periods:
        result = _call_ak("stock_yjkb_em", ak.stock_yjkb_em, date=period)
        if result is None or result.empty:
            continue
        for _, row in result.iterrows():
            row_code = normalize_stock_code(str(row.get("股票代码", "")))
            if normalized_code and row_code != normalized_code:
                continue
            items.append(
                ExpressItem(
                    code=row_code,
                    report_period=format_date_value(period),
                    announce_date=format_date_value(row.get("公告日期", "")),
                    revenue=_float_value(row.get("营业收入-营业收入")),
                    operating_profit=None,
                    total_profit=None,
                    net_profit=_float_value(row.get("净利润-净利润")),
                    eps=_float_value(row.get("每股收益")),
                    roe=_float_value(row.get("净资产收益率")),
                )
            )
    return sorted(items, key=lambda item: (item.code, item.report_period, item.announce_date))


def get_forecasts(code: str, report_period: str, start_period: str, end_period: str) -> list[ForecastItem]:
    normalized_code = normalize_stock_code(code)
    periods = _period_candidates(report_period, start_period, end_period)
    items: list[ForecastItem] = []
    for period in periods:
        result = _call_ak("stock_yjyg_em", ak.stock_yjyg_em, date=period)
        if result is None or result.empty:
            continue
        for _, row in result.iterrows():
            row_code = normalize_stock_code(str(row.get("股票代码", "")))
            if normalized_code and row_code != normalized_code:
                continue
            forecast_value = _float_value(row.get("预测数值"))
            pct_chg = _float_value(row.get("业绩变动幅度"))
            items.append(
                ForecastItem(
                    code=row_code,
                    report_period=format_date_value(period),
                    forecast_type=_text_value(row.get("预告类型", "")),
                    forecast_summary=_text_value(row.get("业绩变动原因", "")),
                    net_profit_min=forecast_value,
                    net_profit_max=forecast_value,
                    pct_chg_min=pct_chg,
                    pct_chg_max=pct_chg,
                )
            )
    return sorted(items, key=lambda item: (item.code, item.report_period, item.forecast_type))


def get_stock_finance_indicators(code: str, codes: str, report_period: str, start_period: str, end_period: str) -> list[StockFinanceIndicatorItem]:
    request_codes = [normalize_stock_code(item) for item in ([code] if code else codes.split(",")) if normalize_stock_code(item)]
    periods = [format_date_value(item) for item in _period_candidates(report_period, start_period, end_period)]
    items: list[StockFinanceIndicatorItem] = []
    for request_code in request_codes:
        result = _call_ak("stock_financial_analysis_indicator_em", ak.stock_financial_analysis_indicator_em, symbol=_em_secucode(request_code), indicator="按报告期")
        if result is None or result.empty:
            continue
        for _, row in result.iterrows():
            row_period = format_date_value(_column_value(row, ("REPORT_DATE", "报告期", "日期")))
            if periods and row_period not in periods:
                continue
            items.append(
                StockFinanceIndicatorItem(
                    code=request_code,
                    report_period=row_period,
                    roe=_float_value(_column_value(row, ("ROE_AVG", "JROE", "净资产收益率", "加权净资产收益率"))),
                    roa=_float_value(_column_value(row, ("ROA", "JROA", "总资产净利率"))),
                    gross_margin=_float_value(_column_value(row, ("XSMLL", "销售毛利率", "毛利率"))),
                    net_margin=_float_value(_column_value(row, ("XSJLL", "销售净利率", "净利率"))),
                    asset_turnover=_float_value(_column_value(row, ("TOAZZL", "总资产周转率"))),
                    current_ratio=_float_value(_column_value(row, ("LD", "流动比率"))),
                    debt_to_asset=_float_value(_column_value(row, ("ZCFZL", "资产负债率"))),
                )
            )
    return sorted(items, key=lambda item: (item.code, item.report_period))


def get_main_business(code: str, report_period: str, start_period: str, end_period: str, classification: str) -> list[MainBusinessItem]:
    normalized_code = normalize_stock_code(code)
    periods = [format_date_value(item) for item in _period_candidates(report_period, start_period, end_period)]
    result = _call_ak("stock_zygc_em", ak.stock_zygc_em, symbol=_ak_secucode(normalized_code))
    if result is None or result.empty:
        return []
    class_map = {"product": "按产品分类", "region": "按地区分类", "industry": "按行业分类"}
    wanted_class = class_map.get(classification, "")
    items: list[MainBusinessItem] = []
    for _, row in result.iterrows():
        row_period = format_date_value(row.get("报告日期", ""))
        row_class_text = _text_value(row.get("分类类型", ""))
        if periods and row_period not in periods:
            continue
        if wanted_class and row_class_text != wanted_class:
            continue
        actual_classification = "product" if row_class_text == "按产品分类" else "region" if row_class_text == "按地区分类" else "industry"
        items.append(
            MainBusinessItem(
                code=normalized_code,
                report_period=row_period,
                classification=actual_classification,
                segment_name=_text_value(row.get("主营构成", "")),
                revenue=_float_value(row.get("主营收入")),
                cost=_float_value(row.get("主营成本")),
                profit=_float_value(row.get("主营利润")),
                revenue_ratio=_float_value(row.get("收入比例")),
            )
        )
    return sorted(items, key=lambda item: (item.code, item.report_period, item.classification, item.segment_name))


def _financial_statement_frame(code: str, report_type: str) -> pd.DataFrame:
    symbol = _ak_secucode(code)
    if report_type == "balance_sheet":
        return _call_ak("stock_balance_sheet_by_report_em", ak.stock_balance_sheet_by_report_em, symbol=symbol)
    if report_type == "cash_flow":
        return _call_ak("stock_cash_flow_sheet_by_report_em", ak.stock_cash_flow_sheet_by_report_em, symbol=symbol)
    return _call_ak("stock_profit_sheet_by_report_em", ak.stock_profit_sheet_by_report_em, symbol=symbol)


def get_stock_financial_statements(codes: list[str], report_period: str, start_period: str, end_period: str, report_type: str) -> list[StockFinancialStatementItem]:
    periods = [format_date_value(item) for item in _period_candidates(report_period, start_period, end_period)]
    items: list[StockFinancialStatementItem] = []
    for code in codes:
        normalized_code = normalize_stock_code(code)
        result = _financial_statement_frame(normalized_code, report_type)
        if result is None or result.empty:
            continue
        for _, row in result.iterrows():
            row_period = format_date_value(_column_value(row, ("REPORT_DATE", "报告期", "日期")))
            if periods and row_period not in periods:
                continue
            items.append(
                StockFinancialStatementItem(
                    code=normalized_code,
                    report_period=row_period,
                    report_type=report_type,
                    announce_date=format_date_value(_column_value(row, ("NOTICE_DATE", "公告日期", "UPDATE_DATE"))),
                    revenue=_float_value(_column_value(row, ("TOTAL_OPERATE_INCOME", "OPERATE_INCOME", "营业总收入", "营业收入"))),
                    operating_profit=_float_value(_column_value(row, ("OPERATE_PROFIT", "营业利润"))),
                    total_profit=_float_value(_column_value(row, ("TOTAL_PROFIT", "利润总额"))),
                    net_profit=_float_value(_column_value(row, ("NETPROFIT", "PARENT_NETPROFIT", "净利润"))),
                    total_assets=_float_value(_column_value(row, ("TOTAL_ASSETS", "资产总计"))),
                    total_liabilities=_float_value(_column_value(row, ("TOTAL_LIABILITIES", "负债合计"))),
                    equity=_float_value(_column_value(row, ("TOTAL_EQUITY", "所有者权益合计"))),
                )
            )
    return sorted(items, key=lambda item: (item.code, item.report_period, item.report_type, item.announce_date))


def get_company_profile(code: str) -> StockProfileItem | None:
    normalized_code = normalize_stock_code(code)
    result = _call_ak("stock_profile_cninfo", ak.stock_profile_cninfo, symbol=normalized_code)
    if result is None or result.empty:
        return None
    row = result.iloc[0]
    return StockProfileItem(
        code=normalized_code,
        company_name=_text_value(row.get("A股简称", "")),
        full_name=_text_value(row.get("公司名称", "")),
        chairman=_text_value(row.get("法人代表", "")),
        manager="",
        website=_text_value(row.get("官方网站", "")),
        employee_count=None,
        main_business=_text_value(row.get("主营业务", "")),
        office=_text_value(row.get("办公地址", "")),
    )


def get_research_reports(code: str, report_date: str, start_date: str, end_date: str) -> list[ResearchReportItem]:
    normalized_code = normalize_stock_code(code)
    actual_start, actual_end = _date_range_from_request(report_date, start_date, end_date, 365)
    result = _call_ak("stock_research_report_em", ak.stock_research_report_em, symbol=normalized_code)
    if result is None or result.empty:
        return []
    items: list[ResearchReportItem] = []
    for _, row in result.iterrows():
        row_date = format_date_value(row.get("日期", ""))
        if not _date_in_window(row_date, actual_start, actual_end):
            continue
        items.append(
            ResearchReportItem(
                code=normalize_stock_code(str(row.get("股票代码", normalized_code))),
                report_date=row_date,
                institution=_text_value(row.get("机构", "")),
                analyst="",
                rating=_text_value(row.get("东财评级", "")),
                target_price=None,
                title=_text_value(row.get("报告名称", "")),
            )
        )
    return sorted(items, key=lambda item: (item.code, item.report_date, item.institution, item.title))


def get_surveys(code: str, survey_date: str, start_date: str, end_date: str) -> list[SurveyItem]:
    normalized_code = normalize_stock_code(code)
    actual_start, actual_end = _date_range_from_request(survey_date, start_date, end_date, 365)
    result = _call_ak("stock_jgdy_detail_em", ak.stock_jgdy_detail_em, date=actual_start.replace("-", ""))
    if result is None or result.empty:
        return []
    items: list[SurveyItem] = []
    for _, row in result.iterrows():
        row_code = normalize_stock_code(str(row.get("代码", "")))
        row_date = format_date_value(row.get("调研日期", ""))
        if normalized_code and row_code != normalized_code:
            continue
        if not _date_in_window(row_date, actual_start, actual_end):
            continue
        items.append(
            SurveyItem(
                code=row_code,
                survey_date=row_date,
                org_name=_text_value(row.get("调研机构", "")),
                survey_method=_text_value(row.get("接待方式", "")),
                topic="",
                announcement_date=format_date_value(row.get("公告日期", "")),
            )
        )
    return sorted(items, key=lambda item: (item.code, item.survey_date, item.org_name, item.announcement_date))


def get_stock_daily_snapshot_full(trade_date: str) -> list[StockQuoteItem]:
    actual_trade_date = format_date_value(trade_date)
    if actual_trade_date == "":
        return []
    if actual_trade_date != datetime.now().strftime("%Y-%m-%d"):
        return []
    result = _call_ak("stock_zh_a_spot_em", ak.stock_zh_a_spot_em)
    if result is None or result.empty:
        return []
    items: list[StockQuoteItem] = []
    for _, row in result.iterrows():
        code = normalize_stock_code(str(row.get("代码", "")))
        if code == "":
            continue
        pre_close = _float_value(row.get("昨收", row.get("昨日收盘")))
        close = _float_value(row.get("最新价", row.get("收盘")))
        change = _float_value(row.get("涨跌额"))
        pct_chg = _float_value(row.get("涨跌幅"))
        if change is None and close is not None and pre_close is not None:
            change = close - pre_close
        if pct_chg is None and change is not None and pre_close not in {None, 0}:
            pct_chg = change / pre_close * 100
        items.append(
            StockQuoteItem(
                code=code,
                trade_time=actual_trade_date,
                freq="1d",
                open=_float_value(row.get("今开", row.get("开盘"))),
                high=_float_value(row.get("最高")),
                low=_float_value(row.get("最低")),
                close=close,
                pre_close=pre_close,
                change=change,
                pct_chg=pct_chg,
                volume=_float_value(row.get("成交量")),
                amount=_float_value(row.get("成交额")),
                adjust="none",
            )
        )
    return sorted(items, key=lambda item: item.code)

