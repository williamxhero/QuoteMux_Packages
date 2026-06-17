from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from quotemux.infra.cache.store import build_cache_path, filter_frame_by_date_range, filter_frame_by_datetime_range, latest_n_rows, merge_cache_frame, read_cache_frame, write_cache_frame
from quotemux.infra.common import build_time_bounds, format_date_value, format_datetime_value, normalize_index_code, normalize_stock_code
from quotemux.runtime_core.quality import calibrate_quote_units
from quotemux.infra.provider_runtime.core import call_provider_api
from platform_models import BoardQuoteItem, DragonTigerItem, ExpressItem, ShareholderCountItem, StockFinanceIndicatorItem, IndexMemberItem, IndexQuoteItem, StockQuoteItem

import sys
_saved_paths = [path for path in sys.path if "quotemux_packages" in path or ("packages" in path and "site-packages" not in path and "dist-packages" not in path)]
for path in _saved_paths:
    try:
        sys.path.remove(path)
    except ValueError:
        pass

try:
    import efinance as ef
except Exception:
    ef = None
finally:
    for path in reversed(_saved_paths):
        sys.path.insert(0, path)


EFINANCE_FREQ_MAP = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "60m": 60,
    "1d": 101,
    "1w": 102,
    "1mo": 103,
}
EFINANCE_ADJUST_MAP = {
    "none": 0,
    "qfq": 1,
    "hfq": 2,
}
DEFAULT_LOOKBACK_DAYS = 30


def _is_available() -> bool:
    return ef is not None


def _require_available() -> None:
    if not _is_available():
        raise RuntimeError("efinance 不可用")


def _call_ef(api_name: str, func, *args, **kwargs):
    _require_available()
    return call_provider_api("efinance", api_name, func, *args, **kwargs)


def _float_value(value: object) -> float | None:
    number = pd.to_numeric(value, errors="coerce")
    return float(number) if pd.notna(number) else None


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

def _date_window(date_value: str, start_date: str, end_date: str, lookback_days: int) -> tuple[str, str]:
    actual_date = format_date_value(date_value)
    if actual_date:
        return actual_date, actual_date
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


def _period_candidates(report_period: str, start_period: str, end_period: str) -> list[str]:
    actual_period = format_date_value(report_period)
    if actual_period:
        return [actual_period]
    actual_start = format_date_value(start_period)
    actual_end = format_date_value(end_period)
    if actual_start == "" or actual_end == "":
        return []
    start_dt = datetime.strptime(actual_start, "%Y-%m-%d")
    end_dt = datetime.strptime(actual_end, "%Y-%m-%d")
    rows: list[str] = []
    for year in range(start_dt.year, end_dt.year + 1):
        for month, day in ((3, 31), (6, 30), (9, 30), (12, 31)):
            current = datetime(year, month, day)
            if start_dt <= current <= end_dt:
                rows.append(current.strftime("%Y-%m-%d"))
    return rows


def _snapshot_trade_date(row: pd.Series, fallback_date: str) -> str:
    if "最新交易日" in row and pd.notna(row["最新交易日"]):
        return format_date_value(row["最新交易日"])
    if "更新时间" in row and pd.notna(row["更新时间"]):
        return format_date_value(row["更新时间"])
    return fallback_date


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


def _fetch_stock_history_frame(code: str, freq: str, start_dt: datetime, end_dt: datetime, adjust: str) -> pd.DataFrame:
    result = _call_ef(
        "stock.get_quote_history",
        ef.stock.get_quote_history,
        normalize_stock_code(code),
        start_dt.strftime("%Y%m%d"),
        end_dt.strftime("%Y%m%d"),
        EFINANCE_FREQ_MAP[freq],
        EFINANCE_ADJUST_MAP.get(adjust, 0),
        suppress_error=True,
    )
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
    work = work[["code", "trade_time", "freq", "open", "high", "low", "close", "volume", "amount"]]
    work = work.dropna(subset=["trade_time"])
    work, _ = calibrate_quote_units(work)
    return work.drop_duplicates(subset=["code", "trade_time", "freq"], keep="last").sort_values("trade_time").reset_index(drop=True)


def _fetch_index_history_frame(index_code: str, freq: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    result = _call_ef(
        "stock.get_quote_history",
        ef.stock.get_quote_history,
        normalize_index_code(index_code),
        start_dt.strftime("%Y%m%d"),
        end_dt.strftime("%Y%m%d"),
        EFINANCE_FREQ_MAP[freq],
        0,
        suppress_error=True,
    )
    if result is None or result.empty:
        return pd.DataFrame()
    work = result.copy()
    work["index_code"] = work["股票代码"].astype(str).str.zfill(6)
    work["trade_time"] = pd.to_datetime(work["日期"], errors="coerce")
    work["freq"] = freq
    work["open"] = pd.to_numeric(work["开盘"], errors="coerce")
    work["high"] = pd.to_numeric(work["最高"], errors="coerce")
    work["low"] = pd.to_numeric(work["最低"], errors="coerce")
    work["close"] = pd.to_numeric(work["收盘"], errors="coerce")
    work["volume"] = pd.to_numeric(work["成交量"], errors="coerce")
    work["amount"] = pd.to_numeric(work["成交额"], errors="coerce")
    work = work[["index_code", "trade_time", "freq", "open", "high", "low", "close", "volume", "amount"]]
    work = work.dropna(subset=["trade_time"])
    work, _ = calibrate_quote_units(work)
    return work.drop_duplicates(subset=["index_code", "trade_time", "freq"], keep="last").sort_values("trade_time").reset_index(drop=True)


def _fetch_board_history_frame(board_code: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    result = _call_ef("stock.get_history_bill", ef.stock.get_history_bill, board_code)
    if result is None or result.empty:
        return pd.DataFrame()
    work = result.copy()
    for column in ["日期", "收盘价", "涨跌幅"]:
        if column not in work.columns:
            return pd.DataFrame()
    work["board_code"] = str(board_code).upper()
    work["trade_time"] = pd.to_datetime(work["日期"], errors="coerce")
    work["freq"] = "1d"
    work["close"] = pd.to_numeric(work["收盘价"], errors="coerce")
    work["pct_chg"] = pd.to_numeric(work["涨跌幅"], errors="coerce")
    valid_pct = work["pct_chg"].notna() & (work["pct_chg"] != -100)
    work["pre_close"] = pd.NA
    work.loc[valid_pct, "pre_close"] = work.loc[valid_pct, "close"] / (1 + work.loc[valid_pct, "pct_chg"] / 100)
    work["change"] = work["close"] - work["pre_close"]
    work["open"] = pd.NA
    work["high"] = pd.NA
    work["low"] = pd.NA
    work["volume"] = pd.NA
    work["amount"] = pd.NA
    work = work[["board_code", "trade_time", "freq", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "volume", "amount"]]
    work = work.dropna(subset=["trade_time"])
    work = work[(work["trade_time"] >= start_dt) & (work["trade_time"] <= end_dt)]
    return work.drop_duplicates(subset=["board_code", "trade_time", "freq"], keep="last").sort_values("trade_time").reset_index(drop=True)


def _frame_to_stock_quotes(df: pd.DataFrame, freq: str, adjust: str) -> list[StockQuoteItem]:
    items: list[StockQuoteItem] = []
    if df.empty:
        return items
    work = df.sort_values("trade_time").copy()
    work["pre_close"] = work["close"].shift(1)
    work["change"] = work["close"] - work["pre_close"]
    work["pct_chg"] = work["change"] / work["pre_close"] * 100
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


def _frame_to_board_quotes(df: pd.DataFrame) -> list[BoardQuoteItem]:
    items: list[BoardQuoteItem] = []
    if df.empty:
        return items
    for _, row in df.sort_values("trade_time").iterrows():
        items.append(
            BoardQuoteItem(
                board_code=str(row["board_code"]),
                trade_time=format_datetime_value(row["trade_time"], "1d"),
                freq="1d",
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
        cache_path = build_cache_path("efinance", ["stocks", "quotes"], {"code": normalized_code, "freq": freq, "adjust": adjust})
        cache_df = read_cache_frame(cache_path)
        filtered_cache = filter_frame_by_datetime_range(cache_df, "trade_time", start_dt, end_dt)
        if filtered_cache.empty or (count and len(filtered_cache) < count):
            fetched_df = _fetch_stock_history_frame(normalized_code, freq, start_dt, end_dt, adjust)
            if not fetched_df.empty:
                cache_df = merge_cache_frame(cache_df, fetched_df, ["code", "trade_time", "freq"], ["trade_time"])
                write_cache_frame(cache_path, cache_df)
        filtered_df = filter_frame_by_datetime_range(cache_df, "trade_time", start_dt, end_dt)
        filtered_df = latest_n_rows(filtered_df, "trade_time", count)
        items.extend(_frame_to_stock_quotes(filtered_df, freq, adjust))
    return items


def get_board_quotes(board_codes: list[str], freq: str, trade_date: str, start_date: str, end_date: str, start_time: str, end_time: str, count: int | None) -> list[BoardQuoteItem]:
    del start_time
    del end_time
    if freq != "1d":
        return []
    start_dt, end_dt = _resolve_time_window(trade_date, start_date, end_date, "", "", count, False)
    items: list[BoardQuoteItem] = []
    for board_code in board_codes:
        normalized_code = str(board_code).upper()
        cache_path = build_cache_path("efinance", ["boards", "quotes"], {"board_code": normalized_code, "freq": freq})
        cache_df = read_cache_frame(cache_path)
        filtered_cache = filter_frame_by_datetime_range(cache_df, "trade_time", start_dt, end_dt)
        if filtered_cache.empty or (count and len(filtered_cache) < count):
            fetched_df = _fetch_board_history_frame(normalized_code, start_dt, end_dt)
            if not fetched_df.empty:
                cache_df = merge_cache_frame(cache_df, fetched_df, ["board_code", "trade_time", "freq"], ["trade_time"])
                write_cache_frame(cache_path, cache_df)
        filtered_df = filter_frame_by_datetime_range(cache_df, "trade_time", start_dt, end_dt)
        filtered_df = latest_n_rows(filtered_df, "trade_time", count)
        items.extend(_frame_to_board_quotes(filtered_df))
    return items


def get_index_quotes(index_codes: list[str], freq: str, trade_date: str, start_date: str, end_date: str, count: int | None) -> list[IndexQuoteItem]:
    start_dt, end_dt = _resolve_time_window(trade_date, start_date, end_date, "", "", count, False)
    items: list[IndexQuoteItem] = []
    for index_code in index_codes:
        normalized_code = normalize_index_code(index_code)
        cache_path = build_cache_path("efinance", ["indexes", "quotes"], {"index_code": normalized_code, "freq": freq})
        cache_df = read_cache_frame(cache_path)
        filtered_cache = filter_frame_by_datetime_range(cache_df, "trade_time", start_dt, end_dt)
        if filtered_cache.empty or (count and len(filtered_cache) < count):
            fetched_df = _fetch_index_history_frame(normalized_code, freq, start_dt, end_dt)
            if not fetched_df.empty:
                cache_df = merge_cache_frame(cache_df, fetched_df, ["index_code", "trade_time", "freq"], ["trade_time"])
                write_cache_frame(cache_path, cache_df)
        filtered_df = filter_frame_by_datetime_range(cache_df, "trade_time", start_dt, end_dt)
        filtered_df = latest_n_rows(filtered_df, "trade_time", count)
        items.extend(_frame_to_index_quotes(filtered_df, freq))
    return items


def get_index_members(index_code: str, trade_date: str) -> list[IndexMemberItem]:
    del trade_date
    result = _call_ef("stock.get_members", ef.stock.get_members, normalize_index_code(index_code))
    if result is None or result.empty:
        return []
    work = result.copy()
    work["index_code"] = work["指数代码"].astype(str).str.zfill(6)
    work["code"] = work["股票代码"].astype(str).str.zfill(6)
    work["name"] = work["股票名称"].fillna("").astype(str)
    work["weight"] = pd.to_numeric(work["股票权重"], errors="coerce")
    items: list[IndexMemberItem] = []
    for _, row in work.iterrows():
        items.append(
            IndexMemberItem(
                index_code=str(row["index_code"]),
                code=str(row["code"]),
                name=str(row["name"]),
                weight=float(row["weight"]) if pd.notna(row["weight"]) else None,
                trade_date="",
            )
        )
    return items


def get_stock_daily_snapshot_full(trade_date: str) -> list[StockQuoteItem]:
    actual_trade_date = format_date_value(trade_date)
    if actual_trade_date == "":
        return []
    result = _call_ef("stock.get_realtime_quotes", ef.stock.get_realtime_quotes, None)
    if result is None or result.empty:
        return []
    items: list[StockQuoteItem] = []
    for _, row in result.iterrows():
        code = normalize_stock_code(str(row.get("代码", "")))
        row_trade_date = _snapshot_trade_date(row, actual_trade_date)
        if code == "" or row_trade_date != actual_trade_date:
            continue
        pre_close = _float_value(row.get("昨日收盘", row.get("昨收")))
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
                trade_time=row_trade_date,
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


def get_dragon_tiger(trade_date: str, start_date: str, end_date: str, code: str, limit: int) -> list[DragonTigerItem]:
    actual_start, actual_end = _date_window(trade_date, start_date, end_date, 30)
    result = _call_ef("stock.get_daily_billboard", ef.stock.get_daily_billboard, actual_start, actual_end)
    if result is None or result.empty:
        return []
    normalized_code = normalize_stock_code(code)
    items: list[DragonTigerItem] = []
    for _, row in result.iterrows():
        row_code = normalize_stock_code(str(row.get("股票代码", "")))
        if normalized_code and row_code != normalized_code:
            continue
        items.append(
            DragonTigerItem(
                trade_date=format_date_value(row.get("上榜日期", "")),
                code=row_code,
                name=_text_value(row.get("股票名称", "")),
                reason=_text_value(row.get("上榜原因", row.get("解读", ""))),
                buy_amount=_float_value(row.get("龙虎榜买入额")),
                sell_amount=_float_value(row.get("龙虎榜卖出额")),
                net_amount=_float_value(row.get("龙虎榜净买额")),
            )
        )
    return sorted(items, key=lambda item: (item.trade_date, item.code, item.reason))[:limit]


def get_shareholder_count(code: str, trade_date: str, start_date: str, end_date: str) -> list[ShareholderCountItem]:
    normalized_code = normalize_stock_code(code)
    periods = _period_candidates(trade_date, start_date, end_date)
    if periods == []:
        periods = [""]
    items: list[ShareholderCountItem] = []
    for period in periods:
        result = _call_ef("stock.get_latest_holder_number", ef.stock.get_latest_holder_number, period or None)
        if result is None or result.empty:
            continue
        for _, row in result.iterrows():
            row_code = normalize_stock_code(str(row.get("股票代码", "")))
            if normalized_code and row_code != normalized_code:
                continue
            items.append(
                ShareholderCountItem(
                    code=row_code,
                    trade_date=format_date_value(row.get("股东户数统计截止日", period)),
                    holder_count=_int_value(row.get("股东人数")),
                    avg_holding=_float_value(row.get("户均持股数量")),
                )
            )
    return sorted(items, key=lambda item: (item.code, item.trade_date))


def _company_performance_frame(period: str) -> pd.DataFrame:
    return _call_ef("stock.get_all_company_performance", ef.stock.get_all_company_performance, period or None)


def get_express(code: str, report_period: str, start_period: str, end_period: str) -> list[ExpressItem]:
    normalized_code = normalize_stock_code(code)
    periods = _period_candidates(report_period, start_period, end_period)
    if periods == []:
        periods = [""]
    items: list[ExpressItem] = []
    for period in periods:
        result = _company_performance_frame(period)
        if result is None or result.empty:
            continue
        for _, row in result.iterrows():
            row_code = normalize_stock_code(str(row.get("股票代码", "")))
            if normalized_code and row_code != normalized_code:
                continue
            items.append(
                ExpressItem(
                    code=row_code,
                    report_period=format_date_value(period or row.get("报告日期", "")),
                    announce_date=format_date_value(row.get("公告日期", "")),
                    revenue=_float_value(row.get("营业收入")),
                    operating_profit=None,
                    total_profit=None,
                    net_profit=_float_value(row.get("净利润")),
                    eps=_float_value(row.get("每股收益")),
                    roe=_float_value(row.get("净资产收益率")),
                )
            )
    return sorted(items, key=lambda item: (item.code, item.report_period, item.announce_date))


def get_stock_finance_indicators(code: str, codes: str, report_period: str, start_period: str, end_period: str) -> list[StockFinanceIndicatorItem]:
    request_codes = [normalize_stock_code(item) for item in ([code] if code else codes.split(",")) if normalize_stock_code(item)]
    periods = _period_candidates(report_period, start_period, end_period)
    if periods == []:
        periods = [""]
    items: list[StockFinanceIndicatorItem] = []
    for period in periods:
        result = _company_performance_frame(period)
        if result is None or result.empty:
            continue
        for _, row in result.iterrows():
            row_code = normalize_stock_code(str(row.get("股票代码", "")))
            if request_codes and row_code not in request_codes:
                continue
            items.append(
                StockFinanceIndicatorItem(
                    code=row_code,
                    report_period=format_date_value(period or row.get("报告日期", "")),
                    roe=_float_value(row.get("净资产收益率")),
                    roa=None,
                    gross_margin=_float_value(row.get("销售毛利率")),
                    net_margin=None,
                    asset_turnover=None,
                    current_ratio=None,
                    debt_to_asset=None,
                )
            )
    return sorted(items, key=lambda item: (item.code, item.report_period))

