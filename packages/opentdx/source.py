from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from functools import lru_cache
from threading import local

import pandas as pd

from quotemux.infra.cache.store import build_cache_path, filter_frame_by_datetime_range, latest_n_rows, merge_cache_frame, read_cache_frame, write_cache_frame
from quotemux.common import intraday_quote_cache_needs_refresh
from quotemux.infra.common import INTRADAY_RULES, add_quote_metrics, aggregate_ohlc, build_time_bounds, format_datetime_value, normalize_index_code, normalize_stock_code
from quotemux.infra.provider_runtime.core import call_provider_api
from platform_models import IndexQuoteItem, StockQuoteItem

import sys
_saved_paths = [path for path in sys.path if "quotemux_packages" in path or ("packages" in path and "site-packages" not in path and "dist-packages" not in path)]
for path in _saved_paths:
    try:
        sys.path.remove(path)
    except ValueError:
        pass

try:
    from opentdx import ADJUST, MARKET, PERIOD, TdxClient
except Exception:
    ADJUST = None
    MARKET = None
    PERIOD = None
    TdxClient = None
finally:
    for path in reversed(_saved_paths):
        sys.path.insert(0, path)


DEFAULT_LOOKBACK_DAYS = 10
MINUTES_PER_TRADE_DAY = 242
DAILY_FREQS = {"1d", "1w", "1mo"}
_CLIENT_STATE = local()


def _is_available() -> bool:
    return TdxClient is not None and MARKET is not None and PERIOD is not None and ADJUST is not None


def _require_available() -> None:
    if not _is_available():
        raise RuntimeError("OpenTDX 不可用")


def _market_from_code(code: str):
    normalized = normalize_stock_code(code)
    if normalized.startswith(("4", "8", "920")):
        return MARKET.BJ
    if normalized.startswith(("5", "6", "9")):
        return MARKET.SH
    return MARKET.SZ


def _market_from_index_code(index_code: str):
    normalized = normalize_index_code(index_code)
    if normalized.startswith("399"):
        return MARKET.SZ
    if normalized.startswith("8"):
        return MARKET.BJ
    return MARKET.SH


def _adjust_from_text(adjust: str):
    if adjust == "qfq":
        return ADJUST.QFQ
    if adjust == "hfq":
        return ADJUST.HFQ
    return ADJUST.NONE


def _period_from_freq(freq: str):
    if freq == "1w":
        return PERIOD.WEEKLY
    if freq == "1mo":
        return PERIOD.MONTHLY
    return PERIOD.DAILY


def _estimate_bar_count(start_dt: datetime, end_dt: datetime) -> int:
    # OpenTDX 从最新一根开始倒序返回，补历史日期时必须覆盖目标日到今天的区间。
    latest_date = max(end_dt.date(), datetime.now().date())
    span_days = max(1, (latest_date - start_dt.date()).days + 1)
    return min(60000, max(MINUTES_PER_TRADE_DAY, span_days * MINUTES_PER_TRADE_DAY))


def _estimate_daily_count(start_dt: datetime, end_dt: datetime) -> int:
    span_days = max(1, (end_dt.date() - start_dt.date()).days + 1)
    return min(8000, max(260, span_days + 30))


@lru_cache(maxsize=1)
def _client_factory():
    _require_available()
    return TdxClient


def _call_tdx(api_name: str, func, *args, **kwargs):
    _require_available()

    client = getattr(_CLIENT_STATE, "client", None)
    if client is None:
        # provider worker 内按线程复用连接，避免全市场补采为每只股票重复登录公共节点。
        client = _client_factory()()
        _CLIENT_STATE.client = client

    def _invoke():
        return func(client, *args, **kwargs)

    try:
        return call_provider_api("opentdx", api_name, _invoke)
    except Exception:
        try:
            client.__exit__(None, None, None)
        finally:
            _CLIENT_STATE.client = None
        raise


def _fetch_stock_intraday_frame(code: str, start_dt: datetime, end_dt: datetime, adjust: str) -> pd.DataFrame:
    records = _call_tdx(
        "stock_kline",
        lambda client, market, normalized, bar_count, adjust_value: client.stock_kline(
            market,
            normalized,
            PERIOD.MINS,
            start=0,
            count=bar_count,
            times=1,
            adjust=adjust_value,
        ),
        _market_from_code(code),
        normalize_stock_code(code),
        _estimate_bar_count(start_dt, end_dt),
        _adjust_from_text(adjust),
    )
    if not records:
        return pd.DataFrame()

    out = pd.DataFrame(records)
    if out.empty:
        return out
    out["code"] = normalize_stock_code(code)
    time_column = "date_time" if "date_time" in out.columns else "datetime"
    out["trade_time"] = pd.to_datetime(out[time_column], errors="coerce")
    out["freq"] = "1m"
    out["adjust"] = adjust
    out["open"] = pd.to_numeric(out["open"], errors="coerce")
    out["high"] = pd.to_numeric(out["high"], errors="coerce")
    out["low"] = pd.to_numeric(out["low"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["volume"] = pd.to_numeric(out["vol"], errors="coerce")
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    out = out[["code", "trade_time", "freq", "open", "high", "low", "close", "volume", "amount", "adjust"]]
    out = out.dropna(subset=["trade_time"])
    out = filter_frame_by_datetime_range(out, "trade_time", start_dt, end_dt)
    out = out.drop_duplicates(subset=["code", "trade_time", "freq"], keep="last")
    return out.sort_values("trade_time").reset_index(drop=True)


def _fetch_stock_daily_frame(code: str, freq: str, start_dt: datetime, end_dt: datetime, adjust: str) -> pd.DataFrame:
    normalized = normalize_stock_code(code)
    records = _call_tdx(
        "stock_kline",
        lambda client, market, stock_code, period, bar_count, adjust_value: client.stock_kline(
            market,
            stock_code,
            period,
            start=0,
            count=bar_count,
            times=1,
            adjust=adjust_value,
        ),
        _market_from_code(normalized),
        normalized,
        _period_from_freq(freq),
        _estimate_daily_count(start_dt, end_dt),
        _adjust_from_text(adjust),
    )
    if not records:
        return pd.DataFrame()
    out = pd.DataFrame(records)
    if out.empty:
        return out
    out["code"] = normalized
    time_column = "date_time" if "date_time" in out.columns else "datetime"
    out["trade_time"] = pd.to_datetime(out[time_column], errors="coerce")
    out["freq"] = freq
    out["adjust"] = adjust
    out["open"] = pd.to_numeric(out["open"], errors="coerce")
    out["high"] = pd.to_numeric(out["high"], errors="coerce")
    out["low"] = pd.to_numeric(out["low"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["volume"] = pd.to_numeric(out["vol"], errors="coerce")
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    out = out[["code", "trade_time", "freq", "open", "high", "low", "close", "volume", "amount", "adjust"]]
    out = out.dropna(subset=["trade_time"])
    out = add_quote_metrics(out)
    out = filter_frame_by_datetime_range(out, "trade_time", start_dt, end_dt)
    out = out.drop_duplicates(subset=["code", "trade_time", "freq"], keep="last")
    return out.sort_values("trade_time").reset_index(drop=True)


def _fetch_index_daily_frame(index_code: str, freq: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    normalized = normalize_index_code(index_code)
    records = _call_tdx(
        "index_kline",
        lambda client, market, current_code, period, bar_count: client.stock_kline(
            market,
            current_code,
            period,
            start=0,
            count=bar_count,
            times=1,
            adjust=ADJUST.NONE,
        ),
        _market_from_index_code(normalized),
        normalized,
        _period_from_freq(freq),
        _estimate_daily_count(start_dt, end_dt),
    )
    if not records:
        return pd.DataFrame()
    out = pd.DataFrame(records)
    if out.empty:
        return out
    out["index_code"] = normalized
    time_column = "date_time" if "date_time" in out.columns else "datetime"
    out["trade_time"] = pd.to_datetime(out[time_column], errors="coerce")
    out["freq"] = freq
    out["open"] = pd.to_numeric(out["open"], errors="coerce")
    out["high"] = pd.to_numeric(out["high"], errors="coerce")
    out["low"] = pd.to_numeric(out["low"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["volume"] = pd.to_numeric(out["vol"], errors="coerce")
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    out = out[["index_code", "trade_time", "freq", "open", "high", "low", "close", "volume", "amount"]]
    out = out.dropna(subset=["trade_time"])
    out = add_quote_metrics(out)
    out = filter_frame_by_datetime_range(out, "trade_time", start_dt, end_dt)
    out = out.drop_duplicates(subset=["index_code", "trade_time", "freq"], keep="last")
    return out.sort_values("trade_time").reset_index(drop=True)


def _frame_to_stock_quotes(df: pd.DataFrame, freq: str) -> list[StockQuoteItem]:
    if df.empty:
        return []
    items: list[StockQuoteItem] = []
    for _, row in df.sort_values("trade_time").iterrows():
        items.append(
            StockQuoteItem(
                code=str(row["code"]),
                trade_time=format_datetime_value(row["trade_time"], freq),
                freq=str(row["freq"]),
                open=float(row["open"]) if pd.notna(row["open"]) else None,
                high=float(row["high"]) if pd.notna(row["high"]) else None,
                low=float(row["low"]) if pd.notna(row["low"]) else None,
                close=float(row["close"]) if pd.notna(row["close"]) else None,
                pre_close=float(row["pre_close"]) if "pre_close" in row and pd.notna(row["pre_close"]) else None,
                change=float(row["change"]) if "change" in row and pd.notna(row["change"]) else None,
                pct_chg=float(row["pct_chg"]) if "pct_chg" in row and pd.notna(row["pct_chg"]) else None,
                volume=float(row["volume"]) if "volume" in row and pd.notna(row["volume"]) else None,
                amount=float(row["amount"]) if pd.notna(row["amount"]) else None,
                adjust=str(row["adjust"]),
                is_suspended=False,
                is_st=False,
            )
        )
    return items


def _frame_to_index_quotes(df: pd.DataFrame, freq: str) -> list[IndexQuoteItem]:
    if df.empty:
        return []
    items: list[IndexQuoteItem] = []
    for _, row in df.sort_values("trade_time").iterrows():
        items.append(
            IndexQuoteItem(
                index_code=str(row["index_code"]),
                trade_time=format_datetime_value(row["trade_time"], freq),
                freq=str(row["freq"]),
                open=float(row["open"]) if pd.notna(row["open"]) else None,
                high=float(row["high"]) if pd.notna(row["high"]) else None,
                low=float(row["low"]) if pd.notna(row["low"]) else None,
                close=float(row["close"]) if pd.notna(row["close"]) else None,
                pre_close=float(row["pre_close"]) if "pre_close" in row and pd.notna(row["pre_close"]) else None,
                change=float(row["change"]) if "change" in row and pd.notna(row["change"]) else None,
                pct_chg=float(row["pct_chg"]) if "pct_chg" in row and pd.notna(row["pct_chg"]) else None,
                volume=float(row["volume"]) if "volume" in row and pd.notna(row["volume"]) else None,
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
    if freq not in INTRADAY_RULES and freq not in DAILY_FREQS:
        return []

    intraday = freq in INTRADAY_RULES
    request_start_dt, request_end_dt = build_time_bounds(trade_date, start_date, end_date, start_time, end_time, count, intraday)
    if request_start_dt is None and request_end_dt is None:
        request_end_dt = datetime.now()
        request_start_dt = request_end_dt - timedelta(days=DEFAULT_LOOKBACK_DAYS if intraday else 400)
    elif request_start_dt is None:
        request_start_dt = request_end_dt - timedelta(days=DEFAULT_LOOKBACK_DAYS if intraday else 400)
    elif request_end_dt is None:
        request_end_dt = datetime.now()

    def _load_code(code: str) -> list[StockQuoteItem]:
        normalized_code = normalize_stock_code(code)
        cache_path = build_cache_path("opentdx", ["stocks", "quotes"], {"code": normalized_code, "adjust": adjust, "source_freq": "1m" if intraday else freq})
        cache_df = read_cache_frame(cache_path)
        need_refresh = True
        if not cache_df.empty:
            filtered_cache = filter_frame_by_datetime_range(cache_df, "trade_time", request_start_dt, request_end_dt)
            need_refresh = filtered_cache.empty or intraday_quote_cache_needs_refresh(filtered_cache, freq, request_start_dt, request_end_dt, count)
        if need_refresh:
            try:
                fetched_df = _fetch_stock_intraday_frame(normalized_code, request_start_dt, request_end_dt, adjust) if intraday else _fetch_stock_daily_frame(normalized_code, freq, request_start_dt, request_end_dt, adjust)
            except Exception:
                # 单只股票连接失败不能丢弃同批其他股票，缺口交给后续 provider 或下一轮补采。
                fetched_df = pd.DataFrame()
            if not fetched_df.empty:
                merged_df = merge_cache_frame(cache_df, fetched_df, ["code", "trade_time", "freq"], ["trade_time"])
                write_cache_frame(cache_path, merged_df)
                cache_df = merged_df
        filtered_df = filter_frame_by_datetime_range(cache_df, "trade_time", request_start_dt, request_end_dt)
        if filtered_df.empty:
            return []
        filtered_df["trade_time"] = pd.to_datetime(filtered_df["trade_time"], errors="coerce")
        agg_df = add_quote_metrics(aggregate_ohlc(filtered_df, freq)) if intraday else filtered_df
        agg_df["code"] = normalized_code
        agg_df["freq"] = freq
        agg_df["adjust"] = adjust
        agg_df = latest_n_rows(agg_df, "trade_time", count)
        return _frame_to_stock_quotes(agg_df, freq)

    items: list[StockQuoteItem] = []
    if intraday and len(codes) > 1:
        # 每只股票使用独立客户端和缓存文件，四路并发与 provider 运行时上限一致。
        with ThreadPoolExecutor(max_workers=min(4, len(codes))) as executor:
            batch_items = list(executor.map(_load_code, codes))
        # 批内已有成功结果时，顺序重试少数瞬时失败代码，避免整条 fallback 链被放大。
        if any(batch_items):
            batch_items = [code_items if code_items else _load_code(code) for code, code_items in zip(codes, batch_items)]
        for code_items in batch_items:
            items.extend(code_items)
        return items

    for code in codes:
        items.extend(_load_code(code))
    return items


def get_index_quotes(index_codes: list[str], freq: str, trade_date: str, start_date: str, end_date: str, count: int | None) -> list[IndexQuoteItem]:
    if freq not in DAILY_FREQS:
        return []

    request_start_dt, request_end_dt = build_time_bounds(trade_date, start_date, end_date, "", "", count, False)
    if request_start_dt is None and request_end_dt is None:
        request_end_dt = datetime.now()
        request_start_dt = request_end_dt - timedelta(days=400)
    elif request_start_dt is None:
        request_start_dt = request_end_dt - timedelta(days=400)
    elif request_end_dt is None:
        request_end_dt = datetime.now()

    items: list[IndexQuoteItem] = []
    for index_code in index_codes:
        normalized_code = normalize_index_code(index_code)
        cache_path = build_cache_path("opentdx", ["indexes", "quotes"], {"index_code": normalized_code, "freq": freq})
        cache_df = read_cache_frame(cache_path)
        filtered_cache = filter_frame_by_datetime_range(cache_df, "trade_time", request_start_dt, request_end_dt)
        if filtered_cache.empty or (count and len(filtered_cache) < count):
            fetched_df = _fetch_index_daily_frame(normalized_code, freq, request_start_dt, request_end_dt)
            if not fetched_df.empty:
                cache_df = merge_cache_frame(cache_df, fetched_df, ["index_code", "trade_time", "freq"], ["trade_time"])
                write_cache_frame(cache_path, cache_df)
        filtered_df = filter_frame_by_datetime_range(cache_df, "trade_time", request_start_dt, request_end_dt)
        filtered_df = latest_n_rows(filtered_df, "trade_time", count)
        items.extend(_frame_to_index_quotes(filtered_df, freq))
    return items

