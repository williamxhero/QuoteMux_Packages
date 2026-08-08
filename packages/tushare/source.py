from __future__ import annotations

from datetime import datetime, timedelta
from functools import lru_cache
import math
import threading

import pandas as pd

from quotemux.infra.cache.store import build_cache_path, filter_frame_by_date_range, filter_frame_by_datetime_range, latest_n_rows, merge_cache_frame, plan_missing_ranges, read_cache_frame, write_cache_frame
from quotemux.infra.config import DATE_FORMAT
from quotemux.infra.provider_config import get_provider_api_key, get_provider_config_value
from quotemux.common import intraday_quote_cache_needs_refresh
from platform_models import AdjFactorItem, BoardCatalogItem, BoardCategoryItem, BoardMemberHistoryItem, BoardMemberItem, BoardMoneyFlowItem, BoardQuoteItem, BSECodeMappingItem, ExpressItem, ForecastItem, IndexCatalogItem, IndexMemberItem, IndexQuoteItem, MarketCapitalFlowItem, NameHistoryItem, ShareholderChangeItem, StockBasicInfo, StockFinancialStatementItem, StockMarginItem, StockMoneyFlowItem, StockQuoteItem, TechnicalFactorItem, TradingCalendarItem, TradingSessionItem
from quotemux.infra.common import INTRADAY_RULES, aggregate_ohlc, add_quote_metrics, build_time_bounds, format_date_value, format_datetime_value, index_code_to_ts, normalize_index_code, normalize_stock_code, stock_code_to_ts
from .rate_limit import call_tushare_api

import sys
_saved_paths = [path for path in sys.path if "quotemux_packages" in path or ("packages" in path and "site-packages" not in path and "dist-packages" not in path)]
for path in _saved_paths:
    try:
        sys.path.remove(path)
    except ValueError:
        pass

try:
    import tushare as ts
except Exception:
    ts = None
finally:
    for path in reversed(_saved_paths):
        sys.path.insert(0, path)


TS_FREQ_MAP = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "60m": "60min",
    "1d": "D",
    "1w": "W",
    "1mo": "M",
}
TS_INDEX_MARKETS = ("CSI", "SSE", "SZSE", "SW", "CICC", "OTH")
TS_STOCK_LIST_STATUS = ("L", "D", "P")
_PRO_BAR_TOKEN_LOCK = threading.Lock()
DEFAULT_TUSHARE_REQUEST_TIMEOUT_SECONDS = 10.0


def _tushare_request_timeout_seconds() -> float:
    try:
        value = float(get_provider_config_value("timeout_seconds"))
    except ValueError:
        return DEFAULT_TUSHARE_REQUEST_TIMEOUT_SECONDS
    if not math.isfinite(value) or value < 1.0:
        return DEFAULT_TUSHARE_REQUEST_TIMEOUT_SECONDS
    return value


@lru_cache(maxsize=16)
def _build_ts_pro(api_key: str, timeout_seconds: float):
    return ts.pro_api(api_key, timeout=timeout_seconds)


def get_ts_pro():
    api_key = get_provider_api_key()
    if ts is None or api_key == "":
        return None
    return _build_ts_pro(api_key, _tushare_request_timeout_seconds())


def _normalize_index_market(market: str) -> str:
    if not market:
        return ""
    return market.strip().lower()


def _resolve_index_markets(market: str) -> list[str]:
    text = market.strip().upper()
    if text == "":
        return list(TS_INDEX_MARKETS)
    if text == "A_SHARE":
        return ["CSI", "SSE", "SZSE", "SW", "CICC"]
    if text in TS_INDEX_MARKETS:
        return [text]
    return []


def _to_tushare_date(value: str) -> str:
    return format_date_value(value).replace("-", "")


def _stock_exchange_from_ts_code(ts_code: str) -> str:
    text = str(ts_code).upper()
    if text.endswith(".SH"):
        return "SSE"
    if text.endswith(".SZ"):
        return "SZSE"
    if text.endswith(".BJ"):
        return "BSE"
    return ""


def _stock_market_from_row(market_text: str, exchange: str, code: str) -> str:
    text = str(market_text).lower()
    if exchange == "BSE" or code.startswith(("4", "8")) or code.startswith("920"):
        return "beijing"
    if "科创" in text or code.startswith("688"):
        return "star_market"
    if "创业" in text or code.startswith(("300", "301")):
        return "chi_next"
    return "main_board"


def _stock_list_status(status: str) -> str:
    if status == "D":
        return "delisted"
    if status == "P":
        return "pending"
    return "listed"


def _stock_statuses(list_status: str, include_delisted: bool) -> tuple[str, ...]:
    if list_status == "listed":
        return ("L",)
    if list_status == "delisted":
        return ("D",)
    if list_status == "pending":
        return ("P",)
    if include_delisted:
        return TS_STOCK_LIST_STATUS
    return ("L",)


def _name_indicates_st(name: str) -> bool:
    upper_name = name.upper().replace(" ", "")
    return upper_name.startswith("ST") or upper_name.startswith("*ST")


def _fetch_stock_basic_frame(status: str) -> pd.DataFrame:
    pro = get_ts_pro()
    if pro is None:
        return pd.DataFrame()
    try:
        df = call_tushare_api(
            "stock_basic",
            pro.stock_basic,
            exchange="",
            list_status=status,
            fields="ts_code,symbol,name,area,industry,market,list_date,delist_date,list_status",
        )
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    for column in ["ts_code", "symbol", "name", "area", "industry", "market", "list_date", "delist_date", "list_status"]:
        if column not in work.columns:
            work[column] = ""
    work["code"] = work["symbol"].fillna("").astype(str).str.zfill(6)
    work["exchange"] = work["ts_code"].map(_stock_exchange_from_ts_code)
    work["market2"] = work.apply(lambda row: _stock_market_from_row(str(row["market"]), str(row["exchange"]), str(row["code"])), axis=1)
    work["list_status2"] = work["list_status"].fillna("").astype(str).map(_stock_list_status)
    return work[["code", "name", "exchange", "market2", "list_status2", "list_date", "delist_date", "industry", "area"]]


def _load_stock_basic_frame(status: str, refresh: bool = False) -> pd.DataFrame:
    cache_path = build_cache_path("tushare", ["stocks", "catalog"], {"status": status})
    cache_df = read_cache_frame(cache_path)
    if refresh:
        fetched_df = _fetch_stock_basic_frame(status)
        if fetched_df.empty:
            return fetched_df
        write_cache_frame(cache_path, fetched_df)
        return fetched_df
    if cache_df.empty:
        fetched_df = _fetch_stock_basic_frame(status)
        if not fetched_df.empty:
            write_cache_frame(cache_path, fetched_df)
            cache_df = fetched_df
    return cache_df


def _apply_bse_code_mappings(frame: pd.DataFrame, mappings: list[BSECodeMappingItem]) -> pd.DataFrame:
    if frame.empty or mappings == []:
        return frame
    work = frame.copy()
    present_codes = set(work["code"].astype(str))
    for mapping in mappings:
        if mapping.new_code not in present_codes:
            continue
        old_code_rows = work["code"].astype(str) == mapping.old_code
        if not old_code_rows.any():
            old_code_item = work[work["code"].astype(str) == mapping.new_code].iloc[[0]].copy()
            old_code_item.loc[:, "code"] = mapping.old_code
            work = pd.concat([work, old_code_item], ignore_index=True)
            present_codes.add(mapping.old_code)
            old_code_rows = work["code"].astype(str) == mapping.old_code
        work.loc[old_code_rows, "list_status2"] = "delisted"
        work.loc[old_code_rows, "delist_date"] = mapping.effective_date
    return work


def get_stock_catalog(codes: list[str], name: str, exchange: str, list_status: str, include_delisted: bool, limit: int, offset: int, refresh: bool = False) -> list[StockBasicInfo]:
    statuses = _stock_statuses(list_status, include_delisted)
    frames = [_load_stock_basic_frame(status, refresh) for status in statuses]
    if refresh and include_delisted and statuses == TS_STOCK_LIST_STATUS and (frames[0].empty or frames[1].empty):
        return []
    frames = [frame for frame in frames if not frame.empty]
    if frames == []:
        return []
    work = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["code"], keep="last")
    # 北交所改号后目录只保留新代码；以 Tushare 改号表补充旧代码终止记录。
    work = _apply_bse_code_mappings(work, get_bse_code_mappings("", "", "active"))
    work = work.drop_duplicates(subset=["code"], keep="last")
    normalized_codes = [normalize_stock_code(code) for code in codes if normalize_stock_code(code)]
    if normalized_codes:
        work = work[work["code"].isin(normalized_codes)]
    if name:
        work = work[work["name"].fillna("").astype(str).str.contains(name, case=False, na=False)]
    if exchange:
        work = work[work["exchange"] == exchange]
    if list_status:
        work = work[work["list_status2"] == list_status]
    work = work.sort_values("code").iloc[offset: offset + limit]
    items: list[StockBasicInfo] = []
    for _, row in work.iterrows():
        items.append(
            StockBasicInfo(
                code=str(row["code"]),
                name=str(row["name"]),
                exchange=str(row["exchange"]),
                market=str(row["market2"]),
                list_status=str(row["list_status2"]),
                list_date=format_date_value(row["list_date"]),
                delist_date=format_date_value(row["delist_date"]),
                industry=str(row["industry"] or ""),
                listing_board=str(row["market2"] or ""),
                area=str(row["area"] or ""),
            )
        )
    return items


def get_stock_basic(code: str) -> StockBasicInfo | None:
    items = get_stock_catalog([normalize_stock_code(code)], "", "", "", True, 1, 0)
    return items[0] if items else None


def get_stock_name_history(code: str, start_date: str, end_date: str) -> list[NameHistoryItem]:
    pro = get_ts_pro()
    if pro is None:
        return []
    request_kwargs: dict[str, str] = {}
    ts_code = stock_code_to_ts(code)
    if ts_code != "":
        request_kwargs["ts_code"] = ts_code
    actual_start_date = _to_tushare_date(start_date)
    actual_end_date = _to_tushare_date(end_date)
    if actual_start_date != "":
        request_kwargs["start_date"] = actual_start_date
    if actual_end_date != "":
        request_kwargs["end_date"] = actual_end_date
    try:
        if ts_code != "":
            df = call_tushare_api("namechange", pro.namechange, **request_kwargs)
        else:
            frames: list[pd.DataFrame] = []
            offset = 0
            while True:
                page = call_tushare_api("namechange", pro.namechange, limit=5000, offset=offset, **request_kwargs)
                if page is None or page.empty:
                    break
                frames.append(page)
                if len(page) < 5000:
                    break
                offset += len(page)
            df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    except Exception:
        return []
    if df is None or df.empty:
        return []
    work = df.copy()
    for column in ["ts_code", "name", "start_date", "end_date", "ann_date"]:
        if column not in work.columns:
            work[column] = ""
    items: list[NameHistoryItem] = []
    for _, row in work.sort_values("start_date").iterrows():
        normalized = normalize_stock_code(str(row["ts_code"]))
        if normalized == "":
            continue
        items.append(
            NameHistoryItem(
                code=normalized,
                name=str(row["name"]),
                start_date=format_date_value(row["start_date"]),
                end_date=format_date_value(row["end_date"]),
                ann_date=format_date_value(row["ann_date"]),
            )
        )
    return items


def get_adj_factors(code: str, start_date: str, end_date: str, base_date: str) -> list[AdjFactorItem]:
    del base_date
    pro = get_ts_pro()
    ts_code = stock_code_to_ts(code)
    if pro is None or ts_code == "":
        return []
    start_value = format_date_value(start_date)
    end_value = format_date_value(end_date)
    if start_value == "" or end_value == "":
        return []
    expanded_start = (datetime.strptime(start_value, "%Y-%m-%d") - timedelta(days=10)).strftime("%Y%m%d")
    expanded_end = (datetime.strptime(end_value, "%Y-%m-%d") + timedelta(days=10)).strftime("%Y%m%d")
    try:
        df = call_tushare_api("adj_factor", pro.adj_factor, ts_code=ts_code, start_date=expanded_start, end_date=expanded_end)
    except Exception:
        return []
    if df is None or df.empty:
        return []
    work = df.copy()
    for column in ["trade_date", "adj_factor"]:
        if column not in work.columns:
            work[column] = None
    work["trade_date"] = work["trade_date"].map(format_date_value)
    work["adj_factor"] = pd.to_numeric(work["adj_factor"], errors="coerce")
    work = work.dropna(subset=["adj_factor"]).drop_duplicates(subset=["trade_date"], keep="last").sort_values("trade_date")
    try:
        daily = call_tushare_api(
            "daily",
            pro.daily,
            ts_code=ts_code,
            start_date=_to_tushare_date(start_value),
            end_date=_to_tushare_date(end_value),
            fields="trade_date",
        )
    except Exception:
        daily = pd.DataFrame()
    requested_dates = []
    if daily is not None and not daily.empty and "trade_date" in daily.columns:
        requested_dates = sorted({format_date_value(value) for value in daily["trade_date"] if format_date_value(value) != ""})
    if requested_dates == []:
        try:
            bak_daily = call_tushare_api(
                "bak_daily",
                pro.bak_daily,
                ts_code=ts_code,
                start_date=_to_tushare_date(start_value),
                end_date=_to_tushare_date(end_value),
                fields="trade_date",
            )
        except Exception:
            bak_daily = pd.DataFrame()
        if bak_daily is not None and not bak_daily.empty and "trade_date" in bak_daily.columns:
            requested_dates = sorted(
                {format_date_value(value) for value in bak_daily["trade_date"] if format_date_value(value) != ""}
            )
    if requested_dates:
        factor_by_date = dict(zip(work["trade_date"], work["adj_factor"]))
        factor_dates = list(work["trade_date"])
        for trade_date in requested_dates:
            if trade_date in factor_by_date:
                continue
            previous_dates = [value for value in factor_dates if value < trade_date]
            next_dates = [value for value in factor_dates if value > trade_date]
            if previous_dates == [] or next_dates == []:
                continue
            previous_factor = factor_by_date[previous_dates[-1]]
            next_factor = factor_by_date[next_dates[0]]
            if previous_factor == next_factor:
                factor_by_date[trade_date] = previous_factor
        work = pd.DataFrame(
            [
                {"trade_date": trade_date, "adj_factor": factor_by_date[trade_date]}
                for trade_date in sorted(factor_by_date)
            ]
        )
    work = work[(work["trade_date"] >= start_value) & (work["trade_date"] <= end_value)]
    normalized = normalize_stock_code(code)
    items: list[AdjFactorItem] = []
    for _, row in work.sort_values("trade_date").iterrows():
        items.append(
            AdjFactorItem(
                code=normalized,
                trade_date=format_date_value(row["trade_date"]).replace("-", ""),
                adj_factor=float(row["adj_factor"]) if pd.notna(row["adj_factor"]) else None,
            )
        )
    return items


def _board_code_to_ts(board_code: str) -> str:
    text = str(board_code).strip().upper()
    if text == "":
        return ""
    if "." in text:
        return text
    return f"{text}.TI"


def _board_ref_item(board_code: str) -> tuple[str, str]:
    if not str(board_code).upper().startswith("BK"):
        return "", ""
    try:
        from quotemux.infra.db.client import query_dataframe

        frame = query_dataframe(
            """
            select name, board_type
            from ref.board
            where board_code = %s
            """,
            (str(board_code).upper(),),
        )
    except Exception:
        return "", ""
    if frame.empty:
        return "", ""
    row = frame.iloc[0]
    return str(row["name"]).strip(), str(row["board_type"]).strip()


def _tushare_board_code_from_name(board_name: str, board_type: str) -> str:
    if board_name == "":
        return ""
    frames = [_load_board_catalog_frame(index_type) for index_type in ("N", "I")]
    frames = [frame for frame in frames if not frame.empty]
    if frames == []:
        return ""
    work = pd.concat(frames, ignore_index=True)
    matched = work[work["name"].astype(str).str.strip() == board_name]
    if matched.empty:
        return ""
    candidates = [str(row["board_code"]).upper() for _, row in matched.iterrows()]
    if board_type == "concept":
        prefixes = ("885", "886", "883", "884", "881", "877")
    elif board_type == "industry":
        prefixes = ("884", "881", "877", "861", "871", "700")
    else:
        prefixes = ("884", "881", "877", "885", "886", "861", "871", "700", "883")
    for prefix in prefixes:
        for candidate in candidates:
            if candidate.startswith(prefix):
                return candidate
    return sorted(candidates)[0]


def _resolve_board_code_pair(board_code: str) -> tuple[str, str]:
    output_code = _board_code_to_ts(board_code).split(".", 1)[0]
    if not output_code.startswith("BK"):
        return output_code, output_code
    board_name, board_type = _board_ref_item(output_code)
    provider_code = _tushare_board_code_from_name(board_name, board_type)
    if provider_code == "":
        return output_code, output_code
    return output_code, provider_code


def _board_category_from_code(board_code: str) -> str:
    text = str(board_code).upper()
    if text.startswith(("881", "877")):
        return "industry"
    if text.startswith(("885", "886", "BK")):
        return "concept"
    return ""


def _fetch_board_catalog_frame(index_type: str) -> pd.DataFrame:
    pro = get_ts_pro()
    if pro is None:
        return pd.DataFrame()
    try:
        df = call_tushare_api("ths_index", pro.ths_index, type=index_type)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    for column in ["ts_code", "name", "list_date", "type"]:
        if column not in work.columns:
            work[column] = ""
    work["board_code"] = work["ts_code"].fillna("").astype(str).str.split(".").str[0]
    work["category"] = work["board_code"].map(_board_category_from_code)
    work["status"] = "active"
    work["start_date"] = work["list_date"].fillna("").astype(str)
    work["end_date"] = ""
    return work[["board_code", "name", "category", "status", "start_date", "end_date"]]


def _load_board_catalog_frame(index_type: str) -> pd.DataFrame:
    cache_path = build_cache_path("tushare", ["boards", "catalog"], {"type": index_type})
    cache_df = read_cache_frame(cache_path)
    if cache_df.empty or "start_date" not in cache_df.columns:
        fetched_df = _fetch_board_catalog_frame(index_type)
        if not fetched_df.empty:
            write_cache_frame(cache_path, fetched_df)
            cache_df = fetched_df
    return cache_df


def get_board_catalog(category: str, market: str, status: str, limit: int, offset: int) -> list[BoardCatalogItem]:
    if market and market != "a_share":
        return []
    frames = [_load_board_catalog_frame(index_type) for index_type in ("N", "I")]
    frames = [frame for frame in frames if not frame.empty]
    if frames == []:
        return []
    work = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["board_code"], keep="last")
    if category:
        work = work[work["category"] == category]
    if status:
        work = work[work["status"] == status]
    work = work.sort_values("board_code").iloc[offset: offset + limit]
    return [
        BoardCatalogItem(
            board_code=str(row["board_code"]),
            board_name=str(row["name"]),
            category=str(row["category"]),
            market="a_share",
            status=str(row["status"]),
            start_date=str(row.get("start_date", "")),
            end_date=str(row.get("end_date", "")),
        )
        for _, row in work.iterrows()
    ]


def get_board_profile(board_code: str) -> BoardCatalogItem | None:
    for item in get_board_catalog("", "a_share", "", 100000, 0):
        if item.board_code == board_code:
            return item
    return None


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


def _fetch_board_members_frame(board_code: str, output_board_code: str = "") -> pd.DataFrame:
    pro = get_ts_pro()
    ts_code = _board_code_to_ts(board_code)
    if pro is None or ts_code == "":
        return pd.DataFrame()
    try:
        df = call_tushare_api("ths_member", pro.ths_member, ts_code=ts_code)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    for column in ["ts_code", "con_code", "con_name", "weight", "in_date", "out_date"]:
        if column not in work.columns:
            work[column] = ""
    work["board_code"] = output_board_code or ts_code.split(".", 1)[0]
    work["code"] = work["con_code"].map(normalize_stock_code)
    work["name"] = work["con_name"].fillna("").astype(str)
    return work[["board_code", "code", "name", "weight", "in_date", "out_date"]]


def _load_board_members_frame(board_code: str) -> pd.DataFrame:
    normalized, provider_code = _resolve_board_code_pair(board_code)
    cache_path = build_cache_path("tushare", ["boards", "members"], {"board_code": normalized})
    cache_df = read_cache_frame(cache_path)
    if cache_df.empty:
        fetched_df = _fetch_board_members_frame(provider_code, normalized)
        if not fetched_df.empty:
            write_cache_frame(cache_path, fetched_df)
            cache_df = fetched_df
    return cache_df


def get_board_members(board_code: str, trade_date: str) -> list[BoardMemberItem]:
    work = _load_board_members_frame(board_code)
    if work.empty:
        return []
    target_date = _to_tushare_date(trade_date)
    if target_date:
        work = work[
            (work["in_date"].fillna("").astype(str) <= target_date)
            & ((work["out_date"].fillna("").astype(str) == "") | (work["out_date"].fillna("").astype(str) >= target_date))
        ]
    items: list[BoardMemberItem] = []
    for _, row in work.sort_values("code").iterrows():
        items.append(
            BoardMemberItem(
                board_code=str(row["board_code"]),
                code=str(row["code"]),
                name=str(row["name"]),
                weight=float(row["weight"]) if pd.notna(row["weight"]) and str(row["weight"]) != "" else None,
                join_date=format_date_value(row["in_date"]),
            )
        )
    return items


def get_board_member_history(board_code: str, start_date: str, end_date: str) -> list[BoardMemberHistoryItem]:
    work = _load_board_members_frame(board_code)
    if work.empty:
        return []
    start_text = _to_tushare_date(start_date)
    end_text = _to_tushare_date(end_date)
    items: list[BoardMemberHistoryItem] = []
    for _, row in work.iterrows():
        in_date = str(row["in_date"] or "")
        out_date = str(row["out_date"] or "")
        if in_date and (start_text == "" or in_date >= start_text) and (end_text == "" or in_date <= end_text):
            items.append(BoardMemberHistoryItem(board_code=str(row["board_code"]), code=str(row["code"]), name=str(row["name"]), effective_date=format_date_value(in_date), action="add"))
        if out_date and (start_text == "" or out_date >= start_text) and (end_text == "" or out_date <= end_text):
            items.append(BoardMemberHistoryItem(board_code=str(row["board_code"]), code=str(row["code"]), name=str(row["name"]), effective_date=format_date_value(out_date), action="remove"))
        if in_date == "" and out_date == "":
            baseline_date = "19000101"
            if (start_text == "" or baseline_date >= start_text) and (end_text == "" or baseline_date <= end_text):
                items.append(BoardMemberHistoryItem(board_code=str(row["board_code"]), code=str(row["code"]), name=str(row["name"]), effective_date=format_date_value(baseline_date), action="add"))
    return sorted(items, key=lambda item: (item.effective_date, item.code, item.action))


def get_industry_catalog(level: str, source: str, limit: int, offset: int) -> list[BoardCatalogItem]:
    pro = get_ts_pro()
    if pro is None:
        return []
    source_text = source.strip().upper() or "SW2021"
    try:
        frame = call_tushare_api("index_classify", pro.index_classify, src=source_text)
    except Exception:
        return []
    if frame is None or frame.empty:
        return []
    work = frame.copy()
    for column in ["index_code", "industry_name", "level", "is_pub"]:
        if column not in work.columns:
            work[column] = ""
    level_text = level.strip().upper()
    if level_text != "":
        work = work[work["level"].fillna("").astype(str).str.upper() == level_text]
    work = work[work["is_pub"].fillna("").astype(str) != "0"]
    work = work.drop_duplicates("index_code").sort_values("index_code").iloc[offset : offset + limit]
    items: list[BoardCatalogItem] = []
    for _, row in work.iterrows():
        index_code = str(row["index_code"]).strip()
        if index_code == "":
            continue
        items.append(
            BoardCatalogItem(
                board_code=index_code.split(".", 1)[0],
                board_name=str(row["industry_name"] or ""),
                category=f"industry_{source_text.lower()}_{str(row['level']).lower()}",
                market="a_share",
                status="active",
            )
        )
    return items


def get_industry_member_history(board_code: str, start_date: str, end_date: str) -> list[BoardMemberHistoryItem]:
    pro = get_ts_pro()
    normalized_code = str(board_code).strip().upper().split(".", 1)[0]
    if pro is None or normalized_code == "":
        return []
    try:
        frame = call_tushare_api("index_member", pro.index_member, index_code=f"{normalized_code}.SI")
    except Exception:
        return []
    if frame is None or frame.empty:
        return []
    start_text = _to_tushare_date(start_date)
    end_text = _to_tushare_date(end_date)
    items: list[BoardMemberHistoryItem] = []
    for _, row in frame.iterrows():
        code = normalize_stock_code(str(row.get("con_code", "")))
        in_date = format_date_value(row.get("in_date"))
        out_date = format_date_value(row.get("out_date"))
        if code == "":
            continue
        overlaps_window = (end_text == "" or in_date == "" or in_date <= end_text) and (out_date == "" or start_text == "" or out_date >= start_text)
        if not overlaps_window:
            continue
        baseline_date = start_text if start_text != "" and (in_date == "" or in_date < start_text) else in_date
        if baseline_date != "":
            items.append(BoardMemberHistoryItem(board_code=normalized_code, code=code, name="", effective_date=format_date_value(baseline_date), action="add"))
        if out_date != "" and (start_text == "" or out_date >= start_text) and (end_text == "" or out_date <= end_text):
            items.append(BoardMemberHistoryItem(board_code=normalized_code, code=code, name="", effective_date=format_date_value(out_date), action="remove"))
    return sorted(items, key=lambda item: (item.effective_date, item.code, item.action))


def _fetch_board_quotes_frame(board_code: str, start_value: str, end_value: str, output_board_code: str = "") -> pd.DataFrame:
    pro = get_ts_pro()
    ts_code = _board_code_to_ts(board_code)
    if pro is None or ts_code == "":
        return pd.DataFrame()
    fetcher = getattr(pro, "ths_daily", None)
    if fetcher is None:
        return pd.DataFrame()
    try:
        df = call_tushare_api("ths_daily", fetcher, ts_code=ts_code, start_date=start_value, end_date=end_value)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    for column in ["trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_change", "vol", "avg_price", "amount"]:
        if column not in work.columns:
            work[column] = None
    work["board_code"] = output_board_code or ts_code.split(".", 1)[0]
    work["trade_time"] = pd.to_datetime(work["trade_date"], errors="coerce")
    work["volume"] = pd.to_numeric(work["vol"], errors="coerce") if "vol" in work.columns else None
    work["amount"] = pd.to_numeric(work["amount"], errors="coerce")
    amount_missing = work["amount"].isna()
    work.loc[amount_missing, "amount"] = pd.to_numeric(work.loc[amount_missing, "avg_price"], errors="coerce") * pd.to_numeric(work.loc[amount_missing, "vol"], errors="coerce") * 100
    work["change"] = pd.to_numeric(work["change"], errors="coerce")
    work["pct_chg"] = pd.to_numeric(work["pct_change"], errors="coerce")
    return work[["board_code", "trade_time", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "volume", "amount"]]


def get_board_quotes(board_codes: list[str], freq: str, trade_date: str, start_date: str, end_date: str, start_time: str, end_time: str, count: int | None) -> list[BoardQuoteItem]:
    del start_time
    del end_time
    request_start_dt, request_end_dt = build_time_bounds(trade_date, start_date, end_date, "", "", count, False)
    request_start = request_start_dt.strftime(DATE_FORMAT) if request_start_dt is not None else ""
    request_end = request_end_dt.strftime(DATE_FORMAT) if request_end_dt is not None else ""
    if request_start == "" and request_end == "":
        request_end = datetime.now().strftime(DATE_FORMAT)
        request_start = (datetime.now() - timedelta(days=400)).strftime(DATE_FORMAT)
    elif request_start == "":
        request_start = request_end
    elif request_end == "":
        request_end = request_start
    items: list[BoardQuoteItem] = []
    for board_code in board_codes:
        normalized, provider_code = _resolve_board_code_pair(board_code)
        cache_path = build_cache_path("tushare", ["boards", "quotes"], {"board_code": normalized})
        cache_df = read_cache_frame(cache_path)
        missing_ranges = plan_missing_ranges(cache_df, "trade_time", request_start, request_end, "day")
        fetched_frames = [_fetch_board_quotes_frame(provider_code, missing_start, missing_end, normalized) for missing_start, missing_end in missing_ranges]
        fetched_frames = [frame for frame in fetched_frames if not frame.empty]
        if cache_df.empty and fetched_frames == []:
            fetched_df = _fetch_board_quotes_frame(provider_code, request_start, request_end, normalized)
            if not fetched_df.empty:
                fetched_frames.append(fetched_df)
        if fetched_frames:
            cache_df = merge_cache_frame(cache_df, pd.concat(fetched_frames, ignore_index=True), ["board_code", "trade_time"], ["trade_time"])
            write_cache_frame(cache_path, cache_df)
        filtered_df = filter_frame_by_date_range(cache_df, "trade_time", request_start, request_end)
        if filtered_df.empty:
            continue
        filtered_df["trade_time"] = pd.to_datetime(filtered_df["trade_time"])
        if freq == "1d":
            daily_df = latest_n_rows(filtered_df.sort_values("trade_time"), "trade_time", count)
            for _, row in daily_df.iterrows():
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
            continue
        agg_df = add_quote_metrics(aggregate_ohlc(filtered_df.drop(columns=["board_code"]), freq))
        if count:
            agg_df = agg_df.tail(count)
        for _, row in agg_df.iterrows():
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


def get_board_daily_money_flow_snapshot(trade_date: str, scope: str, limit: int, offset: int) -> list[BoardMoneyFlowItem]:
    scopes = [scope] if scope in {"industry", "concept"} else ["concept", "industry"]
    items: list[BoardMoneyFlowItem] = []
    target_count = limit + offset
    for current_scope in scopes:
        catalog_items = get_board_catalog(current_scope, "a_share", "active", 10000, 0)
        if current_scope == "industry":
            catalog_items = [item for item in catalog_items if item.board_code.startswith("881")]
        for catalog_item in catalog_items:
            items.extend(get_board_money_flow(catalog_item.board_code, trade_date, "", "", current_scope))
            if len(items) >= target_count:
                break
        if len(items) >= target_count:
            break
    return sorted(items, key=lambda item: (item.board_code, item.trade_date))[offset: offset + limit]


def get_market_sessions(codes: str) -> list[TradingSessionItem]:
    items: list[TradingSessionItem] = []
    for code in [normalize_stock_code(item) for item in codes.split(",") if normalize_stock_code(item)]:
        items.append(TradingSessionItem(code=code, session_name="pre_open", start_time="09:15:00", end_time="09:25:00", timezone="Asia/Shanghai"))
        items.append(TradingSessionItem(code=code, session_name="continuous", start_time="09:30:00", end_time="11:30:00", timezone="Asia/Shanghai"))
        items.append(TradingSessionItem(code=code, session_name="continuous", start_time="13:00:00", end_time="14:57:00", timezone="Asia/Shanghai"))
        items.append(TradingSessionItem(code=code, session_name="closing_call", start_time="14:57:00", end_time="15:00:00", timezone="Asia/Shanghai"))
        items.append(TradingSessionItem(code=code, session_name="after_hours", start_time="15:00:00", end_time="15:30:00", timezone="Asia/Shanghai"))
    return items


def _fetch_index_catalog_frame(market: str) -> pd.DataFrame:
    pro = get_ts_pro()
    if pro is None:
        return pd.DataFrame()
    try:
        df = call_tushare_api("index_basic", pro.index_basic, market=market)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    for column in ["ts_code", "name", "category", "market", "publisher", "list_date", "exp_date"]:
        if column not in work.columns:
            work[column] = ""
    work["index_code"] = work["ts_code"].map(normalize_index_code)
    work["index_name"] = work["name"].fillna("").astype(str)
    work["category"] = work["category"].fillna("").astype(str)
    work["market2"] = work["market"].fillna("").astype(str).map(_normalize_index_market)
    work["publisher2"] = work["publisher"].fillna("").astype(str)
    work["list_date2"] = work["list_date"].fillna("").astype(str)
    work["status"] = work["exp_date"].fillna("").astype(str).map(lambda value: "inactive" if value else "active")
    return work[["index_code", "index_name", "category", "market2", "publisher2", "list_date2", "status"]]


def get_index_catalog(index_code: str, category: str, market: str, publisher: str, status: str) -> list[IndexCatalogItem]:
    selected_markets = _resolve_index_markets(market)
    if market and not selected_markets:
        return []
    frames: list[pd.DataFrame] = []
    for market_code in selected_markets:
        cache_path = build_cache_path("tushare", ["indexes", "catalog"], {"market": market_code.lower()})
        cache_df = read_cache_frame(cache_path)
        if cache_df.empty:
            fetched_df = _fetch_index_catalog_frame(market_code)
            if not fetched_df.empty:
                write_cache_frame(cache_path, fetched_df)
                cache_df = fetched_df
        if not cache_df.empty:
            frames.append(cache_df)
    if not frames:
        return []
    work = merge_cache_frame(pd.DataFrame(), pd.concat(frames, ignore_index=True), ["index_code"], ["index_code"])
    normalized_code = normalize_index_code(index_code)
    if normalized_code:
        work = work[work["index_code"] == normalized_code]
    if category:
        work = work[work["category"] == category]
    if publisher:
        work = work[work["publisher2"] == publisher]
    if status:
        work = work[work["status"] == status]
    items: list[IndexCatalogItem] = []
    for _, row in work.sort_values("index_code").iterrows():
        items.append(
            IndexCatalogItem(
                index_code=str(row["index_code"]),
                index_name=str(row["index_name"]),
                category=str(row["category"]),
                market=str(row["market2"]),
                publisher=str(row["publisher2"]),
                list_date=format_date_value(row["list_date2"]),
                status=str(row["status"]),
            )
        )
    return items


from .market_topics import get_block_trades, get_connect_active_top10, get_connect_capital_flow, get_connect_quotas, get_dragon_tiger, get_dragon_tiger_institutions, get_hot_money_details, get_hot_money_profiles, get_market_open_auctions
from .stock_chips import get_chip_distribution, get_chip_performance
from .stock_finance import get_audits, get_disclosure_dates, get_dividends, get_express, get_forecasts, get_main_business, get_repurchases, get_rights_issues, get_share_changes, get_unlock_schedules
from .stock_financial_pit import get_stock_financial_pit_period
from .stock_ownership import get_ccass_holding_details, get_ccass_holdings, get_hk_connect_holdings, get_pledge_details, get_pledge_stats, get_shareholder_count, get_shareholder_top10
from .stocks import get_auctions, get_bse_code_mappings, get_company_profile, get_hk_connect_targets, get_management_rewards, get_managers, get_nine_turn, get_premarket, get_rank_broker_monthly_picks, get_rank_research_reports, get_research_reports, get_stock_ah_comparisons, get_stock_archive, get_stock_daily_basic, get_stock_daily_market_value, get_stock_daily_valuation, get_stock_finance_indicators, get_stock_risk_flags, get_surveys


def _fetch_index_quotes_frame(index_code: str, start_value: str, end_value: str) -> pd.DataFrame:
    pro = get_ts_pro()
    if pro is None:
        return pd.DataFrame()
    try:
        df = call_tushare_api("index_daily", pro.index_daily, ts_code=index_code_to_ts(index_code), start_date=start_value, end_date=end_value)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy().sort_values("trade_date")
    for column in ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]:
        if column not in work.columns:
            work[column] = None
    work["index_code"] = work["ts_code"].map(normalize_index_code)
    work["trade_time"] = pd.to_datetime(work["trade_date"])
    work["volume2"] = pd.to_numeric(work["vol"], errors="coerce") if "vol" in work.columns else None
    return work[["index_code", "trade_time", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "volume2", "amount"]]


def get_index_quotes(
    index_codes: list[str],
    freq: str,
    trade_date: str,
    start_date: str,
    end_date: str,
    count: int | None,
) -> list[IndexQuoteItem]:
    request_start_dt, request_end_dt = build_time_bounds(trade_date, start_date, end_date, "", "", count, False)
    request_start = request_start_dt.strftime(DATE_FORMAT) if request_start_dt is not None else ""
    request_end = request_end_dt.strftime(DATE_FORMAT) if request_end_dt is not None else ""
    if request_start == "" and request_end == "":
        request_end = datetime.now().strftime(DATE_FORMAT)
        request_start = (datetime.now() - timedelta(days=400)).strftime(DATE_FORMAT)
    elif request_start == "":
        request_start = request_end
    elif request_end == "":
        request_end = request_start
    items: list[IndexQuoteItem] = []
    for index_code in index_codes:
        normalized = normalize_index_code(index_code)
        cache_path = build_cache_path("tushare", ["indexes", "quotes"], {"index_code": normalized})
        cache_df = read_cache_frame(cache_path)
        missing_ranges = plan_missing_ranges(cache_df, "trade_time", request_start, request_end, "day")
        fetched_frames: list[pd.DataFrame] = []
        for missing_start, missing_end in missing_ranges:
            fetched_df = _fetch_index_quotes_frame(normalized, missing_start, missing_end)
            if not fetched_df.empty:
                fetched_frames.append(fetched_df)
        if cache_df.empty and not fetched_frames:
            fetched_df = _fetch_index_quotes_frame(normalized, request_start, request_end)
            if not fetched_df.empty:
                fetched_frames.append(fetched_df)
        if fetched_frames:
            merged_cache = merge_cache_frame(cache_df, pd.concat(fetched_frames, ignore_index=True), ["index_code", "trade_time"], ["trade_time"])
            write_cache_frame(cache_path, merged_cache)
            cache_df = merged_cache
        filtered_df = filter_frame_by_date_range(cache_df, "trade_time", request_start, request_end)
        if filtered_df.empty:
            continue
        filtered_df["trade_time"] = pd.to_datetime(filtered_df["trade_time"])
        agg_df = add_quote_metrics(aggregate_ohlc(filtered_df.rename(columns={"volume2": "volume"}), freq))
        if count:
            agg_df = agg_df.tail(count)
        for _, row in agg_df.iterrows():
            items.append(
                IndexQuoteItem(
                    index_code=normalized,
                    trade_time=format_datetime_value(row["trade_time"], freq),
                    freq=freq,
                    open=float(row["open"]) if pd.notna(row["open"]) else None,
                    high=float(row["high"]) if pd.notna(row["high"]) else None,
                    low=float(row["low"]) if pd.notna(row["low"]) else None,
                    close=float(row["close"]) if pd.notna(row["close"]) else None,
                    pre_close=float(row["pre_close"]) if pd.notna(row["pre_close"]) else None,
                    change=float(row["change"]) if pd.notna(row["change"]) else None,
                    pct_chg=float(row["pct_chg"]) if pd.notna(row["pct_chg"]) else None,
                    volume=float(row["volume"]) if "volume" in row and pd.notna(row["volume"]) else None,
                    amount=float(row["amount"]) if pd.notna(row["amount"]) else None,
                )
            )
    return items


def _fetch_index_members_frame(index_code: str, start_value: str, end_value: str) -> pd.DataFrame:
    pro = get_ts_pro()
    if pro is None:
        return pd.DataFrame()
    try:
        df = call_tushare_api("index_weight", pro.index_weight, index_code=index_code_to_ts(index_code), start_date=start_value, end_date=end_value)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    for column in ["index_code", "con_code", "trade_date", "weight"]:
        if column not in work.columns:
            work[column] = None
    work["index_code2"] = work["index_code"].map(normalize_index_code)
    work["code"] = work["con_code"].map(normalize_stock_code)
    work["trade_date2"] = work["trade_date"].fillna("").astype(str)
    return work[["index_code2", "code", "trade_date2", "weight"]]


def get_index_members(index_code: str, trade_date: str) -> list[IndexMemberItem]:
    normalized = normalize_index_code(index_code)
    actual_trade_date = format_date_value(trade_date)
    if actual_trade_date:
        target_day = datetime.strptime(actual_trade_date, "%Y-%m-%d")
        start_value = target_day.replace(day=1).strftime(DATE_FORMAT)
        end_value = (target_day.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        end_text = end_value.strftime(DATE_FORMAT)
    else:
        end_text = datetime.now().strftime(DATE_FORMAT)
        start_value = (datetime.now() - timedelta(days=370)).strftime(DATE_FORMAT)
    cache_path = build_cache_path("tushare", ["indexes", "members"], {"index_code": normalized})
    cache_df = read_cache_frame(cache_path)
    missing_ranges = plan_missing_ranges(cache_df, "trade_date2", start_value, end_text, "day")
    fetched_frames: list[pd.DataFrame] = []
    for missing_start, missing_end in missing_ranges:
        fetched_df = _fetch_index_members_frame(normalized, missing_start, missing_end)
        if not fetched_df.empty:
            fetched_frames.append(fetched_df)
    if cache_df.empty and not fetched_frames:
        fetched_df = _fetch_index_members_frame(normalized, start_value, end_text)
        if not fetched_df.empty:
            fetched_frames.append(fetched_df)
    if fetched_frames:
        merged_cache = merge_cache_frame(cache_df, pd.concat(fetched_frames, ignore_index=True), ["index_code2", "code", "trade_date2"], ["trade_date2", "code"])
        write_cache_frame(cache_path, merged_cache)
        cache_df = merged_cache
    filtered_df = filter_frame_by_date_range(cache_df, "trade_date2", start_value, end_text)
    if filtered_df.empty:
        return []
    if actual_trade_date:
        exact_trade_date = actual_trade_date.replace("-", "")
        exact_df = filtered_df[filtered_df["trade_date2"] == exact_trade_date]
        if exact_df.empty:
            candidate_df = filtered_df[filtered_df["trade_date2"] <= exact_trade_date]
            if candidate_df.empty:
                filtered_df = pd.DataFrame()
            else:
                latest_trade_date = candidate_df["trade_date2"].max()
                filtered_df = candidate_df[candidate_df["trade_date2"] == latest_trade_date]
        else:
            filtered_df = exact_df
    else:
        latest_trade_date = filtered_df["trade_date2"].max()
        filtered_df = filtered_df[filtered_df["trade_date2"] == latest_trade_date]
    if filtered_df.empty:
        return []
    items: list[IndexMemberItem] = []
    for _, row in filtered_df.sort_values(["trade_date2", "code"]).iterrows():
        items.append(
            IndexMemberItem(
                index_code=str(row["index_code2"]),
                code=str(row["code"]),
                name="",
                weight=float(row["weight"]) if pd.notna(row["weight"]) else None,
                trade_date=format_date_value(str(row["trade_date2"])),
            )
        )
    return items


def _fetch_stock_quotes_frame(code: str, freq: str, start_dt: datetime | None, end_dt: datetime | None, adjust: str) -> pd.DataFrame:
    api_key = get_provider_api_key()
    if ts is None or api_key == "" or freq == "tick":
        return pd.DataFrame()
    try:
        with _PRO_BAR_TOKEN_LOCK:
            ts.set_token(api_key)
            df = call_tushare_api(
                "pro_bar",
                ts.pro_bar,
                ts_code=stock_code_to_ts(code),
                start_date=start_dt.strftime(DATE_FORMAT) if start_dt else "",
                end_date=end_dt.strftime(DATE_FORMAT) if end_dt else "",
                asset="E",
                adj=None if adjust == "none" else adjust,
                freq=TS_FREQ_MAP.get(freq, "D"),
            )
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    time_column = "trade_time" if "trade_time" in df.columns else "trade_date"
    volume_column = "vol" if "vol" in df.columns else "volume"
    work = df.copy().sort_values(time_column)
    work["code"] = normalize_stock_code(code)
    work["trade_time"] = pd.to_datetime(work[time_column])
    work["freq"] = freq
    work["adjust"] = adjust
    work["volume2"] = pd.to_numeric(work[volume_column], errors="coerce") if volume_column in work.columns else None
    if freq in {"1d", "1w", "1mo"}:
        work["volume2"] = work["volume2"] * 100
    if "amount" in work.columns:
        work["amount"] = pd.to_numeric(work["amount"], errors="coerce") * 1000
    work["is_suspended"] = False
    work["is_st"] = False
    if "name" in work.columns:
        work["is_st"] = work["name"].fillna("").astype(str).map(_name_indicates_st)
    return work[["code", "trade_time", "freq", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "volume2", "amount", "adjust", "is_suspended", "is_st"]]


def _fetch_stock_daily_snapshot_frame(trade_date: str) -> pd.DataFrame:
    pro = get_ts_pro()
    if pro is None:
        return pd.DataFrame()
    try:
        df = call_tushare_api("daily", pro.daily, trade_date=trade_date.replace("-", ""))
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["code"] = work["ts_code"].astype(str).str.split(".").str[0]
    work["trade_time"] = pd.to_datetime(work["trade_date"])
    work["freq"] = "1d"
    work["adjust"] = "none"
    work["volume2"] = pd.to_numeric(work["vol"], errors="coerce") * 100 if "vol" in work.columns else None
    for column in ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "amount"]:
        if column not in work.columns:
            work[column] = None
    work["amount"] = pd.to_numeric(work["amount"], errors="coerce") * 1000
    work["is_suspended"] = False
    work["is_st"] = False
    return work[["code", "trade_time", "freq", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "volume2", "amount", "adjust", "is_suspended", "is_st"]]


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
                pre_close=float(row["pre_close"]) if pd.notna(row["pre_close"]) else None,
                change=float(row["change"]) if pd.notna(row["change"]) else None,
                pct_chg=float(row["pct_chg"]) if pd.notna(row["pct_chg"]) else None,
                volume=float(row["volume2"]) if pd.notna(row["volume2"]) else None,
                amount=float(row["amount"]) if pd.notna(row["amount"]) else None,
                adjust=str(row["adjust"]),
                is_suspended=bool(row["is_suspended"]) if "is_suspended" in row and pd.notna(row["is_suspended"]) else False,
                is_st=bool(row["is_st"]) if "is_st" in row and pd.notna(row["is_st"]) else False,
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
    request_start_dt, request_end_dt = build_time_bounds(trade_date, start_date, end_date, start_time, end_time, count, freq in INTRADAY_RULES)
    items: list[StockQuoteItem] = []
    for code in codes:
        cache_path = build_cache_path("tushare", ["stocks", "quotes"], {"code": normalize_stock_code(code), "freq": freq, "adjust": adjust})
        cache_df = read_cache_frame(cache_path)
        fetch_start_dt = request_start_dt
        fetch_end_dt = request_end_dt
        if fetch_start_dt is None and fetch_end_dt is None:
            fetch_end_dt = datetime.now()
            fetch_start_dt = fetch_end_dt - timedelta(days=30)
        range_start = fetch_start_dt.strftime("%Y%m%d") if fetch_start_dt else ""
        range_end = fetch_end_dt.strftime("%Y%m%d") if fetch_end_dt else ""
        missing_ranges = plan_missing_ranges(cache_df, "trade_time", range_start, range_end, "day")
        filtered_cache = filter_frame_by_datetime_range(cache_df, "trade_time", request_start_dt, request_end_dt)
        if intraday_quote_cache_needs_refresh(filtered_cache, freq, request_start_dt, request_end_dt, count):
            missing_ranges = [(range_start, range_end)]
        fetched_frames: list[pd.DataFrame] = []
        for missing_start, missing_end in missing_ranges:
            start_dt = datetime.strptime(missing_start, "%Y%m%d")
            end_dt = datetime.strptime(missing_end, "%Y%m%d") + timedelta(hours=23, minutes=59, seconds=59)
            fetched_df = _fetch_stock_quotes_frame(code, freq, start_dt, end_dt, adjust)
            if not fetched_df.empty:
                fetched_frames.append(fetched_df)
        if cache_df.empty and not fetched_frames:
            fetched_df = _fetch_stock_quotes_frame(code, freq, fetch_start_dt, fetch_end_dt, adjust)
            if not fetched_df.empty:
                fetched_frames.append(fetched_df)
        if fetched_frames:
            merged_cache = merge_cache_frame(cache_df, pd.concat(fetched_frames, ignore_index=True), ["code", "trade_time", "freq"], ["trade_time"])
            write_cache_frame(cache_path, merged_cache)
            cache_df = merged_cache
        filtered_df = filter_frame_by_datetime_range(cache_df, "trade_time", request_start_dt, request_end_dt)
        filtered_df = latest_n_rows(filtered_df, "trade_time", count)
        items.extend(_frame_to_stock_quotes(filtered_df, freq))
    return items


def get_stock_daily_snapshot(trade_date: str) -> list[StockQuoteItem]:
    actual_trade_date = format_date_value(trade_date)
    if actual_trade_date == "":
        return []
    cache_path = build_cache_path("tushare", ["stocks", "quotes", "daily-snapshot"], {"trade_date": actual_trade_date.replace("-", "")})
    cache_df = read_cache_frame(cache_path)
    if cache_df.empty:
        fetched_df = _fetch_stock_daily_snapshot_frame(actual_trade_date)
        if not fetched_df.empty:
            write_cache_frame(cache_path, fetched_df)
            cache_df = fetched_df
    if cache_df.empty:
        return []
    filtered_df = filter_frame_by_date_range(cache_df, "trade_time", actual_trade_date, actual_trade_date)
    return _frame_to_stock_quotes(filtered_df, "1d")


def get_stock_daily_snapshot_full(trade_date: str) -> list[StockQuoteItem]:
    return get_stock_daily_snapshot(trade_date)


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - 100 / (1 + rs)


def get_technical_factors(code: str, trade_date: str, start_date: str, end_date: str, adjust: str) -> list[TechnicalFactorItem]:
    quote_items = get_stock_quotes([code], "1d", trade_date, start_date, end_date, "", "", None, adjust)
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


def _fetch_money_flow_frame(code: str, start_value: str, end_value: str, view: str) -> pd.DataFrame:
    pro = get_ts_pro()
    if pro is None:
        return pd.DataFrame()
    try:
        df = call_tushare_api("moneyflow", pro.moneyflow, ts_code=stock_code_to_ts(code), start_date=start_value, end_date=end_value)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["code"] = normalize_stock_code(code)
    work["view"] = view
    for column_name in ("buy_sm_amount", "buy_md_amount", "buy_lg_amount", "buy_elg_amount"):
        if column_name not in work.columns:
            work[column_name] = 0
    work["main_inflow"] = _amount_wan_to_yuan(work["buy_lg_amount"].fillna(0) + work["buy_elg_amount"].fillna(0))
    work["main_outflow"] = _amount_wan_to_yuan(work["sell_lg_amount"].fillna(0) + work["sell_elg_amount"].fillna(0))
    work["net_inflow"] = _amount_wan_to_yuan(work["net_mf_amount"])
    work["active_buy_amount"] = _amount_wan_to_yuan(sum(work[column_name].fillna(0) for column_name in ("buy_sm_amount", "buy_md_amount", "buy_lg_amount", "buy_elg_amount")))
    return work[["code", "trade_date", "view", "main_inflow", "main_outflow", "net_inflow", "active_buy_amount"]]


def _money_flow_frame_from_raw(frame: pd.DataFrame, view: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    work = frame.copy()
    code_column = "ts_code" if "ts_code" in work.columns else "code"
    if code_column not in work.columns or "trade_date" not in work.columns:
        return pd.DataFrame()
    work["code"] = work[code_column].map(normalize_stock_code)
    work["trade_date"] = work["trade_date"].map(format_date_value)
    work["view"] = view
    for column_name in ("buy_sm_amount", "buy_md_amount", "buy_lg_amount", "buy_elg_amount", "sell_lg_amount", "sell_elg_amount", "net_mf_amount"):
        if column_name not in work.columns:
            work[column_name] = 0
    work["main_inflow"] = _amount_wan_to_yuan(work["buy_lg_amount"].fillna(0) + work["buy_elg_amount"].fillna(0))
    work["main_outflow"] = _amount_wan_to_yuan(work["sell_lg_amount"].fillna(0) + work["sell_elg_amount"].fillna(0))
    work["net_inflow"] = _amount_wan_to_yuan(work["net_mf_amount"])
    work["active_buy_amount"] = _amount_wan_to_yuan(sum(work[column_name].fillna(0) for column_name in ("buy_sm_amount", "buy_md_amount", "buy_lg_amount", "buy_elg_amount")))
    return work[["code", "trade_date", "view", "main_inflow", "main_outflow", "net_inflow", "active_buy_amount"]]


def _fetch_money_flow_daily_frame(trade_date: str, view: str) -> pd.DataFrame:
    pro = get_ts_pro()
    actual_trade_date = format_date_value(trade_date)
    if pro is None or actual_trade_date == "":
        return pd.DataFrame()
    try:
        frame = call_tushare_api("moneyflow", pro.moneyflow, trade_date=actual_trade_date.replace("-", ""))
    except Exception:
        return pd.DataFrame()
    return _money_flow_frame_from_raw(frame, view)


def _money_flow_items_from_frame(frame: pd.DataFrame) -> list[StockMoneyFlowItem]:
    items: list[StockMoneyFlowItem] = []
    for _, row in frame.sort_values(["code", "trade_date"]).iterrows():
        items.append(
            StockMoneyFlowItem(
                code=str(row["code"]),
                trade_date=str(row["trade_date"]),
                view=str(row["view"]),
                main_inflow=float(row["main_inflow"]) if pd.notna(row["main_inflow"]) else None,
                main_outflow=float(row["main_outflow"]) if pd.notna(row["main_outflow"]) else None,
                net_inflow=float(row["net_inflow"]) if pd.notna(row["net_inflow"]) else None,
                active_buy_amount=float(row["active_buy_amount"]) if "active_buy_amount" in row and pd.notna(row["active_buy_amount"]) else None,
            )
        )
    return items


def get_stock_money_flow(code: str, trade_date: str, start_date: str, end_date: str, view: str) -> list[StockMoneyFlowItem]:
    actual_start = trade_date or start_date
    actual_end = trade_date or end_date
    if not actual_start and not actual_end:
        actual_end = datetime.now().strftime(DATE_FORMAT)
        actual_start = (datetime.now() - timedelta(days=30)).strftime(DATE_FORMAT)
    elif not actual_start:
        actual_start = actual_end
    elif not actual_end:
        actual_end = actual_start
    cache_path = build_cache_path("tushare", ["stocks", "indicators", "money-flow"], {"code": normalize_stock_code(code), "view": view})
    cache_df = read_cache_frame(cache_path)
    missing_ranges = plan_missing_ranges(cache_df, "trade_date", actual_start, actual_end, "day")
    fetched_frames: list[pd.DataFrame] = []
    for missing_start, missing_end in missing_ranges:
        fetched_df = _fetch_money_flow_frame(code, missing_start, missing_end, view)
        if not fetched_df.empty:
            fetched_frames.append(fetched_df)
    if cache_df.empty and not fetched_frames:
        fetched_df = _fetch_money_flow_frame(code, actual_start, actual_end, view)
        if not fetched_df.empty:
            fetched_frames.append(fetched_df)
    if fetched_frames:
        merged_cache = merge_cache_frame(cache_df, pd.concat(fetched_frames, ignore_index=True), ["code", "trade_date", "view"], ["trade_date"])
        write_cache_frame(cache_path, merged_cache)
        cache_df = merged_cache
    filtered_df = filter_frame_by_date_range(cache_df, "trade_date", actual_start, actual_end)
    if filtered_df.empty or "trade_date" not in filtered_df.columns:
        return []
    return _money_flow_items_from_frame(filtered_df)


def get_stock_money_flow_batch(codes: str, trade_date: str, view: str) -> list[StockMoneyFlowItem]:
    actual_codes = [normalize_stock_code(item) for item in codes.split(",") if normalize_stock_code(item)]
    actual_trade_date = format_date_value(trade_date)
    if actual_codes == [] or actual_trade_date == "":
        return []
    unique_codes = list(dict.fromkeys(actual_codes))
    cache_path = build_cache_path("tushare", ["stocks", "indicators", "money-flow-daily"], {"trade_date": actual_trade_date, "view": view})
    cache_df = read_cache_frame(cache_path)
    if cache_df.empty:
        fetched_df = _fetch_money_flow_daily_frame(actual_trade_date, view)
        if not fetched_df.empty:
            write_cache_frame(cache_path, fetched_df)
            cache_df = fetched_df
    if not cache_df.empty and {"code", "trade_date", "view"}.issubset(set(cache_df.columns)):
        filtered_df = cache_df[(cache_df["code"].isin(unique_codes)) & (cache_df["trade_date"] == actual_trade_date) & (cache_df["view"] == view)]
        if not filtered_df.empty:
            return _money_flow_items_from_frame(filtered_df)
    items: list[StockMoneyFlowItem] = []
    for code in unique_codes:
        items.extend(get_stock_money_flow(code, actual_trade_date, "", "", view))
    return sorted(items, key=lambda item: (item.code, item.trade_date, item.view))


def get_stock_money_flow_snapshot(trade_date: str, view: str) -> list[StockMoneyFlowItem]:
    actual_trade_date = format_date_value(trade_date)
    if actual_trade_date == "":
        return []
    cache_path = build_cache_path(
        "tushare",
        ["stocks", "indicators", "money-flow-snapshot"],
        {"trade_date": actual_trade_date, "view": view},
    )
    cache_df = read_cache_frame(cache_path)
    if _money_flow_snapshot_cache_needs_fetch(cache_df, actual_trade_date):
        cache_df = _fetch_money_flow_daily_frame(actual_trade_date, view)
        if not cache_df.empty:
            write_cache_frame(cache_path, cache_df)
    if cache_df.empty:
        return []
    return _money_flow_items_from_frame(cache_df)


def _money_flow_snapshot_cache_needs_fetch(frame: pd.DataFrame, trade_date: str) -> bool:
    required_columns = {"code", "trade_date", "view", "active_buy_amount"}
    if frame.empty or not required_columns.issubset(frame.columns):
        return True
    snapshot = frame[frame["trade_date"].map(format_date_value) == trade_date]
    return snapshot.empty or snapshot["active_buy_amount"].isna().all()


def _margin_items_from_frame(frame: pd.DataFrame) -> list[StockMarginItem]:
    if frame.empty or "ts_code" not in frame.columns or "trade_date" not in frame.columns:
        return []
    items: list[StockMarginItem] = []
    for _, row in frame.iterrows():
        code = normalize_stock_code(str(row["ts_code"]))
        if code == "":
            continue
        items.append(
            StockMarginItem(
                code=code,
                trade_date=format_date_value(row["trade_date"]),
                financing_balance=float(row["rzye"]) if "rzye" in row and pd.notna(row["rzye"]) else None,
                financing_buy=float(row["rzmre"]) if "rzmre" in row and pd.notna(row["rzmre"]) else None,
                financing_repay=float(row["rzche"]) if "rzche" in row and pd.notna(row["rzche"]) else None,
                securities_lending_balance=float(row["rqye"]) if "rqye" in row and pd.notna(row["rqye"]) else None,
                securities_lending_volume=float(row["rqyl"]) if "rqyl" in row and pd.notna(row["rqyl"]) else None,
                securities_lending_repay=float(row["rqchl"]) if "rqchl" in row and pd.notna(row["rqchl"]) else None,
                securities_lending_sell=float(row["rqmcl"]) if "rqmcl" in row and pd.notna(row["rqmcl"]) else None,
                total_margin_balance=float(row["rzrqye"]) if "rzrqye" in row and pd.notna(row["rzrqye"]) else None,
            )
        )
    return sorted(items, key=lambda item: (item.code, item.trade_date))


def get_stock_margin_snapshot(trade_date: str) -> list[StockMarginItem]:
    actual_trade_date = format_date_value(trade_date)
    if actual_trade_date == "":
        return []
    cache_path = build_cache_path("tushare", ["stocks", "indicators", "margin-snapshot"], {"trade_date": actual_trade_date})
    cache_df = read_cache_frame(cache_path)
    if cache_df.empty:
        pro = get_ts_pro()
        if pro is None:
            return []
        try:
            cache_df = call_tushare_api("margin_detail", pro.margin_detail, trade_date=actual_trade_date.replace("-", ""))
        except Exception:
            return []
        if cache_df is None or cache_df.empty:
            return []
        write_cache_frame(cache_path, cache_df)
    return _margin_items_from_frame(cache_df)


def _express_items_from_snapshot_frame(frame: pd.DataFrame) -> list[ExpressItem]:
    if frame.empty or "ts_code" not in frame.columns:
        return []
    items: list[ExpressItem] = []
    for _, row in frame.iterrows():
        code = normalize_stock_code(str(row["ts_code"]))
        report_period = format_date_value(row["end_date"]) if "end_date" in row else ""
        announce_date = format_date_value(row["ann_date"]) if "ann_date" in row else ""
        if code == "" or report_period == "" or announce_date == "":
            continue
        items.append(
            ExpressItem(
                code=code,
                report_period=report_period,
                announce_date=announce_date,
                revenue=float(row["revenue"]) if "revenue" in row and pd.notna(row["revenue"]) else None,
                operating_profit=float(row["operate_profit"]) if "operate_profit" in row and pd.notna(row["operate_profit"]) else None,
                total_profit=float(row["total_profit"]) if "total_profit" in row and pd.notna(row["total_profit"]) else None,
                net_profit=float(row["n_income"]) if "n_income" in row and pd.notna(row["n_income"]) else None,
                eps=float(row["diluted_eps"]) if "diluted_eps" in row and pd.notna(row["diluted_eps"]) else None,
                roe=float(row["diluted_roe"]) if "diluted_roe" in row and pd.notna(row["diluted_roe"]) else None,
            )
        )
    return sorted(items, key=lambda item: (item.announce_date, item.code, item.report_period))


def get_stock_express_snapshot(announce_date: str) -> list[ExpressItem]:
    actual_announce_date = format_date_value(announce_date)
    if actual_announce_date == "":
        return []
    cache_path = build_cache_path("tushare", ["stocks", "finance", "express-snapshot"], {"announce_date": actual_announce_date})
    cache_df = read_cache_frame(cache_path)
    if cache_df.empty:
        pro = get_ts_pro()
        if pro is None:
            return []
        try:
            cache_df = call_tushare_api("express", pro.express, start_date=actual_announce_date.replace("-", ""), end_date=actual_announce_date.replace("-", ""))
        except Exception:
            return []
        if cache_df is None or cache_df.empty:
            return []
        write_cache_frame(cache_path, cache_df)
    return _express_items_from_snapshot_frame(cache_df)


def _forecast_items_from_snapshot_frame(frame: pd.DataFrame) -> list[ForecastItem]:
    if frame.empty or "ts_code" not in frame.columns:
        return []
    items: list[ForecastItem] = []
    for _, row in frame.iterrows():
        code = normalize_stock_code(str(row["ts_code"]))
        report_period = format_date_value(row["end_date"]) if "end_date" in row else ""
        announce_date = format_date_value(row["ann_date"]) if "ann_date" in row else ""
        if code == "" or report_period == "" or announce_date == "":
            continue
        items.append(
            ForecastItem(
                code=code,
                report_period=report_period,
                announce_date=announce_date,
                forecast_type=str(row["type"]) if "type" in row and pd.notna(row["type"]) else "",
                forecast_summary=str(row["summary"]) if "summary" in row and pd.notna(row["summary"]) else "",
                net_profit_min=float(row["net_profit_min"]) if "net_profit_min" in row and pd.notna(row["net_profit_min"]) else None,
                net_profit_max=float(row["net_profit_max"]) if "net_profit_max" in row and pd.notna(row["net_profit_max"]) else None,
                pct_chg_min=float(row["p_change_min"]) if "p_change_min" in row and pd.notna(row["p_change_min"]) else None,
                pct_chg_max=float(row["p_change_max"]) if "p_change_max" in row and pd.notna(row["p_change_max"]) else None,
            )
        )
    return sorted(items, key=lambda item: (item.announce_date, item.code, item.report_period, item.forecast_type))


def get_stock_forecast_snapshot(announce_date: str) -> list[ForecastItem]:
    actual_announce_date = format_date_value(announce_date)
    if actual_announce_date == "":
        return []
    cache_path = build_cache_path("tushare", ["stocks", "finance", "forecast-snapshot"], {"announce_date": actual_announce_date})
    cache_df = read_cache_frame(cache_path)
    if cache_df.empty:
        pro = get_ts_pro()
        if pro is None:
            return []
        try:
            cache_df = call_tushare_api("forecast", pro.forecast, ann_date=actual_announce_date.replace("-", ""))
        except Exception:
            return []
        if cache_df is None or cache_df.empty:
            return []
        write_cache_frame(cache_path, cache_df)
    return _forecast_items_from_snapshot_frame(cache_df)


def _first_existing_column(frame: pd.DataFrame, column_names: tuple[str, ...]) -> object:
    for column_name in column_names:
        if column_name in frame.columns:
            return frame[column_name]
    return None


def _amount_wan_to_yuan(value: object) -> object:
    if value is None:
        return None
    return pd.to_numeric(value, errors="coerce") * 10000


def board_code_to_ts(board_code: str) -> str:
    text = board_code.strip().upper()
    if not text:
        return ""
    if "." in text:
        return text
    return f"{text}.TI"


def _fetch_board_money_flow_frame(board_code: str, start_value: str, end_value: str, scope: str) -> pd.DataFrame:
    pro = get_ts_pro()
    if pro is None:
        return pd.DataFrame()
    fetch_name = "moneyflow_ind_ths" if scope == "industry" else "moneyflow_cnt_ths"
    fetcher = getattr(pro, fetch_name, None)
    if fetcher is None:
        return pd.DataFrame()
    try:
        df = call_tushare_api(fetch_name, fetcher, ts_code=board_code_to_ts(board_code), start_date=start_value, end_date=end_value)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    code_column = "ts_code" if "ts_code" in work.columns else "code"
    work["board_code"] = work[code_column].astype(str).str.split(".").str[0]
    work["scope"] = scope
    work["inflow"] = _amount_wan_to_yuan(_first_existing_column(work, ("net_buy_amount", "buy_amount", "buy_elg_amount")))
    work["outflow"] = _amount_wan_to_yuan(_first_existing_column(work, ("net_sell_amount", "sell_amount", "sell_elg_amount")))
    work["net_inflow"] = _amount_wan_to_yuan(_first_existing_column(work, ("net_amount", "net_buy", "net_mf_amount", "net_inflow")))
    return work[["board_code", "trade_date", "scope", "inflow", "outflow", "net_inflow"]]


def get_board_money_flow(board_code: str, trade_date: str, start_date: str, end_date: str, scope: str) -> list[BoardMoneyFlowItem]:
    actual_start = trade_date or start_date
    actual_end = trade_date or end_date
    if not actual_start and not actual_end:
        actual_end = datetime.now().strftime(DATE_FORMAT)
        actual_start = (datetime.now() - timedelta(days=30)).strftime(DATE_FORMAT)
    elif not actual_start:
        actual_start = actual_end
    elif not actual_end:
        actual_end = actual_start
    cache_path = build_cache_path("tushare", ["boards", "indicators", "money-flow"], {"board_code": board_code, "scope": scope})
    cache_df = read_cache_frame(cache_path)
    missing_ranges = plan_missing_ranges(cache_df, "trade_date", actual_start, actual_end, "day")
    fetched_frames: list[pd.DataFrame] = []
    for missing_start, missing_end in missing_ranges:
        fetched_df = _fetch_board_money_flow_frame(board_code, missing_start, missing_end, scope)
        if not fetched_df.empty:
            fetched_frames.append(fetched_df)
    if cache_df.empty and not fetched_frames:
        fetched_df = _fetch_board_money_flow_frame(board_code, actual_start, actual_end, scope)
        if not fetched_df.empty:
            fetched_frames.append(fetched_df)
    if fetched_frames:
        merged_cache = merge_cache_frame(cache_df, pd.concat(fetched_frames, ignore_index=True), ["board_code", "trade_date", "scope"], ["trade_date"])
        write_cache_frame(cache_path, merged_cache)
        cache_df = merged_cache
    filtered_df = filter_frame_by_date_range(cache_df, "trade_date", actual_start, actual_end)
    if filtered_df.empty or "trade_date" not in filtered_df.columns:
        return []
    items: list[BoardMoneyFlowItem] = []
    for _, row in filtered_df.sort_values("trade_date").iterrows():
        items.append(
            BoardMoneyFlowItem(
                board_code=str(row["board_code"]),
                trade_date=str(row["trade_date"]),
                scope=str(row["scope"]),
                inflow=float(row["inflow"]) if pd.notna(row["inflow"]) else None,
                outflow=float(row["outflow"]) if pd.notna(row["outflow"]) else None,
                net_inflow=float(row["net_inflow"]) if pd.notna(row["net_inflow"]) else None,
            )
        )
    return items


def _fetch_market_capital_flow_frame(start_value: str, end_value: str) -> pd.DataFrame:
    pro = get_ts_pro()
    if pro is None:
        return pd.DataFrame()
    fetcher = getattr(pro, "moneyflow_mkt_dc", None)
    if fetcher is None:
        return pd.DataFrame()
    try:
        df = call_tushare_api("moneyflow_mkt_dc", fetcher, start_date=start_value, end_date=end_value)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["market"] = "all"
    work["main_inflow"] = None
    work["main_outflow"] = None
    if {"buy_elg_amount", "buy_lg_amount", "sell_elg_amount", "sell_lg_amount"}.issubset(work.columns):
        work["main_inflow"] = work["buy_elg_amount"].fillna(0) + work["buy_lg_amount"].fillna(0)
        work["main_outflow"] = work["sell_elg_amount"].fillna(0) + work["sell_lg_amount"].fillna(0)
    work["net_inflow"] = work["net_amount"] if "net_amount" in work.columns else None
    if work[["main_inflow", "main_outflow"]].isna().any().any():
        flow_frames = [_fetch_money_flow_daily_frame(day, "main") for day in sorted(work["trade_date"].astype(str).unique())]
        flow_frames = [frame for frame in flow_frames if not frame.empty]
        if flow_frames:
            daily_flow = pd.concat(flow_frames, ignore_index=True)
            if {"trade_date", "main_inflow", "main_outflow", "net_inflow"}.issubset(daily_flow.columns):
                aggregate_flow = daily_flow.groupby("trade_date", as_index=False)[["main_inflow", "main_outflow", "net_inflow"]].sum()
                work = work.drop(columns=["main_inflow", "main_outflow"]).merge(aggregate_flow, on="trade_date", how="left", suffixes=("", "_stock"))
                if "net_inflow_stock" in work.columns:
                    work["net_inflow"] = work["net_inflow"].fillna(work["net_inflow_stock"])
                    work = work.drop(columns=["net_inflow_stock"])
    return work[["trade_date", "market", "main_inflow", "main_outflow", "net_inflow"]]


def _market_capital_flow_cache_needs_fetch(cache_df: pd.DataFrame, start_value: str, end_value: str) -> bool:
    filtered_df = filter_frame_by_date_range(cache_df, "trade_date", start_value, end_value)
    required_columns = {"trade_date", "market", "main_inflow", "main_outflow", "net_inflow"}
    if filtered_df.empty or not required_columns.issubset(filtered_df.columns):
        return True
    return bool(filtered_df[["main_inflow", "main_outflow", "net_inflow"]].isna().any().any())


def get_market_capital_flow(trade_date: str, start_date: str, end_date: str) -> list[MarketCapitalFlowItem]:
    actual_start = trade_date or start_date
    actual_end = trade_date or end_date
    if not actual_start and not actual_end:
        actual_end = datetime.now().strftime(DATE_FORMAT)
        actual_start = (datetime.now() - timedelta(days=30)).strftime(DATE_FORMAT)
    elif not actual_start:
        actual_start = actual_end
    elif not actual_end:
        actual_end = actual_start
    cache_path = build_cache_path("tushare", ["markets", "indicators", "main-capital-flow"], {"market": "all"})
    cache_df = read_cache_frame(cache_path)
    missing_ranges = plan_missing_ranges(cache_df, "trade_date", actual_start, actual_end, "day")
    fetched_frames: list[pd.DataFrame] = []
    for missing_start, missing_end in missing_ranges:
        fetched_df = _fetch_market_capital_flow_frame(missing_start, missing_end)
        if not fetched_df.empty:
            fetched_frames.append(fetched_df)
    if cache_df.empty and not fetched_frames:
        fetched_df = _fetch_market_capital_flow_frame(actual_start, actual_end)
        if not fetched_df.empty:
            fetched_frames.append(fetched_df)
    if not fetched_frames and _market_capital_flow_cache_needs_fetch(cache_df, actual_start, actual_end):
        fetched_df = _fetch_market_capital_flow_frame(actual_start, actual_end)
        if not fetched_df.empty:
            fetched_frames.append(fetched_df)
    if fetched_frames:
        merged_cache = merge_cache_frame(cache_df, pd.concat(fetched_frames, ignore_index=True), ["trade_date", "market"], ["trade_date"])
        write_cache_frame(cache_path, merged_cache)
        cache_df = merged_cache
    filtered_df = filter_frame_by_date_range(cache_df, "trade_date", actual_start, actual_end)
    if filtered_df.empty or "trade_date" not in filtered_df.columns:
        return []
    filtered_df = filtered_df.drop_duplicates(subset=["trade_date", "market"], keep="last")
    items: list[MarketCapitalFlowItem] = []
    for _, row in filtered_df.sort_values("trade_date").iterrows():
        items.append(
            MarketCapitalFlowItem(
                trade_date=str(row["trade_date"]),
                market=str(row["market"]),
                main_inflow=float(row["main_inflow"]) if pd.notna(row["main_inflow"]) else None,
                main_outflow=float(row["main_outflow"]) if pd.notna(row["main_outflow"]) else None,
                net_inflow=float(row["net_inflow"]) if pd.notna(row["net_inflow"]) else None,
            )
        )
    return items


def trade_calendar_fetch_exchange(exchange: str) -> str:
    if exchange == "BSE":
        return "SSE"
    if exchange == "HKEX":
        return ""
    return exchange


def _fetch_trading_calendar_frame(exchange: str, start_value: str, end_value: str) -> pd.DataFrame:
    pro = get_ts_pro()
    if pro is None:
        return pd.DataFrame()
    fetch_exchange = trade_calendar_fetch_exchange(exchange)
    if not fetch_exchange:
        return pd.DataFrame()
    try:
        df = call_tushare_api("trade_cal", pro.trade_cal, exchange=fetch_exchange, start_date=start_value, end_date=end_value)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["exchange"] = exchange
    work["trade_date"] = work["cal_date"]
    return work[["exchange", "trade_date", "is_open"]]


def get_trading_calendar(exchange: str, start_date: str, end_date: str, is_open: bool | None) -> list[TradingCalendarItem]:
    actual_end = end_date or datetime.now().strftime(DATE_FORMAT)
    actual_start = start_date or (datetime.now() - timedelta(days=365)).strftime(DATE_FORMAT)
    cache_path = build_cache_path("tushare", ["markets", "calendar", "trading"], {"exchange": exchange})
    cache_df = read_cache_frame(cache_path)
    missing_ranges = plan_missing_ranges(cache_df, "trade_date", actual_start, actual_end, "day")
    fetched_frames: list[pd.DataFrame] = []
    for missing_start, missing_end in missing_ranges:
        fetched_df = _fetch_trading_calendar_frame(exchange, missing_start, missing_end)
        if not fetched_df.empty:
            fetched_frames.append(fetched_df)
    if cache_df.empty and not fetched_frames:
        fetched_df = _fetch_trading_calendar_frame(exchange, actual_start, actual_end)
        if not fetched_df.empty:
            fetched_frames.append(fetched_df)
    if fetched_frames:
        merged_cache = merge_cache_frame(cache_df, pd.concat(fetched_frames, ignore_index=True), ["exchange", "trade_date"], ["trade_date"])
        write_cache_frame(cache_path, merged_cache)
        cache_df = merged_cache
    filtered_df = filter_frame_by_date_range(cache_df, "trade_date", actual_start, actual_end)
    if is_open is not None:
        filtered_df = filtered_df[filtered_df["is_open"].astype(str) == ("1" if is_open else "0")]
    items: list[TradingCalendarItem] = []
    for _, row in filtered_df.sort_values("trade_date").iterrows():
        items.append(
            TradingCalendarItem(
                exchange=str(row["exchange"]),
                trade_date=str(row["trade_date"]),
                is_open=str(row["is_open"]) == "1",
            )
        )
    return items


def _fetch_financial_frame(code: str, start_value: str, end_value: str, report_type: str) -> pd.DataFrame:
    pro = get_ts_pro()
    if pro is None:
        return pd.DataFrame()
    if report_type == "income_statement":
        fetch_name = "income"
        fields = "ts_code,ann_date,end_date,total_revenue,operate_profit,total_profit,n_income"
    elif report_type == "balance_sheet":
        fetch_name = "balancesheet"
        fields = "ts_code,ann_date,end_date,total_assets,total_liab,total_hldr_eqy_exc_min_int"
    else:
        fetch_name = "cashflow"
        fields = "ts_code,ann_date,end_date"
    fetcher = getattr(pro, fetch_name, None)
    if fetcher is None:
        return pd.DataFrame()
    try:
        df = call_tushare_api(fetch_name, fetcher, ts_code=stock_code_to_ts(code), start_date=start_value, end_date=end_value, fields=fields)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["code"] = normalize_stock_code(code)
    work["report_period"] = work["end_date"]
    work["report_type"] = report_type
    work["announce_date"] = work["ann_date"]
    work["revenue"] = work["total_revenue"] if "total_revenue" in work.columns else None
    work["operating_profit"] = work["operate_profit"] if "operate_profit" in work.columns else None
    work["total_profit"] = work["total_profit"] if "total_profit" in work.columns else None
    work["net_profit"] = work["n_income"] if "n_income" in work.columns else None
    work["total_assets2"] = work["total_assets"] if "total_assets" in work.columns else None
    work["total_liabilities2"] = work["total_liab"] if "total_liab" in work.columns else None
    work["equity2"] = work["total_hldr_eqy_exc_min_int"] if "total_hldr_eqy_exc_min_int" in work.columns else None
    return work[["code", "report_period", "report_type", "announce_date", "revenue", "operating_profit", "total_profit", "net_profit", "total_assets2", "total_liabilities2", "equity2"]]


def get_stock_financial_statements(
    codes: list[str],
    report_period: str,
    start_period: str,
    end_period: str,
    report_type: str,
) -> list[StockFinancialStatementItem]:
    start_value = start_period or report_period
    end_value = end_period or report_period
    if not start_value and not end_value:
        end_value = datetime.now().strftime("%Y1231")
        start_value = f"{datetime.now().year - 2}0101"
    elif not start_value:
        start_value = end_value
    elif not end_value:
        end_value = start_value
    items: list[StockFinancialStatementItem] = []
    for code in codes:
        cache_path = build_cache_path("tushare", ["stocks", "finance", "statements"], {"code": normalize_stock_code(code), "report_type": report_type})
        cache_df = read_cache_frame(cache_path)
        missing_ranges = plan_missing_ranges(cache_df, "report_period", start_value, end_value, "quarter")
        fetched_frames: list[pd.DataFrame] = []
        for missing_start, missing_end in missing_ranges:
            fetched_df = _fetch_financial_frame(code, missing_start, missing_end, report_type)
            if not fetched_df.empty:
                fetched_frames.append(fetched_df)
        if cache_df.empty and not fetched_frames:
            fetched_df = _fetch_financial_frame(code, start_value, end_value, report_type)
            if not fetched_df.empty:
                fetched_frames.append(fetched_df)
        if fetched_frames:
            merged_cache = merge_cache_frame(cache_df, pd.concat(fetched_frames, ignore_index=True), ["code", "report_period", "report_type", "announce_date"], ["report_period", "announce_date"])
            write_cache_frame(cache_path, merged_cache)
            cache_df = merged_cache
        filtered_df = filter_frame_by_date_range(cache_df, "report_period", start_value, end_value)
        required_columns = {"code", "report_period", "report_type", "announce_date"}
        if filtered_df.empty or not required_columns.issubset(set(filtered_df.columns)):
            continue
        for _, row in filtered_df.sort_values(["report_period", "announce_date"]).iterrows():
            items.append(
                StockFinancialStatementItem(
                    code=str(row["code"]),
                    report_period=str(row["report_period"]),
                    report_type=str(row["report_type"]),
                    announce_date=str(row["announce_date"]),
                    revenue=float(row["revenue"]) if pd.notna(row["revenue"]) else None,
                    operating_profit=float(row["operating_profit"]) if pd.notna(row["operating_profit"]) else None,
                    total_profit=float(row["total_profit"]) if pd.notna(row["total_profit"]) else None,
                    net_profit=float(row["net_profit"]) if pd.notna(row["net_profit"]) else None,
                    total_assets=float(row["total_assets2"]) if pd.notna(row["total_assets2"]) else None,
                    total_liabilities=float(row["total_liabilities2"]) if pd.notna(row["total_liabilities2"]) else None,
                    equity=float(row["equity2"]) if pd.notna(row["equity2"]) else None,
                )
            )
    return items

def get_concept_catalog(category: str, market: str, status: str, limit: int, offset: int):
    return get_board_catalog(category, market, status, limit, offset)

def get_concept_profile(concept_id: str):
    return get_board_profile(concept_id)

def get_concept_categories(parent_code: str, level: int | None):
    return get_board_categories(parent_code, level)

def get_concept_members(concept_id: str, trade_date: str):
    return get_board_members(concept_id, trade_date)

def get_concept_member_history(concept_id: str, start_date: str, end_date: str):
    return get_board_member_history(concept_id, start_date, end_date)

def get_concept_quotes(concept_ids: list[str], freq: str, trade_date: str, start_date: str, end_date: str, start_time: str, end_time: str, count: int | None):
    return get_board_quotes(concept_ids, freq, trade_date, start_date, end_date, start_time, end_time, count)

def get_concept_money_flow(concept_id: str, trade_date: str, start_date: str, end_date: str, scope: str):
    return get_board_money_flow(concept_id, trade_date, start_date, end_date, scope)

def get_concept_daily_money_flow_snapshot(trade_date: str, scope: str, limit: int, offset: int):
    return get_board_daily_money_flow_snapshot(trade_date, scope, limit, offset)

