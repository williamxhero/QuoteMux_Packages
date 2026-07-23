from __future__ import annotations

import pandas as pd

from platform_models import BoardMemberItem, BoardMoneyFlowItem, BoardQuoteItem, HLSignalItem, NineTurnItem, ShareholderChangeItem, ShareholderCountItem, StockQuoteItem, TechnicalFactorItem, TradingCalendarItem
from quotemux.infra.common import normalize_stock_code


BOARD_STOCK_INDUSTRY_MAP = {
    "BK1326": "半导体",
    "BK1327": "元器件",
    "BK1329": "半导体",
    "BK1332": "化工原料",
}


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


def _normalized_calendar_items(items: list[TradingCalendarItem]) -> list[TradingCalendarItem]:
    from quotemux.infra.common import format_date_value

    by_key: dict[tuple[str, str], TradingCalendarItem] = {}
    for item in items:
        normalized_date = format_date_value(item.trade_date)
        key = (item.exchange, normalized_date)
        if key not in by_key:
            by_key[key] = item.model_copy(update={"trade_date": normalized_date})
    return sorted(by_key.values(), key=lambda item: (item.exchange, item.trade_date))


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


def _local_stock_bar_1m_items(code: str, start_date: str, end_date: str) -> list[StockQuoteItem]:
    from quotemux.infra.common import format_datetime_value
    from quotemux.infra.db.market_reads import load_stock_intraday_frame

    if code == "" or start_date == "" or end_date == "":
        return []
    frame = load_stock_intraday_frame([code], f"{start_date} 00:00:00", f"{end_date} 23:59:59", "1m")
    if frame.empty:
        return []
    items: list[StockQuoteItem] = []
    for _, row in frame.iterrows():
        items.append(
            StockQuoteItem(
                code=str(row["code"]),
                trade_time=format_datetime_value(row["trade_time"], "1m"),
                freq="1m",
                open=float(row["open"]) if pd.notna(row["open"]) else None,
                high=float(row["high"]) if pd.notna(row["high"]) else None,
                low=float(row["low"]) if pd.notna(row["low"]) else None,
                close=float(row["close"]) if pd.notna(row["close"]) else None,
                volume=float(row["volume"]) if pd.notna(row["volume"]) else None,
                amount=float(row["amount"]) if pd.notna(row["amount"]) else None,
            )
        )
    return items


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


def get_nine_turn(code: str, freq: str, trade_date: str, start_date: str, end_date: str) -> list[NineTurnItem]:
    from quotemux.infra.common import format_date_value

    normalized = normalize_stock_code(code)
    actual_freq = freq or "daily"
    quote_freq = "1d" if actual_freq in {"D", "daily"} else actual_freq
    if normalized == "":
        return []
    actual_trade_date = format_date_value(trade_date or end_date or start_date)
    actual_start_date = format_date_value(start_date)
    actual_end_date = format_date_value(end_date)
    request_start = actual_start_date or format_date_value(trade_date)
    request_end = actual_end_date or format_date_value(trade_date)
    seed_days = -60 if quote_freq == "1d" else 0
    seed_start = _offset_date_text(request_start, seed_days) if request_start and seed_days != 0 else request_start
    quote_items = _local_stock_bar_1m_items(normalized, seed_start or request_start, request_end) if quote_freq == "1m" else []
    if quote_items == []:
        quote_items = _quote_mux().stocks.get_quotes(_stock_quotes_request(normalized, quote_freq, "", seed_start or request_start, request_end, "none"))
    if quote_items == []:
        return []
    frame = pd.DataFrame([item.model_dump() for item in quote_items])
    frame["trade_key"] = frame["trade_time"].astype(str)
    frame["trade_date"] = frame["trade_key"].str[:10]
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["close"]).sort_values("trade_key").reset_index(drop=True)
    if frame.empty:
        return []
    up_count = 0
    down_count = 0
    rows: list[NineTurnItem] = []
    for index, row in frame.iterrows():
        if index < 4:
            setup_index = 0
            signal = ""
        else:
            previous_close = frame.iloc[index - 4]["close"]
            current_close = row["close"]
            if current_close > previous_close:
                up_count += 1
                down_count = 0
                setup_index = up_count
                signal = "nine_up" if up_count >= 9 else ""
            elif current_close < previous_close:
                down_count += 1
                up_count = 0
                setup_index = down_count
                signal = "nine_down" if down_count >= 9 else ""
            else:
                up_count = 0
                down_count = 0
                setup_index = 0
                signal = ""
        trade_key = str(row["trade_key"])
        trade_day = format_date_value(trade_key[:10])
        if actual_trade_date and trade_day != actual_trade_date:
            continue
        if actual_start_date and trade_day < actual_start_date:
            continue
        if actual_end_date and trade_day > actual_end_date:
            continue
        rows.append(NineTurnItem(code=normalized, trade_time=trade_day if quote_freq == "1d" else trade_key, freq=actual_freq, setup_index=int(setup_index), countdown_index=None, signal=signal))
    return rows


def get_previous_trading_days(exchange: str, trade_date: str, n: int) -> list[TradingCalendarItem]:
    from quotemux.infra.common import format_date_value, parse_date_text

    actual_trade_date = format_date_value(trade_date)
    if n <= 0:
        return []
    start_date = _offset_date_text(actual_trade_date, -max(n * 8, 30))
    items = _normalized_calendar_items(_quote_mux().markets.get_trading_calendar(_trading_calendar_request(exchange, start_date, actual_trade_date, None)))
    open_items = [item for item in items if item.is_open and item.trade_date < actual_trade_date]
    start_day = parse_date_text(start_date)
    trade_day = parse_date_text(actual_trade_date)
    if len(open_items) < n and start_day is not None and trade_day is not None:
        yearly_items = get_yearly_trading_calendar(exchange, start_day.year, trade_day.year)
        open_items = [item for item in yearly_items if item.is_open and start_date <= item.trade_date < actual_trade_date]
    return open_items[-n:]


def get_next_trading_days(exchange: str, trade_date: str, n: int) -> list[TradingCalendarItem]:
    from datetime import date

    from quotemux.infra.common import format_date_value, parse_date_text

    actual_trade_date = format_date_value(trade_date)
    if n <= 0:
        return []
    trade_day = parse_date_text(actual_trade_date)
    end_date = ""
    if trade_day is not None:
        try:
            next_year_day = trade_day.replace(year=trade_day.year + 1)
        except ValueError:
            next_year_day = date(trade_day.year + 1, 2, 28)
        end_date = next_year_day.strftime("%Y-%m-%d")
    items = _normalized_calendar_items(_quote_mux().markets.get_trading_calendar(_trading_calendar_request(exchange, actual_trade_date, end_date, None)))
    open_items = [item for item in items if item.is_open and item.trade_date > actual_trade_date]
    end_day = parse_date_text(end_date)
    if len(open_items) < n and trade_day is not None and end_day is not None:
        yearly_items = get_yearly_trading_calendar(exchange, trade_day.year, end_day.year)
        open_items = [item for item in yearly_items if item.is_open and actual_trade_date < item.trade_date <= end_date]
    return open_items[:n]


def get_yearly_trading_calendar(exchange: str, start_year: int, end_year: int) -> list[TradingCalendarItem]:
    return _normalized_calendar_items(_quote_mux().markets.get_trading_calendar(_trading_calendar_request(exchange, f"{start_year}-01-01", f"{end_year}-12-31", None)))


def get_board_members(board_code: str, trade_date: str) -> list[BoardMemberItem]:
    del trade_date
    normalized = str(board_code).upper()
    from quotemux.infra.db.client import query_dataframe

    frame = query_dataframe(
        """
        select
            stock.code,
            stock.name
        from ref.board board
        join ref.stock stock
          on stock.industry = board.name
        where board.board_code = %s
          and board.board_type = 'industry'
          and board.status = 'active'
          and (stock.delisted_date is null or stock.delisted_date >= current_date)
        order by stock.code
        """,
        (normalized,),
    )
    if frame.empty:
        return []
    return [
        BoardMemberItem(
            board_code=normalized,
            code=str(row["code"]).zfill(6),
            name=str(row["name"]),
            join_date="",
        )
        for _, row in frame.iterrows()
    ]


def _board_quote_date_window(trade_date: str, start_date: str, end_date: str) -> tuple[str, str]:
    from quotemux.infra.common import format_date_value

    actual_trade_date = format_date_value(trade_date)
    if actual_trade_date != "":
        return actual_trade_date, actual_trade_date
    actual_start = format_date_value(start_date)
    actual_end = format_date_value(end_date)
    if actual_start == "":
        actual_start = actual_end
    if actual_end == "":
        actual_end = actual_start
    return actual_start, actual_end


def _board_quote_frame(board_codes: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    from quotemux.infra.db.client import query_dataframe

    if board_codes == [] or start_date == "" or end_date == "":
        return pd.DataFrame()
    return query_dataframe(
        """
        with target_member_rows as (
            select
                membership.board_code,
                membership.stock_code as code,
                membership.weight,
                membership.valid_from,
                membership.valid_to
            from ref.board_stock_membership membership
            where membership.board_code = any(%s)
              and membership.valid_from <= %s::date
              and (membership.valid_to is null or membership.valid_to >= %s::date)
        ),
        fallback_member_dates as (
            select
                membership.board_code,
                max(membership.valid_from) as valid_from
            from ref.board_stock_membership membership
            where membership.board_code = any(%s)
              and membership.valid_from <= %s::date
            group by membership.board_code
        ),
        fallback_member_rows as (
            select
                membership.board_code,
                membership.stock_code as code,
                membership.weight,
                membership.valid_from,
                membership.valid_to
            from ref.board_stock_membership membership
            join fallback_member_dates latest
              on latest.board_code = membership.board_code
             and latest.valid_from = membership.valid_from
            where not exists (
                select 1
                from target_member_rows target
                where target.board_code = membership.board_code
            )
        ),
        member_rows as (
            select * from target_member_rows
            union all
            select * from fallback_member_rows
        ),
        daily_rows as (
            select
                member_rows.board_code,
                stock_rows.trade_date,
                stock_rows.code,
                stock_rows.close,
                stock_rows.volume,
                stock_rows.amount,
                lag(stock_rows.close) over (partition by member_rows.board_code, stock_rows.code order by stock_rows.trade_date) as pre_close
            from member_rows
            join fact.stock_daily_1d stock_rows on stock_rows.code = member_rows.code
            where stock_rows.trade_date between %s and %s
              and stock_rows.is_suspended = false
              and stock_rows.is_st = false
              and stock_rows.close is not null
              and stock_rows.amount is not null
              and stock_rows.amount > 0
        ),
        metric_rows as (
            select
                board_code,
                trade_date,
                volume,
                amount,
                (close - pre_close) / nullif(pre_close, 0) * 100 as pct_chg
            from daily_rows
            where pre_close is not null
        ),
        aggregate_rows as (
            select
                board_code,
                trade_date,
                sum(volume) as volume,
                sum(amount) as amount,
                sum(pct_chg * amount) / nullif(sum(amount), 0) as pct_chg,
                count(*) as stock_count
            from metric_rows
            group by board_code, trade_date
        )
        select
            aggregate_rows.board_code,
            coalesce(board_ref.name, '') as board_name,
            aggregate_rows.trade_date::text as trade_time,
            aggregate_rows.volume,
            aggregate_rows.amount,
            aggregate_rows.pct_chg
        from aggregate_rows
        left join ref.board board_ref on board_ref.board_code = aggregate_rows.board_code
        where aggregate_rows.stock_count > 0
        order by aggregate_rows.board_code, aggregate_rows.trade_date
        """,
        (board_codes, end_date, end_date, board_codes, end_date, start_date, end_date),
    )


def _board_quote_query_start_date(start_date: str) -> str:
    return _offset_date_text(start_date, -45)


def _canonical_concept_quote_frame(concept_ids: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    from quotemux.infra.db.client import query_dataframe

    if concept_ids == [] or start_date == "" or end_date == "":
        return pd.DataFrame()
    return query_dataframe(
        """
        with snapshot_dates as (
            select
                membership.concept_id,
                coalesce(
                    max(membership.valid_from) filter (where membership.valid_from <= %s::date),
                    min(membership.valid_from)
                ) as valid_from
            from ref.concept_stock_membership membership
            where membership.concept_id = any(%s)
            group by membership.concept_id
        ),
        member_rows as (
            select
                membership.concept_id,
                membership.stock_market,
                membership.stock_code as code
            from ref.concept_stock_membership membership
            join snapshot_dates snapshot
              on snapshot.concept_id = membership.concept_id
             and snapshot.valid_from = membership.valid_from
            where membership.valid_to is null or membership.valid_to >= %s::date
        ),
        daily_rows as (
            select
                member_rows.concept_id,
                stock_rows.trade_date,
                stock_rows.code,
                stock_rows.close,
                stock_rows.volume,
                stock_rows.amount,
                stock_rows.pct_chg,
                lag(stock_rows.close) over (
                    partition by member_rows.concept_id, stock_rows.market, stock_rows.code
                    order by stock_rows.trade_date
                ) as pre_close
            from member_rows
            join fact.stock_daily_1d stock_rows
              on stock_rows.market = member_rows.stock_market
             and stock_rows.code = member_rows.code
            where stock_rows.trade_date between %s::date and %s::date
              and coalesce(stock_rows.is_suspended, false) = false
              and coalesce(stock_rows.is_st, false) = false
              and stock_rows.close is not null
              and stock_rows.amount is not null
              and stock_rows.amount > 0
        ),
        aggregate_rows as (
            select
                concept_id,
                trade_date,
                sum(volume) as volume,
                sum(amount) as amount,
                sum(pct_chg * amount) filter (where pct_chg is not null)
                    / nullif(sum(amount) filter (where pct_chg is not null), 0) as pct_chg,
                count(*) filter (where pre_close is not null) as stock_count
            from daily_rows
            group by concept_id, trade_date
        )
        select
            aggregate_rows.concept_id as board_code,
            coalesce(concept_ref.name, '') as board_name,
            aggregate_rows.trade_date::text as trade_time,
            aggregate_rows.volume,
            aggregate_rows.amount,
            aggregate_rows.pct_chg
        from aggregate_rows
        left join ref.concept concept_ref on concept_ref.concept_id = aggregate_rows.concept_id
        where aggregate_rows.stock_count > 0
        order by aggregate_rows.concept_id, aggregate_rows.trade_date
        """,
        (end_date, concept_ids, end_date, start_date, end_date),
    )


def _canonical_concept_quote_snapshot_frame(concept_ids: list[str], trade_date: str) -> pd.DataFrame:
    from quotemux.infra.db.client import query_dataframe

    if concept_ids == [] or trade_date == "":
        return pd.DataFrame()
    return query_dataframe(
        """
        with snapshot_dates as (
            select
                membership.concept_id,
                coalesce(
                    max(membership.valid_from) filter (where membership.valid_from <= %s::date),
                    min(membership.valid_from)
                ) as valid_from
            from ref.concept_stock_membership membership
            where membership.concept_id = any(%s)
            group by membership.concept_id
        ),
        member_rows as (
            select
                membership.concept_id,
                membership.stock_market,
                membership.stock_code as code
            from ref.concept_stock_membership membership
            join snapshot_dates snapshot
              on snapshot.concept_id = membership.concept_id
             and snapshot.valid_from = membership.valid_from
            where membership.valid_to is null or membership.valid_to >= %s::date
        ),
        previous_trade_date as (
            select max(trade_date) as trade_date
            from fact.stock_daily_1d
            where trade_date < %s::date
        ),
        daily_rows as (
            select
                member_rows.concept_id,
                stock_rows.trade_date,
                stock_rows.close,
                stock_rows.volume,
                stock_rows.amount,
                stock_rows.pct_chg,
                previous_rows.close as pre_close
            from member_rows
            join fact.stock_daily_1d stock_rows
              on stock_rows.market = member_rows.stock_market
             and stock_rows.code = member_rows.code
             and stock_rows.trade_date = %s::date
            left join fact.stock_daily_1d previous_rows
              on previous_rows.market = stock_rows.market
             and previous_rows.code = stock_rows.code
             and previous_rows.trade_date = (select trade_date from previous_trade_date)
            where coalesce(stock_rows.is_suspended, false) = false
              and coalesce(stock_rows.is_st, false) = false
              and stock_rows.close is not null
              and stock_rows.amount is not null
              and stock_rows.amount > 0
        ),
        aggregate_rows as (
            select
                concept_id,
                trade_date,
                sum(volume) as volume,
                sum(amount) as amount,
                sum(pct_chg * amount) filter (where pct_chg is not null)
                    / nullif(sum(amount) filter (where pct_chg is not null), 0) as pct_chg,
                count(*) filter (where pre_close is not null) as stock_count
            from daily_rows
            group by concept_id, trade_date
        )
        select
            aggregate_rows.concept_id as board_code,
            coalesce(concept_ref.name, '') as board_name,
            aggregate_rows.trade_date::text as trade_time,
            aggregate_rows.volume,
            aggregate_rows.amount,
            aggregate_rows.pct_chg
        from aggregate_rows
        left join ref.concept concept_ref on concept_ref.concept_id = aggregate_rows.concept_id
        where aggregate_rows.stock_count > 0
        order by aggregate_rows.concept_id, aggregate_rows.trade_date
        """,
        (trade_date, concept_ids, trade_date, trade_date, trade_date),
    )


def _industry_board_quote_frame(start_date: str, end_date: str, board_codes: list[str]) -> pd.DataFrame:
    from quotemux.infra.db.client import query_dataframe

    if start_date == "" or end_date == "":
        return pd.DataFrame()
    industry_names = [item.removeprefix("INDUSTRY:") for item in board_codes if item.startswith("INDUSTRY:")]
    industry_filter = "and stock_ref.industry = any(%s)" if industry_names != [] else ""
    params: tuple[object, ...]
    if industry_names != []:
        params = (start_date, end_date, industry_names)
    else:
        params = (start_date, end_date)
    return query_dataframe(
        f"""
        with daily_rows as (
            select
                stock_ref.industry,
                stock_rows.trade_date,
                stock_rows.market,
                stock_rows.code,
                stock_rows.open,
                stock_rows.high,
                stock_rows.low,
                stock_rows.close,
                stock_rows.volume,
                stock_rows.amount,
                coalesce(
                    stock_rows.pre_close,
                    lag(stock_rows.close) over (
                        partition by stock_rows.market, stock_rows.code
                        order by stock_rows.trade_date
                    )
                ) as pre_close
            from fact.stock_daily_1d stock_rows
            join ref.stock stock_ref
              on stock_ref.market = stock_rows.market
             and stock_ref.code = stock_rows.code
            where stock_rows.trade_date between %s::date and %s::date
              and stock_ref.industry <> ''
              {industry_filter}
              and coalesce(stock_rows.is_suspended, false) = false
              and coalesce(stock_rows.is_st, false) = false
              and stock_rows.open is not null
              and stock_rows.high is not null
              and stock_rows.low is not null
              and stock_rows.close is not null
              and stock_rows.amount is not null
              and stock_rows.amount > 0
        ),
        aggregate_rows as (
            select
                industry,
                trade_date,
                sum(open * amount) / nullif(sum(amount) filter (where pre_close is not null), 0) as open,
                sum(high * amount) / nullif(sum(amount) filter (where pre_close is not null), 0) as high,
                sum(low * amount) / nullif(sum(amount) filter (where pre_close is not null), 0) as low,
                sum(close * amount) / nullif(sum(amount) filter (where pre_close is not null), 0) as close,
                sum(pre_close * amount) / nullif(sum(amount) filter (where pre_close is not null), 0) as pre_close,
                sum(volume) as volume,
                sum(amount) as amount,
                count(*) filter (where pre_close is not null) as stock_count
            from daily_rows
            where pre_close is not null
            group by industry, trade_date
        )
        select
            'INDUSTRY:' || industry as board_code,
            industry as board_name,
            trade_date::text as trade_time,
            open,
            high,
            low,
            close,
            pre_close,
            close - pre_close as change,
            (close - pre_close) / nullif(pre_close, 0) * 100 as pct_chg,
            volume,
            amount
        from aggregate_rows
        where stock_count > 0
        order by industry, trade_date
        """,
        params,
    )


def _derived_quote_items(frame: pd.DataFrame, start_date: str, end_date: str, count: int | None) -> list[BoardQuoteItem]:
    if frame.empty:
        return []
    work = frame.copy()
    work["pct_chg"] = pd.to_numeric(work["pct_chg"], errors="coerce")
    work["amount"] = pd.to_numeric(work["amount"], errors="coerce")
    work["volume"] = pd.to_numeric(work["volume"], errors="coerce")
    work = work[(work["trade_time"] >= start_date) & (work["trade_time"] <= end_date)]
    if count:
        work = work.sort_values("trade_time").groupby("board_code", group_keys=False).tail(count)
    return [
        BoardQuoteItem(
            board_code=str(row["board_code"]),
            board_name=str(row["board_name"]),
            trade_time=str(row["trade_time"]),
            freq="1d",
            pct_chg=float(row["pct_chg"]) if pd.notna(row["pct_chg"]) else None,
            volume=float(row["volume"]) if pd.notna(row["volume"]) else None,
            amount=float(row["amount"]) if pd.notna(row["amount"]) else None,
        )
        for _, row in work.iterrows()
    ]


def _industry_quote_items(frame: pd.DataFrame, start_date: str, end_date: str, count: int | None) -> list[BoardQuoteItem]:
    if frame.empty:
        return []
    work = frame.copy()
    metric_columns = ("open", "high", "low", "close", "pre_close", "change", "pct_chg", "amount", "volume")
    for column_name in metric_columns:
        work[column_name] = pd.to_numeric(work[column_name], errors="coerce")
    work = work[(work["trade_time"] >= start_date) & (work["trade_time"] <= end_date)]
    if count:
        work = work.sort_values("trade_time").groupby("board_code", group_keys=False).tail(count)
    return [
        BoardQuoteItem(
            board_code=str(row["board_code"]),
            board_name=str(row["board_name"]),
            trade_time=str(row["trade_time"]),
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
        for _, row in work.iterrows()
    ]


def get_industry_board_quotes(board_codes: list[str], freq: str, trade_date: str, start_date: str, end_date: str, start_time: str, end_time: str, count: int | None) -> list[BoardQuoteItem]:
    del start_time, end_time
    if freq != "1d":
        return []
    actual_start, actual_end = _board_quote_date_window(trade_date, start_date, end_date)
    if actual_start == "" or actual_end == "":
        return []
    query_start = _board_quote_query_start_date(actual_start)
    if query_start == "":
        return []
    return _industry_quote_items(_industry_board_quote_frame(query_start, actual_end, board_codes), actual_start, actual_end, count)


def _board_quote_snapshot_frame(board_codes: list[str], trade_date: str) -> pd.DataFrame:
    from quotemux.infra.db.client import query_dataframe

    if board_codes == [] or trade_date == "":
        return pd.DataFrame()
    return query_dataframe(
        """
        with target_member_rows as (
            select
                membership.board_code,
                membership.stock_code as code,
                membership.weight,
                membership.valid_from,
                membership.valid_to
            from ref.board_stock_membership membership
            where membership.board_code = any(%s)
              and membership.valid_from <= %s::date
              and (membership.valid_to is null or membership.valid_to >= %s::date)
        ),
        fallback_member_dates as (
            select
                membership.board_code,
                max(membership.valid_from) as valid_from
            from ref.board_stock_membership membership
            where membership.board_code = any(%s)
              and membership.valid_from <= %s::date
            group by membership.board_code
        ),
        fallback_member_rows as (
            select
                membership.board_code,
                membership.stock_code as code,
                membership.weight,
                membership.valid_from,
                membership.valid_to
            from ref.board_stock_membership membership
            join fallback_member_dates latest
              on latest.board_code = membership.board_code
             and latest.valid_from = membership.valid_from
            where not exists (
                select 1
                from target_member_rows target
                where target.board_code = membership.board_code
            )
        ),
        member_rows as (
            select * from target_member_rows
            union all
            select * from fallback_member_rows
        ),
        previous_trade_date as (
            select max(trade_date) as trade_date
            from fact.stock_daily_1d
            where trade_date < %s::date
        ),
        daily_rows as (
            select
                member_rows.board_code,
                stock_rows.trade_date,
                stock_rows.code,
                stock_rows.close,
                stock_rows.volume,
                stock_rows.amount,
                previous_rows.close as pre_close
            from member_rows
            join fact.stock_daily_1d stock_rows
              on stock_rows.code = member_rows.code
             and stock_rows.trade_date = %s::date
            left join fact.stock_daily_1d previous_rows
              on previous_rows.code = stock_rows.code
             and previous_rows.trade_date = (select trade_date from previous_trade_date)
            where stock_rows.is_suspended = false
              and stock_rows.is_st = false
              and stock_rows.close is not null
              and stock_rows.amount is not null
              and stock_rows.amount > 0
        ),
        metric_rows as (
            select
                board_code,
                trade_date,
                volume,
                amount,
                (close - pre_close) / nullif(pre_close, 0) * 100 as pct_chg
            from daily_rows
            where pre_close is not null
        ),
        aggregate_rows as (
            select
                board_code,
                trade_date,
                sum(volume) as volume,
                sum(amount) as amount,
                sum(pct_chg * amount) / nullif(sum(amount), 0) as pct_chg,
                count(*) as stock_count
            from metric_rows
            group by board_code, trade_date
        )
        select
            aggregate_rows.board_code,
            coalesce(board_ref.name, '') as board_name,
            aggregate_rows.trade_date::text as trade_time,
            aggregate_rows.volume,
            aggregate_rows.amount,
            aggregate_rows.pct_chg
        from aggregate_rows
        left join ref.board board_ref on board_ref.board_code = aggregate_rows.board_code
        where aggregate_rows.stock_count > 0
        order by aggregate_rows.board_code, aggregate_rows.trade_date
        """,
        (board_codes, trade_date, trade_date, board_codes, trade_date, trade_date, trade_date),
    )


def get_board_quotes(board_codes: list[str], freq: str, trade_date: str, start_date: str, end_date: str, start_time: str, end_time: str, count: int | None) -> list[BoardQuoteItem]:
    del start_time
    del end_time
    if freq != "1d":
        return []
    normalized_codes = [str(item).upper() for item in board_codes if str(item).upper()]
    normalized_codes = list(dict.fromkeys(normalized_codes))
    if normalized_codes == []:
        return []
    actual_start, actual_end = _board_quote_date_window(trade_date, start_date, end_date)
    if actual_start == "" or actual_end == "":
        return []
    if actual_start == actual_end:
        frame = _board_quote_snapshot_frame(normalized_codes, actual_end)
    else:
        query_start = _board_quote_query_start_date(actual_start)
        if query_start == "":
            return []
        frame = _board_quote_frame(normalized_codes, query_start, actual_end)
    if frame.empty:
        return []
    work = frame.copy()
    work["pct_chg"] = pd.to_numeric(work["pct_chg"], errors="coerce")
    work["amount"] = pd.to_numeric(work["amount"], errors="coerce")
    work["volume"] = pd.to_numeric(work["volume"], errors="coerce")
    work = work[(work["trade_time"] >= actual_start) & (work["trade_time"] <= actual_end)]
    if count:
        work = work.sort_values("trade_time").groupby("board_code", group_keys=False).tail(count)
    items: list[BoardQuoteItem] = []
    for _, row in work.iterrows():
        items.append(
            BoardQuoteItem(
                board_code=str(row["board_code"]),
                board_name=str(row["board_name"]),
                trade_time=str(row["trade_time"]),
                freq="1d",
                open=None,
                high=None,
                low=None,
                close=None,
                pre_close=None,
                change=None,
                pct_chg=float(row["pct_chg"]) if pd.notna(row["pct_chg"]) else None,
                volume=float(row["volume"]) if pd.notna(row["volume"]) else None,
                amount=float(row["amount"]) if pd.notna(row["amount"]) else None,
            )
        )
    return items


def _sum_optional_column(frame: pd.DataFrame, column_name: str) -> float | None:
    if column_name not in frame.columns:
        return None
    values = pd.to_numeric(frame[column_name], errors="coerce")
    if not values.notna().any():
        return None
    return float(values.sum())


def _amount_wan_to_yuan(value: object) -> object:
    if value is None:
        return None
    return pd.to_numeric(value, errors="coerce") * 10000


def _amount_yuan_to_yi(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value) / 100000000.0


def _tushare_daily_stock_money_flow(date_value: str, member_codes: set[str]) -> pd.DataFrame:
    from quotemux.infra.common import stock_code_to_ts
    from quotemux.infra.provider_config import get_provider_api_key
    from quotemux.infra.tushare.rate_limit import call_tushare_api

    if member_codes == set():
        return pd.DataFrame()
    try:
        import tushare as ts
    except Exception:
        return pd.DataFrame()
    api_key = get_provider_api_key()
    if api_key == "":
        return pd.DataFrame()
    pro = ts.pro_api(api_key)
    request_date = date_value.replace("-", "")
    try:
        frame = call_tushare_api("moneyflow", pro.moneyflow, trade_date=request_date)
    except Exception:
        return pd.DataFrame()
    if frame is None or frame.empty:
        return pd.DataFrame()
    work = frame.copy()
    if "ts_code" not in work.columns or "trade_date" not in work.columns:
        return pd.DataFrame()
    member_ts_codes = {stock_code_to_ts(code) for code in member_codes if stock_code_to_ts(code) != ""}
    work = work[work["ts_code"].astype(str).isin(member_ts_codes)]
    if work.empty:
        return pd.DataFrame()
    for column_name in ("buy_lg_amount", "buy_elg_amount", "sell_lg_amount", "sell_elg_amount", "net_mf_amount"):
        if column_name not in work.columns:
            work[column_name] = 0
    result = pd.DataFrame(
        [
            {
                "trade_date": date_value,
                "main_inflow": _amount_wan_to_yuan(pd.to_numeric(work["buy_lg_amount"], errors="coerce").fillna(0).sum() + pd.to_numeric(work["buy_elg_amount"], errors="coerce").fillna(0).sum()),
                "main_outflow": _amount_wan_to_yuan(pd.to_numeric(work["sell_lg_amount"], errors="coerce").fillna(0).sum() + pd.to_numeric(work["sell_elg_amount"], errors="coerce").fillna(0).sum()),
                "net_inflow": _amount_wan_to_yuan(pd.to_numeric(work["net_mf_amount"], errors="coerce").fillna(0).sum()),
            }
        ]
    )
    return result


def _aggregate_board_money_flow(normalized: str, scope: str, date_value: str, codes: list[str]) -> BoardMoneyFlowItem | None:
    fast_frame = _tushare_daily_stock_money_flow(date_value, set(codes))
    if fast_frame.empty:
        return None
    inflow = _sum_optional_column(fast_frame, "main_inflow")
    outflow = _sum_optional_column(fast_frame, "main_outflow")
    net_inflow = _sum_optional_column(fast_frame, "net_inflow")
    if inflow is None and outflow is None and net_inflow is None:
        return None
    return BoardMoneyFlowItem(board_code=normalized, trade_date=date_value, scope=scope, inflow=_amount_yuan_to_yi(inflow), outflow=_amount_yuan_to_yi(outflow), net_inflow=_amount_yuan_to_yi(net_inflow))


def get_board_money_flow(board_code: str, trade_date: str, start_date: str, end_date: str, scope: str) -> list[BoardMoneyFlowItem]:
    from datetime import timedelta

    from quotemux.infra.common import format_date_value, parse_date_text

    normalized = str(board_code).upper()
    if normalized == "" or scope != "board":
        return []
    quote_mux = _quote_mux()
    actual_trade_date = format_date_value(trade_date)
    actual_start = format_date_value(start_date)
    actual_end = format_date_value(end_date)
    if actual_trade_date != "":
        date_values = [actual_trade_date]
    else:
        if actual_start == "" and actual_end == "":
            return []
        if actual_start == "":
            actual_start = actual_end
        if actual_end == "":
            actual_end = actual_start
        start_day = parse_date_text(actual_start)
        end_day = parse_date_text(actual_end)
        if start_day is None or end_day is None or start_day > end_day:
            return []
        date_values = []
        current_day = start_day
        while current_day <= end_day:
            date_values.append(current_day.strftime("%Y-%m-%d"))
            current_day += timedelta(days=1)
    rows: list[BoardMoneyFlowItem] = []
    for date_value in date_values:
        member_items = quote_mux.concepts.get_members(normalized, date_value)
        codes = [item.code for item in member_items if item.code != ""]
        if codes == []:
            continue
        fast_item = _aggregate_board_money_flow(normalized, scope, date_value, codes)
        if fast_item is not None:
            rows.append(fast_item)
            continue
        flow_items = quote_mux.stocks.get_money_flow_batch(",".join(codes), date_value, "main")
        if flow_items == []:
            continue
        frame = pd.DataFrame([item.model_dump() for item in flow_items])
        if frame.empty:
            continue
        inflow = _sum_optional_column(frame, "main_inflow")
        outflow = _sum_optional_column(frame, "main_outflow")
        net_inflow = _sum_optional_column(frame, "net_inflow")
        if inflow is None and outflow is None and net_inflow is None:
            continue
        rows.append(
            BoardMoneyFlowItem(
                board_code=normalized,
                trade_date=date_value,
                scope=scope,
                inflow=_amount_yuan_to_yi(inflow),
                outflow=_amount_yuan_to_yi(outflow),
                net_inflow=_amount_yuan_to_yi(net_inflow),
            )
        )
    if rows == []:
        return []
    return sorted(rows, key=lambda item: item.trade_date)

def get_concept_members(concept_id: str, trade_date: str):
    return get_board_members(concept_id, trade_date)

def get_concept_quotes(concept_ids: list[str], freq: str, trade_date: str, start_date: str, end_date: str, start_time: str, end_time: str, count: int | None):
    normalized_ids = list(dict.fromkeys(str(item).strip().upper() for item in concept_ids if str(item).strip() != ""))
    if normalized_ids != [] and all(item.startswith("C") and item[1:].isdigit() for item in normalized_ids):
        if freq != "1d":
            return []
        actual_start, actual_end = _board_quote_date_window(trade_date, start_date, end_date)
        if actual_start == "" or actual_end == "":
            return []
        if actual_start == actual_end:
            return _derived_quote_items(_canonical_concept_quote_snapshot_frame(normalized_ids, actual_end), actual_start, actual_end, count)
        query_start = _board_quote_query_start_date(actual_start)
        if query_start == "":
            return []
        return _derived_quote_items(_canonical_concept_quote_frame(normalized_ids, query_start, actual_end), actual_start, actual_end, count)
    return get_board_quotes(normalized_ids, freq, trade_date, start_date, end_date, start_time, end_time, count)

def get_concept_money_flow(concept_id: str, trade_date: str, start_date: str, end_date: str, scope: str):
    actual_scope = "board" if scope in {"", "concept"} else scope
    return get_board_money_flow(concept_id, trade_date, start_date, end_date, actual_scope)


def _money_flow_snapshot_board_codes(trade_date: str) -> list[str]:
    from quotemux.infra.common import format_date_value
    from quotemux.infra.db.client import query_dataframe

    actual_trade_date = format_date_value(trade_date)
    if actual_trade_date == "":
        return []
    frame = query_dataframe(
        """
        select distinct membership.board_code
        from ref.board_stock_membership membership
        where membership.valid_from <= %s::date
          and (membership.valid_to is null or membership.valid_to >= %s::date)
        order by membership.board_code
        """,
        (actual_trade_date, actual_trade_date),
    )
    if frame.empty:
        return []
    return [str(row["board_code"]).upper() for _, row in frame.iterrows() if str(row["board_code"]).strip() != ""]


def get_concept_daily_money_flow_snapshot(trade_date: str, scope: str, limit: int, offset: int) -> list[BoardMoneyFlowItem]:
    from quotemux.infra.common import format_date_value

    actual_trade_date = format_date_value(trade_date)
    if actual_trade_date == "" or scope not in {"", "concept", "board"}:
        return []
    board_codes = _money_flow_snapshot_board_codes(actual_trade_date)
    rows: list[BoardMoneyFlowItem] = []
    for board_code in board_codes:
        rows.extend(get_board_money_flow(board_code, actual_trade_date, "", "", "board"))
    return rows[offset: offset + limit]

