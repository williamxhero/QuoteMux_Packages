from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import re

import duckdb

from platform_models import BoardCatalogItem, BoardMemberItem, BoardQuoteItem, LimitOrderAmountItem
from quotemux.infra.provider_config import get_provider_config_value


DEFAULT_WAREHOUSE_ROOT = "/data/crawler-provider/data/warehouse"
CONCEPT_PROVIDER_BY_BOARD_TYPE = {"ths": "ths", "em": "eastmoney"}


def _warehouse_root() -> Path:
    configured = get_provider_config_value("warehouse_root").strip()
    return Path(configured or DEFAULT_WAREHOUSE_ROOT)


def _date_text(value: str) -> str:
    text = str(value).strip()
    if text == "":
        return ""
    if re.fullmatch(r"[0-9]{8}", text):
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", text):
        return text
    return ""


def _month_text(trade_date: str) -> str:
    return trade_date[:7]


def _market_for_code(code: str) -> str:
    if code.startswith(("60", "688", "900")):
        return "sh"
    if code.startswith(("00", "30", "200")):
        return "sz"
    if code.startswith(("43", "83", "87", "88", "920")):
        return "bj"
    if code.startswith("6"):
        return "sh"
    if code.startswith(("0", "3")):
        return "sz"
    return "bj"


def _limit_side(value: object) -> str:
    text = str(value).strip()
    if text == "limit_up":
        return "up"
    if text == "limit_down":
        return "down"
    return text


def _float_value(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "nan", "nat"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _read_rows(path: Path, sql: str) -> list[tuple]:
    if not path.exists():
        return []
    connection = duckdb.connect(database=":memory:", read_only=False)
    try:
        return connection.execute(sql, [path.as_posix()]).fetchall()
    finally:
        connection.close()


def _latest_file(pattern: str) -> Path | None:
    files = sorted(_warehouse_root().glob(pattern))
    if files == []:
        return None
    return files[-1]


def _limit_pool_path(trade_date: str) -> Path:
    return _warehouse_root() / "limit_pool" / f"month={_month_text(trade_date)}" / f"limit_pool_{trade_date}.parquet"


def _limit_order_book_path(trade_date: str) -> Path:
    return _warehouse_root() / "limit_order_book" / f"month={_month_text(trade_date)}" / f"limit_order_book_{trade_date}.parquet"


def _concept_trend_path(provider: str, trade_date: str) -> Path:
    return _warehouse_root() / "concept_trend" / f"provider={provider}" / f"month={_month_text(trade_date)}" / f"concept_trend_{trade_date}.parquet"


def _concept_member_path(provider: str, concept_code: str, trade_date: str) -> Path:
    return _warehouse_root() / "concept_members" / f"provider={provider}" / f"month={_month_text(trade_date)}" / f"concept_members_{concept_code}_{trade_date}.parquet"


def _latest_concept_catalog_path(provider: str) -> Path | None:
    return _latest_file(f"concept_catalog/provider={provider}/month=*/concept_catalog_*.parquet")


def _latest_concept_trend_path(provider: str) -> Path | None:
    return _latest_file(f"concept_trend/provider={provider}/month=*/concept_trend_*.parquet")


def _latest_concept_member_path(provider: str, concept_code: str) -> Path | None:
    return _latest_file(f"concept_members/provider={provider}/month=*/concept_members_{concept_code}_*.parquet")


def _concept_provider(board_type: str) -> str:
    return CONCEPT_PROVIDER_BY_BOARD_TYPE.get(board_type.strip().lower(), board_type.strip().lower())


def _date_range(start_date: str, end_date: str) -> list[str]:
    start_text = _date_text(start_date)
    end_text = _date_text(end_date)
    if start_text == "" and end_text == "":
        return []
    if start_text == "":
        start_text = end_text
    if end_text == "":
        end_text = start_text
    start_day = datetime.strptime(start_text, "%Y-%m-%d").date()
    end_day = datetime.strptime(end_text, "%Y-%m-%d").date()
    if start_day > end_day:
        return []
    dates: list[str] = []
    current = start_day
    while current <= end_day:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates


def _concept_trend_paths(provider: str, trade_date: str, start_date: str, end_date: str) -> list[Path]:
    actual_trade_date = _date_text(trade_date)
    if actual_trade_date != "":
        path = _concept_trend_path(provider, actual_trade_date)
        return [path] if path.exists() else []
    date_values = _date_range(start_date, end_date)
    if date_values != []:
        paths = [_concept_trend_path(provider, date_value) for date_value in date_values]
        return [path for path in paths if path.exists()]
    latest_path = _latest_concept_trend_path(provider)
    return [] if latest_path is None else [latest_path]


def _concept_selector(value: str) -> tuple[str, str]:
    text = str(value).strip()
    if ":" not in text:
        return "", text
    board_type, concept_code = text.split(":", 1)
    return board_type.strip().lower(), concept_code.strip()


def _selector_map(values: list[str]) -> dict[str, set[str]]:
    provider_codes: dict[str, set[str]] = {}
    all_provider_codes: set[str] = set()
    for value in values:
        board_type, concept_code = _concept_selector(value)
        if concept_code == "":
            continue
        if board_type == "":
            all_provider_codes.add(concept_code)
            continue
        provider_codes.setdefault(_concept_provider(board_type), set()).add(concept_code)
    if all_provider_codes:
        for provider in CONCEPT_PROVIDER_BY_BOARD_TYPE.values():
            provider_codes.setdefault(provider, set()).update(all_provider_codes)
    return provider_codes


def _board_type(provider: str) -> str:
    if provider == "eastmoney":
        return "em"
    return provider


def _limit_pool_items(trade_date: str) -> list[LimitOrderAmountItem]:
    rows = _read_rows(
        _limit_pool_path(trade_date),
        """
        select trading_date, limit_type, stock_code, stock_name
        from read_parquet(?)
        order by limit_type, stock_code
        """,
    )
    return [
        LimitOrderAmountItem(
            code=str(row[2]).zfill(6),
            trade_date=_date_text(str(row[0])) or trade_date,
            limit_side=_limit_side(row[1]),
            market=_market_for_code(str(row[2]).zfill(6)),
            close=None,
            limit_price=None,
            order_price=None,
            order_volume=None,
            order_amount=None,
            captured_at="",
        )
        for row in rows
        if re.fullmatch(r"[0-9]{6}", str(row[2]).zfill(6))
    ]


def get_limit_stock_candidates(trade_date: str) -> list[LimitOrderAmountItem]:
    actual_trade_date = _date_text(trade_date)
    if actual_trade_date == "":
        raise ValueError("trade_date 不能为空")
    return _limit_pool_items(actual_trade_date)


def get_limit_order_amount(*args: object) -> list[LimitOrderAmountItem]:
    trade_date = _date_text(str(args[0] if len(args) == 1 else args[1] if len(args) >= 2 else ""))
    if trade_date == "":
        raise ValueError("trade_date 不能为空")
    rows = _read_rows(
        _limit_order_book_path(trade_date),
        """
        select trading_date, limit_type, stock_code, stock_name, market,
               buy_1_price, buy_1_volume, buy_1_amount,
               sell_1_price, sell_1_volume, sell_1_amount
        from read_parquet(?)
        order by limit_type, stock_code
        """,
    )
    if rows == []:
        return _limit_pool_items(trade_date)
    items: list[LimitOrderAmountItem] = []
    for row in rows:
        code = str(row[2]).zfill(6)
        limit_side = _limit_side(row[1])
        if not re.fullmatch(r"[0-9]{6}", code):
            continue
        if limit_side == "up":
            order_price = _float_value(row[5])
            order_volume = _float_value(row[6])
            order_amount = _float_value(row[7])
        else:
            order_price = _float_value(row[8])
            order_volume = _float_value(row[9])
            order_amount = _float_value(row[10])
        items.append(
            LimitOrderAmountItem(
                code=code,
                trade_date=_date_text(str(row[0])) or trade_date,
                limit_side=limit_side,
                market=str(row[4]).strip() or _market_for_code(code),
                close=order_price,
                limit_price=order_price,
                order_price=order_price,
                order_volume=order_volume,
                order_amount=order_amount,
                captured_at=f"{trade_date} 15:00:00",
            )
        )
    return sorted(items, key=lambda item: (item.trade_date, item.limit_side, item.code))


def get_concept_catalog(category: str, market: str, status: str, limit: int, offset: int) -> list[BoardCatalogItem]:
    if category not in {"", "concept"} or status not in {"", "active"}:
        return []
    requested_market = market.strip().lower()
    providers = tuple(CONCEPT_PROVIDER_BY_BOARD_TYPE.values()) if requested_market in {"", "a_share"} else (_concept_provider(requested_market),)
    items: list[BoardCatalogItem] = []
    for provider in providers:
        path = _latest_concept_catalog_path(provider)
        if path is None:
            continue
        rows = _read_rows(
            path,
            """
            select provider, trading_date, concept_code, concept_name, constituent_count
            from read_parquet(?)
            order by concept_code
            """,
        )
        for row in rows:
            provider_text = str(row[0]).strip()
            board_type = _board_type(provider_text)
            items.append(
                BoardCatalogItem(
                    board_code=str(row[2]).strip(),
                    board_name=str(row[3]).strip(),
                    category="concept",
                    market=board_type,
                    status="active",
                )
            )
    return sorted(items, key=lambda item: (item.market, item.board_code))[offset : offset + int(limit)]


def get_concept_quotes(concept_ids: list[str], freq: str, trade_date: str, start_date: str, end_date: str, start_time: str, end_time: str, count: int | None) -> list[BoardQuoteItem]:
    del start_time, end_time
    if freq != "1d":
        return []
    requested_dates = set(_date_range(trade_date or start_date, trade_date or end_date))
    provider_codes = _selector_map([str(item) for item in concept_ids])
    if provider_codes == {}:
        return []
    items: list[BoardQuoteItem] = []
    for provider, concept_codes in provider_codes.items():
        for path in _concept_trend_paths(provider, trade_date, start_date, end_date):
            rows = _read_rows(
                path,
                """
                select provider, trade_date, concept_code,
                       open_price, low_price, high_price, close_price,
                       net_inflow_amount, volume_hand, turnover_amount
                from read_parquet(?)
                order by concept_code
                """,
            )
            for row in rows:
                concept_code = str(row[2]).strip()
                row_date = _date_text(str(row[1]))
                if concept_code not in concept_codes:
                    continue
                if requested_dates and row_date not in requested_dates:
                    continue
                items.append(
                    BoardQuoteItem(
                        board_code=concept_code,
                        trade_time=row_date,
                        freq="1d",
                        open=_float_value(row[3]),
                        low=_float_value(row[4]),
                        high=_float_value(row[5]),
                        close=_float_value(row[6]),
                        pre_close=None,
                        change=None,
                        pct_chg=None,
                        volume=_float_value(row[8]),
                        amount=_float_value(row[9]),
                    )
                )
    sorted_items = sorted(items, key=lambda item: (item.board_code, item.trade_time))
    if count is not None and count > 0:
        return sorted_items[-count:]
    return sorted_items


def get_concept_members(concept_id: str, trade_date: str) -> list[BoardMemberItem]:
    board_type, concept_code = _concept_selector(concept_id)
    actual_trade_date = _date_text(trade_date)
    if concept_code == "":
        return []
    items: list[BoardMemberItem] = []
    providers = (_concept_provider(board_type),) if board_type != "" else tuple(CONCEPT_PROVIDER_BY_BOARD_TYPE.values())
    for provider in providers:
        path = _concept_member_path(provider, concept_code, actual_trade_date) if actual_trade_date != "" else _latest_concept_member_path(provider, concept_code)
        if path is None:
            continue
        rows = _read_rows(
            path,
            """
            select provider, trading_date, concept_code, stock_code
            from read_parquet(?)
            order by stock_code
            """,
        )
        for row in rows:
            if str(row[2]).strip() != concept_code:
                continue
            items.append(
                BoardMemberItem(
                    board_code=concept_code,
                    code=str(row[3]).zfill(6),
                    name="",
                    join_date=_date_text(str(row[1])),
                )
            )
    return sorted(items, key=lambda item: (item.board_code, item.code))
