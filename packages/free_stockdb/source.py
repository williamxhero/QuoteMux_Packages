from __future__ import annotations

from datetime import date, datetime
import math
from typing import Any

import msgpack
import requests

from platform_models import StockQuoteItem
from quotemux.infra.provider_config import get_provider_config_value
from quotemux.infra.provider_runtime.core import call_provider_api


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7899
DAILY_FREQ = "1d"


class FreeStockdbError(RuntimeError):
    """Raised when the bounded free-stockdb HTTP contract cannot be read."""


def _endpoint() -> tuple[str, int]:
    host = get_provider_config_value("host").strip() or DEFAULT_HOST
    port_text = get_provider_config_value("port").strip() or str(DEFAULT_PORT)
    try:
        port = int(port_text)
    except ValueError as exc:
        raise FreeStockdbError("free-stockdb port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise FreeStockdbError("free-stockdb port is outside 1..65535")
    return host, port


def _normalise_code(value: object) -> str:
    text = str(value).strip()
    if text.endswith((".SH", ".SZ", ".BJ")):
        text = text[:-3]
    if not text.isdigit() or len(text) != 6:
        raise FreeStockdbError(f"invalid stock code: {value!r}")
    return text


def _date_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.strftime("%Y%m%d")
    if isinstance(value, date):
        return value.strftime("%Y%m%d")
    text = str(value).strip()
    if len(text) == 8 and text.isdigit():
        try:
            datetime.strptime(text, "%Y%m%d")
        except ValueError as exc:
            raise FreeStockdbError(f"invalid trade date: {value!r}") from exc
        return text
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        try:
            return datetime.strptime(text, "%Y-%m-%d").strftime("%Y%m%d")
        except ValueError as exc:
            raise FreeStockdbError(f"invalid trade date: {value!r}") from exc
    raise FreeStockdbError(f"trade date must be YYYYMMDD or YYYY-MM-DD: {value!r}")


def _exact_date(trade_date: str, start_date: str, end_date: str) -> str:
    supplied = [_date_text(value) for value in (trade_date, start_date, end_date) if str(value or "").strip()]
    if not supplied:
        raise FreeStockdbError("an exact historical trade_date is required")
    if len(set(supplied)) != 1:
        raise FreeStockdbError("free-stockdb source accepts one exact historical date only")
    return supplied[0]


def _number(row: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = row.get(name)
        if value is None or value == "":
            continue
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise FreeStockdbError(f"non-numeric {name} in free-stockdb row") from exc
        if not math.isfinite(number):
            raise FreeStockdbError(f"non-finite {name} in free-stockdb row")
        return number
    return None


def _decode_response(response: requests.Response) -> Any:
    response.raise_for_status()
    try:
        return msgpack.unpackb(response.content, raw=False, strict_map_key=False)
    except Exception as exc:
        raise FreeStockdbError("free-stockdb response is not valid MessagePack") from exc


def _row_from_payload(payload: Any) -> dict[str, Any] | None:
    if payload is None or payload == [] or payload == {}:
        return None
    if isinstance(payload, dict):
        # The exact-key endpoint returns one row. Some deployments wrap it in
        # a one-element data/result container; accept those without broadening
        # the query beyond one exact key.
        for key in ("data", "result", "value"):
            if key in payload and isinstance(payload[key], (dict, list)):
                return _row_from_payload(payload[key])
        return {str(key): value for key, value in payload.items()}
    if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
        return {str(key): value for key, value in payload[0].items()}
    raise FreeStockdbError("free-stockdb exact-key response has an unexpected shape")


def _query(code: str, trade_date: str, *, session: requests.Session | None = None) -> dict[str, Any] | None:
    return call_provider_api(
        "free_stockdb",
        "quotes.daily",
        lambda: _query_http(code, trade_date, session=session),
    )


def _query_http(code: str, trade_date: str, *, session: requests.Session | None = None) -> dict[str, Any] | None:
    host, port = _endpoint()
    url = f"http://{host}:{port}/"
    response = (session or requests).get(
        url,
        params={"cmd": "get", "t": f"日k:{code}:{trade_date}"},
        timeout=10,
    )
    return _row_from_payload(_decode_response(response))


def _quote(code: str, trade_date: str, row: dict[str, Any], adjust: str) -> StockQuoteItem:
    row_date = row.get("date", trade_date)
    actual_date = _date_text(row_date)
    if actual_date != trade_date:
        raise FreeStockdbError(f"free-stockdb row date mismatch for {code}")
    row_code = row.get("code", code)
    if _normalise_code(row_code) != code:
        raise FreeStockdbError(f"free-stockdb row code mismatch for {code}")
    close = _number(row, "close")
    pre_close = _number(row, "pre_close", "preclose")
    change = _number(row, "change")
    pct_chg = _number(row, "pct_chg", "pct_change")
    if change is None and close is not None and pre_close is not None:
        change = close - pre_close
    return StockQuoteItem(
        code=code,
        trade_time=f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]} 00:00:00",
        freq=DAILY_FREQ,
        open=_number(row, "open"),
        high=_number(row, "high"),
        low=_number(row, "low"),
        close=close,
        pre_close=pre_close,
        change=change,
        pct_chg=pct_chg,
        # free-stockdb's verified row uses shares and yuan; do not rescale.
        volume=_number(row, "volume", "vol"),
        amount=_number(row, "amount"),
        adjust=adjust or "none",
        is_suspended=bool(row.get("is_suspended", row.get("suspended", False))),
        is_st=bool(row.get("is_st", False)),
    )


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
    """Return exact-date daily rows for explicitly requested stock codes.

    This intentionally does not implement ranges, latest-date discovery,
    weekly/monthly bars, intraday data, or a native free-stockdb SDK import.
    A missing exact key is an empty result for that code; transport/decoding
    failures remain errors so the runtime can distinguish them from a miss.
    """
    del start_time, end_time, count
    if freq != DAILY_FREQ:
        return []
    if not codes:
        return []
    exact_date = _exact_date(trade_date, start_date, end_date)
    items: list[StockQuoteItem] = []
    for raw_code in codes:
        code = _normalise_code(raw_code)
        row = _query(code, exact_date)
        if row is not None:
            items.append(_quote(code, exact_date, row, adjust))
    return items
