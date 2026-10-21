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
        if isinstance(value, bool):
            raise FreeStockdbError(f"invalid boolean {name} in free-stockdb row")
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
    if "date" not in row or "code" not in row:
        raise FreeStockdbError("free-stockdb row must include code and date identity")
    if row.get("is_st") not in (None, False, 0, "", "0") or row.get("is_suspended", row.get("suspended")) not in (None, False, 0, "", "0"):
        raise FreeStockdbError("free-stockdb endpoint does not provide verified ST/suspension flags")
    row_date = row["date"]
    actual_date = _date_text(row_date)
    if actual_date != trade_date:
        raise FreeStockdbError(f"free-stockdb row date mismatch for {code}")
    row_code = row["code"]
    if _normalise_code(row_code) != code:
        raise FreeStockdbError(f"free-stockdb row code mismatch for {code}")
    values = {name: _number(row, *names) for name, names in {
        "open": ("open",), "high": ("high",), "low": ("low",), "close": ("close",),
        "pre_close": ("pre_close", "preclose"), "volume": ("volume", "vol"), "amount": ("amount",),
    }.items()}
    if any(values[key] is None for key in ("open", "high", "low", "close", "pre_close", "volume", "amount")):
        raise FreeStockdbError("free-stockdb row is missing required OHLCVA fields")
    if values["volume"] < 0 or values["amount"] < 0:
        raise FreeStockdbError("free-stockdb row volume and amount must be non-negative")
    if not (values["low"] <= min(values["open"], values["close"]) <= max(values["open"], values["close"]) <= values["high"]):
        raise FreeStockdbError("free-stockdb row has invalid OHLC bounds")
    pre_close, volume, amount = values["pre_close"], values["volume"], values["amount"]
    change = _number(row, "change")
    pct_chg = _number(row, "pct_chg", "pct_change")
    if change is None:
        change = values["close"] - values["pre_close"]
    trade_time = trade_date[:4] + "-" + trade_date[4:6] + "-" + trade_date[6:]
    return StockQuoteItem(
        code=code,
        trade_time=trade_time,
        freq=DAILY_FREQ,
        open=values["open"], high=values["high"], low=values["low"], close=values["close"],
        pre_close=pre_close, change=change, pct_chg=pct_chg,
        volume=volume, amount=amount,
        adjust="none",
        is_suspended=False,
        is_st=False,
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
    del start_time, end_time
    if freq != DAILY_FREQ:
        return []
    if not codes:
        return []
    if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 0):
        raise FreeStockdbError("count must be a non-negative integer")
    if adjust not in {"", "none"}:
        raise FreeStockdbError("free-stockdb supports unadjusted daily quotes only")
    exact_date = _exact_date(trade_date, start_date, end_date)
    items: list[StockQuoteItem] = []
    for raw_code in codes:
        code = _normalise_code(raw_code)
        row = _query(code, exact_date)
        if row is not None:
            items.append(_quote(code, exact_date, row, adjust))
    return items[:count] if count is not None else items
