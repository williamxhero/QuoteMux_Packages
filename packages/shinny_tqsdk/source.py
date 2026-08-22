from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
import math
import time
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from quotemux.infra.provider_config import get_provider_config_value, get_provider_secret_value
from quotemux.infra.provider_runtime.core import call_provider_api

if TYPE_CHECKING:
    from platform_models import (
        FutureContractCatalogItem,
        FutureContractRealtimeQuoteItem,
        FutureMainContractMappingItem,
        FutureRealtimeQuoteItem,
    )


DEFAULT_TIMEOUT_SECONDS = 5.0
CHINA_TIMEZONE = ZoneInfo("Asia/Shanghai")


def _load_tqsdk() -> tuple[type[Any], type[Any]]:
    try:
        from tqsdk import TqApi, TqAuth
    except ImportError as exc:
        raise RuntimeError("TqSdk 不可用；请安装 shinny_tqsdk package 的 requirements.txt") from exc
    return TqApi, TqAuth


def _timeout_seconds() -> float:
    try:
        timeout = float(get_provider_config_value("timeout_seconds"))
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_SECONDS
    if not math.isfinite(timeout) or timeout <= 0:
        return DEFAULT_TIMEOUT_SECONDS
    return timeout


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _optional_int(value: object) -> int | None:
    numeric = _optional_float(value)
    return None if numeric is None else int(numeric)


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    try:
        if not math.isfinite(float(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1"}:
            return True
        if text in {"false", "0"}:
            return False
        return None
    try:
        return bool(value)
    except (TypeError, ValueError):
        return None


def _quote_time(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    return _text(value)


def _metadata_datetime(value: object) -> str:
    """Format TqSdk metadata epoch seconds as the documented China-market time."""
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(CHINA_TIMEZONE)
        return value.strftime("%Y-%m-%d %H:%M:%S")
    value = _plain_value(value)
    if value is None:
        return ""
    timestamp = _optional_float(value)
    if timestamp is not None:
        return datetime.fromtimestamp(timestamp, CHINA_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")
    return _text(value)


def _text(value: object) -> str:
    value = _plain_value(value)
    if value is None:
        return ""
    try:
        if math.isnan(float(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _plain_value(value: object) -> object:
    """Keep provider metadata intact while making pandas/numpy values API-serializable."""
    if value is None:
        return None
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()  # type: ignore[union-attr]
        except (TypeError, ValueError):
            pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain_value(item) for item in value]
    try:
        if bool(value != value):
            return None
    except (TypeError, ValueError):
        pass
    if str(value) in {"nan", "NaT", "<NA>"}:
        return None
    return str(value)


def _row_value(row: Mapping[str, object], name: str) -> object:
    return _plain_value(row.get(name))


def _require_credentials() -> tuple[str, str]:
    username = get_provider_config_value("username").strip()
    password = get_provider_secret_value("password")
    if username == "" or password == "":
        raise RuntimeError("shinny_tqsdk requires non-empty username and password configuration")
    return username, password


def _future_realtime_quote_item(**kwargs: object) -> FutureRealtimeQuoteItem:
    from platform_models import FutureRealtimeQuoteItem

    return FutureRealtimeQuoteItem(**kwargs)


def _future_contract_catalog_item(**kwargs: object) -> FutureContractCatalogItem:
    from platform_models import FutureContractCatalogItem

    return FutureContractCatalogItem(**kwargs)


def _future_main_contract_mapping_item(**kwargs: object) -> FutureMainContractMappingItem:
    from platform_models import FutureMainContractMappingItem

    return FutureMainContractMappingItem(**kwargs)


def _future_contract_realtime_quote_item(**kwargs: object) -> FutureContractRealtimeQuoteItem:
    from platform_models import FutureContractRealtimeQuoteItem

    return FutureContractRealtimeQuoteItem(**kwargs)


def _wait_for_quotes(api: Any, quotes: Sequence[Any], timeout: float) -> None:
    deadline = time.time() + timeout
    while True:
        if all(_quote_time(getattr(quote, "datetime", None)) != "" for quote in quotes):
            return
        if not api.wait_update(deadline=deadline) or time.time() >= deadline:
            raise TimeoutError(f"shinny_tqsdk did not receive all quote updates within {timeout:g} seconds")


def _normalize_products(products: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    normalized = [(str(product).strip(), str(exchange).strip()) for product, exchange in products]
    if any(product == "" or exchange == "" for product, exchange in normalized):
        raise ValueError("each future product must include non-empty product_code and exchange")
    return list(dict.fromkeys(normalized))


def _provider_symbol(exchange: str, product_code: str) -> str:
    return f"KQ.m@{exchange}.{product_code}"


def _quote_list(api: Any, symbols: list[str]) -> list[Any]:
    if symbols == []:
        return []
    return list(api.get_quote_list(symbols))


def _quote_product_code(quote: Any, fallback_symbol: str) -> str:
    product_code = _text(getattr(quote, "product_id", None))
    if product_code != "":
        return product_code
    contract = fallback_symbol.partition(".")[2]
    for index, character in enumerate(contract):
        if character.isdigit():
            return contract[:index]
    return contract


def _quote_exchange(quote: Any, fallback_symbol: str) -> str:
    return _text(getattr(quote, "exchange_id", None)) or fallback_symbol.partition(".")[0]


def _real_contract_symbol(quote: Any, fallback_symbol: str) -> str:
    return _text(getattr(quote, "instrument_id", None)) or fallback_symbol


def get_future_main_continuous_realtime(
    products: list[tuple[str, str]],
) -> list[FutureRealtimeQuoteItem]:
    """Return bounded, read-only real-time snapshots for main-continuous products."""
    normalized_products = _normalize_products(products)
    if normalized_products == []:
        return []
    username, password = _require_credentials()
    timeout = _timeout_seconds()

    def _invoke() -> list[FutureRealtimeQuoteItem]:
        TqApi, TqAuth = _load_tqsdk()
        api = TqApi(auth=TqAuth(username, password))
        try:
            symbols = [_provider_symbol(exchange, product_code) for product_code, exchange in normalized_products]
            quotes = _quote_list(api, symbols)
            _wait_for_quotes(api, quotes, timeout)
            return [
                _future_realtime_quote_item(
                    product_code=product_code,
                    exchange=exchange,
                    provider_symbol=provider_symbol,
                    contract_symbol=_text(getattr(quote, "underlying_symbol", None)),
                    quote_time=_quote_time(getattr(quote, "datetime", None)),
                    last_price=_optional_float(getattr(quote, "last_price", None)),
                    open=_optional_float(getattr(quote, "open", None)),
                    high=_optional_float(getattr(quote, "highest", None)),
                    low=_optional_float(getattr(quote, "lowest", None)),
                    pre_close=_optional_float(getattr(quote, "pre_close", None)),
                    volume=_optional_float(getattr(quote, "volume", None)),
                    open_interest=_optional_float(getattr(quote, "open_interest", None)),
                    bid_price1=_optional_float(getattr(quote, "bid_price1", None)),
                    ask_price1=_optional_float(getattr(quote, "ask_price1", None)),
                    trading_status=_text(getattr(quote, "trading_status", None)),
                )
                for (product_code, exchange), provider_symbol, quote in zip(normalized_products, symbols, quotes, strict=True)
            ]
        finally:
            api.close()

    return call_provider_api("shinny_tqsdk", "futures.main_continuous.realtime", _invoke)


def get_future_contract_catalog(
    products: list[tuple[str, str]] | None = None,
    include_expired: bool = False,
) -> list[FutureContractCatalogItem]:
    """Return raw TqSdk FUTURE metadata, optionally limited to requested products."""
    requested_products = set(_normalize_products(products or []))
    username, password = _require_credentials()

    def _invoke() -> list[FutureContractCatalogItem]:
        TqApi, TqAuth = _load_tqsdk()
        api = TqApi(auth=TqAuth(username, password))
        try:
            symbols = list(api.query_quotes(ins_class="FUTURE", expired=include_expired))
            metadata_frame = api.query_symbol_info(symbols)
            rows = metadata_frame.to_dict("records")
            if requested_products:
                rows = [
                    row
                    for row in rows
                    if (_text(_row_value(row, "product_id")), _text(_row_value(row, "exchange_id"))) in requested_products
                ]
            return [
                _future_contract_catalog_item(
                    provider_symbol=_text(_row_value(row, "instrument_id")),
                    contract_symbol=_text(_row_value(row, "instrument_id")),
                    product_code=_text(_row_value(row, "product_id")),
                    exchange=_text(_row_value(row, "exchange_id")),
                    name=_text(_row_value(row, "instrument_name")),
                    ins_class=_text(_row_value(row, "ins_class")),
                    underlying_symbol=_text(_row_value(row, "underlying_symbol")),
                    expired=_optional_bool(_row_value(row, "expired")),
                    expire_datetime=_metadata_datetime(row.get("expire_datetime")),
                    metadata_time=_text(_row_value(row, "datetime")),
                    last_exercise_datetime=_metadata_datetime(row.get("last_exercise_datetime")),
                    delivery_year=_optional_int(_row_value(row, "delivery_year")),
                    delivery_month=_optional_int(_row_value(row, "delivery_month")),
                    volume_multiple=_optional_float(_row_value(row, "volume_multiple")),
                    price_tick=_optional_float(_row_value(row, "price_tick")),
                    price_decs=_optional_int(_row_value(row, "price_decs")),
                    max_limit_order_volume=_optional_int(_row_value(row, "max_limit_order_volume")),
                    max_market_order_volume=_optional_int(_row_value(row, "max_market_order_volume")),
                    min_limit_order_volume=_optional_int(_row_value(row, "min_limit_order_volume")),
                    min_market_order_volume=_optional_int(_row_value(row, "min_market_order_volume")),
                    raw_metadata={str(key): _plain_value(value) for key, value in row.items()},
                )
                for row in rows
            ]
        finally:
            api.close()

    return call_provider_api("shinny_tqsdk", "futures.contracts.catalog", _invoke)


def get_future_main_contract_mapping(
    products: list[tuple[str, str]] | None = None,
) -> list[FutureMainContractMappingItem]:
    """Return current TqSdk main-contract mappings; empty products asks TqSdk for all products."""
    requested_products = _normalize_products(products or [])
    username, password = _require_credentials()
    timeout = _timeout_seconds()

    def _invoke() -> list[FutureMainContractMappingItem]:
        TqApi, TqAuth = _load_tqsdk()
        api = TqApi(auth=TqAuth(username, password))
        try:
            if requested_products:
                provider_symbols = [_provider_symbol(exchange, product_code) for product_code, exchange in requested_products]
                quotes = _quote_list(api, provider_symbols)
                _wait_for_quotes(api, quotes, timeout)
                return [
                    _future_main_contract_mapping_item(
                        product_code=product_code,
                        exchange=exchange,
                        provider_symbol=provider_symbol,
                        contract_symbol=_text(getattr(quote, "underlying_symbol", None)),
                        updated_time=_quote_time(getattr(quote, "datetime", None)),
                    )
                    for (product_code, exchange), provider_symbol, quote in zip(requested_products, provider_symbols, quotes, strict=True)
                ]

            contracts = [str(symbol) for symbol in api.query_cont_quotes()]
            quotes = _quote_list(api, contracts)
            _wait_for_quotes(api, quotes, timeout)
            return [
                _future_main_contract_mapping_item(
                    product_code=_quote_product_code(quote, provider_symbol),
                    exchange=_quote_exchange(quote, provider_symbol),
                    provider_symbol=_provider_symbol(_quote_exchange(quote, provider_symbol), _quote_product_code(quote, provider_symbol)),
                    contract_symbol=_real_contract_symbol(quote, provider_symbol),
                    updated_time=_quote_time(getattr(quote, "datetime", None)),
                )
                for provider_symbol, quote in zip(contracts, quotes, strict=True)
            ]
        finally:
            api.close()

    return call_provider_api("shinny_tqsdk", "futures.contracts.main_mapping", _invoke)


def get_future_contract_realtime_quotes(
    contract_symbols: list[str],
) -> list[FutureContractRealtimeQuoteItem]:
    """Return five-level, read-only snapshots for explicitly requested real delivery contracts."""
    symbols = list(dict.fromkeys(str(symbol).strip() for symbol in contract_symbols))
    if any(symbol == "" or symbol.startswith("KQ.m@") or "." not in symbol for symbol in symbols):
        raise ValueError("contract_symbols must contain real delivery symbols in EXCHANGE.contract format")
    if symbols == []:
        return []
    username, password = _require_credentials()
    timeout = _timeout_seconds()

    def _invoke() -> list[FutureContractRealtimeQuoteItem]:
        TqApi, TqAuth = _load_tqsdk()
        api = TqApi(auth=TqAuth(username, password))
        try:
            quotes = _quote_list(api, symbols)
            _wait_for_quotes(api, quotes, timeout)
            return [
                _future_contract_realtime_quote_item(
                    provider_symbol=provider_symbol,
                    contract_symbol=_real_contract_symbol(quote, provider_symbol),
                    product_code=_quote_product_code(quote, provider_symbol),
                    exchange=_quote_exchange(quote, provider_symbol),
                    quote_time=_quote_time(getattr(quote, "datetime", None)),
                    last_price=_optional_float(getattr(quote, "last_price", None)),
                    open=_optional_float(getattr(quote, "open", None)),
                    high=_optional_float(getattr(quote, "highest", None)),
                    low=_optional_float(getattr(quote, "lowest", None)),
                    pre_close=_optional_float(getattr(quote, "pre_close", None)),
                    pre_settlement=_optional_float(getattr(quote, "pre_settlement", None)),
                    settlement=_optional_float(getattr(quote, "settlement", None)),
                    average=_optional_float(getattr(quote, "average", None)),
                    volume=_optional_float(getattr(quote, "volume", None)),
                    amount=_optional_float(getattr(quote, "amount", None)),
                    open_interest=_optional_float(getattr(quote, "open_interest", None)),
                    **{
                        f"{side}_{field}{level}": _optional_float(getattr(quote, f"{side}_{field}{level}", None))
                        for side in ("bid", "ask")
                        for field in ("price", "volume")
                        for level in range(1, 6)
                    },
                    upper_limit=_optional_float(getattr(quote, "upper_limit", None)),
                    lower_limit=_optional_float(getattr(quote, "lower_limit", None)),
                    trading_status=_text(getattr(quote, "trading_status", None)),
                    expired=_optional_bool(getattr(quote, "expired", None)),
                )
                for provider_symbol, quote in zip(symbols, quotes, strict=True)
            ]
        finally:
            api.close()

    return call_provider_api("shinny_tqsdk", "futures.contract.realtime", _invoke)
