from __future__ import annotations

from pydantic import TypeAdapter, ValidationError

from platform_models.p0_fundamentals import (
    CapitalP0Request,
    CompanyP0Request,
    P0Data,
    P0Page,
    P0Request,
    StatementsP0Request,
)
from platform_models.migration_contracts import (
    EtfProfileRequest,
    ExpressEventsRequest,
    ForecastEventsRequest,
    MigrationRequest,
)

from quotemux_packages.eastmoney_official.capital import fetch_capital
from quotemux_packages.eastmoney_official.company import fetch_company
from quotemux_packages.eastmoney_official.errors import P0ProviderError
from quotemux_packages.eastmoney_official.etf_profile import query_etf_profile
from quotemux_packages.eastmoney_official.finance_events import query_finance_events
from quotemux_packages.eastmoney_official.statements import fetch_statements


_REQUEST_ADAPTER = TypeAdapter(P0Request)
_MIGRATION_REQUEST_ADAPTER = TypeAdapter(MigrationRequest)


def query(payload: object) -> P0Page[P0Data]:
    try:
        request = _REQUEST_ADAPTER.validate_python(payload)
    except ValidationError as exc:
        raise P0ProviderError(
            "contract_error", f"P0 request 不符合 contract: {exc}"
        ) from exc
    if isinstance(request, CompanyP0Request):
        return fetch_company(request)
    if isinstance(request, CapitalP0Request):
        return fetch_capital(request)
    if isinstance(request, StatementsP0Request):
        return fetch_statements(request)
    raise P0ProviderError("contract_error", "未知 P0 capability")


def query_migration(payload: object):
    try:
        request = _MIGRATION_REQUEST_ADAPTER.validate_python(payload)
    except ValidationError as exc:
        raise P0ProviderError(
            "contract_error", f"migration request 不符合 contract: {exc}"
        ) from exc
    if isinstance(request, (ForecastEventsRequest, ExpressEventsRequest)):
        return query_finance_events(request)
    if isinstance(request, EtfProfileRequest):
        return query_etf_profile(request)
    raise P0ProviderError("contract_error", "eastmoney_official 不支持该 migration capability")
