from __future__ import annotations

from platform_models.p0_fundamentals import (
    CompanyP0Data,
    CompanyP0Request,
    P0Page,
    P0Record,
)

from quotemux_packages.eastmoney_official.common import (
    date_text,
    optional_string,
    prefixed_security_code,
    raw_hash,
    utc_now_text,
    validate_response_identity,
)
from quotemux_packages.eastmoney_official.errors import P0ProviderError
from quotemux_packages.eastmoney_official.http import get_json
from quotemux_packages.eastmoney_official.policies import (
    COMPANY_MAX_RESPONSE_BYTES,
    CONNECT_TIMEOUT_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
)


SOURCE = "eastmoney_hsf10_company_survey"


def fetch_company(request: CompanyP0Request) -> P0Page[CompanyP0Data]:
    url = f"https://emweb.securities.eastmoney.com/PC_HSF10/CompanySurvey/PageAjax?code={request.market}{request.code}"
    payload = get_json(
        url,
        connect_timeout=CONNECT_TIMEOUT_SECONDS,
        request_timeout=REQUEST_TIMEOUT_SECONDS,
        max_bytes=COMPANY_MAX_RESPONSE_BYTES,
    )
    fetched_at = utc_now_text()
    profiles = payload.get("jbzl")
    if not isinstance(profiles, list):
        raise P0ProviderError("schema_error", "company response 缺少 jbzl 数组")
    if profiles == []:
        return P0Page[CompanyP0Data](
            capability_id=request.capability_id,
            data_version=request.data_version,
            provider=request.provider,
            source=SOURCE,
            source_version=request.source_version,
            fetched_at=fetched_at,
            confirmed_empty=True,
            next_cursor="",
            records=[],
        )
    if len(profiles) != 1 or not isinstance(profiles[0], dict):
        raise P0ProviderError(
            "schema_error", "company response 的 jbzl 必须只有一个对象"
        )
    profile = dict(profiles[0])
    validate_response_identity(profile, request)
    issue_rows = payload.get("fxxg", [])
    if not isinstance(issue_rows, list):
        raise P0ProviderError("schema_error", "company response 的 fxxg 必须是数组")
    issue = issue_rows[0] if issue_rows != [] else {}
    if not isinstance(issue, dict):
        raise P0ProviderError("schema_error", "company response 的 fxxg[0] 必须是对象")
    projection = dict(profile)
    if "LISTING_DATE" in issue:
        projection["LISTING_DATE"] = issue["LISTING_DATE"]
    if "FOUND_DATE" in issue:
        projection["FOUND_DATE"] = issue["FOUND_DATE"]
    industry_path = optional_string(profile, "EM2016")
    data = CompanyP0Data(
        code=request.code,
        market=request.market,
        security_code=prefixed_security_code(request),
        company_name=optional_string(profile, "SECURITY_NAME_ABBR"),
        company_full_name=optional_string(profile, "ORG_NAME"),
        security_type=optional_string(profile, "SECURITY_TYPE"),
        trade_market=optional_string(profile, "TRADE_MARKET"),
        industry_system="eastmoney_em2016",
        industry_code="",
        industry_name=industry_path.rsplit("-", 1)[-1] if industry_path != "" else "",
        industry_path=industry_path,
        industry_csrc_path=optional_string(profile, "INDUSTRYCSRC1"),
        listing_date=date_text(
            issue.get("LISTING_DATE"), "LISTING_DATE", required=False
        ),
        found_date=date_text(issue.get("FOUND_DATE"), "FOUND_DATE", required=False),
    )
    event_id = f"{prefixed_security_code(request)}:company_survey"
    return P0Page[CompanyP0Data](
        capability_id=request.capability_id,
        data_version=request.data_version,
        provider=request.provider,
        source=SOURCE,
        source_version=request.source_version,
        fetched_at=fetched_at,
        confirmed_empty=False,
        next_cursor="",
        records=[
            P0Record[CompanyP0Data](
                source_event_id=event_id,
                raw_hash=raw_hash(projection),
                raw_projection=projection,
                data=data,
            )
        ],
    )
