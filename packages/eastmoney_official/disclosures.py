from __future__ import annotations

import re

from platform_models.p0_fundamentals import (
    LegacyEastmoneyReportDisclosureData,
    LegacyEastmoneyReportDisclosuresRequest,
    P0Page,
    P0Record,
)

from quotemux_packages.eastmoney_official.common import (
    date_text,
    ensure_unique_event_ids,
    next_cursor,
    optional_string,
    prefixed_security_code,
    raw_hash,
    request_page,
    required_string,
    utc_now_text,
)
from quotemux_packages.eastmoney_official.errors import P0ProviderError
from quotemux_packages.eastmoney_official.http import get_json
from quotemux_packages.eastmoney_official.policies import (
    CONNECT_TIMEOUT_SECONDS,
    DISCLOSURES_MAX_PAGES,
    DISCLOSURES_MAX_RESPONSE_BYTES,
    DISCLOSURES_PAGE_SIZE,
    REQUEST_TIMEOUT_SECONDS,
)


SOURCE = "eastmoney_notice_security_ann"
_YEAR_PATTERN = re.compile(r"(19|20)\d{2}")


def fetch_disclosures(
    request: LegacyEastmoneyReportDisclosuresRequest,
) -> P0Page[LegacyEastmoneyReportDisclosureData]:
    page = request_page(request, DISCLOSURES_MAX_PAGES)
    url = (
        "https://np-anotice-stock.eastmoney.com/api/security/ann"
        f"?sr=-1&page_size={DISCLOSURES_PAGE_SIZE}&page_index={page}"
        f"&ann_type=A&client_source=web&stock_list={request.code}&f_node=1&s_node=0"
    )
    payload = get_json(
        url,
        connect_timeout=CONNECT_TIMEOUT_SECONDS,
        request_timeout=REQUEST_TIMEOUT_SECONDS,
        max_bytes=DISCLOSURES_MAX_RESPONSE_BYTES,
    )
    data_node = payload.get("data")
    if not isinstance(data_node, dict):
        raise P0ProviderError("schema_error", "disclosures response 缺少 data 对象")
    items = data_node.get("list")
    if not isinstance(items, list):
        raise P0ProviderError(
            "schema_error", "disclosures response 缺少 data.list 数组"
        )
    has_more = len(items) == DISCLOSURES_PAGE_SIZE
    if has_more and page == DISCLOSURES_MAX_PAGES:
        raise P0ProviderError("contract_error", "disclosures 达到最大页，禁止截断")
    fetched_at = utc_now_text()
    records: list[P0Record[LegacyEastmoneyReportDisclosureData]] = []
    for item in items:
        if not isinstance(item, dict):
            raise P0ProviderError("schema_error", "disclosures item 必须是对象")
        row = dict(item)
        article_code = required_string(row, "art_code")
        title = _title(row)
        columns = row.get("columns")
        if not isinstance(columns, list):
            raise P0ProviderError("schema_error", "disclosures columns 必须是数组")
        inferred = _infer_report(title, columns)
        if inferred is None:
            continue
        report_kind, report_period = inferred
        if report_period < request.range_start or report_period > request.range_end:
            continue
        projection = row
        records.append(
            P0Record[LegacyEastmoneyReportDisclosureData](
                source_event_id=article_code,
                raw_hash=raw_hash(projection),
                raw_projection=projection,
                data=LegacyEastmoneyReportDisclosureData(
                    code=request.code,
                    market=request.market,
                    security_code=prefixed_security_code(request),
                    report_period=report_period,
                    report_kind=report_kind,
                    notice_date=date_text(
                        row.get("notice_date"), "notice_date", required=False
                    ),
                    notice_title=title,
                    article_code=article_code,
                ),
            )
        )
    ensure_unique_event_ids([record.source_event_id for record in records])
    return P0Page[LegacyEastmoneyReportDisclosureData](
        capability_id=request.capability_id,
        data_version=request.data_version,
        provider=request.provider,
        source=SOURCE,
        source_version=request.source_version,
        fetched_at=fetched_at,
        confirmed_empty=records == [] and not has_more,
        next_cursor=next_cursor(request, page + 1) if has_more else "",
        records=records,
    )


def _title(row: dict[str, object]) -> str:
    title_ch = optional_string(row, "title_ch")
    if title_ch != "":
        return title_ch
    return required_string(row, "title")


def _infer_report(title: str, columns: list[object]) -> tuple[str, str] | None:
    column_codes: set[str] = set()
    for column in columns:
        if not isinstance(column, dict):
            raise P0ProviderError("schema_error", "disclosures column 必须是对象")
        code = optional_string(column, "column_code")
        if code != "":
            column_codes.add(code)
    normalized_title = title.replace(" ", "")
    year_match = _YEAR_PATTERN.search(normalized_title)
    if year_match is None:
        return None
    if "业绩快报" in normalized_title:
        return None
    year = int(year_match.group(0))
    if "001001001003001" in column_codes or "第一季度报告" in normalized_title:
        return "q1", f"{year:04}-03-31"
    if {
        "001001001002001",
        "001001001002002",
    } & column_codes or "半年度报告" in normalized_title:
        return "h1", f"{year:04}-06-30"
    if "001001001004001" in column_codes or "第三季度报告" in normalized_title:
        return "q3", f"{year:04}-09-30"
    if {
        "001001001001001",
        "001001001001002",
    } & column_codes or "年度报告" in normalized_title:
        return "annual", f"{year:04}-12-31"
    return None
