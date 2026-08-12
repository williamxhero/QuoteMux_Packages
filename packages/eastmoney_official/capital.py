from __future__ import annotations

from platform_models.p0_fundamentals import (
    CapitalP0Data,
    CapitalP0Request,
    P0Page,
    P0Record,
)

from quotemux_packages.eastmoney_official.common import (
    date_text,
    ensure_unique_event_ids,
    next_cursor,
    optional_int64,
    optional_string,
    prefixed_security_code,
    raw_hash,
    request_page,
    utc_now_text,
    validate_response_identity,
)
from quotemux_packages.eastmoney_official.errors import P0ProviderError
from quotemux_packages.eastmoney_official.http import get_json
from quotemux_packages.eastmoney_official.policies import (
    CAPITAL_MAX_PAGES,
    CAPITAL_MAX_RESPONSE_BYTES,
    CAPITAL_PAGE_SIZE,
    CONNECT_TIMEOUT_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
)


SOURCE = "eastmoney_hsf10_capital_structure"


def fetch_capital(request: CapitalP0Request) -> P0Page[CapitalP0Data]:
    page = request_page(request, CAPITAL_MAX_PAGES)
    url = f"https://emweb.securities.eastmoney.com/PC_HSF10/CapitalStockStructure/PageAjax?code={request.market}{request.code}"
    payload = get_json(
        url,
        connect_timeout=CONNECT_TIMEOUT_SECONDS,
        request_timeout=REQUEST_TIMEOUT_SECONDS,
        max_bytes=CAPITAL_MAX_RESPONSE_BYTES,
    )
    fetched_at = utc_now_text()
    rows = payload.get("lngbbd")
    if not isinstance(rows, list):
        raise P0ProviderError("schema_error", "capital response 缺少 lngbbd 数组")
    parsed: list[tuple[str, dict[str, object], CapitalP0Data]] = []
    for item in rows:
        if not isinstance(item, dict):
            raise P0ProviderError("schema_error", "lngbbd item 必须是对象")
        row = dict(item)
        validate_response_identity(row, request)
        change_date = date_text(row.get("END_DATE"), "END_DATE", required=True)
        if change_date < request.range_start or change_date > request.range_end:
            continue
        data = CapitalP0Data(
            code=request.code,
            market=request.market,
            security_code=prefixed_security_code(request),
            change_date=change_date,
            total_shares=optional_int64(row, "TOTAL_SHARES"),
            unlimited_shares=optional_int64(row, "UNLIMITED_SHARES"),
            free_shares=optional_int64(row, "FREE_SHARES"),
            listed_a_shares=optional_int64(row, "LISTED_A_SHARES"),
            limited_shares=optional_int64(row, "LIMITED_SHARES"),
            change_reason=optional_string(row, "CHANGE_REASON"),
        )
        parsed.append((change_date, row, data))
    parsed.sort(key=lambda item: (item[0], raw_hash(item[1])))
    if len(parsed) > CAPITAL_PAGE_SIZE * CAPITAL_MAX_PAGES:
        raise P0ProviderError("contract_error", "capital 历史超过最大页，禁止截断")
    event_ids = [
        f"{prefixed_security_code(request)}:{change_date}"
        for change_date, _, _ in parsed
    ]
    ensure_unique_event_ids(event_ids)
    start = (page - 1) * CAPITAL_PAGE_SIZE
    if page > 1 and start >= len(parsed):
        raise P0ProviderError("contract_error", "capital cursor 已超出结果范围")
    end = min(start + CAPITAL_PAGE_SIZE, len(parsed))
    has_more = end < len(parsed)
    if has_more and page == CAPITAL_MAX_PAGES:
        raise P0ProviderError("contract_error", "capital 达到最大页，禁止截断")
    records = [
        P0Record[CapitalP0Data](
            source_event_id=f"{prefixed_security_code(request)}:{change_date}",
            raw_hash=raw_hash(row),
            raw_projection=row,
            data=data,
        )
        for change_date, row, data in parsed[start:end]
    ]
    return P0Page[CapitalP0Data](
        capability_id=request.capability_id,
        data_version=request.data_version,
        provider=request.provider,
        source=SOURCE,
        source_version=request.source_version,
        fetched_at=fetched_at,
        confirmed_empty=page == 1 and records == [],
        next_cursor=next_cursor(request, page + 1) if has_more else "",
        records=records,
    )
