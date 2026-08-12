from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation
from urllib.parse import urlencode

from platform_models.p0_fundamentals import (
    ORIGINAL_VALUE_UNIT,
    P0Page,
    P0Record,
    StatementP0Data,
    StatementsP0Request,
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
    REQUEST_TIMEOUT_SECONDS,
    STATEMENTS_MAX_PAGES,
    STATEMENTS_MAX_REPORT_PERIODS,
    STATEMENTS_MAX_RESPONSE_BYTES,
    STATEMENTS_PAGE_SIZE,
)


SOURCE = "eastmoney_datacenter_financial_statements"
REPORT_NAMES = {
    "balance": "RPT_DMSK_FN_BALANCE",
    "income": "RPT_DMSK_FN_INCOME",
    "cashflow": "RPT_DMSK_FN_CASHFLOW",
}


def fetch_statements(request: StatementsP0Request) -> P0Page[StatementP0Data]:
    _validate_report_period_count(request)
    page = request_page(request, STATEMENTS_MAX_PAGES)
    payload = get_json(
        _build_url(request, page),
        connect_timeout=CONNECT_TIMEOUT_SECONDS,
        request_timeout=REQUEST_TIMEOUT_SECONDS,
        max_bytes=STATEMENTS_MAX_RESPONSE_BYTES,
    )
    if payload.get("success") is False:
        message = payload.get("message")
        if isinstance(message, str) and "返回数据为空" in message:
            return _empty_page(request)
        raise P0ProviderError("request_error", f"statements response failed: {message}")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise P0ProviderError("schema_error", "statements response 缺少 result 对象")
    items = result.get("data")
    if not isinstance(items, list):
        raise P0ProviderError(
            "schema_error", "statements response 缺少 result.data 数组"
        )
    page_count = _page_count(result)
    if page_count is None and len(items) == STATEMENTS_PAGE_SIZE:
        raise P0ProviderError("schema_error", "statements 满页响应缺少 page_count")
    if page_count is not None and page_count > STATEMENTS_MAX_PAGES:
        raise P0ProviderError("contract_error", "statements 页数超过最大页")
    has_more = page_count > page if page_count is not None else False
    if has_more and page == STATEMENTS_MAX_PAGES:
        raise P0ProviderError("contract_error", "statements 达到最大页，禁止截断")
    fetched_at = utc_now_text()
    records: list[P0Record[StatementP0Data]] = []
    for item in items:
        if not isinstance(item, dict):
            raise P0ProviderError("schema_error", "statements item 必须是对象")
        row = dict(item)
        required_string(row, "SECURITY_CODE")
        _validate_identity(row, request)
        report_period = date_text(row.get("REPORT_DATE"), "REPORT_DATE", required=True)
        if report_period < request.range_start or report_period > request.range_end:
            continue
        source_event_id = f"{prefixed_security_code(request)}:{request.statement_type}:{report_period}"
        data = _statement_data(row, request, report_period)
        records.append(
            P0Record[StatementP0Data](
                source_event_id=source_event_id,
                raw_hash=raw_hash(row),
                raw_projection=row,
                data=data,
            )
        )
    ensure_unique_event_ids([record.source_event_id for record in records])
    return P0Page[StatementP0Data](
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


def _empty_page(request: StatementsP0Request) -> P0Page[StatementP0Data]:
    return P0Page[StatementP0Data](
        capability_id=request.capability_id,
        data_version=request.data_version,
        provider=request.provider,
        source=SOURCE,
        source_version=request.source_version,
        fetched_at=utc_now_text(),
        confirmed_empty=True,
        next_cursor="",
        records=[],
    )


def _build_url(request: StatementsP0Request, page: int) -> str:
    report_filter = (
        f'(SECURITY_CODE="{request.code}")'
        f"(REPORT_DATE>='{request.range_start}')"
        f"(REPORT_DATE<='{request.range_end}')"
    )
    params = urlencode(
        {
            "sortColumns": "REPORT_DATE",
            "sortTypes": "-1",
            "pageSize": str(STATEMENTS_PAGE_SIZE),
            "pageNumber": str(page),
            "reportName": REPORT_NAMES[request.statement_type],
            "columns": "ALL",
            "source": "WEB",
            "client": "WEB",
            "filter": report_filter,
        }
    )
    return f"https://datacenter-web.eastmoney.com/api/data/v1/get?{params}"


def _page_count(result: dict[str, object]) -> int | None:
    for field_name in ("pages", "pageCount"):
        parsed = _positive_int(result.get(field_name))
        if parsed is not None:
            return parsed
    total = _positive_int(result.get("count"))
    if total is None:
        total = _positive_int(result.get("total"))
    if total is None:
        total = _positive_int(result.get("totalCount"))
    page_size = _positive_int(result.get("pageSize"))
    if total is None or page_size is None:
        return None
    return (total + page_size - 1) // page_size


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _validate_identity(row: dict[str, object], request: StatementsP0Request) -> None:
    if required_string(row, "SECURITY_CODE") != request.code:
        raise P0ProviderError("contract_error", "statements SECURITY_CODE 与请求不一致")
    secucode = optional_string(row, "SECUCODE")
    if secucode != "" and secucode != f"{request.code}.{request.market}":
        raise P0ProviderError("contract_error", "statements SECUCODE 与请求不一致")


def _statement_data(
    row: dict[str, object],
    request: StatementsP0Request,
    report_period: str,
) -> StatementP0Data:
    return StatementP0Data(
        code=request.code,
        market=request.market,
        security_code=prefixed_security_code(request),
        statement_type=request.statement_type,
        report_period=report_period,
        announce_date=date_text(row.get("NOTICE_DATE"), "NOTICE_DATE", required=False),
        unit_identity=ORIGINAL_VALUE_UNIT,
        total_assets=_metric(
            row, ("TOTAL_ASSETS",), request.statement_type == "balance"
        ),
        total_liabilities=_metric(
            row, ("TOTAL_LIABILITIES",), request.statement_type == "balance"
        ),
        total_equity=_metric(row, ("TOTAL_PARENT_EQUITY", "TOTAL_EQUITY"), False),
        cash_and_equivalents=_metric(row, ("MONETARYFUNDS", "MONETARY_FUNDS"), False),
        accounts_receivable=_metric(
            row, ("ACCOUNTS_RECE", "ACCOUNTS_RECEIVABLE"), False
        ),
        inventory=_metric(row, ("INVENTORY",), False),
        operating_revenue=_metric(
            row,
            ("OPERATE_INCOME", "TOTAL_OPERATE_INCOME"),
            request.statement_type == "income",
        ),
        operating_profit=_metric(row, ("OPERATE_PROFIT",), False),
        total_profit=_metric(row, ("TOTAL_PROFIT",), False),
        net_profit=_metric(row, ("NETPROFIT", "NET_PROFIT"), False),
        net_profit_parent=_metric(row, ("PARENT_NETPROFIT", "NETPROFIT_PARENT"), False),
        basic_eps=_metric(row, ("BASIC_EPS",), False),
        net_operating_cash_flow=_metric(
            row,
            ("NETCASH_OPERATE", "NET_CASHFLOW_OPERATE"),
            request.statement_type == "cashflow",
        ),
        net_investing_cash_flow=_metric(
            row, ("NETCASH_INVEST", "NET_CASHFLOW_INVEST"), False
        ),
        net_financing_cash_flow=_metric(
            row, ("NETCASH_FINANCE", "NET_CASHFLOW_FINANCE"), False
        ),
        cash_flow_net_increase=_metric(
            row, ("CCE_ADD", "CASH_EQUIVALENTS_INCREASE"), False
        ),
    )


def _metric(
    row: dict[str, object], field_names: tuple[str, ...], required: bool
) -> Decimal | None:
    for field_name in field_names:
        value = row.get(field_name)
        if value is None or value == "":
            continue
        if isinstance(value, bool) or isinstance(value, float):
            raise P0ProviderError("parse_error", f"{field_name} 禁止 float")
        if not isinstance(value, (str, int)):
            raise P0ProviderError("parse_error", f"{field_name} 不是 Decimal 可解析值")
        try:
            parsed = Decimal(str(value).strip())
        except InvalidOperation as exc:
            raise P0ProviderError(
                "parse_error", f"{field_name} 不是 Decimal 可解析值"
            ) from exc
        if not parsed.is_finite():
            raise P0ProviderError("parse_error", f"{field_name} 必须是有限 Decimal")
        return parsed
    if required:
        raise P0ProviderError(
            "schema_error", f"缺少必需金额字段: {'/'.join(field_names)}"
        )
    return None


def _validate_report_period_count(request: StatementsP0Request) -> None:
    start = date.fromisoformat(request.range_start)
    end = date.fromisoformat(request.range_end)
    count = 0
    for year in range(start.year, end.year + 1):
        for month, day in ((3, 31), (6, 30), (9, 30), (12, 31)):
            period = date(year, month, day)
            if start <= period <= end:
                count += 1
    if count > STATEMENTS_MAX_REPORT_PERIODS:
        raise P0ProviderError("contract_error", "statements 报告期段超过 160 个季度")
