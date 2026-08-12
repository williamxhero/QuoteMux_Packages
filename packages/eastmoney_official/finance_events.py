from __future__ import annotations

from urllib.parse import urlencode

from platform_models.migration_contracts import (
    ExpressEventsRequest,
    FinancialEventData,
    ForecastEventsRequest,
    MigrationPage,
    MigrationRecord,
)
from quotemux_packages.eastmoney_official.errors import P0ProviderError
from quotemux_packages.eastmoney_official.http import get_json
from quotemux_packages.eastmoney_official.migration_common import (
    date_text,
    datetime_text,
    decimal_value,
    next_cursor,
    optional_string,
    raw_hash,
    request_page,
    required_string,
    utc_now_text,
)
from quotemux_packages.eastmoney_official.policies import (
    CONNECT_TIMEOUT_SECONDS,
    FINANCE_EVENTS_MAX_PAGES,
    FINANCE_EVENTS_MAX_RESPONSE_BYTES,
    FINANCE_EVENTS_PAGE_SIZE,
    REQUEST_TIMEOUT_SECONDS,
)


SOURCE = "eastmoney_datacenter_financial_events"
_REPORT = {
    "stocks.finance.forecasts": "RPT_PUBLIC_OP_NEWPREDICT",
    "stocks.finance.express": "RPT_FCI_PERFORMANCEE",
}


def query_finance_events(
    request: ForecastEventsRequest | ExpressEventsRequest,
) -> MigrationPage[FinancialEventData]:
    page = request_page(request, FINANCE_EVENTS_MAX_PAGES)
    params = {
        "sortColumns": "NOTICE_DATE",
        "sortTypes": "-1",
        "pageSize": str(FINANCE_EVENTS_PAGE_SIZE),
        "pageNumber": str(page),
        "reportName": _REPORT[request.capability_id],
        "columns": "ALL",
        "source": "WEB",
        "client": "WEB",
        "filter": f"(SECURITY_CODE='{request.code}')(REPORT_DATE>='{request.range_start}')(REPORT_DATE<='{request.range_end}')",
    }
    payload = get_json(
        "https://datacenter-web.eastmoney.com/api/data/v1/get?" + urlencode(params),
        connect_timeout=CONNECT_TIMEOUT_SECONDS,
        request_timeout=REQUEST_TIMEOUT_SECONDS,
        max_bytes=FINANCE_EVENTS_MAX_RESPONSE_BYTES,
    )
    if payload.get("success") is False:
        message = str(payload.get("message", ""))
        if "返回数据为空" in message:
            return _page(request, [], "", True)
        raise P0ProviderError("request_error", f"Eastmoney finance events failed: {message}")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise P0ProviderError("schema_error", "finance events 缺少 result 对象")
    rows = result.get("data")
    pages = result.get("pages")
    if not isinstance(rows, list):
        raise P0ProviderError("schema_error", "finance events 缺少 result.data 数组")
    if isinstance(pages, bool) or not isinstance(pages, int) or pages < 0:
        raise P0ProviderError("schema_error", "finance events 缺少合法 pages")
    if pages > FINANCE_EVENTS_MAX_PAGES:
        raise P0ProviderError("contract_error", "finance events 超过最大页")
    has_more = page < pages
    records = [_record(request, item) for item in rows]
    ids = [record.source_event_id for record in records]
    if len(ids) != len(set(ids)):
        raise P0ProviderError("contract_error", "finance events 同页事件身份重复")
    return _page(
        request,
        records,
        next_cursor(request, page + 1) if has_more else "",
        records == [] and not has_more,
    )


def _record(
    request: ForecastEventsRequest | ExpressEventsRequest, item: object
) -> MigrationRecord[FinancialEventData]:
    if not isinstance(item, dict):
        raise P0ProviderError("schema_error", "finance event item 必须是对象")
    row = dict(item)
    code = required_string(row, "SECURITY_CODE")
    if code != request.code:
        raise P0ProviderError("contract_error", "finance event code 与请求不匹配")
    report_period = date_text(row.get("REPORT_DATE"), "REPORT_DATE")
    if report_period < request.range_start or report_period > request.range_end:
        raise P0ProviderError("contract_error", "finance event report period 越界")
    raw_metric_code = optional_string(row, "PREDICT_FINANCE_CODE")
    metric_code = raw_metric_code.lstrip("0") or raw_metric_code
    metric_name = optional_string(row, "PREDICT_FINANCE")
    subtype = optional_string(row, "PREDICT_TYPE", "FORECAST_TYPE", "CHANGE_REASON", "YJBB_TYPE")
    summary = optional_string(row, "PREDICT_CONTENT", "CHANGE_REASON_EXPLAIN", "PERFORMANCE_CHANGE", "REASON", "SUMMARY")
    is_revision = any("修正" in value or "更正" in value for value in (subtype, summary))
    base_type = "forecast" if request.capability_id.endswith("forecasts") else "express"
    event_type = f"{base_type}_revision" if is_revision else base_type
    event_identity = optional_string(row, "INFO_CODE", "ART_CODE", "NOTICE_ID", "ANNOUNCEMENT_ID", "ID")
    if event_identity:
        source_event_id = f"{request.market}.{request.code}:{base_type}:{event_identity}"
    else:
        source_event_id = f"{request.market}.{request.code}:{base_type}:{report_period}:{date_text(row.get('NOTICE_DATE'), 'NOTICE_DATE', required=False)}:{raw_metric_code or subtype or 'unknown'}"
    amount_lower = decimal_value(row, "PREDICT_AMT_LOWER", "FORECAST_NETPROFIT_LOWER", "NETPROFIT_LOWER")
    amount_upper = decimal_value(row, "PREDICT_AMT_UPPER", "FORECAST_NETPROFIT_UPPER", "NETPROFIT_UPPER")
    yoy_lower = decimal_value(row, "ADD_AMP_LOWER", "PREDICT_RATIO_LOWER", "NETPROFIT_YOY_LOWER", "INCREASE_JZ_LOWER")
    yoy_upper = decimal_value(row, "ADD_AMP_UPPER", "PREDICT_RATIO_UPPER", "NETPROFIT_YOY_UPPER", "INCREASE_JZ_UPPER")
    flags: list[str] = []
    # Eastmoney 004 是归母净利润。缺码或其他指标即使金额字段同名，也不能猜成 004。
    parent = base_type == "forecast" and metric_code == "4"
    excl = base_type == "forecast" and (metric_code == "5" or "扣非" in metric_name or "扣除非经常性损益" in metric_name)
    revenue_forecast = base_type == "forecast" and (metric_code == "6" or "营业收入" in metric_name)
    if base_type == "forecast" and not parent:
        flags.append(f"non_comparable_forecast_metric:{raw_metric_code or 'unknown'}")
    values = (amount_lower, amount_upper, yoy_lower, yoy_upper)
    if base_type == "forecast" and all(value is None for value in values):
        flags.append("text_only_forecast")
    if base_type == "forecast" and not any((parent, excl, revenue_forecast)) and any(value is not None for value in values):
        flags.append(f"unmapped_forecast_metric:{raw_metric_code or 'unknown'}")
    direction = _direction(summary)
    data = FinancialEventData(
        code=request.code,
        market=request.market,
        security_code=f"{request.market}.{request.code}",
        report_period=report_period,
        notice_date=date_text(row.get("NOTICE_DATE"), "NOTICE_DATE", required=False),
        notice_time=datetime_text(row.get("NOTICE_DATE"), "NOTICE_DATE"),
        event_type=event_type,
        event_subtype=subtype,
        is_revision=is_revision,
        notice_title=optional_string(row, "NOTICE_TITLE", "ANNOUNCEMENT_TITLE", "TITLE"),
        notice_url=optional_string(row, "NOTICE_URL", "ANNOUNCEMENT_URL", "PDF_URL", "URL"),
        notice_summary=summary,
        forecast_metric_code=raw_metric_code,
        forecast_metric_name=metric_name,
        forecast_summary=summary,
        forecast_direction=direction,
        forecast_amount_lower=amount_lower,
        forecast_amount_upper=amount_upper,
        forecast_yoy_lower=yoy_lower,
        forecast_yoy_upper=yoy_upper,
        net_profit_lower=amount_lower if parent else None,
        net_profit_upper=amount_upper if parent else None,
        net_profit_yoy_lower=yoy_lower if parent else None,
        net_profit_yoy_upper=yoy_upper if parent else None,
        net_profit_excl_nonrecurring_lower=amount_lower if excl else None,
        net_profit_excl_nonrecurring_upper=amount_upper if excl else None,
        net_profit_excl_nonrecurring_yoy_lower=yoy_lower if excl else None,
        net_profit_excl_nonrecurring_yoy_upper=yoy_upper if excl else None,
        operating_revenue_lower=amount_lower if revenue_forecast else None,
        operating_revenue_upper=amount_upper if revenue_forecast else None,
        operating_revenue_yoy_lower=yoy_lower if revenue_forecast else None,
        operating_revenue_yoy_upper=yoy_upper if revenue_forecast else None,
        forecast_amount_unit=optional_string(row, "CURRENCY", "AMOUNT_UNIT", "UNIT"),
        operating_revenue=decimal_value(row, "TOTAL_OPERATE_INCOME", "OPERATE_INCOME") if base_type == "express" else None,
        operating_revenue_yoy=decimal_value(row, "YSTZ", "OPERATE_INCOME_YOY", "TOTAL_OPERATE_INCOME_YOY") if base_type == "express" else None,
        net_profit=decimal_value(row, "NETPROFIT", "NET_PROFIT") if base_type == "express" else None,
        net_profit_parent=decimal_value(row, "PARENT_NETPROFIT", "NETPROFIT_PARENT") if base_type == "express" else None,
        net_profit_yoy=decimal_value(row, "JLRTBZCL", "NETPROFIT_YOY", "PARENT_NETPROFIT_YOY") if base_type == "express" else None,
        basic_eps=decimal_value(row, "BASIC_EPS", "BASIC_EARNINGS_PER_SHARE") if base_type == "express" else None,
        bps=decimal_value(row, "PARENT_BVPS", "BPS", "BVPS", "NET_ASSET_PER_SHARE") if base_type == "express" else None,
        roe=decimal_value(row, "WEIGHTAVG_ROE", "ROE") if base_type == "express" else None,
        data_quality_flags=flags,
    )
    return MigrationRecord[FinancialEventData](
        source_event_id=source_event_id,
        raw_hash=raw_hash(row),
        raw_projection=row,
        data=data,
    )


def _page(request, records, cursor: str, confirmed_empty: bool):
    return MigrationPage[FinancialEventData](
        capability_id=request.capability_id,
        data_version=request.data_version,
        provider=request.provider,
        source=SOURCE,
        source_version=request.source_version,
        fetched_at=utc_now_text(),
        confirmed_empty=confirmed_empty,
        next_cursor=cursor,
        records=records,
    )


def _direction(summary: str) -> str:
    if "扭亏" in summary:
        return "turnaround"
    if any(text in summary for text in ("预增", "增加", "增长")):
        return "increase"
    if any(text in summary for text in ("预减", "下降", "减少")):
        return "decrease"
    if any(text in summary for text in ("亏损", "续亏")):
        return "loss"
    if "不确定" in summary:
        return "uncertain"
    return "unknown" if summary else ""
