from __future__ import annotations

from decimal import Decimal

import pytest

from platform_models.migration_contracts import (
    EtfProfileRequest,
    ExpressEventsRequest,
    ForecastEventsRequest,
)
from quotemux_packages.eastmoney_official import etf_profile, finance_events
from quotemux_packages.eastmoney_official.errors import P0ProviderError


def _forecast(cursor: str = "") -> ForecastEventsRequest:
    return ForecastEventsRequest(
        capability_id="stocks.finance.forecasts",
        provider="eastmoney_official",
        provider_instance_id="eastmoney_official-default",
        code="600000",
        market="SH",
        range_start="2024-01-01",
        range_end="2026-12-31",
        cursor=cursor,
        data_version="quotemux.stocks.finance.forecasts.v2",
        source_version="eastmoney.datacenter.financial_forecast.v1",
    )


def _express() -> ExpressEventsRequest:
    return ExpressEventsRequest(
        capability_id="stocks.finance.express",
        provider="eastmoney_official",
        provider_instance_id="eastmoney_official-default",
        code="600000",
        market="SH",
        range_start="2024-01-01",
        range_end="2026-12-31",
        cursor="",
        data_version="quotemux.stocks.finance.express.v2",
        source_version="eastmoney.datacenter.financial_express.v1",
    )


def _etf() -> EtfProfileRequest:
    return EtfProfileRequest(
        capability_id="funds.etf.profile",
        provider="eastmoney_official",
        provider_instance_id="eastmoney_official-default",
        code="510300",
        market="SH",
        cursor="",
        data_version="quotemux.funds.etf.profile.v1",
        source_version="eastmoney.fundf10.profile.v1",
    )


def test_forecast_non_004_metric_never_populates_parent_profit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "success": True,
        "result": {
            "pages": 1,
            "data": [
                {
                    "SECURITY_CODE": "600000",
                    "REPORT_DATE": "2025-12-31",
                    "NOTICE_DATE": "2026-01-20 18:00:00",
                    "INFO_CODE": "forecast-005",
                    "PREDICT_FINANCE_CODE": "005",
                    "PREDICT_FINANCE": "扣除非经常性损益后的净利润",
                    "PREDICT_AMT_LOWER": "100.00",
                    "PREDICT_AMT_UPPER": "120.00",
                    "ADD_AMP_LOWER": "10",
                    "ADD_AMP_UPPER": "20",
                    "PREDICT_TYPE": "预增修正",
                    "PREDICT_CONTENT": "预计增加",
                    "CURRENCY": "CNY",
                }
            ],
        },
    }
    monkeypatch.setattr(finance_events, "get_json", lambda *args, **kwargs: payload)
    page = finance_events.query_finance_events(_forecast())
    data = page.records[0].data
    assert data.event_type == "forecast_revision"
    assert data.is_revision is True
    assert data.forecast_metric_code == "005"
    assert data.forecast_amount_lower == Decimal("100.00")
    assert data.net_profit_lower is None
    assert data.net_profit_excl_nonrecurring_lower == Decimal("100.00")
    assert "non_comparable_forecast_metric:005" in data.data_quality_flags
    assert page.records[0].raw_projection["PREDICT_FINANCE_CODE"] == "005"
    assert len(page.records[0].raw_hash) == 64


def test_forecast_004_and_express_fields_are_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forecast_payload = {
        "success": True,
        "result": {
            "pages": 1,
            "data": [
                {
                    "SECURITY_CODE": "600000",
                    "REPORT_DATE": "2025-12-31",
                    "NOTICE_DATE": "2026-01-20",
                    "INFO_CODE": "forecast-004",
                    "PREDICT_FINANCE_CODE": "004",
                    "PREDICT_FINANCE": "归属于上市公司股东的净利润",
                    "PREDICT_AMT_LOWER": "200",
                    "PREDICT_AMT_UPPER": "240",
                }
            ],
        },
    }
    monkeypatch.setattr(finance_events, "get_json", lambda *args, **kwargs: forecast_payload)
    forecast = finance_events.query_finance_events(_forecast()).records[0].data
    assert forecast.net_profit_lower == Decimal("200")
    assert forecast.net_profit_excl_nonrecurring_lower is None

    express_payload = {
        "success": True,
        "result": {
            "pages": 1,
            "data": [
                {
                    "SECURITY_CODE": "600000",
                    "REPORT_DATE": "2025-12-31",
                    "NOTICE_DATE": "2026-01-21",
                    "INFO_CODE": "express-1",
                    "TOTAL_OPERATE_INCOME": "1000.25",
                    "YSTZ": "5.5",
                    "NETPROFIT": "200",
                    "PARENT_NETPROFIT": "190",
                    "JLRTBZCL": "8.5",
                    "BASIC_EPS": "1.23",
                    "PARENT_BVPS": "10.5",
                    "WEIGHTAVG_ROE": "12.1",
                }
            ],
        },
    }
    monkeypatch.setattr(finance_events, "get_json", lambda *args, **kwargs: express_payload)
    express = finance_events.query_finance_events(_express()).records[0].data
    assert express.operating_revenue == Decimal("1000.25")
    assert express.net_profit_parent == Decimal("190")
    assert express.basic_eps == Decimal("1.23")
    assert express.bps == Decimal("10.5")
    assert express.roe == Decimal("12.1")


def test_finance_events_confirmed_empty_pagination_and_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        finance_events,
        "get_json",
        lambda *args, **kwargs: {"success": False, "message": "返回数据为空"},
    )
    assert finance_events.query_finance_events(_forecast()).confirmed_empty is True

    def page_payload(url: str, **kwargs: object) -> dict[str, object]:
        page = 2 if "pageNumber=2" in url else 1
        return {
            "success": True,
            "result": {
                "pages": 2,
                "data": [
                    {
                        "SECURITY_CODE": "600000",
                        "REPORT_DATE": "2025-12-31",
                        "NOTICE_DATE": "2026-01-20",
                        "INFO_CODE": f"event-{page}",
                        "PREDICT_FINANCE_CODE": "004",
                    }
                ],
            },
        }

    monkeypatch.setattr(finance_events, "get_json", page_payload)
    first = finance_events.query_finance_events(_forecast())
    second = finance_events.query_finance_events(_forecast(first.next_cursor))
    assert first.next_cursor
    assert second.next_cursor == ""
    assert first.records[0].source_event_id != second.records[0].source_event_id

    monkeypatch.setattr(
        finance_events,
        "get_json",
        lambda *args, **kwargs: {"success": True, "result": {"data": []}},
    )
    with pytest.raises(P0ProviderError) as error:
        finance_events.query_finance_events(_forecast())
    assert error.value.kind == "schema_error"


def test_etf_profile_is_not_catalog_and_preserves_found_listing_distinction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = """
    <label>成立日期：<span>2012-05-04</span></label>
    <table>
      <tr><th>基金简称</th><td>沪深300ETF</td></tr>
      <tr><th>基金全称</th><td>华泰柏瑞沪深300交易型开放式指数证券投资基金</td></tr>
      <tr><th>基金类型</th><td>指数型-股票</td></tr>
      <tr><th>上市日期</th><td>2012-05-28</td></tr>
    </table>
    """
    monkeypatch.setattr(
        "quotemux_packages.eastmoney_official.http_text.get_text",
        lambda *args, **kwargs: text,
    )
    page = etf_profile.query_etf_profile(_etf())
    data = page.records[0].data
    assert data.name == "沪深300ETF"
    assert data.full_name.startswith("华泰柏瑞")
    assert data.found_date == "2012-05-04"
    assert data.listing_date == "2012-05-28"
    assert page.records[0].source_event_id == "SH.510300:fund_detail_profile"
    assert "management" not in page.records[0].raw_projection


def test_etf_invalid_date_is_parse_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "quotemux_packages.eastmoney_official.http_text.get_text",
        lambda *args, **kwargs: "<label>成立日期：<span>2026-13-40</span></label>",
    )
    with pytest.raises(P0ProviderError) as error:
        etf_profile.query_etf_profile(_etf())
    assert error.value.kind == "parse_error"
