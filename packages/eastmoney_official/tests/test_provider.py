from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
import json
from pathlib import Path

import pytest

from platform_models.p0_fundamentals import (
    CapitalP0Request,
    CompanyP0Request,
    LegacyEastmoneyReportDisclosuresRequest,
    StatementsP0Request,
)
from quotemux_packages.eastmoney_official import query as provider_query
from quotemux_packages.eastmoney_official import (
    capital,
    company,
    disclosures,
    policies,
    statements,
)
from quotemux_packages.eastmoney_official.common import next_cursor
from quotemux_packages.eastmoney_official.errors import P0ProviderError


def _company_request(code: str = "600000", market: str = "SH") -> CompanyP0Request:
    return CompanyP0Request(
        capability_id="stocks.profile.company",
        provider="eastmoney_official",
        code=code,
        market=market,
        range_start="",
        range_end="",
        cursor="",
        data_version="quotemux.stocks.profile.company.v1",
        source_version="eastmoney.hsf10.company_survey.v1",
    )


def _capital_request(
    cursor: str = "", code: str = "600000", market: str = "SH"
) -> CapitalP0Request:
    return CapitalP0Request(
        capability_id="stocks.corporate_actions.share_changes",
        provider="eastmoney_official",
        code=code,
        market=market,
        range_start="1990-01-01",
        range_end="2030-12-31",
        cursor=cursor,
        data_version="quotemux.stocks.corporate_actions.share_changes.v1",
        source_version="eastmoney.hsf10.capital_structure.v1",
    )


def _disclosure_request(
    code: str = "600000", market: str = "SH", cursor: str = ""
) -> LegacyEastmoneyReportDisclosuresRequest:
    return LegacyEastmoneyReportDisclosuresRequest(
        capability_id="stocks.finance.report_disclosures",
        provider="eastmoney_official",
        code=code,
        market=market,
        range_start="2019-01-01",
        range_end="2026-12-31",
        cursor=cursor,
        data_version="quotemux.stocks.finance.report_disclosures.v1",
        source_version="eastmoney.notice.security_ann.v1",
    )


def _statements_request(
    statement_type: str = "balance",
    cursor: str = "",
    range_start: str = "1990-01-01",
    code: str = "600000",
    market: str = "SH",
) -> StatementsP0Request:
    return StatementsP0Request(
        capability_id="stocks.finance.statements",
        provider="eastmoney_official",
        code=code,
        market=market,
        range_start=range_start,
        range_end="2026-12-31",
        statement_type=statement_type,
        cursor=cursor,
        data_version="quotemux.stocks.finance.statements.v1",
        source_version="eastmoney.datacenter.financial_statements.v1",
    )


def test_company_fixture_and_bj_confirmed_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "jbzl": [
            {
                "SECUCODE": "600000.SH",
                "SECURITY_CODE": "600000",
                "SECURITY_NAME_ABBR": "浦发银行",
                "ORG_NAME": "上海浦东发展银行股份有限公司",
                "SECURITY_TYPE": "上交所主板A股",
                "TRADE_MARKET": "上海证券交易所",
                "EM2016": "金融-银行-股份制与城商行",
                "INDUSTRYCSRC1": "金融业-货币金融服务",
            }
        ],
        "fxxg": [
            {"LISTING_DATE": "1999-11-10 00:00:00", "FOUND_DATE": "1992-10-19 00:00:00"}
        ],
    }
    monkeypatch.setattr(company, "get_json", lambda *args, **kwargs: payload)
    page = company.fetch_company(_company_request())
    assert page.records[0].data.industry_name == "股份制与城商行"
    assert page.records[0].data.listing_date == "1999-11-10"
    assert page.records[0].source_event_id == "SH.600000:company_survey"

    monkeypatch.setattr(company, "get_json", lambda *args, **kwargs: {"jbzl": []})
    empty = company.fetch_company(_company_request("430017", "BJ"))
    assert empty.confirmed_empty is True
    assert empty.records == []


def test_single_provider_entry_dispatches_typed_company_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "jbzl": [
            {
                "SECUCODE": "600000.SH",
                "SECURITY_CODE": "600000",
                "SECURITY_NAME_ABBR": "浦发银行",
                "ORG_NAME": "上海浦东发展银行股份有限公司",
            }
        ],
        "fxxg": [],
    }
    monkeypatch.setattr(company, "get_json", lambda *args, **kwargs: payload)
    page = provider_query(_company_request().model_dump(mode="json"))
    assert page.capability_id == "stocks.profile.company"
    assert page.provider == "eastmoney_official"
    assert page.records[0].data.security_code == "SH.600000"


def test_capital_cursor_pages_are_replayable_without_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(capital, "utc_now_text", lambda: "2026-08-11T10:00:00Z")
    start = date(2000, 1, 1)
    rows = []
    for index in range(501):
        change_date = start + timedelta(days=index)
        rows.append(
            {
                "SECUCODE": "600000.SH",
                "SECURITY_CODE": "600000",
                "END_DATE": change_date.isoformat(),
                "TOTAL_SHARES": 33305838300 + index,
                "UNLIMITED_SHARES": 33305838300,
                "FREE_SHARES": 33305838300,
                "LISTED_A_SHARES": 33305838300,
                "LIMITED_SHARES": 0,
                "CHANGE_REASON": "历史变更",
            }
        )
    monkeypatch.setattr(capital, "get_json", lambda *args, **kwargs: {"lngbbd": rows})
    first = capital.fetch_capital(_capital_request())
    second = capital.fetch_capital(_capital_request(first.next_cursor))
    replay = capital.fetch_capital(_capital_request(first.next_cursor))
    assert len(first.records) == 500
    assert len(second.records) == 1
    assert second.model_dump(mode="json") == replay.model_dump(mode="json")
    assert {item.source_event_id for item in first.records}.isdisjoint(
        {item.source_event_id for item in second.records}
    )


def test_capital_rejects_float_and_max_page_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    float_row = {
        "SECUCODE": "600000.SH",
        "SECURITY_CODE": "600000",
        "END_DATE": "2025-10-27",
        "TOTAL_SHARES": 1.5,
    }
    monkeypatch.setattr(
        capital, "get_json", lambda *args, **kwargs: {"lngbbd": [float_row]}
    )
    with pytest.raises(P0ProviderError, match="int64") as error:
        capital.fetch_capital(_capital_request())
    assert error.value.kind == "parse_error"

    start = date(1990, 1, 1)
    too_many = [
        {
            "SECUCODE": "600000.SH",
            "SECURITY_CODE": "600000",
            "END_DATE": (start + timedelta(days=index)).isoformat(),
        }
        for index in range(10001)
    ]
    monkeypatch.setattr(
        capital, "get_json", lambda *args, **kwargs: {"lngbbd": too_many}
    )
    with pytest.raises(P0ProviderError, match="禁止截断") as error:
        capital.fetch_capital(_capital_request())
    assert error.value.kind == "contract_error"


def test_disclosure_classification_keeps_600781_regression_and_excludes_express(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "data": {
            "list": [
                {
                    "art_code": "AN20190801",
                    "title_ch": "上海辅仁2019年半年度报告",
                    "notice_date": "2019-08-01",
                    "columns": [{"column_code": "001001001002001"}],
                },
                {
                    "art_code": "AN20240331",
                    "title_ch": "浦发银行2023年度业绩快报",
                    "notice_date": "2024-03-31",
                    "columns": [{"column_code": "001001001001001"}],
                },
            ]
        }
    }
    monkeypatch.setattr(disclosures, "get_json", lambda *args, **kwargs: payload)
    page = disclosures.fetch_disclosures(_disclosure_request("600781"))
    assert [record.data.report_period for record in page.records] == ["2019-06-30"]
    assert page.records[0].data.report_kind == "h1"


def test_disclosure_page_20_with_more_data_is_contract_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _disclosure_request()
    page_20_request = _disclosure_request(cursor=next_cursor(request, 20))
    item = {
        "art_code": "AN20240331",
        "title_ch": "浦发银行2023年年度报告",
        "notice_date": "2024-03-31",
        "columns": [{"column_code": "001001001001001"}],
    }
    monkeypatch.setattr(
        disclosures,
        "get_json",
        lambda *args, **kwargs: {"data": {"list": [item] * 500}},
    )
    with pytest.raises(P0ProviderError, match="禁止截断") as error:
        disclosures.fetch_disclosures(page_20_request)
    assert error.value.kind == "contract_error"


def test_disclosure_cursor_pages_have_no_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def disclosure_payload(url: str, **kwargs: object) -> dict[str, object]:
        page = 2 if "page_index=2" in url else 1
        count = 1 if page == 2 else 500
        return {
            "data": {
                "list": [
                    {
                        "art_code": f"AN{page:02}{index:04}",
                        "title_ch": f"浦发银行2023年年度报告第{page:02}{index:04}版",
                        "notice_date": "2024-03-31",
                        "columns": [{"column_code": "001001001001001"}],
                    }
                    for index in range(count)
                ]
            }
        }

    monkeypatch.setattr(disclosures, "get_json", disclosure_payload)
    first = disclosures.fetch_disclosures(_disclosure_request())
    second = disclosures.fetch_disclosures(
        _disclosure_request(cursor=first.next_cursor)
    )
    assert len(first.records) == 500
    assert len(second.records) == 1
    assert {item.source_event_id for item in first.records}.isdisjoint(
        {item.source_event_id for item in second.records}
    )


def test_statements_preserve_decimal_and_raw_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = {
        "SECURITY_CODE": "600000",
        "SECUCODE": "600000.SH",
        "REPORT_DATE": "2023-12-31 00:00:00",
        "NOTICE_DATE": "2024-04-30 00:00:00",
        "TOTAL_ASSETS": "200.00",
        "TOTAL_LIABILITIES": "120.0",
    }
    monkeypatch.setattr(
        statements,
        "get_json",
        lambda *args, **kwargs: {
            "success": True,
            "result": {"pages": 1, "data": [row]},
        },
    )
    first = statements.fetch_statements(_statements_request())
    assert first.records[0].data.total_assets == Decimal("200.00")
    assert first.records[0].data.unit_identity == "eastmoney_source_original_unscaled"
    assert first.records[0].raw_projection["TOTAL_ASSETS"] == "200.00"

    revised = dict(row, TOTAL_ASSETS="201.00")
    monkeypatch.setattr(
        statements,
        "get_json",
        lambda *args, **kwargs: {
            "success": True,
            "result": {"pages": 1, "data": [revised]},
        },
    )
    second = statements.fetch_statements(_statements_request())
    assert first.records[0].source_event_id == second.records[0].source_event_id
    assert first.records[0].raw_hash != second.records[0].raw_hash


def test_statements_reject_float_missing_metadata_and_oversized_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    float_row = {
        "SECURITY_CODE": "600000",
        "SECUCODE": "600000.SH",
        "REPORT_DATE": "2023-12-31",
        "TOTAL_ASSETS": 200.0,
        "TOTAL_LIABILITIES": "120",
    }
    monkeypatch.setattr(
        statements,
        "get_json",
        lambda *args, **kwargs: {
            "success": True,
            "result": {"pages": 1, "data": [float_row]},
        },
    )
    with pytest.raises(P0ProviderError, match="禁止 float") as error:
        statements.fetch_statements(_statements_request())
    assert error.value.kind == "parse_error"

    full_page = [dict(float_row, TOTAL_ASSETS="200") for _ in range(500)]
    monkeypatch.setattr(
        statements,
        "get_json",
        lambda *args, **kwargs: {"success": True, "result": {"data": full_page}},
    )
    with pytest.raises(P0ProviderError, match="缺少 page_count") as error:
        statements.fetch_statements(_statements_request())
    assert error.value.kind == "schema_error"

    with pytest.raises(P0ProviderError, match="160") as error:
        statements.fetch_statements(_statements_request(range_start="1980-01-01"))
    assert error.value.kind == "contract_error"


def test_bj_confirmed_empty_for_capital_disclosures_and_statements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(capital, "get_json", lambda *args, **kwargs: {"lngbbd": []})
    capital_page = capital.fetch_capital(_capital_request(code="430017", market="BJ"))
    assert capital_page.confirmed_empty is True

    monkeypatch.setattr(
        disclosures, "get_json", lambda *args, **kwargs: {"data": {"list": []}}
    )
    disclosure_page = disclosures.fetch_disclosures(
        _disclosure_request(code="430017", market="BJ")
    )
    assert disclosure_page.confirmed_empty is True

    monkeypatch.setattr(
        statements,
        "get_json",
        lambda *args, **kwargs: {"success": False, "message": "返回数据为空"},
    )
    statement_page = statements.fetch_statements(
        _statements_request(code="430017", market="BJ")
    )
    assert statement_page.confirmed_empty is True


def test_statements_cursor_has_no_overlap_and_page_limit_is_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = {
        1: {
            "SECURITY_CODE": "600000",
            "SECUCODE": "600000.SH",
            "REPORT_DATE": "2023-12-31",
            "TOTAL_ASSETS": "200",
            "TOTAL_LIABILITIES": "120",
        },
        2: {
            "SECURITY_CODE": "600000",
            "SECUCODE": "600000.SH",
            "REPORT_DATE": "2022-12-31",
            "TOTAL_ASSETS": "180",
            "TOTAL_LIABILITIES": "110",
        },
    }

    def page_payload(url: str, **kwargs: object) -> dict[str, object]:
        page = 2 if "pageNumber=2" in url else 1
        return {"success": True, "result": {"pages": 2, "data": [rows[page]]}}

    monkeypatch.setattr(statements, "get_json", page_payload)
    first = statements.fetch_statements(_statements_request())
    second = statements.fetch_statements(_statements_request(cursor=first.next_cursor))
    assert first.records[0].source_event_id != second.records[0].source_event_id
    assert second.next_cursor == ""

    monkeypatch.setattr(
        statements,
        "get_json",
        lambda *args, **kwargs: {"success": True, "result": {"pages": 21, "data": []}},
    )
    with pytest.raises(P0ProviderError, match="最大页") as error:
        statements.fetch_statements(_statements_request())
    assert error.value.kind == "contract_error"


def test_manifest_is_one_entry_single_instance_and_exact_capabilities() -> None:
    manifest_path = Path(__file__).resolve().parents[1] / "quotemux_package.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["package_id"] == "eastmoney_official"
    assert manifest["supports_multi_instance"] is False
    assert manifest["handler_targets"] == {
        "query": "quotemux_packages.eastmoney_official:query",
        "query_migration": "quotemux_packages.eastmoney_official:query_migration",
    }
    assert {item["capability_id"] for item in manifest["capabilities"]} == {
        "stocks.profile.company",
        "stocks.corporate_actions.share_changes",
        "stocks.finance.statements",
        "stocks.finance.forecasts",
        "stocks.finance.express",
        "funds.etf.profile",
    }
    assert {item["handler_name"] for item in manifest["capabilities"]} == {"query", "query_migration"}


def test_provider_bounds_are_frozen() -> None:
    assert policies.CONNECT_TIMEOUT_SECONDS == 10.0
    assert policies.REQUEST_TIMEOUT_SECONDS == 60.0
    assert policies.COMPANY_MAX_RESPONSE_BYTES == 1 * 1024 * 1024
    assert policies.CAPITAL_MAX_RESPONSE_BYTES == 32 * 1024 * 1024
    assert policies.DISCLOSURES_MAX_RESPONSE_BYTES == 8 * 1024 * 1024
    assert policies.STATEMENTS_MAX_RESPONSE_BYTES == 32 * 1024 * 1024
    assert policies.CAPITAL_PAGE_SIZE == 500
    assert policies.CAPITAL_MAX_PAGES == 20
    assert policies.DISCLOSURES_PAGE_SIZE == 500
    assert policies.DISCLOSURES_MAX_PAGES == 20
    assert policies.STATEMENTS_PAGE_SIZE == 500
    assert policies.STATEMENTS_MAX_PAGES == 20
    assert policies.STATEMENTS_MAX_REPORT_PERIODS == 160
