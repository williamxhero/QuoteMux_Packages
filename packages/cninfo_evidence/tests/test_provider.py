from __future__ import annotations

import json
from pathlib import Path

import pytest

from platform_models.p0_fundamentals import ReportDisclosuresP0Request
from quotemux.config_runtime.models import SourceInstanceConfig
from quotemux.source_packages.instance_context import use_source_instance
from quotemux_packages.cninfo_evidence import source
from quotemux_packages.cninfo_evidence.errors import CninfoEvidenceProviderError


def _request(
    document_kind: str = "annual", report_period: str = "2025-12-31"
) -> ReportDisclosuresP0Request:
    return ReportDisclosuresP0Request(
        capability_id="stocks.finance.report_disclosures",
        provider="cninfo_evidence",
        code="600000",
        market="SH",
        report_period=report_period,
        document_kind=document_kind,
        range_start=report_period,
        range_end=report_period,
        cursor="",
        data_version="quotemux.stocks.finance.report_disclosures.v2",
        source_version="cninfo_disclosure/v1",
    )


def _instance(base_url: str = "http://evidence.invalid:8815") -> SourceInstanceConfig:
    return SourceInstanceConfig(
        instance_id="cninfo_evidence-default",
        package_id="cninfo_evidence",
        display_name="CNInfo evidence",
        enabled=True,
        priority=1,
        timeout_seconds=None,
        config_values={"base_url": base_url},
        secret_values={},
        tags=(),
    )


def _evidence_payload(
    document_kind: str = "annual", report_period: str = "2025-12-31"
) -> dict[str, object]:
    return {
        "evidence": [
            {
                "evidence_id": "cninfo:1200000001",
                "code": "600000",
                "report_period": report_period,
                "document_kind": document_kind,
                "published_at": "2026-03-20 18:03:00+08:00",
                "title": "2025年年度报告",
                "source_url": "https://static.cninfo.com.cn/finalpage.pdf",
                "content_hash": "a" * 64,
                "source_version": "cninfo_disclosure/v1",
            }
        ],
        "count": 1,
        "source_version": "cninfo_disclosure/v1",
    }


@pytest.mark.parametrize(
    ("document_kind", "report_period", "report_kind"),
    [
        ("annual", "2025-12-31", "annual"),
        ("quarter1", "2025-03-31", "q1"),
        ("semiannual", "2025-06-30", "h1"),
        ("quarter3", "2025-09-30", "q3"),
    ],
)
def test_maps_versioned_evidence_without_pdf_or_body(
    monkeypatch: pytest.MonkeyPatch,
    document_kind: str,
    report_period: str,
    report_kind: str,
) -> None:
    payload = _evidence_payload(document_kind, report_period)
    monkeypatch.setattr(source, "get_json", lambda *args, **kwargs: payload)
    with use_source_instance(_instance()):
        page = source.query(
            _request(document_kind, report_period).model_dump(mode="json")
        )
    assert page.provider == "cninfo_evidence"
    assert page.source == "news_crawler_cninfo_formal_report_evidence"
    assert page.source_version == "cninfo_disclosure/v1"
    assert page.next_cursor == ""
    record = page.records[0]
    assert record.source_event_id == "cninfo:1200000001"
    assert record.data.report_kind == report_kind
    assert record.data.notice_date == "2026-03-20"
    assert record.data.evidence_id == record.source_event_id
    assert record.data.content_hash == "a" * 64
    assert set(record.raw_projection) == {
        "evidence_id",
        "code",
        "report_period",
        "document_kind",
        "published_at",
        "title",
        "source_url",
        "content_hash",
        "source_version",
    }
    assert "pdf" not in record.raw_projection
    assert "body" not in record.raw_projection
    assert "content" not in record.raw_projection


def test_confirmed_empty_and_exact_request_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[str] = []

    def fake_get_json(url: str, **kwargs: object) -> dict[str, object]:
        captured.append(url)
        return {"evidence": [], "count": 0, "source_version": "cninfo_disclosure/v1"}

    monkeypatch.setattr(source, "get_json", fake_get_json)
    with use_source_instance(_instance("http://example.invalid:8815/")):
        page = source.query(_request().model_dump(mode="json"))
    assert page.confirmed_empty is True
    assert page.records == []
    assert captured == [
        "http://example.invalid:8815/api/v1/disclosures?code=600000&report_period=2025-12-31&document_kind=annual"
    ]


@pytest.mark.parametrize(
    ("mutate", "kind"),
    [
        (lambda payload: payload.update(source_version="cninfo_disclosure/v2"), "contract_error"),
        (lambda payload: payload.update(count=2), "schema_error"),
        (lambda payload: payload["evidence"][0].update(code="000001"), "contract_error"),
        (lambda payload: payload["evidence"][0].update(content_hash="not-a-hash"), "schema_error"),
        (lambda payload: payload["evidence"][0].pop("source_url"), "schema_error"),
    ],
)
def test_schema_and_contract_failures_are_typed(
    monkeypatch: pytest.MonkeyPatch, mutate, kind: str
) -> None:
    payload = _evidence_payload()
    mutate(payload)
    monkeypatch.setattr(source, "get_json", lambda *args, **kwargs: payload)
    with use_source_instance(_instance()):
        with pytest.raises(CninfoEvidenceProviderError) as error:
            source.query(_request().model_dump(mode="json"))
    assert error.value.kind == kind


def test_missing_base_url_is_contract_error() -> None:
    with use_source_instance(_instance("")):
        with pytest.raises(CninfoEvidenceProviderError) as error:
            source.query(_request().model_dump(mode="json"))
    assert error.value.kind == "contract_error"


def test_request_rejects_range_and_document_kind_mismatch() -> None:
    payload = _request().model_dump(mode="json")
    payload["range_end"] = "2026-12-31"
    with pytest.raises(ValueError, match="单一 report_period"):
        ReportDisclosuresP0Request.model_validate(payload)
    payload = _request().model_dump(mode="json")
    payload["document_kind"] = "quarter1"
    with pytest.raises(ValueError, match="不匹配"):
        ReportDisclosuresP0Request.model_validate(payload)


def test_manifest_is_single_provider_and_requires_explicit_base_url() -> None:
    path = Path(__file__).resolve().parents[1] / "quotemux_package.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["package_id"] == "cninfo_evidence"
    assert manifest["supports_multi_instance"] is False
    assert manifest["handler_targets"] == {
        "query": "quotemux_packages.cninfo_evidence:query"
    }
    assert [item["capability_id"] for item in manifest["capabilities"]] == [
        "stocks.finance.report_disclosures"
    ]
    assert manifest["config_schema"] == [
        {
            "name": "base_url",
            "field_type": "string",
            "title": "Evidence API base URL",
            "description": "由 active profile 显式配置的 8815 evidence 服务地址，不含 /api/v1/disclosures。",
            "required": True,
            "default_value": "",
        }
    ]
