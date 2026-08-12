from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import urlencode

from pydantic import ValidationError

from platform_models.p0_fundamentals import (
    P0Page,
    P0Record,
    ReportDisclosureP0Data,
    ReportDisclosuresP0Request,
    canonical_json_sha256,
)
from quotemux.infra.provider_config import get_provider_config_value
from quotemux_packages.cninfo_evidence.errors import CninfoEvidenceProviderError
from quotemux_packages.cninfo_evidence.http import get_json
from quotemux_packages.cninfo_evidence.policies import (
    CONNECT_TIMEOUT_SECONDS,
    MAX_RESPONSE_BYTES,
    REQUEST_TIMEOUT_SECONDS,
)


SOURCE = "news_crawler_cninfo_formal_report_evidence"
SOURCE_VERSION = "cninfo_disclosure/v1"
_REPORT_KIND = {
    "annual": "annual",
    "quarter1": "q1",
    "semiannual": "h1",
    "quarter3": "q3",
}


def query(payload: object) -> P0Page[ReportDisclosureP0Data]:
    try:
        request = ReportDisclosuresP0Request.model_validate(payload)
    except ValidationError as exc:
        raise CninfoEvidenceProviderError(
            "contract_error", f"CNInfo evidence request 不符合 contract: {exc}"
        ) from exc
    base_url = get_provider_config_value("base_url").rstrip("/")
    if base_url == "":
        raise CninfoEvidenceProviderError(
            "contract_error", "cninfo_evidence instance 未配置 base_url"
        )
    query_text = urlencode(
        {
            "code": request.code,
            "report_period": request.report_period,
            "document_kind": request.document_kind,
        }
    )
    response = get_json(
        f"{base_url}/api/v1/disclosures?{query_text}",
        connect_timeout=CONNECT_TIMEOUT_SECONDS,
        request_timeout=REQUEST_TIMEOUT_SECONDS,
        max_bytes=MAX_RESPONSE_BYTES,
    )
    return _build_page(request, response)


def _build_page(
    request: ReportDisclosuresP0Request, response: dict[str, object]
) -> P0Page[ReportDisclosureP0Data]:
    source_version = _required_string(response, "source_version")
    if source_version != request.source_version or source_version != SOURCE_VERSION:
        raise CninfoEvidenceProviderError(
            "contract_error", "CNInfo evidence source_version 不匹配"
        )
    evidence = response.get("evidence")
    count = response.get("count")
    if not isinstance(evidence, list):
        raise CninfoEvidenceProviderError(
            "schema_error", "CNInfo evidence 响应缺少 evidence 数组"
        )
    if isinstance(count, bool) or not isinstance(count, int) or count != len(evidence):
        raise CninfoEvidenceProviderError(
            "schema_error", "CNInfo evidence count 与 evidence 不一致"
        )
    records: list[P0Record[ReportDisclosureP0Data]] = []
    event_ids: set[str] = set()
    for item in evidence:
        if not isinstance(item, dict):
            raise CninfoEvidenceProviderError(
                "schema_error", "CNInfo evidence item 必须是对象"
            )
        row = dict(item)
        evidence_id = _required_string(row, "evidence_id")
        code = _required_string(row, "code")
        report_period = _required_string(row, "report_period")
        document_kind = _required_string(row, "document_kind")
        item_source_version = _required_string(row, "source_version")
        if (
            code != request.code
            or report_period != request.report_period
            or document_kind != request.document_kind
        ):
            raise CninfoEvidenceProviderError(
                "contract_error", "CNInfo evidence item 身份与请求不匹配"
            )
        if item_source_version != source_version:
            raise CninfoEvidenceProviderError(
                "contract_error", "CNInfo evidence item source_version 不匹配"
            )
        if evidence_id in event_ids:
            raise CninfoEvidenceProviderError(
                "contract_error", "CNInfo evidence_id 重复"
            )
        event_ids.add(evidence_id)
        published_at = _required_string(row, "published_at")
        notice_date = _published_date(published_at)
        content_hash = _required_string(row, "content_hash")
        if len(content_hash) != 64 or any(
            character not in "0123456789abcdef" for character in content_hash
        ):
            raise CninfoEvidenceProviderError(
                "schema_error", "CNInfo evidence content_hash 不是小写 SHA-256"
            )
        projection = row
        records.append(
            P0Record[ReportDisclosureP0Data](
                source_event_id=evidence_id,
                raw_hash=canonical_json_sha256(projection),
                raw_projection=projection,
                data=ReportDisclosureP0Data(
                    code=request.code,
                    market=request.market,
                    security_code=f"{request.market}.{request.code}",
                    report_period=request.report_period,
                    report_kind=_REPORT_KIND[request.document_kind],
                    notice_date=notice_date,
                    notice_title=_required_string(row, "title"),
                    article_code=evidence_id,
                    evidence_id=evidence_id,
                    published_at=published_at,
                    source_url=_required_string(row, "source_url"),
                    content_hash=content_hash,
                ),
            )
        )
    return P0Page[ReportDisclosureP0Data](
        capability_id=request.capability_id,
        data_version=request.data_version,
        provider=request.provider,
        source=SOURCE,
        source_version=source_version,
        fetched_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        confirmed_empty=records == [],
        next_cursor="",
        records=records,
    )


def _required_string(payload: dict[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or value.strip() == "":
        raise CninfoEvidenceProviderError(
            "schema_error", f"CNInfo evidence 缺少字符串字段 {name}"
        )
    return value.strip()


def _published_date(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CninfoEvidenceProviderError(
            "parse_error", "CNInfo evidence published_at 不是 ISO 时间"
        ) from exc
    if parsed.tzinfo is None:
        raise CninfoEvidenceProviderError(
            "schema_error", "CNInfo evidence published_at 必须包含时区"
        )
    return parsed.date().isoformat()
