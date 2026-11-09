"""Native, scoped exchange observations; these are not published authority inputs."""

from __future__ import annotations

import base64
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Protocol
from urllib.parse import urlencode


PROVIDER = "exchange_official"
CONTRACT_VERSION = "exchange-official-lifecycle-v1"
CAPTURE_FORMAT = "exchange-official-stock-authority-capture-v1"
SZSE_B_SHARE_POLICY = "szse-b-share-code-range-202603-v1"
SZSE_B_SHARE_RULE_URL = "https://www.szse.cn/marketServices/technicalservice/doc/P020260306733846760075.pdf"
SZSE_B_SHARE_RULE_SHA256 = "1acbb35917105252edd5502377cc8ccfdd566d7cd62f37fc3838cf11483336c4"


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


class ExchangeCaptureError(RuntimeError):
    """The entire attempted observation failed, retaining all obtained raw pages."""

    def __init__(self, kind: str, message: str, *, receipts: tuple[PageReceipt, ...] = ()) -> None:
        super().__init__(message)
        self.kind = kind
        self.receipts = receipts
        self.artifact_dir = ""


@dataclass(frozen=True)
class CaptureLimits:
    requested_page_size: int = 200
    max_pages_per_partition: int = 200
    max_total_pages: int = 600
    total_timeout: float = 600.0
    connect_timeout: float = 5.0
    request_timeout: float = 20.0
    max_response_bytes: int = 2 * 1024 * 1024
    max_total_response_bytes: int = 64 * 1024 * 1024
    verify_first_page: bool = True

    def validate(self) -> None:
        for name, upper in (
            ("requested_page_size", 1000),
            ("max_pages_per_partition", 1000),
            ("max_total_pages", 5000),
            ("max_response_bytes", 16 * 1024 * 1024),
            ("max_total_response_bytes", 512 * 1024 * 1024),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper:
                raise ExchangeCaptureError("contract_error", f"invalid {name}: require integer 1..{upper}")
        for name, upper in (("connect_timeout", 30), ("request_timeout", 120), ("total_timeout", 3600)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= upper:
                raise ExchangeCaptureError("contract_error", f"invalid {name}: require finite seconds in (0,{upper}]")
        if self.verify_first_page is not True:
            raise ExchangeCaptureError("contract_error", "first-page verification cannot be disabled")


@dataclass(frozen=True)
class HttpRequest:
    endpoint: str
    parameters: tuple[tuple[str, str], ...]
    headers: tuple[tuple[str, str], ...]

    @property
    def url(self) -> str:
        return self.endpoint + "?" + urlencode(self.parameters)


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes
    headers: tuple[tuple[str, str], ...] = ()


class HttpTransport(Protocol):
    def __call__(self, request: HttpRequest, *, connect_timeout: float, request_timeout: float,
                 max_bytes: int, deadline_monotonic: float) -> HttpResponse: ...


@dataclass(frozen=True)
class Pagination:
    page_no: int
    page_size: int
    page_count: int
    total: int

    def to_payload(self) -> dict[str, int]:
        return {"page_no": self.page_no, "page_size": self.page_size,
                "page_count": self.page_count, "total": self.total}


@dataclass(frozen=True)
class PageReceipt:
    partition_id: str
    purpose: str
    request: HttpRequest
    started_at_utc: str
    completed_at_utc: str
    response: HttpResponse
    pagination: Pagination | None = None
    schema_fields: tuple[str, ...] = ()
    schema_sha256: str = ""
    selected_page_sha256: str = ""

    @property
    def raw_body(self) -> bytes:
        return self.response.body

    @property
    def raw_sha256(self) -> str:
        return hashlib.sha256(self.raw_body).hexdigest()

    @property
    def raw_bytes(self) -> int:
        return len(self.raw_body)

    def to_payload(self) -> dict[str, object]:
        return {
            "partition_id": self.partition_id, "purpose": self.purpose,
            "method": "GET", "endpoint": self.request.endpoint,
            "parameters": dict(self.request.parameters), "url": self.request.url,
            "request_headers": dict(self.request.headers),
            "started_at_utc": self.started_at_utc, "completed_at_utc": self.completed_at_utc,
            "http_status": self.response.status, "response_headers": [list(pair) for pair in self.response.headers],
            "raw_bytes": len(self.raw_body), "raw_sha256": self.raw_sha256,
            "raw_base64": base64.b64encode(self.raw_body).decode("ascii"),
            "pagination": self.pagination.to_payload() if self.pagination else None,
            "schema_fields": list(self.schema_fields), "schema_sha256": self.schema_sha256,
            "selected_page_sha256": self.selected_page_sha256,
        }


@dataclass(frozen=True)
class RowReference:
    partition_id: str
    page_no: int
    row_index: int
    raw_sha256: str

    def to_payload(self) -> dict[str, object]:
        return {"partition_id": self.partition_id, "page_no": self.page_no,
                "row_index": self.row_index, "raw_sha256": self.raw_sha256}


@dataclass(frozen=True)
class NormalizedAuthorityItem:
    market: str
    code: str
    name: str
    list_status: str
    listed_date: str
    delisted_date: str
    listing_board: str
    owned_fields: tuple[str, ...]
    evidence_refs: tuple[RowReference, ...]
    industry: str = ""
    area: str = ""

    def to_payload(self) -> dict[str, object]:
        return {"market": self.market, "code": self.code, "name": self.name,
                "list_status": self.list_status, "listed_date": self.listed_date,
                "delisted_date": self.delisted_date, "industry": self.industry,
                "listing_board": self.listing_board, "area": self.area,
                "owned_fields": list(self.owned_fields),
                "evidence_refs": [ref.to_payload() for ref in self.evidence_refs]}


@dataclass(frozen=True)
class PartitionReceipt:
    partition_id: str
    exchange: str
    list_status: str
    board: str
    row_count: int
    pages: tuple[PageReceipt, ...]
    normalized_sha256: str
    rejected_rows: tuple[RejectedRow, ...] = ()

    def to_payload(self) -> dict[str, object]:
        return {"partition_id": self.partition_id, "exchange": self.exchange,
                "list_status": self.list_status, "board": self.board,
                "row_count": self.row_count, "accepted_row_count": self.row_count - len(self.rejected_rows),
                "rejected_row_count": len(self.rejected_rows),
                "rejected_rows": [row.to_payload() for row in self.rejected_rows],
                "rejected_rows_sha256": canonical_sha256([row.to_payload() for row in self.rejected_rows]),
                "rejection_policy": SZSE_B_SHARE_POLICY if self.exchange == "SZSE" and self.list_status in {"P", "D"} else "reject_all_invalid_or_out_of_scope_rows",
                "normalized_sha256": self.normalized_sha256,
                "pages": [page.to_payload() for page in self.pages]}


@dataclass(frozen=True)
class RejectedRow:
    market: str
    code: str
    evidence_ref: RowReference

    def to_payload(self) -> dict[str, object]:
        return {"market": self.market, "code": self.code, "reason": "out_of_scope_szse_b_share",
                "policy_version": SZSE_B_SHARE_POLICY, "policy_source_url": SZSE_B_SHARE_RULE_URL,
                "policy_source_sha256": SZSE_B_SHARE_RULE_SHA256,
                "evidence_ref": self.evidence_ref.to_payload()}


@dataclass(frozen=True)
class ExchangeAuthorityCapture:
    exchanges: tuple[str, ...]
    capture_started_at_utc: str
    capture_completed_at_utc: str
    partitions: tuple[PartitionReceipt, ...]
    items: tuple[NormalizedAuthorityItem, ...]
    lifecycle_overlaps: tuple[tuple[str, str], ...]
    artifact_dir: str = ""
    limits: CaptureLimits = CaptureLimits()
    transport_mode: str = "caller_supplied_transport"

    @property
    def provider(self) -> str:
        return PROVIDER

    @property
    def contract_version(self) -> str:
        return CONTRACT_VERSION

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "format_version": CAPTURE_FORMAT, "provider": PROVIDER,
            "contract_version": CONTRACT_VERSION,
            "scope": {"exchanges": list(self.exchanges), "security_type": "A_SHARE", "lifecycle": ["L", "P", "D"]},
            "capture_started_at_utc": self.capture_started_at_utc,
            "capture_completed_at_utc": self.capture_completed_at_utc,
            "limits": asdict(self.limits),
            "transport_mode": self.transport_mode,
            "identity_policy": {"version": "official-exchange-stock-catalog-compatibility-v1",
                                "szse_stock_segments": ["000xxx-004xxx", "300xxx-309xxx"],
                                "szse_code_rule_url": SZSE_B_SHARE_RULE_URL,
                                "szse_code_rule_sha256": SZSE_B_SHARE_RULE_SHA256,
                                "depositary_receipts": "preserve_existing_stock_catalog_compatibility"},
            "observation_kind": "bounded_paged_observation_not_atomic_source_snapshot",
            "partitions": [part.to_payload() for part in self.partitions],
            "items": [item.to_payload() for item in self.items],
            "accounting": {"raw_row_count": sum(part.row_count for part in self.partitions),
                           "rejected_row_count": sum(len(part.rejected_rows) for part in self.partitions), "item_count": len(self.items),
                           "lifecycle_overlap_count": len(self.lifecycle_overlaps),
                           "lifecycle_overlaps": [{"market": market, "code": code, "from": "L", "to": "P"}
                                                  for market, code in self.lifecycle_overlaps]},
        }
        payload["capture_sha256"] = canonical_sha256(payload)
        return payload
