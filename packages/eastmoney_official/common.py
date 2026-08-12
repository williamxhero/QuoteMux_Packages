from __future__ import annotations

from base64 import urlsafe_b64decode, urlsafe_b64encode
from datetime import date, datetime, timezone
from hashlib import sha256
import json

from platform_models.p0_fundamentals import P0Request, canonical_json_sha256

from quotemux_packages.eastmoney_official.errors import P0ProviderError


INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1


def utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def prefixed_security_code(request: P0Request) -> str:
    return f"{request.market}.{request.code}"


def eastmoney_security_code(request: P0Request) -> str:
    return f"{request.code}.{request.market}"


def validate_response_identity(row: dict[str, object], request: P0Request) -> None:
    security_code = required_string(row, "SECURITY_CODE")
    if security_code != request.code:
        raise P0ProviderError("contract_error", "响应 SECURITY_CODE 与请求不一致")
    secucode = required_string(row, "SECUCODE")
    if secucode != eastmoney_security_code(request):
        raise P0ProviderError("contract_error", "响应 SECUCODE 与请求不一致")


def required_string(row: dict[str, object], field_name: str) -> str:
    value = row.get(field_name)
    if not isinstance(value, str):
        raise P0ProviderError("schema_error", f"{field_name} 必须是字符串")
    text = value.strip()
    if text == "":
        raise P0ProviderError("schema_error", f"{field_name} 不能为空")
    return text


def optional_string(row: dict[str, object], field_name: str) -> str:
    value = row.get(field_name)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise P0ProviderError("schema_error", f"{field_name} 必须是字符串或 null")
    return value.strip()


def date_text(value: object, field_name: str, *, required: bool) -> str:
    if value is None:
        if required:
            raise P0ProviderError("schema_error", f"{field_name} 缺失")
        return ""
    if isinstance(value, bool):
        raise P0ProviderError("parse_error", f"{field_name} 不是合法日期")
    text = str(value).strip()
    if text == "":
        if required:
            raise P0ProviderError("parse_error", f"{field_name} 不能为空")
        return ""
    compact = text[:10]
    try:
        if len(compact) == 8 and compact.isdigit():
            return datetime.strptime(compact, "%Y%m%d").strftime("%Y-%m-%d")
        return date.fromisoformat(compact).isoformat()
    except ValueError as exc:
        raise P0ProviderError("parse_error", f"{field_name} 不是合法日期") from exc


def optional_int64(row: dict[str, object], field_name: str) -> int | None:
    value = row.get(field_name)
    if value is None or value == "":
        return None
    if isinstance(value, bool) or isinstance(value, float):
        raise P0ProviderError("parse_error", f"{field_name} 必须是 int64 股数")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.strip().lstrip("-").isdigit():
        parsed = int(value.strip())
    else:
        raise P0ProviderError("parse_error", f"{field_name} 必须是 int64 股数")
    if parsed < INT64_MIN or parsed > INT64_MAX:
        raise P0ProviderError("parse_error", f"{field_name} 超出 int64 范围")
    return parsed


def raw_hash(raw_projection: dict[str, object]) -> str:
    try:
        return canonical_json_sha256(raw_projection)
    except (TypeError, ValueError) as exc:
        raise P0ProviderError(
            "contract_error", f"raw_projection 无法 canonical JSON 编码: {exc}"
        ) from exc


def request_page(request: P0Request, max_pages: int) -> int:
    if request.cursor == "":
        return 1
    try:
        decoded = urlsafe_b64decode(
            request.cursor + "=" * (-len(request.cursor) % 4)
        ).decode("utf-8")
        page_text, fingerprint = decoded.split(":", 1)
        page = int(page_text)
    except (ValueError, UnicodeDecodeError) as exc:
        raise P0ProviderError("contract_error", "cursor 非法") from exc
    if fingerprint != _request_fingerprint(request):
        raise P0ProviderError("contract_error", "cursor 与请求范围不一致")
    if page < 2 or page > max_pages:
        raise P0ProviderError("contract_error", "cursor 超出最大页")
    return page


def next_cursor(request: P0Request, page: int) -> str:
    payload = f"{page}:{_request_fingerprint(request)}".encode("utf-8")
    return urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _request_fingerprint(request: P0Request) -> str:
    payload = request.model_dump(mode="json")
    payload["cursor"] = ""
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(canonical).hexdigest()[:24]


def ensure_unique_event_ids(event_ids: list[str]) -> None:
    if len(event_ids) != len(set(event_ids)):
        raise P0ProviderError("contract_error", "同一页出现重复 source_event_id")
