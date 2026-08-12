from __future__ import annotations

from base64 import urlsafe_b64decode, urlsafe_b64encode
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json

from platform_models.provider_contracts import canonical_json_sha256
from quotemux_packages.eastmoney_official.errors import P0ProviderError


def utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def required_string(row: dict[str, object], name: str) -> str:
    value = row.get(name)
    if not isinstance(value, str) or value.strip() == "":
        raise P0ProviderError("schema_error", f"{name} 必须是非空字符串")
    return value.strip()


def optional_string(row: dict[str, object], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value is None:
            continue
        if not isinstance(value, str):
            raise P0ProviderError("schema_error", f"{name} 必须是字符串或 null")
        text = value.strip()
        if text and text != "-":
            return text
    return ""


def date_text(value: object, name: str, *, required: bool = True) -> str:
    if value is None or str(value).strip() == "":
        if required:
            raise P0ProviderError("schema_error", f"{name} 缺失")
        return ""
    text = str(value).strip()[:10]
    try:
        if len(text) == 8 and text.isdigit():
            return datetime.strptime(text, "%Y%m%d").strftime("%Y-%m-%d")
        return datetime.strptime(text, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError as exc:
        raise P0ProviderError("parse_error", f"{name} 不是合法日期") from exc


def datetime_text(value: object, name: str) -> str:
    if value is None or str(value).strip() == "":
        return ""
    text = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text[:19], fmt).isoformat()
        except ValueError:
            pass
    if len(text) >= 10:
        date_text(text, name)
        return f"{text[:10]}T00:00:00"
    raise P0ProviderError("parse_error", f"{name} 不是合法时间")


def decimal_value(row: dict[str, object], *names: str) -> Decimal | None:
    for name in names:
        if name not in row or row[name] in {None, "", "-"}:
            continue
        value = row[name]
        if isinstance(value, bool) or isinstance(value, float):
            raise P0ProviderError("parse_error", f"{name} 禁止 bool/float")
        text = str(value).strip().replace(",", "").replace("%", "")
        try:
            return Decimal(text)
        except InvalidOperation as exc:
            raise P0ProviderError("parse_error", f"{name} 不是合法 Decimal") from exc
    return None


def raw_hash(row: dict[str, object]) -> str:
    try:
        return canonical_json_sha256(row)
    except (TypeError, ValueError) as exc:
        raise P0ProviderError("contract_error", f"raw projection 不可 canonical: {exc}") from exc


def request_page(request: object, max_pages: int) -> int:
    cursor = str(getattr(request, "cursor", ""))
    if cursor == "":
        return 1
    try:
        decoded = urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)).decode("utf-8")
        page_text, fingerprint = decoded.split(":", 1)
        page = int(page_text)
    except (ValueError, UnicodeDecodeError) as exc:
        raise P0ProviderError("contract_error", "cursor 非法") from exc
    if fingerprint != _fingerprint(request) or page < 2 or page > max_pages:
        raise P0ProviderError("contract_error", "cursor 与请求不匹配或超界")
    return page


def next_cursor(request: object, page: int) -> str:
    return urlsafe_b64encode(f"{page}:{_fingerprint(request)}".encode("utf-8")).decode("ascii").rstrip("=")


def _fingerprint(request: object) -> str:
    payload = request.model_dump(mode="json")
    payload["cursor"] = ""
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(data).hexdigest()[:24]
