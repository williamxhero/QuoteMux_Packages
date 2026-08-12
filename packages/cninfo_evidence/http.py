from __future__ import annotations

import http.client
import json
import socket
from urllib.parse import urlsplit

from quotemux_packages.cninfo_evidence.errors import CninfoEvidenceProviderError


def get_json(
    url: str, *, connect_timeout: float, request_timeout: float, max_bytes: int
) -> dict[str, object]:
    target = urlsplit(url)
    if target.scheme not in {"http", "https"} or target.hostname is None:
        raise CninfoEvidenceProviderError(
            "contract_error", "CNInfo evidence base_url 必须是明确的 HTTP(S) 地址"
        )
    path = target.path or "/"
    if target.query:
        path = f"{path}?{target.query}"
    connection = _connection_for(target, connect_timeout)
    try:
        connection.connect()
        if connection.sock is not None:
            connection.sock.settimeout(request_timeout)
        connection.request(
            "GET",
            path,
            headers={
                "Accept": "application/json",
                "User-Agent": "QuoteMux/cninfo_evidence",
            },
        )
        response = connection.getresponse()
        if response.status == 400:
            raise CninfoEvidenceProviderError(
                "contract_error", "CNInfo evidence 拒绝了请求参数"
            )
        if response.status == 429:
            raise CninfoEvidenceProviderError(
                "rate_limit_error", "CNInfo evidence 返回 429"
            )
        if response.status in {401, 403}:
            raise CninfoEvidenceProviderError(
                "permission_error", f"CNInfo evidence 返回 {response.status}"
            )
        if response.status < 200 or response.status >= 300:
            raise CninfoEvidenceProviderError(
                "request_error", f"CNInfo evidence 返回 HTTP {response.status}"
            )
        content_length = response.getheader("Content-Length")
        if content_length is not None:
            try:
                declared_bytes = int(content_length)
            except ValueError as exc:
                raise CninfoEvidenceProviderError(
                    "schema_error", "CNInfo evidence Content-Length 非法"
                ) from exc
            if declared_bytes > max_bytes:
                raise CninfoEvidenceProviderError(
                    "contract_error", "CNInfo evidence 响应超过 max response bytes"
                )
        body = response.read(max_bytes + 1)
        if len(body) > max_bytes:
            raise CninfoEvidenceProviderError(
                "contract_error", "CNInfo evidence 响应超过 max response bytes"
            )
    except CninfoEvidenceProviderError:
        raise
    except (TimeoutError, socket.timeout) as exc:
        raise CninfoEvidenceProviderError(
            "timeout_error", "CNInfo evidence 请求超时"
        ) from exc
    except (OSError, http.client.HTTPException) as exc:
        raise CninfoEvidenceProviderError(
            "request_error", f"CNInfo evidence 请求失败: {exc}"
        ) from exc
    finally:
        connection.close()
    try:
        payload = json.loads(body.decode("utf-8"), parse_float=str)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CninfoEvidenceProviderError(
            "parse_error", "CNInfo evidence 响应不是合法 UTF-8 JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise CninfoEvidenceProviderError(
            "schema_error", "CNInfo evidence 顶层响应必须是对象"
        )
    return payload


def _connection_for(
    target, connect_timeout: float
) -> http.client.HTTPConnection:
    connection_type = (
        http.client.HTTPSConnection
        if target.scheme == "https"
        else http.client.HTTPConnection
    )
    return connection_type(target.hostname, target.port, timeout=connect_timeout)
