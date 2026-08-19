from __future__ import annotations

import http.client
import socket
from urllib.parse import urlsplit

from quotemux_packages.eastmoney_official.errors import P0ProviderError


def get_text(
    url: str, *, connect_timeout: float, request_timeout: float, max_bytes: int
) -> str:
    target = urlsplit(url)
    if target.scheme != "https" or target.hostname is None:
        raise P0ProviderError("contract_error", "Provider 只允许 HTTPS 官方端点")
    connection = http.client.HTTPSConnection(
        target.hostname, target.port, timeout=connect_timeout
    )
    try:
        connection.connect()
        if connection.sock is not None:
            connection.sock.settimeout(request_timeout)
        connection.request(
            "GET",
            target.path or "/",
            headers={"User-Agent": "QuoteMux/eastmoney_official"},
        )
        response = connection.getresponse()
        if response.status == 429:
            raise P0ProviderError("rate_limit_error", "Eastmoney 返回 429")
        if response.status in {401, 403}:
            raise P0ProviderError(
                "permission_error", f"Eastmoney 返回 {response.status}"
            )
        if response.status < 200 or response.status >= 300:
            raise P0ProviderError(
                "request_error", f"Eastmoney 返回 HTTP {response.status}"
            )
        body = response.read(max_bytes + 1)
        if len(body) > max_bytes:
            raise P0ProviderError(
                "contract_error", "Eastmoney 响应超过 max response bytes"
            )
    except P0ProviderError:
        raise
    except (TimeoutError, socket.timeout) as exc:
        raise P0ProviderError("timeout_error", "Eastmoney 请求超时") from exc
    except (OSError, http.client.HTTPException) as exc:
        raise P0ProviderError("request_error", f"Eastmoney 请求失败: {exc}") from exc
    finally:
        connection.close()
    for encoding in ("utf-8", "gb18030"):
        try:
            return body.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise P0ProviderError("parse_error", "Eastmoney HTML 编码无法解析")
