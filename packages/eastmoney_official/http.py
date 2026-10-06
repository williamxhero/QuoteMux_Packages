from __future__ import annotations

import importlib
import json
import socket
import sys
from pathlib import Path
from urllib.parse import urlsplit

from quotemux_packages.eastmoney_official.errors import P0ProviderError


def _load_stdlib_http():
    local_dir = Path(__file__).resolve().parent
    original_path = sys.path[:]
    local_http = sys.modules.get("http")
    if local_http is not None and getattr(local_http, "__file__", None) == __file__:
        del sys.modules["http"]
    try:
        sys.path[:] = [
            entry
            for entry in sys.path
            if Path(entry or ".").resolve() != local_dir
        ]
        stdlib_http = importlib.import_module("http")
        importlib.import_module("http.client")
        return stdlib_http
    finally:
        sys.path[:] = original_path


http = _load_stdlib_http()


def get_json(
    url: str, *, connect_timeout: float, request_timeout: float, max_bytes: int
) -> dict[str, object]:
    target = urlsplit(url)
    if target.scheme != "https" or target.hostname is None:
        raise P0ProviderError("contract_error", "Provider 只允许 HTTPS 官方端点")
    path = target.path or "/"
    if target.query != "":
        path = f"{path}?{target.query}"
    connection = http.client.HTTPSConnection(
        target.hostname, target.port, timeout=connect_timeout
    )
    try:
        connection.connect()
        if connection.sock is not None:
            connection.sock.settimeout(request_timeout)
        connection.request(
            "GET",
            path,
            headers={
                "Accept": "application/json",
                "User-Agent": "QuoteMux/eastmoney_official",
            },
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
        content_length = response.getheader("Content-Length")
        if content_length is not None:
            try:
                declared_bytes = int(content_length)
            except ValueError as exc:
                raise P0ProviderError(
                    "schema_error", "Eastmoney Content-Length 非法"
                ) from exc
            if declared_bytes > max_bytes:
                raise P0ProviderError(
                    "contract_error", "Eastmoney 响应超过 max response bytes"
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
    try:
        text = body.decode("utf-8")
        payload = json.loads(text, parse_float=str)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise P0ProviderError(
            "parse_error", "Eastmoney 响应不是合法 UTF-8 JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise P0ProviderError("schema_error", "Eastmoney 顶层响应必须是对象")
    return payload
