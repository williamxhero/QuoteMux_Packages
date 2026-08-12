from __future__ import annotations

import pytest

from quotemux_packages.eastmoney_official import http
from quotemux_packages.eastmoney_official.errors import P0ProviderError


class _Response:
    def __init__(self, status: int, body: bytes = b"{}") -> None:
        self.status = status
        self._body = body

    def getheader(self, name: str) -> str | None:
        if name == "Content-Length":
            return str(len(self._body))
        return None

    def read(self, size: int) -> bytes:
        return self._body[:size]


class _Connection:
    def __init__(
        self, response: _Response | None = None, error: Exception | None = None
    ) -> None:
        self.response = response
        self.error = error
        self.sock = None

    def connect(self) -> None:
        if self.error is not None:
            raise self.error

    def request(self, *args: object, **kwargs: object) -> None:
        return None

    def getresponse(self) -> _Response:
        assert self.response is not None
        return self.response

    def close(self) -> None:
        return None


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (429, "rate_limit_error"),
        (401, "permission_error"),
        (403, "permission_error"),
        (500, "request_error"),
    ],
)
def test_http_status_errors_are_typed(
    monkeypatch: pytest.MonkeyPatch, status: int, kind: str
) -> None:
    monkeypatch.setattr(
        http.http.client,
        "HTTPSConnection",
        lambda *args, **kwargs: _Connection(_Response(status)),
    )
    with pytest.raises(P0ProviderError) as error:
        http.get_json(
            "https://example.com/test",
            connect_timeout=10,
            request_timeout=60,
            max_bytes=1024,
        )
    assert error.value.kind == kind


def test_timeout_parse_and_response_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        http.http.client,
        "HTTPSConnection",
        lambda *args, **kwargs: _Connection(error=TimeoutError("timeout")),
    )
    with pytest.raises(P0ProviderError) as timeout_error:
        http.get_json(
            "https://example.com/test",
            connect_timeout=10,
            request_timeout=60,
            max_bytes=1024,
        )
    assert timeout_error.value.kind == "timeout_error"

    monkeypatch.setattr(
        http.http.client,
        "HTTPSConnection",
        lambda *args, **kwargs: _Connection(_Response(200, b"not-json")),
    )
    with pytest.raises(P0ProviderError) as parse_error:
        http.get_json(
            "https://example.com/test",
            connect_timeout=10,
            request_timeout=60,
            max_bytes=1024,
        )
    assert parse_error.value.kind == "parse_error"

    monkeypatch.setattr(
        http.http.client,
        "HTTPSConnection",
        lambda *args, **kwargs: _Connection(_Response(200, b"{}" * 513)),
    )
    with pytest.raises(P0ProviderError) as size_error:
        http.get_json(
            "https://example.com/test",
            connect_timeout=10,
            request_timeout=60,
            max_bytes=1024,
        )
    assert size_error.value.kind == "contract_error"
