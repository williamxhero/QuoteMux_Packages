"""Bounded direct HTTPS transport; redirects, caches and retries are not used."""

from __future__ import annotations

import http.client
import socket
import queue
import threading
from collections.abc import Callable
from time import monotonic
from urllib.parse import urlsplit

from quotemux_packages.exchange_official.contracts import ExchangeCaptureError, HttpRequest, HttpResponse


class _DeadlineReader:
    """Check the absolute deadline inside http.client's framing/header loops."""

    def __init__(self, stream, connection_socket, remaining: Callable[[], float]) -> None:
        self._stream = stream
        self._socket = connection_socket
        self._remaining = remaining

    def read1(self, amount: int = -1) -> bytes:
        self._socket.settimeout(self._remaining())
        data = self._stream.read1(65536 if amount < 0 else amount)
        self._remaining()
        return data

    def read(self, amount: int = -1) -> bytes:
        chunks: list[bytes] = []
        left = amount
        while left != 0:
            chunk = self.read1(min(left, 65536) if left > 0 else 65536)
            if not chunk:
                break
            chunks.append(chunk)
            if left > 0:
                left -= len(chunk)
        return b"".join(chunks)

    def readline(self, limit: int = -1) -> bytes:
        # BufferedReader.readline can internally receive arbitrarily many slow
        # bytes while resetting a socket timeout. read1(1) leaves the absolute
        # deadline check in this loop, including status and chunk/trailer lines.
        line = bytearray()
        while limit < 0 or len(line) < limit:
            char = self.read1(1)
            if not char:
                break
            line.extend(char)
            if char == b"\n":
                break
        return bytes(line)

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def __getattr__(self, name):
        return getattr(self._stream, name)


class _DeadlineHTTPResponse(http.client.HTTPResponse):
    def __init__(self, sock, *args, remaining: Callable[[], float], **kwargs) -> None:
        super().__init__(sock, *args, **kwargs)
        self.fp = _DeadlineReader(self.fp, sock, remaining)


def _resolve(host: str, port: int, remaining: Callable[[], float]):
    # OS getaddrinfo does not honor socket timeouts. Only resolution runs in a
    # daemon: it cannot initiate TCP/HTTPS after an abandoned capture. The caller
    # has a hard bounded wait; every connect is still on the guarded caller.
    remaining()
    result: queue.Queue = queue.Queue(maxsize=1)

    def resolve() -> None:
        try:
            result.put(socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM))
        except Exception as exc:
            result.put(exc)

    threading.Thread(target=resolve, name="exchange-official-dns", daemon=True).start()
    try:
        value = result.get(timeout=remaining())
    except queue.Empty as exc:
        raise ExchangeCaptureError("timeout_error", "official exchange DNS deadline exceeded") from exc
    remaining()
    if isinstance(value, Exception):
        raise value
    return value


def _connector(deadline: float):
    def remaining() -> float:
        seconds = deadline - monotonic()
        if seconds <= 0:
            raise ExchangeCaptureError("timeout_error", "official exchange connect deadline exceeded")
        return seconds

    def connect(address, timeout=None, source_address=None):
        addresses = _resolve(address[0], address[1], remaining)
        last_error: OSError | None = None
        for family, kind, protocol, _name, endpoint in addresses:
            remaining()
            connection_socket = socket.socket(family, kind, protocol)
            try:
                connection_socket.settimeout(remaining())
                if source_address:
                    connection_socket.bind(source_address)
                connection_socket.connect(endpoint)
                connection_socket.settimeout(remaining())
                return connection_socket
            except OSError as exc:
                last_error = exc
                connection_socket.close()
            except Exception:
                connection_socket.close()
                raise
        if last_error is not None:
            raise last_error
        raise ExchangeCaptureError("transport_error", "official exchange DNS returned no stream address")

    return connect


def get_response(request: HttpRequest, *, connect_timeout: float, request_timeout: float,
                 max_bytes: int, deadline_monotonic: float) -> HttpResponse:
    target = urlsplit(request.url)
    if target.scheme != "https" or target.hostname not in {"query.sse.com.cn", "www.szse.cn"} or target.port is not None:
        raise ExchangeCaptureError("contract_error", "only the pinned official HTTPS hosts are supported")
    request_deadline = min(deadline_monotonic, monotonic() + request_timeout)

    def remaining() -> float:
        seconds = request_deadline - monotonic()
        if seconds <= 0:
            raise ExchangeCaptureError("timeout_error", "official exchange HTTP deadline exceeded")
        return seconds

    connection = http.client.HTTPSConnection(target.hostname, timeout=min(connect_timeout, remaining()))
    connection._create_connection = _connector(min(request_deadline, monotonic() + connect_timeout))
    try:
        connection.connect()
        connection.response_class = lambda *args, **kwargs: _DeadlineHTTPResponse(*args, remaining=remaining, **kwargs)

        def set_timeout() -> None:
            if connection.sock is not None:
                connection.sock.settimeout(remaining())

        set_timeout()
        connection.request("GET", target.path + "?" + target.query, headers=dict(request.headers))
        set_timeout()
        response = connection.getresponse()
        headers = tuple(response.getheaders())
        declared = response.getheader("Content-Length")
        if declared is not None:
            if not declared.isascii() or not declared.isdigit() or int(declared) > max_bytes:
                raise ExchangeCaptureError("limit_error", "invalid or oversized Content-Length")
        chunks: list[bytes] = []
        size = 0
        while True:
            set_timeout()
            chunk = response.read1(min(65536, max_bytes + 1 - size))
            remaining()
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > max_bytes:
                raise ExchangeCaptureError("limit_error", "official exchange response exceeds byte limit")
        if declared is not None and size != int(declared):
            raise ExchangeCaptureError("transport_error", "official exchange response was truncated")
        return HttpResponse(response.status, b"".join(chunks), headers)
    except ExchangeCaptureError:
        raise
    except (TimeoutError, socket.timeout) as exc:
        raise ExchangeCaptureError("timeout_error", "official exchange HTTPS request timed out") from exc
    except (OSError, http.client.HTTPException) as exc:
        raise ExchangeCaptureError("transport_error", f"official exchange HTTPS request failed: {exc}") from exc
    finally:
        connection.close()
