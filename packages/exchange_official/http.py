"""Bounded direct HTTPS transport; redirects, caches and retries are not used."""

from __future__ import annotations

import http.client
import queue
import socket
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
        self.ended_with_blank_line = False
        self.ended_with_crlf = False

    def read1(self, amount: int = -1) -> bytes:
        self._socket.settimeout(self._remaining())
        # For body reads, let http.client account for the returned bytes before
        # the outer loop checks the deadline and decides if framing is complete.
        return self._stream.read1(65536 if amount < 0 else amount)

    def read(self, amount: int = -1) -> bytes:
        chunks: list[bytes] = []
        left = amount
        while left != 0:
            chunk = self.read1(min(left, 65536) if left > 0 else 65536)
            self._remaining()
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
        self.ended_with_blank_line = False
        self.ended_with_crlf = False
        while limit < 0 or len(line) < limit:
            char = self.read1(1)
            # Let the parser consume a complete header/trailer terminator. The
            # next I/O is still checked, and the body loop checks before return.
            if char == b"\n" and line in (b"", b"\r"):
                self.ended_with_blank_line = True
                self.ended_with_crlf = line == b"\r"
                line.extend(char)
                break
            self._remaining()
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
        self.reader = self.fp = _DeadlineReader(self.fp, sock, remaining)

    def _safe_read(self, amt: int) -> bytes:
        data = super()._safe_read(amt)
        # CPython 3.12/3.13 _get_chunk_left discards these bytes after
        # read1 has exhausted a data chunk; it does not check for CRLF.
        if self.chunked and self.chunk_left == 0 and amt == 2 and data != b"\r\n":
            raise http.client.HTTPException("invalid chunk delimiter")
        return data


class _CompletedResponseTimeout(ExchangeCaptureError):
    def __init__(self, error: ExchangeCaptureError, response: HttpResponse) -> None:
        super().__init__(error.kind, str(error))
        self.response = response


def _resolve(host: str, port: int, remaining: Callable[[], float], dns_allowed: Callable[[], bool]):
    # OS getaddrinfo does not honor socket timeouts. Only resolution runs in a
    # daemon: it cannot initiate TCP/HTTPS after an abandoned capture. The wait
    # uses the remaining budget; every connect is still on the guarded caller.
    remaining()
    result: queue.Queue = queue.Queue(maxsize=1)
    abandoned = threading.Event()

    def resolve() -> None:
        try:
            if not dns_allowed() or abandoned.is_set():
                raise ExchangeCaptureError("timeout_error", "official exchange DNS deadline exceeded")
            result.put(socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM))
        except Exception as exc:
            result.put(exc)

    worker = threading.Thread(target=resolve, name="exchange-official-dns", daemon=True)
    try:
        remaining()
        worker.start()
        value = result.get(timeout=remaining())
        remaining()
        if isinstance(value, Exception):
            raise value
        return value
    except queue.Empty as exc:
        raise ExchangeCaptureError("timeout_error", "official exchange DNS deadline exceeded") from exc
    finally:
        abandoned.set()


def _connector(remaining: Callable[[], float], dns_allowed: Callable[[], bool]):
    def connect(address, timeout=None, source_address=None):
        addresses = _resolve(address[0], address[1], remaining, dns_allowed)
        last_error: OSError | None = None
        for family, kind, protocol, _name, endpoint in addresses:
            remaining()
            connection_socket = socket.socket(family, kind, protocol)
            try:
                if source_address:
                    remaining()
                    connection_socket.bind(source_address)
                connection_socket.settimeout(remaining())
                connection_socket.connect(endpoint)
                remaining()
                return connection_socket
            except OSError as exc:
                last_error = exc
                connection_socket.close()
                remaining()
            except Exception:
                connection_socket.close()
                raise
        if last_error is not None:
            raise last_error
        raise ExchangeCaptureError("transport_error", "official exchange DNS returned no stream address")

    return connect


class _DeadlineHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host: str, remaining: Callable[[], float], connect_remaining: Callable[[], float],
                 dns_allowed: Callable[[], bool]) -> None:
        super().__init__(host, timeout=connect_remaining())
        self._remaining = remaining
        self._connect_remaining = connect_remaining
        self._create_connection = _connector(connect_remaining, dns_allowed)

    def connect(self) -> None:
        self._connect_remaining()
        # Retain stdlib TCP setup and the verified HTTPS context. Splitting TLS
        # setup from the handshake lets us refresh the timeout at that boundary.
        http.client.HTTPConnection.connect(self)
        self._connect_remaining()
        self.sock = self._context.wrap_socket(self.sock, server_hostname=self.host, do_handshake_on_connect=False)
        self.sock.settimeout(self._connect_remaining())
        self.sock.do_handshake()
        self._connect_remaining()

    def send(self, data) -> None:
        # Check again after http.client has built the request bytes, just before
        # sending them (not only before request() begins formatting).
        self.sock.settimeout(self._remaining())
        super().send(data)
        self._remaining()


def get_response(request: HttpRequest, *, connect_timeout: float, request_timeout: float,
                 max_bytes: int, deadline_monotonic: float) -> HttpResponse:
    """Compatibility entry point for the four-keyword transport contract."""
    started = monotonic()
    request_deadline = min(deadline_monotonic, started + request_timeout)
    connect_deadline = min(request_deadline, started + connect_timeout)

    def remaining() -> float:
        seconds = request_deadline - monotonic()
        if seconds <= 0:
            raise ExchangeCaptureError("timeout_error", "official exchange HTTP deadline exceeded")
        return seconds

    def connect_remaining() -> float:
        seconds = connect_deadline - monotonic()
        if seconds <= 0:
            raise ExchangeCaptureError("timeout_error", "official exchange connect deadline exceeded")
        return seconds

    def dns_allowed() -> bool:
        return started <= monotonic() < connect_deadline

    return _get_response(request, max_bytes=max_bytes, remaining=remaining, connect_remaining=connect_remaining,
                         dns_allowed=dns_allowed)


def _get_response(request: HttpRequest, *, max_bytes: int, remaining: Callable[[], float],
                  connect_remaining: Callable[[], float], dns_allowed: Callable[[], bool]) -> HttpResponse:
    connect_remaining()
    target = urlsplit(request.url)
    if target.scheme != "https" or target.hostname not in {"query.sse.com.cn", "www.szse.cn"} or target.port is not None:
        raise ExchangeCaptureError("contract_error", "only the pinned official HTTPS hosts are supported")
    connection = _DeadlineHTTPSConnection(target.hostname, remaining, connect_remaining, dns_allowed)
    response = None
    try:
        connection.connect()
        connection.response_class = lambda *args, **kwargs: _DeadlineHTTPResponse(*args, remaining=remaining, **kwargs)

        def set_timeout() -> None:
            seconds = remaining()
            if connection.sock is not None:
                connection.sock.settimeout(seconds)

        set_timeout()
        connection.request("GET", target.path + "?" + target.query, headers=dict(request.headers))
        set_timeout()
        response = connection.getresponse()
        headers_terminated = response.reader.ended_with_blank_line
        headers = tuple(response.getheaders())
        declared = response.getheader("Content-Length")
        length = None
        if declared is not None:
            if declared.isascii() and declared.isdigit():
                length = int(declared)
            if length is None or length > max_bytes:
                remaining()
                raise ExchangeCaptureError("limit_error", "invalid or oversized Content-Length")
        transfer_encoding = response.getheader("Transfer-Encoding")
        chunks: list[bytes] = []
        size = 0
        complete = not response.chunked and response.length == 0
        deadline_error = None
        try:
            remaining()
            while not complete:
                chunk = response.read1(min(65536, max_bytes + 1 - size))
                if chunk:
                    chunks.append(chunk)
                    size += len(chunk)
                complete = not chunk or (not response.chunked and response.length == 0)
                remaining()
                if size > max_bytes:
                    raise ExchangeCaptureError("limit_error", "official exchange response exceeds byte limit")
        except ExchangeCaptureError as exc:
            if exc.kind != "timeout_error" or not complete or size > max_bytes:
                raise
            deadline_error = exc
        # Do not pass unchecked framing to capture even if the read finished
        # in budget: cleanup/handoff can make the response late afterwards.
        if response.chunked:
            framing_verified = length is None and response.reader.ended_with_crlf
        else:
            framing_verified = transfer_encoding is None and (length is None or size == length)
        if not headers_terminated or not framing_verified:
            if deadline_error is not None:
                raise deadline_error
            if length is not None and size != length:
                raise ExchangeCaptureError("transport_error", "official exchange response was truncated")
            raise ExchangeCaptureError("transport_error", "official exchange response framing was incomplete or invalid")
        original = HttpResponse(response.status, b"".join(chunks), headers)
        if deadline_error is not None:
            raise _CompletedResponseTimeout(deadline_error, original) from deadline_error
        return original
    except ExchangeCaptureError:
        raise
    except (TimeoutError, socket.timeout) as exc:
        raise ExchangeCaptureError("timeout_error", "official exchange HTTPS request timed out") from exc
    except (OSError, http.client.HTTPException) as exc:
        raise ExchangeCaptureError("transport_error", f"official exchange HTTPS request failed: {exc}") from exc
    finally:
        try:
            # For Connection: close, http.client has detached the response from
            # the connection; close its file even when framing is incomplete.
            if response is not None:
                response.close()
        finally:
            connection.close()
