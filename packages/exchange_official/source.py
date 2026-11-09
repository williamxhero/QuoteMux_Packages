"""Capture complete, explicit SHSE/SZSE lifecycle observations without local cache."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from time import monotonic, sleep
from uuid import uuid4

from quotemux_packages.exchange_official.contracts import (
    CONTRACT_VERSION, PROVIDER, CaptureLimits, ExchangeAuthorityCapture, ExchangeCaptureError,
    HttpRequest, HttpResponse, HttpTransport, NormalizedAuthorityItem, PageReceipt,
    Pagination, PartitionReceipt, RejectedRow, RowReference, canonical_sha256,
)


_SSE_SQL = "COMMON_SSE_CP_GPJCTPZ_GPLB_GP_L"
_SSE_FIELDS = frozenset({"A_STOCK_CODE", "COMPANY_CODE", "SEC_NAME_CN", "COMPANY_ABBR",
                         "LIST_DATE", "DELIST_DATE", "STOCK_TYPE", "LIST_BOARD", "STATE_CODE", "STATE_CODE_STOCK"})
_SZSE_FIELDS = {"L": frozenset({"agdm", "agjc", "agssrq", "bk"}),
                "P": frozenset({"zqdm", "zqjc", "ssrq", "ztrq"}),
                "D": frozenset({"zqdm", "zqjc", "ssrq", "zzrq"})}
_OWNED_IDENTITY_FIELDS = ("code", "delisted_date", "list_status", "listed_date", "market", "name")
# Only the current observation's trading date uses this clock. Official listing
# dates remain date values; no historical timezone/DST conversion is performed.
_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class _Partition:
    exchange: str
    status: str
    board: str = ""

    @property
    def key(self) -> str:
        return f"{self.exchange}:{self.status}" + (f":{self.board}" if self.board else "")

    def request(self, page: int, size: int) -> HttpRequest:
        if self.exchange == "SHSE":
            endpoint = "https://query.sse.com.cn/" + ("sseQuery/" if self.status == "L" else "") + "commonQuery.do"
            params = (("sqlId", _SSE_SQL), ("COMPANY_STATUS", {"L": "2,4,5,7,8", "P": "5", "D": "3"}[self.status]),
                      ("STOCK_TYPE", self.board), ("REG_PROVINCE", ""), ("CSRC_CODE", ""), ("STOCK_CODE", ""),
                      ("type", "inParams"), ("isPagination", "true"), ("pageHelp.cacheSize", "1"),
                      ("pageHelp.beginPage", str(page)), ("pageHelp.endPage", str(page)),
                      ("pageHelp.pageSize", str(size)), ("pageHelp.pageNo", str(page)))
            referer = "https://www.sse.com.cn/assortment/stock/list/" + ("share/" if self.status == "L" else "delisting/")
        else:
            tab = "tab2" if self.status == "D" else "tab1"
            endpoint = "https://www.szse.cn/api/report/ShowReport/data"
            params = (("SHOWTYPE", "JSON"), ("CATALOGID", "1110" if self.status == "L" else "1793_ssgs"),
                      ("TABKEY", tab), ("PAGENO", str(page)), (tab + "PAGESIZE", str(size)))
            referer = "https://www.szse.cn/market/product/stock/list/index.html"
        return HttpRequest(endpoint, params, (("Accept", "application/json"),
                           ("Accept-Encoding", "identity"), ("User-Agent", "QuoteMux/exchange_official"), ("Referer", referer)))


@dataclass(frozen=True)
class _Page:
    rows: tuple[dict[str, object], ...]
    pagination: Pagination
    schema_fields: tuple[str, ...]
    schema_sha256: str
    selected_sha256: str


@dataclass
class _CapturedPartition:
    definition: _Partition
    first_page: _Page
    rows: list[NormalizedAuthorityItem]
    states: dict[str, str]
    receipts: list[PageReceipt]
    rejected_rows: list[RejectedRow]


def _scope(exchanges: tuple[str, ...]) -> tuple[str, ...]:
    if (not isinstance(exchanges, tuple) or not exchanges
            or any(not isinstance(value, str) or value not in {"SHSE", "SZSE"} for value in exchanges)
            or tuple(sorted(set(exchanges))) != exchanges):
        raise ExchangeCaptureError("contract_error", "exchanges must be a nonempty unique sorted tuple containing only SHSE/SZSE; global/BJSE are unsupported")
    return exchanges


def capture_stock_authority(exchanges: tuple[str, ...], *, limits: CaptureLimits = CaptureLimits(),
                            output_dir: Path | None = None, transport: HttpTransport | None = None) -> ExchangeAuthorityCapture:
    """Observe all L/P/D pages in an explicit supported scope, or fail the batch.

    No calendar fresh-through or authority input ID is invented here. The caller
    binds this native receipt into v2 authority only after independent acceptance.
    A caller-provided transport is also subject to the strict-public-read guard.
    """
    scope = _scope(exchanges)
    if not isinstance(limits, CaptureLimits):
        raise ExchangeCaptureError("contract_error", "limits must be CaptureLimits")
    limits.validate()
    if output_dir is not None and not isinstance(output_dir, Path):
        raise ExchangeCaptureError("contract_error", "output_dir must be a pathlib.Path or None")
    from quotemux.infra.provider_runtime.core import call_provider_api

    return call_provider_api(PROVIDER, "capture_stock_authority", _capture, scope, limits, output_dir, transport)


class _Attempt:
    def __init__(self, limits: CaptureLimits, output_dir: Path | None, transport: HttpTransport | None) -> None:
        self.limits = limits
        self.deadline = monotonic() + limits.total_timeout
        self.started = datetime.now(UTC).isoformat()
        self.receipts: list[PageReceipt] = []
        self.byte_count = 0
        self.last_request_started_at: float | None = None
        self.artifact_dir: Path | None = None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            self.artifact_dir = output_dir / (datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex)
            self.artifact_dir.mkdir(mode=0o700)
            (self.artifact_dir / "raw").mkdir(mode=0o700)
        if transport is None:
            from quotemux_packages.exchange_official.http import get_response
            transport = get_response
        self.transport = transport

    def remaining(self) -> float:
        seconds = self.deadline - monotonic()
        if seconds <= 0:
            raise ExchangeCaptureError("timeout_error", "total monotonic capture deadline exceeded")
        return seconds

    def fetch(self, partition: _Partition, page: int, purpose: str = "page") -> _Page:
        self.remaining()
        if len(self.receipts) >= self.limits.max_total_pages:
            raise ExchangeCaptureError("limit_error", "total page request limit exceeded")
        request = partition.request(page, self.limits.requested_page_size)
        self._wait_for_request_interval()
        started = datetime.now(UTC).isoformat()
        dispatch_at = monotonic()
        remaining = self.deadline - dispatch_at
        if remaining <= 0:
            raise ExchangeCaptureError("timeout_error", "total monotonic capture deadline exceeded before dispatch")
        self.last_request_started_at = dispatch_at
        response = self.transport(request, connect_timeout=min(self.limits.connect_timeout, remaining),
                                  request_timeout=min(self.limits.request_timeout, remaining),
                                  max_bytes=self.limits.max_response_bytes, deadline_monotonic=self.deadline)
        completed = datetime.now(UTC).isoformat()
        if not isinstance(response, HttpResponse) or not isinstance(response.body, bytes):
            raise ExchangeCaptureError("transport_error", "transport must return HttpResponse with original bytes")
        receipt = PageReceipt(partition.key, purpose, request, started, completed, response)
        self.receipts.append(receipt)
        self.byte_count += len(response.body)
        if len(response.body) > self.limits.max_response_bytes or self.byte_count > self.limits.max_total_response_bytes:
            raise ExchangeCaptureError("limit_error", "capture response byte limit exceeded")
        if self.artifact_dir is not None:
            _write_bytes(self.artifact_dir / "raw" / f"{len(self.receipts):04d}.raw", response.body)
        self.remaining()
        if isinstance(response.status, bool) or response.status != 200:
            raise ExchangeCaptureError("transport_error", f"official exchange returned HTTP {response.status}")
        parsed = _parse_page(partition, page, response.body)
        self.receipts[-1] = replace(receipt, pagination=parsed.pagination, schema_fields=parsed.schema_fields,
                                    schema_sha256=parsed.schema_sha256, selected_page_sha256=parsed.selected_sha256)
        self.remaining()
        return parsed

    def _wait_for_request_interval(self) -> None:
        if self.last_request_started_at is None:
            self.remaining()
            return
        target = self.last_request_started_at + self.limits.min_request_interval_seconds
        if target >= self.deadline:
            raise ExchangeCaptureError("timeout_error", "insufficient total capture budget for the required request interval")
        while True:
            remaining = self.remaining()
            delay = target - monotonic()
            if delay <= 0:
                return
            if delay >= remaining:
                raise ExchangeCaptureError("timeout_error", "insufficient total capture budget for the required request interval")
            # The interval is capped at10s. Check the same total deadline before
            # and after every small wait, including early wakes/OS oversleep.
            sleep(min(delay, 1.0))
            self.remaining()

    def finish_manifest(self, payload: dict[str, object]) -> None:
        if self.artifact_dir is not None:
            body = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
            _write_bytes(self.artifact_dir / "manifest.json", body)


def _write_bytes(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _capture(exchanges: tuple[str, ...], limits: CaptureLimits, output_dir: Path | None,
             transport: HttpTransport | None) -> ExchangeAuthorityCapture:
    attempt = _Attempt(limits, output_dir, transport)
    try:
        definitions = [_Partition(exchange, status, board)
                       for exchange in exchanges for status in ("L", "P", "D")
                       for board in (("1", "8") if exchange == "SHSE" else ("",))]
        captured = [_capture_partition(attempt, definition) for definition in definitions]
        # Recheck every selected first page after all partitions, not merely before
        # later partitions can change. This still is not an atomic source snapshot.
        for current in captured:
            verified = attempt.fetch(current.definition, 1, "verification")
            current.receipts.append(attempt.receipts[-1])
            if verified.selected_sha256 != current.first_page.selected_sha256:
                raise ExchangeCaptureError("coherence_error", f"first page changed during observation: {current.definition.key}")
        items, overlaps = _combine(captured, exchanges)
        partitions = tuple(PartitionReceipt(current.definition.key, current.definition.exchange,
                                             current.definition.status, current.definition.board,
                                             len(current.rows) + len(current.rejected_rows), tuple(current.receipts),
                                             canonical_sha256([item.to_payload() for item in current.rows]), tuple(current.rejected_rows))
                           for current in captured)
        attempt.remaining()
        result = ExchangeAuthorityCapture(exchanges, attempt.started, datetime.now(UTC).isoformat(),
                                          partitions, items, overlaps, str(attempt.artifact_dir or ""), limits,
                                          "direct_official_https" if transport is None else "caller_supplied_transport")
        if attempt.artifact_dir is not None:
            manifest = result.to_payload()
            manifest["result"] = "complete_scoped_native_observation"
            # The native payload includes raw bytes. The disk manifest references
            # the already-fsynced exact bytes instead of duplicating them inline.
            raw_index = {id(receipt): index for index, receipt in enumerate(attempt.receipts, 1)}
            for part, native in zip(result.partitions, manifest["partitions"], strict=True):
                for receipt, page_payload in zip(part.pages, native["pages"], strict=True):
                    page_payload.pop("raw_base64")
                    page_payload["raw_file"] = f"raw/{raw_index[id(receipt)]:04d}.raw"
            manifest["manifest_sha256"] = canonical_sha256(manifest)
            attempt.finish_manifest(manifest)
        return result
    except Exception as exc:
        error = exc if isinstance(exc, ExchangeCaptureError) else ExchangeCaptureError("capture_error", f"official exchange capture failed: {exc}")
        error.receipts = tuple(attempt.receipts)
        error.artifact_dir = str(attempt.artifact_dir or "")
        if attempt.artifact_dir is not None and not (attempt.artifact_dir / "manifest.json").exists():
            pages = []
            for index, receipt in enumerate(attempt.receipts, 1):
                native = receipt.to_payload()
                native.pop("raw_base64")
                relative = f"raw/{index:04d}.raw"
                native["raw_file"] = relative if (attempt.artifact_dir / relative).is_file() else None
                pages.append(native)
            attempt.finish_manifest({"result": "failed", "provider": PROVIDER, "contract_version": CONTRACT_VERSION,
                                     "scope": {"exchanges": list(exchanges)}, "kind": error.kind, "error": str(error),
                                     "transport_mode": "direct_official_https" if transport is None else "caller_supplied_transport",
                                     "limits": asdict(limits),
                                     "capture_started_at_utc": attempt.started,
                                     "capture_completed_at_utc": datetime.now(UTC).isoformat(), "pages": pages})
        if error is exc:
            raise
        raise error from exc


def _capture_partition(attempt: _Attempt, definition: _Partition) -> _CapturedPartition:
    first = attempt.fetch(definition, 1)
    if first.pagination.page_count > attempt.limits.max_pages_per_partition:
        raise ExchangeCaptureError("limit_error", f"partition page count exceeds limit: {definition.key}")
    captured = _CapturedPartition(definition, first, [], {}, [], [])
    seen: set[str] = set()
    for page_no in range(1, max(1, first.pagination.page_count) + 1):
        page = first if page_no == 1 else attempt.fetch(definition, page_no)
        if (page.pagination.page_size, page.pagination.page_count, page.pagination.total, page.schema_sha256) != (
                first.pagination.page_size, first.pagination.page_count, first.pagination.total, first.schema_sha256):
            raise ExchangeCaptureError("coherence_error", f"partition pagination/schema changed: {definition.key}")
        receipt = attempt.receipts[-1]
        captured.receipts.append(receipt)
        for index, row in enumerate(page.rows):
            raw_code = row.get("zqdm")
            is_b_share = (definition.exchange == "SZSE" and definition.status in {"P", "D"}
                          and isinstance(raw_code, str) and re.fullmatch(r"20[0-9]{4}", raw_code, flags=re.ASCII) is not None)
            item = _normalize_row(definition, row, receipt, index, allow_b_share=is_b_share)
            if item.code in seen:
                raise ExchangeCaptureError("identity_error", f"duplicate security within partition {definition.key}: {item.code}")
            seen.add(item.code)
            if is_b_share:
                captured.rejected_rows.append(RejectedRow(item.market, item.code, item.evidence_refs[0]))
                continue
            captured.rows.append(item)
            captured.states[item.code] = str(row.get("STATE_CODE", ""))
    if len(captured.rows) + len(captured.rejected_rows) != first.pagination.total:
        raise ExchangeCaptureError("coherence_error", f"partition raw row accounting mismatch: {definition.key}")
    return captured


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ExchangeCaptureError("schema_error", f"duplicate JSON object key: {key}")
        output[key] = value
    return output


def _parse_page(definition: _Partition, requested_page: int, body: bytes) -> _Page:
    try:
        payload = json.loads(body.decode("utf-8"), object_pairs_hook=_json_object,
                             parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid JSON constant: {value}")))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ExchangeCaptureError("parse_error", "official response is not strict UTF-8 JSON") from exc
    if definition.exchange == "SHSE":
        if (not isinstance(payload, dict) or payload.get("sqlId") != _SSE_SQL
                or payload.get("type") != "inParams" or payload.get("isPagination") != "true"
                or payload.get("actionErrors") != [] or payload.get("actionMessages") != []
                or payload.get("fieldErrors") != {}):
            raise ExchangeCaptureError("schema_error", "SSE envelope/selector/error contract mismatch")
        metadata = payload.get("pageHelp")
        rows = payload.get("result")
        if (not isinstance(metadata, dict) or not isinstance(rows, list)
                or ("data" in metadata and metadata["data"] != rows)
                or (rows == [] and metadata.get("data") != [])):
            raise ExchangeCaptureError("schema_error", "SSE result/pageHelp.data contract mismatch")
        pagination = _pagination(metadata, ("pageNo", "pageSize", "pageCount", "total"))
        schema: object = sorted(rows[0]) if rows and isinstance(rows[0], dict) else []
        required = _SSE_FIELDS
    else:
        if not isinstance(payload, list) or any(not isinstance(report, dict) for report in payload):
            raise ExchangeCaptureError("schema_error", "SZSE envelope must be a report list")
        catalog = "1110" if definition.status == "L" else "1793_ssgs"
        tab = "tab2" if definition.status == "D" else "tab1"
        matches = [report for report in payload if isinstance(report.get("metadata"), dict)
                   and report["metadata"].get("catalogid") == catalog and report["metadata"].get("tabkey") == tab]
        if len(matches) != 1:
            raise ExchangeCaptureError("schema_error", "SZSE requested catalog/tab is absent or duplicated")
        selected = matches[0]
        if "error" not in selected or selected["error"] is not None:
            raise ExchangeCaptureError("schema_error", "SZSE report error is not an explicit success")
        metadata = selected["metadata"]
        rows = selected.get("data")
        if not isinstance(rows, list) or not isinstance(metadata.get("cols"), dict):
            raise ExchangeCaptureError("schema_error", "SZSE rows/declared columns are missing")
        pagination = _pagination(metadata, ("pageno", "pagesize", "pagecount", "recordcount"))
        schema = metadata["cols"]
        required = _SZSE_FIELDS[definition.status]
    if pagination.page_no != requested_page:
        raise ExchangeCaptureError("coherence_error", "response page number differs from issued page")
    expected = 0 if pagination.total == 0 else min(pagination.page_size, pagination.total - (requested_page - 1) * pagination.page_size)
    if expected < 0 or (pagination.total > 0 and requested_page > pagination.page_count) or len(rows) != expected:
        raise ExchangeCaptureError("coherence_error", "response row count differs from actual pagination")
    schema_fields = tuple(sorted(schema))
    if (rows or definition.exchange == "SZSE") and not required <= set(schema_fields):
        raise ExchangeCaptureError("schema_error", f"required row schema missing: {definition.key}")
    for row in rows:
        if not isinstance(row, dict) or not required <= row.keys():
            raise ExchangeCaptureError("schema_error", "non-object row or required row fields missing")
        if tuple(sorted(row)) != schema_fields:
            raise ExchangeCaptureError("schema_error", "row fields differ from the selected page schema")
    schema_sha256 = canonical_sha256(schema)
    selected_sha256 = canonical_sha256({"rows": rows, "pagination": pagination.to_payload(), "schema": schema})
    return _Page(tuple(rows), pagination, schema_fields, schema_sha256, selected_sha256)


def _pagination(metadata: dict[str, object], names: tuple[str, str, str, str]) -> Pagination:
    values = [metadata.get(key) for key in names]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ExchangeCaptureError("schema_error", "pagination must contain actual integer metadata")
    page, size, count, total = values
    if page < 1 or size < 1 or count < 0 or total < 0 or count != (total + size - 1) // size:
        raise ExchangeCaptureError("coherence_error", "pagination metadata is inconsistent")
    return Pagination(page, size, count, total)


class _NameParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in {"a", "u", "b", "strong", "i", "em", "span", "br"}:
            raise ExchangeCaptureError("identity_error", "unexpected markup in official security name")
        if tag == "br":
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag not in {"a", "u", "b", "strong", "i", "em", "span", "br"}:
            raise ExchangeCaptureError("identity_error", "unexpected markup in official security name")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ExchangeCaptureError("schema_error", f"{field} must be original text")
    value = value.strip()
    if not value or value == "-" or "\ufffd" in value or any(ord(char) < 32 and not char.isspace() for char in value):
        raise ExchangeCaptureError("identity_error", f"{field} is blank or invalid")
    return value


def _name(value: object, *, html: bool) -> str:
    original = _text(value, "official name")
    if html:
        parser = _NameParser()
        parser.feed(original)
        parser.close()
        result = " ".join("".join(parser.parts).split())
    else:
        result = " ".join(original.split())
        if "<" in result or ">" in result:
            raise ExchangeCaptureError("identity_error", "unexpected markup in plain official name")
    return _text(result, "normalized official name")


def _date(value: object, field: str, *, required: bool = True) -> str:
    if value is None or value == "" or value == "-":
        if not required:
            return ""
        raise ExchangeCaptureError("identity_error", f"{field} is required")
    if not isinstance(value, str):
        raise ExchangeCaptureError("identity_error", f"{field} must be original date text")
    if re.fullmatch(r"[0-9]{8}", value, flags=re.ASCII):
        text = f"{value[:4]}-{value[4:6]}-{value[6:]}"
    elif re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value, flags=re.ASCII):
        text = value
    else:
        raise ExchangeCaptureError("identity_error", f"{field} has an invalid date format")
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise ExchangeCaptureError("identity_error", f"{field} is not a valid calendar date") from exc


def _normalize_row(definition: _Partition, row: dict[str, object], receipt: PageReceipt,
                   row_index: int, *, allow_b_share: bool = False) -> NormalizedAuthorityItem:
    if definition.exchange == "SHSE":
        code = row["A_STOCK_CODE"]
        if row["COMPANY_CODE"] != code or row["STOCK_TYPE"] != definition.board or row["LIST_BOARD"] != {"1": "1", "8": "2"}[definition.board]:
            raise ExchangeCaptureError("identity_error", "SSE company/code/board identity mismatch")
        state = row["STATE_CODE"]
        allowed = {"L": {"2", "4", "5", "7", "8"}, "P": {"5"}, "D": {"3"}}[definition.status]
        if state not in allowed:
            raise ExchangeCaptureError("identity_error", "SSE lifecycle state contradicts issued selector")
        name = _name(row["COMPANY_ABBR"] if definition.status == "D" else row["SEC_NAME_CN"], html=False)
        listed = _date(row["LIST_DATE"], "LIST_DATE")
        delisted = _date(row["DELIST_DATE"], "DELIST_DATE", required=definition.status == "D")
        board = "主板" if definition.board == "1" else "科创板"
    else:
        code = row["agdm"] if definition.status == "L" else row["zqdm"]
        name = _name(row["agjc"] if definition.status == "L" else row["zqjc"], html=True)
        listed = _date(row["agssrq"] if definition.status == "L" else row["ssrq"], "listing date")
        delisted = _date(row["zzrq"], "termination date") if definition.status == "D" else ""
        board = _text(row["bk"], "bk") if definition.status == "L" else ""
        if board and board not in {"主板", "创业板", "中小板"}:
            raise ExchangeCaptureError("identity_error", "unrecognized directly reported SZSE board")
        if definition.status == "P":
            suspended = _date(row["ztrq"], "suspension of listing date")
            if suspended < listed or suspended > datetime.fromisoformat(receipt.completed_at_utc).astimezone(_SHANGHAI).date().isoformat():
                raise ExchangeCaptureError("identity_error", "invalid suspension-of-listing interval")
    # SZSE's official March2026 code table separates stock domains 000..004
    # and 300..309 from warrants/voting/statistics/399 indexes. Preserve existing
    # stock-catalog compatibility inside the 001/309 depositary-receipt segments.
    prefix = r"6[0-9]{5}" if definition.exchange == "SHSE" else (r"20[0-9]{4}" if allow_b_share else r"(?:00[0-4]|30[0-9])[0-9]{3}")
    if not isinstance(code, str) or re.fullmatch(prefix, code, flags=re.ASCII) is None:
        raise ExchangeCaptureError("identity_error", f"noncanonical or out-of-scope A-share identity: {definition.exchange}:{code}")
    observed = datetime.fromisoformat(receipt.completed_at_utc).astimezone(_SHANGHAI).date().isoformat()
    if listed > observed or (delisted and (delisted < listed or delisted > observed)):
        raise ExchangeCaptureError("identity_error", "official lifecycle date interval is invalid")
    if definition.status != "D" and delisted:
        raise ExchangeCaptureError("identity_error", "non-delisted security has a termination date")
    owned = tuple(sorted((*_OWNED_IDENTITY_FIELDS, *(("listing_board",) if board else ()))))
    ref = RowReference(definition.key, receipt.pagination.page_no, row_index, receipt.raw_sha256)
    return NormalizedAuthorityItem(definition.exchange, code, name, definition.status, listed, delisted, board, owned, (ref,))


def _combine(captured: list[_CapturedPartition], exchanges: tuple[str, ...]) -> tuple[tuple[NormalizedAuthorityItem, ...], tuple[tuple[str, str], ...]]:
    collected: dict[tuple[str, str], NormalizedAuthorityItem] = {}
    overlaps: list[tuple[str, str]] = []
    sse_listed_states = {item.code: current.states[item.code] for current in captured
                         if current.definition.exchange == "SHSE" and current.definition.status == "L" for item in current.rows}
    sse_pending = {item.code for current in captured
                   if current.definition.exchange == "SHSE" and current.definition.status == "P" for item in current.rows}
    if {code for code, state in sse_listed_states.items() if state == "5"} != sse_pending:
        raise ExchangeCaptureError("coherence_error", "SSE explicit P set differs from status5 membership of L selector")
    for current in captured:
        for item in current.rows:
            key = (item.market, item.code)
            previous = collected.get(key)
            if previous is None:
                collected[key] = item
                continue
            if item.market == "SHSE" and previous.list_status == "L" and item.list_status == "P":
                if replace(previous, list_status="P", evidence_refs=()) != replace(item, evidence_refs=()):
                    raise ExchangeCaptureError("identity_error", "SSE L/P overlap has contradictory identity/date/name")
                collected[key] = replace(item, evidence_refs=previous.evidence_refs + item.evidence_refs)
                overlaps.append(key)
            else:
                raise ExchangeCaptureError("identity_error", f"unexplained lifecycle/board overlap: {key}")
    for exchange in exchanges:
        if not any(item.market == exchange and item.list_status == "L" for item in collected.values()):
            raise ExchangeCaptureError("coherence_error", f"listed shard is empty: {exchange}")
    if sum(len(current.rows) + len(current.rejected_rows) for current in captured) != len(collected) + len(overlaps) + sum(len(current.rejected_rows) for current in captured):
        raise ExchangeCaptureError("coherence_error", "aggregate raw/normalized/overlap accounting mismatch")
    return tuple(collected[key] for key in sorted(collected)), tuple(sorted(overlaps))
