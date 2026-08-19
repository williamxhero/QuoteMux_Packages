from __future__ import annotations

import base64
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
import math

import pandas as pd

from platform_models.migration_contracts import (
    IndexMemberAuditData,
    IndexMembersAuditRequest,
    MigrationPage,
    MigrationRecord,
)
from platform_models.provider_contracts import canonical_json_sha256
from quotemux.infra.common import (
    index_code_to_ts,
    normalize_index_code,
    normalize_stock_code,
)
from quotemux_packages.tushare.migration_errors import TushareMigrationError


SOURCE = "tushare_index_weight"
PAGE_SIZE = 500
MAX_RECORDS = 100_000
REQUIRED_COLUMNS = {"index_code", "con_code", "trade_date", "weight"}


def query_index_members(
    request: IndexMembersAuditRequest,
) -> MigrationPage[IndexMemberAuditData]:
    # Lazy import prevents a parallel Provider implementation and keeps the existing
    # Tushare client/config seam authoritative.
    from quotemux_packages.tushare.rate_limit import call_tushare_api
    from quotemux_packages.tushare.source import get_ts_pro

    pro = get_ts_pro()
    if pro is None:
        raise TushareMigrationError(
            "permission_error", "Tushare token 未配置或 SDK 不可用"
        )
    try:
        frame = call_tushare_api(
            "index_weight",
            pro.index_weight,
            index_code=index_code_to_ts(request.index_code),
            start_date=request.range_start.replace("-", ""),
            end_date=request.range_end.replace("-", ""),
        )
    except TimeoutError as exc:
        raise TushareMigrationError(
            "timeout_error", "Tushare index_weight 超时"
        ) from exc
    except TushareMigrationError:
        raise
    except Exception as exc:
        text = str(exc).lower()
        if any(token in text for token in ("权限", "permission", "积分", "token")):
            kind = "permission_error"
        elif any(token in text for token in ("频率", "rate limit", "too many")):
            kind = "rate_limit_error"
        else:
            kind = "request_error"
        raise TushareMigrationError(kind, f"Tushare index_weight 失败: {exc}") from exc
    if frame is None:
        raise TushareMigrationError("schema_error", "index_weight 返回 None")
    if not isinstance(frame, pd.DataFrame):
        raise TushareMigrationError("schema_error", "index_weight 必须返回 DataFrame")
    if frame.empty:
        return _page(request, [], "", True)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise TushareMigrationError(
            "schema_error", f"index_weight 缺少字段: {','.join(sorted(missing))}"
        )

    records = [_record(request, row) for _, row in frame.iterrows()]
    if request.query_mode == "current":
        records = [
            record for record in records if record.data.as_of_date == request.as_of_date
        ]
    records.sort(key=lambda record: (record.data.as_of_date, record.data.code))
    if len(records) > MAX_RECORDS:
        raise TushareMigrationError("contract_error", "index_weight 超过最大记录数")
    event_ids = [record.source_event_id for record in records]
    if len(event_ids) != len(set(event_ids)):
        raise TushareMigrationError("contract_error", "index_weight 事件身份重复")
    if records == []:
        return _page(request, [], "", True)

    page_number = _cursor_page(request)
    offset = (page_number - 1) * PAGE_SIZE
    if offset >= len(records):
        raise TushareMigrationError("contract_error", "index_weight cursor 越界")
    selected = records[offset : offset + PAGE_SIZE]
    has_more = offset + PAGE_SIZE < len(records)
    next_cursor = _encode_cursor(request, page_number + 1) if has_more else ""
    return _page(request, selected, next_cursor, False)


def _record(
    request: IndexMembersAuditRequest, row: pd.Series
) -> MigrationRecord[IndexMemberAuditData]:
    raw_projection = {
        str(key): _canonical_value(value) for key, value in row.to_dict().items()
    }
    actual_index = normalize_index_code(str(row["index_code"]))
    requested_index = normalize_index_code(request.index_code)
    if actual_index != requested_index:
        raise TushareMigrationError("contract_error", "index_code 与请求不匹配")
    code = normalize_stock_code(str(row["con_code"]))
    if code == "":
        raise TushareMigrationError("parse_error", "con_code 无法标准化")
    as_of_date = _date_text(row["trade_date"])
    if not request.range_start <= as_of_date <= request.range_end:
        raise TushareMigrationError("contract_error", "trade_date 超出请求范围")
    try:
        weight = Decimal(str(row["weight"]))
    except (InvalidOperation, ValueError) as exc:
        raise TushareMigrationError("parse_error", "weight 不是 Decimal") from exc
    if not weight.is_finite():
        raise TushareMigrationError("parse_error", "weight 必须为有限 Decimal")
    try:
        data = IndexMemberAuditData(
            index_code=request.index_code,
            code=code,
            as_of_date=as_of_date,
            weight=weight,
            weight_unit="percent",
        )
    except Exception as exc:
        raise TushareMigrationError("parse_error", f"weight 精度非法: {exc}") from exc
    return MigrationRecord[IndexMemberAuditData](
        source_event_id=f"{request.index_code}:{as_of_date}:{code}",
        raw_hash=canonical_json_sha256(raw_projection),
        raw_projection=raw_projection,
        data=data,
    )


def _date_text(value: object) -> str:
    text = str(value).strip().replace(".0", "")
    try:
        return datetime.strptime(text, "%Y%m%d").date().isoformat()
    except ValueError as exc:
        raise TushareMigrationError(
            "parse_error", "trade_date 必须是 YYYYMMDD"
        ) from exc


def _canonical_value(value: object) -> object:
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, float):
        return "" if math.isnan(value) else str(value)
    if hasattr(value, "item"):
        value = value.item()
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _page(
    request: IndexMembersAuditRequest,
    records: list[MigrationRecord[IndexMemberAuditData]],
    next_cursor: str,
    confirmed_empty: bool,
) -> MigrationPage[IndexMemberAuditData]:
    return MigrationPage[IndexMemberAuditData](
        capability_id=request.capability_id,
        data_version=request.data_version,
        provider=request.provider,
        source=SOURCE,
        source_version=request.source_version,
        fetched_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        confirmed_empty=confirmed_empty,
        next_cursor=next_cursor,
        records=records,
    )


def _request_identity(request: IndexMembersAuditRequest) -> str:
    payload = request.model_dump(mode="json", exclude={"cursor"})
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(text.encode("utf-8")).hexdigest()


def _encode_cursor(request: IndexMembersAuditRequest, page: int) -> str:
    payload = {"request": _request_identity(request), "page": page}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _cursor_page(request: IndexMembersAuditRequest) -> int:
    if request.cursor == "":
        return 1
    try:
        padding = "=" * (-len(request.cursor) % 4)
        payload = json.loads(
            base64.urlsafe_b64decode(request.cursor + padding).decode("utf-8")
        )
        page = payload["page"]
        identity = payload["request"]
    except Exception as exc:
        raise TushareMigrationError(
            "contract_error", "index_weight cursor 非法"
        ) from exc
    if isinstance(page, bool) or not isinstance(page, int) or page < 2:
        raise TushareMigrationError("contract_error", "index_weight cursor 页码非法")
    if identity != _request_identity(request):
        raise TushareMigrationError(
            "contract_error", "index_weight cursor 与请求不匹配"
        )
    return page
