from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from platform_models.migration_contracts import IndexMembersAuditRequest
from quotemux_packages.tushare import index_members_contract
from quotemux_packages.tushare.migration_errors import TushareMigrationError
from quotemux_packages.tushare import rate_limit, source


def _request(
    *,
    mode: str = "current",
    start: str = "2025-12-31",
    end: str = "2025-12-31",
    cursor: str = "",
) -> IndexMembersAuditRequest:
    return IndexMembersAuditRequest(
        capability_id="indexes.members",
        provider="tushare",
        provider_instance_id="tushare-default",
        index_code="000300",
        query_mode=mode,
        as_of_date=end,
        range_start=start,
        range_end=end,
        cursor=cursor,
        data_version="quotemux.indexes.members.v2",
        source_version="tushare.index_weight.v1",
    )


def _install_frame(monkeypatch: pytest.MonkeyPatch, frame: pd.DataFrame) -> None:
    pro = type("Pro", (), {"index_weight": lambda self, **kwargs: frame})()
    monkeypatch.setattr(source, "get_ts_pro", lambda: pro)
    monkeypatch.setattr(
        rate_limit,
        "call_tushare_api",
        lambda name, function, **kwargs: function(**kwargs),
    )


def test_current_requires_exact_as_of_and_preserves_decimal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        [
            {
                "index_code": "000300.SH",
                "con_code": "600001.SH",
                "trade_date": "20251231",
                "weight": "1.23456789",
            },
        ]
    )
    _install_frame(monkeypatch, frame)
    page = index_members_contract.query_index_members(_request())
    assert len(page.records) == 1
    assert page.records[0].data.code == "600001"
    assert page.records[0].data.as_of_date == "2025-12-31"
    assert page.records[0].data.weight == Decimal("1.23456789")
    assert page.records[0].data.weight_unit == "percent"
    assert page.records[0].source_event_id == "000300:2025-12-31:600001"
    assert isinstance(page.records[0].raw_projection["weight"], str)


def test_current_never_uses_prior_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    # Exact-date Provider response is empty. The audited contract must not widen
    # the range to borrow an earlier current snapshot.
    _install_frame(monkeypatch, pd.DataFrame())
    page = index_members_contract.query_index_members(_request())
    assert page.confirmed_empty is True
    assert page.records == []


def test_history_is_bounded_paginated_and_replayable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            "index_code": "000300.SH",
            "con_code": f"{index:06d}.SZ",
            "trade_date": "20251231",
            "weight": str(Decimal(index) / Decimal("100")),
        }
        for index in range(1, 502)
    ]
    _install_frame(monkeypatch, pd.DataFrame(rows))
    request = _request(mode="history", start="2025-01-01", end="2025-12-31")
    first = index_members_contract.query_index_members(request)
    second_request = _request(
        mode="history", start="2025-01-01", end="2025-12-31", cursor=first.next_cursor
    )
    second = index_members_contract.query_index_members(second_request)
    replay = index_members_contract.query_index_members(second_request)
    assert len(first.records) == 500
    assert len(second.records) == 1
    assert second.next_cursor == ""
    assert [item.model_dump(mode="json") for item in second.records] == [
        item.model_dump(mode="json") for item in replay.records
    ]
    assert {record.source_event_id for record in first.records}.isdisjoint(
        {record.source_event_id for record in second.records}
    )


def test_empty_schema_permission_and_provider_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_frame(monkeypatch, pd.DataFrame())
    assert (
        index_members_contract.query_index_members(_request()).confirmed_empty is True
    )

    _install_frame(monkeypatch, pd.DataFrame([{"index_code": "000300.SH"}]))
    with pytest.raises(TushareMigrationError) as schema:
        index_members_contract.query_index_members(_request())
    assert schema.value.kind == "schema_error"

    monkeypatch.setattr(source, "get_ts_pro", lambda: None)
    with pytest.raises(TushareMigrationError) as permission:
        index_members_contract.query_index_members(_request())
    assert permission.value.kind == "permission_error"

    pro = type("Pro", (), {"index_weight": lambda self, **kwargs: None})()
    monkeypatch.setattr(source, "get_ts_pro", lambda: pro)
    monkeypatch.setattr(
        rate_limit,
        "call_tushare_api",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("权限不足")),
    )
    with pytest.raises(TushareMigrationError) as provider_permission:
        index_members_contract.query_index_members(_request())
    assert provider_permission.value.kind == "permission_error"


def test_weight_precision_and_cursor_mismatch_are_contract_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_frame(
        monkeypatch,
        pd.DataFrame(
            [
                {
                    "index_code": "000300.SH",
                    "con_code": "600000.SH",
                    "trade_date": "20251231",
                    "weight": "1.123456789",
                }
            ]
        ),
    )
    with pytest.raises(TushareMigrationError) as precision:
        index_members_contract.query_index_members(_request())
    assert precision.value.kind == "parse_error"

    cursor = index_members_contract._encode_cursor(
        _request(mode="history", start="2025-01-01"), 2
    )
    with pytest.raises(TushareMigrationError) as mismatch:
        index_members_contract._cursor_page(
            _request(mode="history", start="2025-02-01", cursor=cursor)
        )
    assert mismatch.value.kind == "contract_error"


def test_request_rejects_unbounded_history() -> None:
    with pytest.raises(ValueError, match="366"):
        _request(mode="history", start="2024-01-01", end="2025-12-31")
