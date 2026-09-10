from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import hashlib
from zoneinfo import ZoneInfo

import pytest

from quotemux_packages.shinny_edb import actual_contract_capture as capture


def _nanos(text: str) -> str:
    return str(int(datetime.fromisoformat(text).replace(tzinfo=ZoneInfo("Asia/Shanghai")).timestamp() * 1_000_000_000))


def _csv(*source_starts: str) -> str:
    rows = ["datetime_nano,open,high,low,close,volume,close_oi"]
    for index, source_start in enumerate(source_starts):
        rows.append(f"{_nanos(source_start)},6000,6002,5997,6001,{12 + index},{34 + index}")
    return "\n".join(rows)


def test_captures_exact_ag2604_window_as_an_immutable_actual_contract_artifact() -> None:
    calls: list[tuple[str, str, str, str]] = []

    def fetch(exchange: str, actual_contract: str, start_time: str, end_time: str) -> str:
        calls.append((exchange, actual_contract, start_time, end_time))
        return _csv("2026-02-02 09:00:00", "2026-02-02 09:01:00")

    result = capture.capture_future_actual_contract_1m(
        product_code="ag",
        exchange="SHFE",
        actual_contract="AG2604",
        start_time="2026-02-02 09:01:00",
        end_time="2026-02-02 09:02:00",
        session_anchor_date="2026-02-02",
        trading_day="2026-02-02",
        captured_at=datetime(2026, 8, 25, 1, 2, 3, tzinfo=timezone.utc),
        fetch_csv=fetch,
    )

    assert calls == [("SHFE", "AG2604", "2026-02-02 09:01:00", "2026-02-02 09:02:00")]
    assert result.request["symbol"] == "SHFE.AG2604"
    assert [item.bar_time for item in result.rows] == ["2026-02-02 09:01:00", "2026-02-02 09:02:00"]
    assert result.rows[0].close == Decimal("6001")
    assert result.rows[0].session_anchor_date == "2026-02-02"
    assert result.rows[0].trading_day == "2026-02-02"
    assert result.timestamp_contract["source_bar_timestamp"] == "minute_start"
    assert result.artifact_sha256 == hashlib.sha256(result.canonical_artifact_bytes()).hexdigest()
    assert result.artifact_envelope()["artifact_sha256"] == result.artifact_sha256


def test_classifies_ag1906_2018_http_403_as_history_unavailable(monkeypatch) -> None:
    class Response:
        status_code = 403
        text = "historical permission required"

    monkeypatch.setattr(capture.requests, "get", lambda *args, **kwargs: Response())

    with pytest.raises(capture.ShinnyEdbHistoryUnavailableError, match="HTTP 403"):
        capture.capture_future_actual_contract_1m(
            product_code="ag",
            exchange="SHFE",
            actual_contract="AG1906",
            start_time="2018-11-29 13:31:00",
            end_time="2018-11-29 13:52:00",
            session_anchor_date="2018-11-29",
            trading_day="2018-11-29",
        )


def test_rejects_truncated_actual_contract_response() -> None:
    with pytest.raises(capture.ShinnyEdbCaptureValidationError, match="truncated"):
        capture.capture_future_actual_contract_1m(
            product_code="ag",
            exchange="SHFE",
            actual_contract="AG2604",
            start_time="2026-02-02 09:01:00",
            end_time="2026-02-02 09:03:00",
            session_anchor_date="2026-02-02",
            trading_day="2026-02-02",
            fetch_csv=lambda *_: _csv("2026-02-02 09:00:00", "2026-02-02 09:01:00"),
        )
