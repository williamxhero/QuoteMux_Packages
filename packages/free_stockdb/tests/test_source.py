from __future__ import annotations

from datetime import date, datetime
import json
from pathlib import Path
import sys
from typing import Any

import msgpack
import pytest
import requests

ROOT = Path(__file__).resolve().parents[3]
QUOTE_MUX_SRC = Path("D:/WILL/STOCK/QuantResearch/QuoteMux/src")
for path in (ROOT, QUOTE_MUX_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from quotemux.source_packages.manifest import SourcePackageManifest
from quotemux_packages.free_stockdb import source


class _Response:
    def __init__(self, payload: Any, status: int = 200) -> None:
        self.content = msgpack.packb(payload, use_bin_type=True)
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status={self.status_code}")


def _row() -> dict[str, Any]:
    return {
        "date": 20261019,
        "code": "000001",
        "name": "平安银行",
        "open": "11.20",
        "high": 11.35,
        "low": 11.10,
        "close": 11.30,
        "pre_close": 11.25,
        "volume": 123456789,
        "amount": 987654321.5,
        "pct_chg": 0.4444,
        "is_st": 0,
    }


def test_manifest_contract() -> None:
    payload = json.loads((ROOT / "packages/free_stockdb/quotemux_package.json").read_text())
    assert payload["package_id"] == "free_stockdb"
    assert payload["capabilities"] == [
        {
            "capability_id": "stocks.quotes.daily",
            "support_level": "native",
            "handler_name": "get_stock_quotes",
            "mergeable": True,
            "notes": "仅支持显式代码与单个精确历史日期；缺失键返回空结果。",
        }
    ]
    assert payload["handler_targets"]["get_stock_quotes"].endswith(":get_stock_quotes")
    assert {field["name"] for field in payload["config_schema"]} == {"host", "port"}
    assert payload["secret_fields"] == []
    manifest = SourcePackageManifest.from_dict(payload, package_root=str(ROOT / "packages/free_stockdb"))
    assert manifest.package_id == "free_stockdb"
    assert manifest.contract_names == ("stocks.quotes.daily",)


def test_exact_date_query_decodes_msgpack_and_preserves_units(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, str], float]] = []

    def fake_get(url: str, *, params: dict[str, str], timeout: float) -> _Response:
        calls.append((url, params, timeout))
        return _Response(_row())

    monkeypatch.setattr(source.requests, "get", fake_get)
    monkeypatch.setattr(source, "get_provider_config_value", lambda name: {"host": "10.0.0.8", "port": "7899"}.get(name, ""))
    result = source.get_stock_quotes(["000001"], "1d", "2026-10-19", "", "", "", "", None, "none")

    assert len(result) == 1
    item = result[0]
    assert item.code == "000001"
    assert item.trade_time == "2026-10-19 00:00:00"
    assert item.volume == 123456789.0
    assert item.amount == 987654321.5
    assert item.change == pytest.approx(0.05)
    assert calls == [("http://10.0.0.8:7899/", {"cmd": "get", "t": "日k:000001:20261019"}, 10)]


def test_missing_exact_key_is_empty_and_each_explicit_code_is_queried(monkeypatch: pytest.MonkeyPatch) -> None:
    requested: list[str] = []

    def fake_get(url: str, *, params: dict[str, str], timeout: float) -> _Response:
        del url, timeout
        requested.append(params["t"])
        return _Response(_row() if params["t"].startswith("日k:000001:") else None)

    monkeypatch.setattr(source.requests, "get", fake_get)
    result = source.get_stock_quotes(["000001", "600000"], "1d", "20261019", "", "", "", "", None, "none")
    assert [item.code for item in result] == ["000001"]
    assert requested == ["日k:000001:20261019", "日k:600000:20261019"]


def test_bounded_inputs_and_transport_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(source.requests, "get", lambda *args, **kwargs: _Response(_row()))
    assert source.get_stock_quotes(["000001"], "5m", "20261019", "", "", "", "", None, "none") == []
    with pytest.raises(source.FreeStockdbError, match="exact historical"):
        source.get_stock_quotes(["000001"], "1d", "", "", "", "", "", None, "none")
    with pytest.raises(source.FreeStockdbError, match="one exact"):
        source.get_stock_quotes(["000001"], "1d", "20261019", "20261018", "", "", "", None, "none")

    def broken(*args: Any, **kwargs: Any) -> _Response:
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(source.requests, "get", broken)
    with pytest.raises(requests.ConnectionError):
        source.get_stock_quotes(["000001"], "1d", "20261019", "", "", "", "", None, "none")
