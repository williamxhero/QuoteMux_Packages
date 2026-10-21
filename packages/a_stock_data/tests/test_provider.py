from __future__ import annotations

import io
import json
import struct
import zipfile
from pathlib import Path

import pytest

from quotemux_packages.a_stock_data import source


def _archive(ymd="220105", *, bj=False, malformed=False):
    def pair(market, code, seq=0, close=10.5):
        record = bytearray(150)
        record[:6] = code.encode()
        struct.pack_into("<H", record, 32, seq)
        record[40:44] = "Demo".encode()
        block = bytearray(512)
        struct.pack_into("<5d", block, 4, 10, 10.1, 11, 9.5, close)
        struct.pack_into("<Q", block, 56, 123456)
        struct.pack_into("<d", block, 72, 987654.25)
        return bytes(record), bytes(block)

    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        for market, code in (("sh", "600519"), ("sz", "000001")):
            rec, block = pair(market, code)
            archive.writestr(f"{market}{ymd}.cod", rec)
            archive.writestr(f"{market}{ymd}.md1", block)
        if bj:
            rec, block = pair("bj", "920000")
            archive.writestr(f"bj{ymd}.cod", rec)
            archive.writestr(f"bj{ymd}.md1", block)
    value = out.getvalue()
    return value[:-3] if malformed else value


def test_bse_boundary_skips_bse_before_first_day(monkeypatch):
    monkeypatch.setitem(source.TDX_MIN_PRICED, "sh", 1)
    monkeypatch.setitem(source.TDX_MIN_PRICED, "sz", 1)
    monkeypatch.setitem(source.TDX_MIN_PRICED, "bj", 1)
    assert {r["market"] for r in source._parse_package(_archive("220105"), "20220105")} == {"sh", "sz"}
    assert {r["market"] for r in source._parse_package(_archive("220506", bj=True), "20220506")} == {"sh", "sz", "bj"}


def test_parser_normalizes_binary_rows_and_units(monkeypatch):
    monkeypatch.setitem(source.TDX_MIN_PRICED, "sh", 1)
    monkeypatch.setitem(source.TDX_MIN_PRICED, "sz", 1)
    monkeypatch.setitem(source.TDX_MIN_PRICED, "bj", 1)
    row = source._parse_package(_archive("220105"), "20220105")[0]
    assert row["trade_date"] == "2022-01-05"
    assert row["code"] == "600519"
    assert row["volume"] == 123456
    assert row["amount"] == 987654.25


@pytest.mark.parametrize("content", [b"not zip", _archive("220105", malformed=True)])
def test_malformed_archive_rejected(content):
    with pytest.raises((zipfile.BadZipFile, RuntimeError)):
        source._parse_package(content, "20220105")


def test_date_validation_and_404(monkeypatch):
    assert source._normalize_date("2022-01-05") == "2022-01-05"
    assert source._normalize_date("20220105") == "2022-01-05"
    with pytest.raises(ValueError):
        source._normalize_date("2022/01/05")
    monkeypatch.setattr(source, "_urllib_get", lambda url: (404, b""))
    with pytest.raises(ValueError):
        source._fetch_package("20220105")


def test_row_normalization_and_handler_contract(monkeypatch):
    monkeypatch.setattr(source, "_cached_package", lambda ymd: (source.TDX_PACKAGE_URL.format(ymd=ymd), _archive("220105")))
    monkeypatch.setitem(source.TDX_MIN_PRICED, "sh", 1)
    monkeypatch.setitem(source.TDX_MIN_PRICED, "sz", 1)
    items = source.get_stock_quotes(["600519"], "1d", "2022-01-05", "", "", "", "", None, "none")
    assert len(items) == 1
    assert items[0].model_dump()["trade_time"] == "2022-01-05"
    assert items[0].model_dump()["freq"] == "1d"


def test_market_identity_not_name_filters_indices_and_duplicate_stock_codes(monkeypatch):
    rows = [
        {"market": "sh", "code": "000001", "name": "上证指数", "trade_date": "2022-01-05", "open": 10, "high": 11, "low": 9, "close": 10, "pre_close": 10, "volume": 1, "amount": 1},
        {"market": "sz", "code": "000001", "name": "平安银行", "trade_date": "2022-01-05", "open": 10, "high": 11, "low": 9, "close": 10, "pre_close": 10, "volume": 1, "amount": 1},
        {"market": "sz", "code": "399001", "name": "非指数名称", "trade_date": "2022-01-05", "open": 10, "high": 11, "low": 9, "close": 10, "pre_close": 10, "volume": 1, "amount": 1},
    ]
    assert [item.code for item in source._to_items(rows)] == ["000001"]
    assert source._to_items([rows[1]])[0].is_st is False
    assert source._to_items([{**rows[1], "name": "*ST平安"}])[0].is_st is True
    assert source.normalize_stock_code("SH600519") == "600519"
    for code in ("SH000001", "BJ600519", "600519.SZ", "399001"):
        with pytest.raises(ValueError):
            source.normalize_stock_code(code)


def test_archive_cache_downloads_once_across_calls(monkeypatch, tmp_path):
    monkeypatch.setenv("QUOTEMUX_A_STOCK_DATA_CACHE", str(tmp_path))
    source._cached_package.cache_clear()
    calls = []
    monkeypatch.setattr(source, "_fetch_package", lambda ymd: (calls.append(ymd) or source.TDX_PACKAGE_URL.format(ymd=ymd), _archive("220105")))
    assert source._cached_package("20220105")[1] == _archive("220105")
    source._cached_package.cache_clear()
    assert source._cached_package("20220105")[1] == _archive("220105")
    assert calls == ["20220105"]
    assert (tmp_path / "20220105.zip").is_file()


def test_exact_single_date_required_before_fetch(monkeypatch):
    monkeypatch.setattr(source, "_cached_package", lambda _ymd: pytest.fail("must reject before IO"))
    with pytest.raises(ValueError, match="exact trade date"):
        source.get_stock_quotes(["600519"], "1d", "", "2022-01-01", "2022-01-05", "", "", None, "none")

    manifest = json.loads((Path(__file__).parents[1] / "quotemux_package.json").read_text())
    assert manifest["handler_targets"]["get_stock_quotes"] == "quotemux_packages.a_stock_data.source:get_stock_quotes"
    assert manifest["capabilities"][0]["capability_id"] == "stocks.quotes.daily"


def test_full_snapshot_returns_exact_day_stock_rows_and_uses_cached_archive(monkeypatch):
    monkeypatch.setitem(source.TDX_MIN_PRICED, "sh", 1)
    monkeypatch.setitem(source.TDX_MIN_PRICED, "sz", 1)
    archive = _archive("220105")
    calls = []
    monkeypatch.setattr(source, "_cached_package", lambda ymd: (calls.append(ymd) or source.TDX_PACKAGE_URL.format(ymd=ymd), archive))

    items = source.get_stock_daily_snapshot_full("2022-01-05")

    assert {item.code for item in items} == {"600519", "000001"}
    assert {item.trade_time for item in items} == {"2022-01-05"}
    assert all(item.freq == "1d" and item.adjust == "none" for item in items)
    assert calls == ["20220105"]


@pytest.mark.parametrize("trade_date", ["", "2022/01/05", "2022-02-30", None])
def test_full_snapshot_rejects_invalid_or_missing_exact_date_before_io(monkeypatch, trade_date):
    monkeypatch.setattr(source, "_cached_package", lambda _ymd: pytest.fail("must reject before IO"))
    with pytest.raises(ValueError, match="trade date"):
        source.get_stock_daily_snapshot_full(trade_date)


def test_manifest_advertises_full_snapshot_capability():
    manifest = json.loads((Path(__file__).parents[1] / "quotemux_package.json").read_text())
    assert manifest["handler_targets"]["get_stock_daily_snapshot_full"] == "quotemux_packages.a_stock_data.source:get_stock_daily_snapshot_full"
    snapshot = [cap for cap in manifest["capabilities"] if cap["capability_id"] == "stocks.quotes.daily_snapshot"]
    assert len(snapshot) == 1
    assert snapshot[0]["handler_name"] == "get_stock_daily_snapshot_full"
    assert snapshot[0]["support_level"] == "native"
