from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from quotemux.config_runtime.models import SourceInstanceConfig
from quotemux.source_packages.instance_context import use_source_instance
from quotemux_packages.shinny_tqsdk import source


def _instance(username: str = "reader", password: str = "secret") -> SourceInstanceConfig:
    return SourceInstanceConfig(
        instance_id="shinny_tqsdk-default",
        package_id="shinny_tqsdk",
        display_name="Shinny TqSdk",
        enabled=True,
        priority=1,
        timeout_seconds=None,
        config_values={"username": username, "timeout_seconds": "3.5"},
        secret_values={"password": password},
        tags=(),
    )


def _patch_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(source, "call_provider_api", lambda _provider, _operation, callback: callback())
    monkeypatch.setattr(source.time, "time", lambda: 100.0)


def test_fetches_main_continuous_snapshot_in_one_group_and_closes_api(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class FakeTqAuth:
        def __init__(self, username: str, password: str) -> None:
            calls["auth"] = (username, password)

    class FakeTqApi:
        def __init__(self, *, auth: FakeTqAuth) -> None:
            pass

        def get_quote_list(self, symbols: list[str]) -> list[SimpleNamespace]:
            calls["symbols"] = symbols
            return [
                SimpleNamespace(
                    underlying_symbol="SHFE.rb2610",
                    datetime="2026-08-20 10:00:00.000000",
                    last_price=3512.0,
                    open=3490.0,
                    highest=3520.0,
                    lowest=3488.0,
                    pre_close=3485.0,
                    volume=1234,
                    open_interest=456789,
                    bid_price1=float("nan"),
                    ask_price1=3513.0,
                    trading_status="CONTINUOUS",
                )
            ]

        def wait_update(self, *, deadline: float) -> bool:
            pytest.fail("a quote with datetime should not need wait_update")

        def close(self) -> None:
            calls["closed"] = True

    _patch_runtime(monkeypatch)
    monkeypatch.setattr(source, "_load_tqsdk", lambda: (FakeTqApi, FakeTqAuth))
    monkeypatch.setattr(source, "_future_realtime_quote_item", lambda **kwargs: kwargs)

    with use_source_instance(_instance()):
        items = source.get_future_main_continuous_realtime([("rb", "SHFE")])

    assert calls["auth"] == ("reader", "secret")
    assert calls["symbols"] == ["KQ.m@SHFE.rb"]
    assert calls["closed"] is True
    assert items[0]["contract_symbol"] == "SHFE.rb2610"
    assert items[0]["bid_price1"] is None


def test_catalog_preserves_raw_tqsdk_metadata_and_closes_api(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class Frame:
        def to_dict(self, orient: str) -> list[dict[str, object]]:
            assert orient == "records"
            return [
                {
                    "instrument_id": "SHFE.rb2610",
                    "product_id": "rb",
                    "exchange_id": "SHFE",
                    "instrument_name": "螺纹钢2610",
                    "ins_class": "FUTURE",
                    "expired": False,
                    "expire_datetime": 1790000000,
                    "last_exercise_datetime": datetime(2026, 8, 21, 10, 30, tzinfo=timezone.utc),
                    "delivery_year": 2026,
                    "delivery_month": 10,
                    "volume_multiple": 10,
                    "price_tick": 1.0,
                    "price_decs": 0,
                    "unsupported": float("nan"),
                }
            ]

    class FakeTqAuth:
        def __init__(self, *_args: str) -> None:
            pass

    class FakeTqApi:
        def __init__(self, *, auth: FakeTqAuth) -> None:
            pass

        def query_quotes(self, **kwargs: object) -> list[str]:
            calls["query_quotes"] = kwargs
            return ["SHFE.rb2610"]

        def query_symbol_info(self, symbols: list[str]) -> Frame:
            calls["query_symbol_info"] = symbols
            return Frame()

        def close(self) -> None:
            calls["closed"] = True

    _patch_runtime(monkeypatch)
    monkeypatch.setattr(source, "_load_tqsdk", lambda: (FakeTqApi, FakeTqAuth))
    monkeypatch.setattr(source, "_future_contract_catalog_item", lambda **kwargs: kwargs)

    with use_source_instance(_instance()):
        items = source.get_future_contract_catalog()

    assert calls["query_quotes"] == {"ins_class": "FUTURE", "expired": False}
    assert calls["query_symbol_info"] == ["SHFE.rb2610"]
    assert calls["closed"] is True
    assert items == [
        {
            "provider_symbol": "SHFE.rb2610",
            "contract_symbol": "SHFE.rb2610",
            "product_code": "rb",
            "exchange": "SHFE",
            "name": "螺纹钢2610",
            "ins_class": "FUTURE",
            "underlying_symbol": "",
            "expired": False,
            "expire_datetime": "2026-09-21 22:13:20",
            "metadata_time": "",
            "last_exercise_datetime": "2026-08-21 18:30:00",
            "delivery_year": 2026,
            "delivery_month": 10,
            "volume_multiple": 10.0,
            "price_tick": 1.0,
            "price_decs": 0,
            "max_limit_order_volume": None,
            "max_market_order_volume": None,
            "min_limit_order_volume": None,
            "min_market_order_volume": None,
            "raw_metadata": {
                "instrument_id": "SHFE.rb2610",
                "product_id": "rb",
                "exchange_id": "SHFE",
                "instrument_name": "螺纹钢2610",
                "ins_class": "FUTURE",
                "expired": False,
                "expire_datetime": 1790000000,
                "last_exercise_datetime": "2026-08-21T10:30:00+00:00",
                "delivery_year": 2026,
                "delivery_month": 10,
                "volume_multiple": 10,
                "price_tick": 1.0,
                "price_decs": 0,
                "unsupported": None,
            },
        }
    ]


def test_catalog_metadata_datetime_formats_epoch_and_datetime_in_china_time() -> None:
    assert source._metadata_datetime(1787295600) == "2026-08-21 15:00:00"
    assert source._metadata_datetime(datetime(2026, 8, 21, 7, tzinfo=timezone.utc)) == "2026-08-21 15:00:00"
    assert source._metadata_datetime(float("nan")) == ""
    assert source._metadata_datetime(None) == ""


def test_catalog_filters_requested_products_and_passes_include_expired(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class Frame:
        def to_dict(self, orient: str) -> list[dict[str, object]]:
            assert orient == "records"
            return [
                {"instrument_id": "SHFE.rb2610", "product_id": "rb", "exchange_id": "SHFE"},
                {"instrument_id": "DCE.m2609", "product_id": "m", "exchange_id": "DCE"},
            ]

    class FakeTqAuth:
        def __init__(self, *_args: str) -> None:
            pass

    class FakeTqApi:
        def __init__(self, *, auth: FakeTqAuth) -> None:
            pass

        def query_quotes(self, **kwargs: object) -> list[str]:
            calls["query_quotes"] = kwargs
            return ["SHFE.rb2610", "DCE.m2609"]

        def query_symbol_info(self, _symbols: list[str]) -> Frame:
            return Frame()

        def close(self) -> None:
            calls["closed"] = True

    _patch_runtime(monkeypatch)
    monkeypatch.setattr(source, "_load_tqsdk", lambda: (FakeTqApi, FakeTqAuth))
    monkeypatch.setattr(source, "_future_contract_catalog_item", lambda **kwargs: kwargs)

    with use_source_instance(_instance()):
        items = source.get_future_contract_catalog([("rb", "SHFE")], include_expired=True)

    assert calls["query_quotes"] == {"ins_class": "FUTURE", "expired": True}
    assert calls["closed"] is True
    assert [item["contract_symbol"] for item in items] == ["SHFE.rb2610"]


def test_requested_main_mapping_uses_main_continuous_quote(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class FakeTqAuth:
        def __init__(self, *_args: str) -> None:
            pass

    class FakeTqApi:
        def __init__(self, *, auth: FakeTqAuth) -> None:
            pass

        def get_quote_list(self, symbols: list[str]) -> list[SimpleNamespace]:
            calls["symbols"] = symbols
            return [SimpleNamespace(underlying_symbol="CFFEX.IF2609", datetime="2026-08-20 10:01:00")]

        def close(self) -> None:
            calls["closed"] = True

    _patch_runtime(monkeypatch)
    monkeypatch.setattr(source, "_load_tqsdk", lambda: (FakeTqApi, FakeTqAuth))
    monkeypatch.setattr(source, "_future_main_contract_mapping_item", lambda **kwargs: kwargs)

    with use_source_instance(_instance()):
        items = source.get_future_main_contract_mapping([("IF", "CFFEX")])

    assert calls["symbols"] == ["KQ.m@CFFEX.IF"]
    assert calls["closed"] is True
    assert items == [
        {
            "product_code": "IF",
            "exchange": "CFFEX",
            "provider_symbol": "KQ.m@CFFEX.IF",
            "contract_symbol": "CFFEX.IF2609",
            "updated_time": "2026-08-20 10:01:00",
        }
    ]


def test_empty_main_mapping_asks_tqsdk_for_all_current_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class FakeTqAuth:
        def __init__(self, *_args: str) -> None:
            pass

    class FakeTqApi:
        def __init__(self, *, auth: FakeTqAuth) -> None:
            pass

        def query_cont_quotes(self) -> list[str]:
            calls["query_cont_quotes"] = True
            return ["DCE.m2609"]

        def get_quote_list(self, symbols: list[str]) -> list[SimpleNamespace]:
            calls["symbols"] = symbols
            return [SimpleNamespace(instrument_id="DCE.m2609", product_id="m", exchange_id="DCE", datetime="2026-08-20 10:03:00")]

        def close(self) -> None:
            calls["closed"] = True

    _patch_runtime(monkeypatch)
    monkeypatch.setattr(source, "_load_tqsdk", lambda: (FakeTqApi, FakeTqAuth))
    monkeypatch.setattr(source, "_future_main_contract_mapping_item", lambda **kwargs: kwargs)

    with use_source_instance(_instance()):
        items = source.get_future_main_contract_mapping([])

    assert calls["query_cont_quotes"] is True
    assert calls["symbols"] == ["DCE.m2609"]
    assert calls["closed"] is True
    assert items[0]["provider_symbol"] == "KQ.m@DCE.m"
    assert items[0]["contract_symbol"] == "DCE.m2609"


def test_contract_realtime_maps_full_book_and_rejects_main_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class FakeTqAuth:
        def __init__(self, *_args: str) -> None:
            pass

    class FakeTqApi:
        def __init__(self, *, auth: FakeTqAuth) -> None:
            pass

        def get_quote_list(self, symbols: list[str]) -> list[SimpleNamespace]:
            calls["symbols"] = symbols
            return [
                SimpleNamespace(
                    instrument_id="SHFE.rb2610",
                    product_id="rb",
                    exchange_id="SHFE",
                    datetime="2026-08-20 10:02:00",
                    last_price=3512.0,
                    highest=3520.0,
                    lowest=3488.0,
                    bid_price1=3511.0,
                    bid_volume1=4,
                    ask_price5=3517.0,
                    ask_volume5=9,
                    upper_limit=float("nan"),
                    lower_limit=4000.0,
                    trading_status="CONTINUOUS",
                    expired=False,
                )
            ]

        def close(self) -> None:
            calls["closed"] = True

    _patch_runtime(monkeypatch)
    monkeypatch.setattr(source, "_load_tqsdk", lambda: (FakeTqApi, FakeTqAuth))
    monkeypatch.setattr(source, "_future_contract_realtime_quote_item", lambda **kwargs: kwargs)

    with use_source_instance(_instance()):
        items = source.get_future_contract_realtime_quotes(["SHFE.rb2610"])

    assert calls["symbols"] == ["SHFE.rb2610"]
    assert calls["closed"] is True
    assert items[0]["high"] == 3520.0
    assert items[0]["bid_price1"] == 3511.0
    assert items[0]["bid_volume1"] == 4.0
    assert items[0]["ask_price5"] == 3517.0
    assert items[0]["ask_volume5"] == 9.0
    assert items[0]["upper_limit"] is None
    with pytest.raises(ValueError, match="real delivery"):
        source.get_future_contract_realtime_quotes(["KQ.m@SHFE.rb"])


def test_missing_credentials_fail_without_loading_tqsdk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(source, "_load_tqsdk", lambda: pytest.fail("should not load TqSdk"))
    with use_source_instance(_instance(username="", password="")):
        with pytest.raises(RuntimeError, match="username and password"):
            source.get_future_main_continuous_realtime([("rb", "SHFE")])


def test_timeout_closes_api_without_returning_empty_quotes(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class FakeTqAuth:
        def __init__(self, *_args: str) -> None:
            pass

    class FakeTqApi:
        def __init__(self, *, auth: FakeTqAuth) -> None:
            pass

        def get_quote_list(self, _symbols: list[str]) -> list[SimpleNamespace]:
            return [SimpleNamespace(datetime="")]

        def wait_update(self, *, deadline: float) -> bool:
            return False

        def close(self) -> None:
            calls["closed"] = True

    _patch_runtime(monkeypatch)
    monkeypatch.setattr(source, "_load_tqsdk", lambda: (FakeTqApi, FakeTqAuth))

    with use_source_instance(_instance()):
        with pytest.raises(TimeoutError, match="did not receive all quote updates"):
            source.get_future_main_continuous_realtime([("rb", "SHFE")])

    assert calls["closed"] is True


def test_manifest_declares_all_read_only_tqsdk_capabilities() -> None:
    path = Path(__file__).resolve().parents[1] / "quotemux_package.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["package_id"] == "shinny_tqsdk"
    assert manifest["secret_fields"] == ["password"]
    assert [item["capability_id"] for item in manifest["capabilities"]] == [
        "futures.contracts.catalog",
        "futures.contracts.main_mapping",
        "futures.quotes.contract.realtime",
        "futures.quotes.main_continuous.realtime",
    ]
    assert set(manifest["handler_targets"]) == {
        "get_future_contract_catalog",
        "get_future_contract_realtime_quotes",
        "get_future_main_contract_mapping",
        "get_future_main_continuous_realtime",
    }
