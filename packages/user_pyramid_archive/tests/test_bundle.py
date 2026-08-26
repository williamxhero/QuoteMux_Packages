from __future__ import annotations

import json
from pathlib import Path

import pytest

from quotemux_packages.user_pyramid_archive.bundle import PRODUCTS, SOURCE_KEY, build_bundle, preflight, validate_authorization


AUTHORIZATION = {"status": "private_research_authorized", "source_thread_id": "01a031ef-006c-7de0-a585-68eecdf769c7",
                 "evidence": "test authorization", "private_server_retention": True, "transformation": True,
                 "private_research": True, "redistribution": False}


def _source(root: Path, rows: dict[str, list[str]], include_tll: bool = True) -> None:
    data = root / "后复权数据"; data.mkdir(parents=True)
    for name in PRODUCTS:
        (data / name).write_text("\n".join(rows.get(name, ["2020/01/02-09:01\t1\t1\t1\t1\t1\t0"])) + "\n", encoding="gbk")
    if include_tll:
        (data / "TLL0.txt").write_text("2020/01/02-09:01\t1\t1\t1\t1\t1\t0\n", encoding="gbk")


def test_tl_mapping_and_authorization(tmp_path: Path) -> None:
    source = tmp_path / "source"; _source(source, {})
    outcome = preflight(source, tmp_path / "bundle", AUTHORIZATION)
    assert PRODUCTS["TL0.txt"][0] == "T"
    assert PRODUCTS["aoL0.txt"] == ("ao", "SHFE")
    assert "TLL0.txt" not in PRODUCTS
    assert outcome["authorization"] == AUTHORIZATION
    (source / "后复权数据" / "TLL0.txt").unlink()
    with pytest.raises(ValueError, match="TLL0"):
        preflight(source, tmp_path / "other", AUTHORIZATION)


def test_conflicts_invalid_and_fact_source_key_are_deterministic(tmp_path: Path) -> None:
    rows = {"agL0.txt": [
        "2020/01/02-09:01\t1\t2\t1\t2\t1\t0",
        "2020/01/02-09:02\t1\t2\t1\t2\t1\t0",
        "2020/01/02-09:02\t1\t3\t1\t2\t1\t0",
        "2020/01/02-09:03\t3\t2\t1\t2\t1\t0",
    ]}
    source = tmp_path / "source"; _source(source, rows)
    first = build_bundle(source, tmp_path / "one", AUTHORIZATION, batch_size=2)
    second = build_bundle(source, tmp_path / "two", AUTHORIZATION, batch_size=3)
    one = json.loads((tmp_path / "one" / "manifest.json").read_text())
    two = json.loads((tmp_path / "two" / "manifest.json").read_text())
    assert first["source_normalized_rowset_sha256"] == second["source_normalized_rowset_sha256"]
    assert first["fact_normalized_rowset_sha256"] == second["fact_normalized_rowset_sha256"]
    assert first["source_normalized_rowset_sha256"] != first["fact_normalized_rowset_sha256"]
    assert one["fact_normalization"]["source_key"] == SOURCE_KEY
    assert one["product_coverage"]["ag"]["conflicting_timestamp_keys"] == 1
    assert one["product_coverage"]["ag"]["invalid_ohlcv_rows"] == 1
    assert one["normalized_row_count"] == two["normalized_row_count"]


def test_authorization_and_expected_hashes_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "source"; _source(source, {})
    rejected = {**AUTHORIZATION, "redistribution": True}
    with pytest.raises(ValueError, match="redistribution"):
        preflight(source, tmp_path / "bad", rejected)
    with pytest.raises(ValueError, match="raw aggregate"):
        build_bundle(source, tmp_path / "wrong", AUTHORIZATION, expected_raw_aggregate_sha256="0" * 64)
