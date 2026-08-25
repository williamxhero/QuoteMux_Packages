from __future__ import annotations

import csv
from datetime import datetime, timedelta
import io
import os
from zoneinfo import ZoneInfo

import requests

from platform_models import FutureBar1mItem
from quotemux.infra.provider_runtime.core import call_provider_api
from quotemux_packages.shinny_edb.actual_contract_capture import (
    ImmutableActualContractCapture,
    ShinnyEdbCaptureError,
    ShinnyEdbCaptureValidationError,
    ShinnyEdbHistoryUnavailableError,
    capture_future_actual_contract_1m,
)


EDB_KLINE_URL = "https://edb.shinnytech.com/md/kline"


def _timeout_seconds() -> float:
    try:
        return max(5.0, float(os.getenv("MHK_SHINNY_EDB_TIMEOUT_SECONDS", "60")))
    except ValueError:
        return 60.0


def _fetch_csv(symbol: str, start_time: str, end_time: str) -> str:
    def _invoke() -> str:
        response = requests.get(
            EDB_KLINE_URL,
            params={
                "period": 60,
                "symbol": symbol,
                "start_time": start_time,
                "end_time": end_time,
            },
            timeout=_timeout_seconds(),
        )
        response.raise_for_status()
        return response.text

    return str(call_provider_api("shinny_edb", "md.kline", _invoke))


def _optional_float(value: object) -> float | None:
    text = str(value or "").strip()
    return None if text == "" else float(text)


def get_future_main_continuous_1m(
    product_code: str,
    exchange: str,
    start_time: str,
    end_time: str,
) -> list[FutureBar1mItem]:
    symbol = f"KQ.m@{exchange}.{product_code}"
    payload = _fetch_csv(symbol, start_time, end_time)
    items: list[FutureBar1mItem] = []
    for row in csv.DictReader(io.StringIO(payload)):
        nano_text = str(row.get("datetime_nano", "")).strip()
        if nano_text == "":
            continue
        # EDB 使用 bar 开始时间；QuoteMux 统一保存 bar 结束时间。
        bar_time = datetime.fromtimestamp(int(nano_text) / 1_000_000_000, ZoneInfo("Asia/Shanghai")).replace(tzinfo=None) + timedelta(minutes=1)
        items.append(
            FutureBar1mItem(
                product_code=product_code,
                exchange=exchange,
                series_type="main_continuous",
                bar_time=bar_time.strftime("%Y-%m-%d %H:%M:%S"),
                open=_optional_float(row.get("open")),
                high=_optional_float(row.get("high")),
                low=_optional_float(row.get("low")),
                close=_optional_float(row.get("close")),
                volume=_optional_float(row.get("volume")),
                open_interest=_optional_float(row.get("close_oi")),
                adjustment_offset=None,
            )
        )
    return items
