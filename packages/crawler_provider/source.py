from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo
import json
import re
import time

import requests

from platform_models import LimitOrderAmountItem


SNAPSHOT_URL = "https://hsmarketwg.eastmoney.com/api/SHSZQuoteSnapshot"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://quote.eastmoney.com/",
}


def _market_for_code(code: str) -> str:
    if code.startswith(("60", "688", "900")):
        return "sh"
    if code.startswith(("00", "30", "200")):
        return "sz"
    if code.startswith(("43", "83", "87", "88", "920")):
        return "bj"
    if code.startswith("6"):
        return "sh"
    if code.startswith(("0", "3")):
        return "sz"
    return "bj"


def _float_value(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "-"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _beijing_today() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")


def _build_limit_stock_candidates(trade_date: str, payload: dict[str, object]) -> list[LimitOrderAmountItem]:
    candidates: list[LimitOrderAmountItem] = []
    for key, limit_side in (("limit_up", "up"), ("limit_down", "down")):
        rows = payload.get(key, [])
        if not isinstance(rows, list):
            raise RuntimeError(f"东方财富涨跌停列表字段不是数组: {key}")
        for row in rows:
            if not isinstance(row, dict):
                raise RuntimeError(f"东方财富涨跌停列表返回了无效行: {key}")
            code = str(row.get("code", "")).strip()
            if not re.match(r"^\d{6}$", code):
                raise RuntimeError(f"东方财富涨跌停列表返回了无效股票代码: {code}")
            candidates.append(
                LimitOrderAmountItem(
                    code=code,
                    trade_date=trade_date,
                    limit_side=limit_side,
                    market=_market_for_code(code),
                    close=None,
                    limit_price=None,
                    captured_at="",
                )
            )
    return sorted(candidates, key=lambda item: (item.limit_side, item.code))


def _snapshot_payload(code: str, timeout_seconds: float) -> dict[str, object]:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.get(
                SNAPSHOT_URL,
                params={"id": code, "callback": "jQuery_limit_order_amount"},
                headers=HEADERS,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            match = re.search(r"\{.*\}", response.text)
            if match is None:
                raise RuntimeError(f"解析股票 {code} 盘口快照失败，未找到 JSON 结构")
            payload = json.loads(match.group(0))
            return payload if isinstance(payload, dict) else {}
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"请求股票 {code} 盘口快照失败: {last_error}")


def _timeout_seconds() -> float:
    # source instance 配置由 worker 注入到环境里时仍可能缺失，这里只保留直接默认值。
    return 5.0


def _build_item(
    code: str,
    trade_date: str,
    limit_side: str,
    close: float | None,
    limit_price: float | None,
    snapshot: dict[str, object],
) -> LimitOrderAmountItem:
    fivequote = snapshot.get("fivequote")
    if not isinstance(fivequote, dict):
        fivequote = {}
    if limit_side == "up":
        order_price = _float_value(fivequote.get("buy1"))
        order_volume = _float_value(fivequote.get("buy1_count"))
    else:
        order_price = _float_value(fivequote.get("sale1"))
        order_volume = _float_value(fivequote.get("sale1_count"))
    order_amount = None
    if order_price is not None and order_volume is not None:
        order_amount = round(order_price * order_volume * 100, 2)
    return LimitOrderAmountItem(
        code=code,
        trade_date=trade_date,
        limit_side=limit_side,
        market=_market_for_code(code),
        close=close,
        limit_price=limit_price,
        order_price=order_price,
        order_volume=order_volume,
        order_amount=order_amount,
        captured_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


def get_limit_order_amount(
    code: str,
    trade_date: str,
    limit_side: str,
    close: float | None,
    limit_price: float | None,
) -> list[LimitOrderAmountItem]:
    actual_code = str(code).strip()
    if not re.match(r"^\d{6}$", actual_code):
        raise ValueError(f"无效股票代码: {actual_code}")
    if limit_side not in {"up", "down"}:
        raise ValueError(f"无效涨跌停方向: {limit_side}")
    snapshot = _snapshot_payload(actual_code, _timeout_seconds())
    return [_build_item(actual_code, trade_date, limit_side, close, limit_price, snapshot)]


def get_limit_stock_candidates(trade_date: str) -> list[LimitOrderAmountItem]:
    from quotemux_packages.crawler_provider.limit_stock_list import crawl_limit_stock_list

    actual_trade_date = str(trade_date)
    if actual_trade_date != _beijing_today():
        raise ValueError(f"涨跌停名单爬虫只支持当天: {actual_trade_date}")
    payload = crawl_limit_stock_list()
    return _build_limit_stock_candidates(actual_trade_date, payload)
