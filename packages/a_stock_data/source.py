from __future__ import annotations

import io
import math
import re
import struct
import urllib.error
import urllib.request
import zipfile
from datetime import date, datetime, timedelta
from typing import Any, Callable

import zlib

from platform_models import StockQuoteItem

TDX_PACKAGE_URL = "https://www.tdx.com.cn/products/data/data/g4day/{ymd}.zip"
TDX_BJ_FIRST_DAY = "20220506"
TDX_MIN_PRICED = {"sh": 10000, "sz": 3000, "bj": 50}
_MARKETS = ("sh", "sz", "bj")


def normalize_stock_code(value: object) -> str:
    text = str(value).strip().upper()
    for suffix in (".SH", ".SZ", ".BJ", ".XSHG", ".XSHE"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return text.removeprefix("SH").removeprefix("SZ").removeprefix("BJ")


def _normalize_date(value: str | date | datetime) -> str:
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            pass
    raise ValueError(f"invalid trade date: {value!r}")


def _parse_package(content: bytes, ymd: str) -> list[dict[str, Any]]:
    """Parse TDX .cod/.md1 pairs without relying on a TDX client."""
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = set(archive.namelist())
        for market in _MARKETS:
            cod_name, md1_name = f"{market}{ymd[2:]}.cod", f"{market}{ymd[2:]}.md1"
            if market == "bj" and ymd < TDX_BJ_FIRST_DAY and cod_name not in names and md1_name not in names:
                continue
            if cod_name not in names or md1_name not in names:
                raise RuntimeError(f"package missing {cod_name}/{md1_name}")
            cod, md1 = archive.read(cod_name), archive.read(md1_name)
            if len(cod) % 150 or len(md1) % 512:
                raise RuntimeError(f"{market} file is not aligned to its record size")
            if len(cod) // 150 != len(md1) // 512:
                raise RuntimeError(f"{market} code/quote record counts differ")
            before, codes, seqs = len(rows), set(), set()
            for offset in range(0, len(cod), 150):
                record = cod[offset : offset + 150]
                try:
                    code = record[:6].rstrip(b"\x00 ").decode("ascii")
                except UnicodeDecodeError as exc:
                    raise RuntimeError(f"{market} code is not ASCII") from exc
                if not re.fullmatch(r"[0-9]{6}", code):
                    raise RuntimeError(f"{market} invalid code {code!r}")
                seq = struct.unpack("<H", record[32:34])[0]
                if code in codes or seq in seqs:
                    raise RuntimeError(f"{market} duplicate code or quote sequence")
                codes.add(code)
                seqs.add(seq)
                block = md1[seq * 512 : (seq + 1) * 512]
                if len(block) != 512:
                    raise RuntimeError(f"{market}{code} quote block out of range")
                prev_close, open_, high, low, close = struct.unpack("<5d", block[4:44])
                volume = struct.unpack("<Q", block[56:64])[0]
                amount = struct.unpack("<d", block[72:80])[0]
                if not all(math.isfinite(v) for v in (prev_close, open_, high, low, close, amount)):
                    raise RuntimeError(f"{market}{code} contains a non-finite value")
                if close <= 0:
                    continue
                raw_name = record[40:72].split(b"\x00")[0]
                try:
                    name = raw_name.decode("gbk").strip()
                except UnicodeDecodeError as exc:
                    raise RuntimeError(f"{market}{code} name is not GBK") from exc
                if not name:
                    raise RuntimeError(f"{market}{code} has a priced row without a name")
                rows.append({
                    "trade_date": f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}", "market": market, "code": code,
                    "name": name, "pre_close": round(prev_close, 4), "open": round(open_, 4),
                    "high": round(high, 4), "low": round(low, 4), "close": round(close, 4),
                    "volume": int(volume), "amount": round(amount, 2),
                })
            if len(rows) - before < TDX_MIN_PRICED[market]:
                raise RuntimeError(f"{market} has too few priced rows: {len(rows) - before}")
    return rows


def _fetch_package(ymd: str, http_get: Callable[[str], tuple[int, bytes]] | None = None) -> tuple[str, bytes]:
    url = TDX_PACKAGE_URL.format(ymd=ymd)
    try:
        status, content = http_get(url) if http_get else _urllib_get(url)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise ValueError(f"no TDX package for {ymd}: non-trading day or not published") from exc
        raise RuntimeError(f"TDX package HTTP {exc.code}") from exc
    if status == 404:
        raise ValueError(f"no TDX package for {ymd}: non-trading day or not published")
    if status < 200 or status >= 300:
        raise RuntimeError(f"TDX package HTTP {status}")
    if not content.startswith(b"PK"):
        raise RuntimeError("TDX response is not a ZIP archive")
    return url, content


def _urllib_get(url: str) -> tuple[int, bytes]:
    request = urllib.request.Request(url, headers={"User-Agent": "QuoteMux/a-stock-data"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return int(response.status), response.read()


def _to_items(rows: list[dict[str, Any]], adjust: str = "none") -> list[StockQuoteItem]:
    return [StockQuoteItem(
        code=normalize_stock_code(row["code"]), trade_time=row["trade_date"], freq="1d",
        open=row["open"], high=row["high"], low=row["low"], close=row["close"],
        pre_close=row["pre_close"], change=round(row["close"] - row["pre_close"], 4),
        pct_chg=round((row["close"] - row["pre_close"]) / row["pre_close"] * 100, 4) if row["pre_close"] else None,
        volume=float(row["volume"]), amount=float(row["amount"]), adjust=adjust,
        is_suspended=False, is_st=False,
    ) for row in rows]


def get_stock_quotes(
    codes: list[str], freq: str, trade_date: str, start_date: str, end_date: str,
    start_time: str, end_time: str, count: int | None, adjust: str,
) -> list[StockQuoteItem]:
    """Fetch exact-date full-market TDX daily bars and filter requested codes.

    A request with no exact date uses the bounded date window, newest first. The
    source is unadjusted; other adjustments are deliberately not fabricated.
    """
    del start_time, end_time
    if freq != "1d" or adjust not in {"", "none"}:
        return []
    requested_date = _normalize_date(trade_date) if trade_date else ""
    if requested_date:
        dates = [requested_date]
    else:
        start = _normalize_date(start_date) if start_date else (date.today() - timedelta(days=30)).isoformat()
        end = _normalize_date(end_date) if end_date else date.today().isoformat()
        dates = [(date.fromisoformat(end) - timedelta(days=i)).isoformat() for i in range((date.fromisoformat(end) - date.fromisoformat(start)).days + 1) if date.fromisoformat(start) <= date.fromisoformat(end) - timedelta(days=i)]
    wanted = {normalize_stock_code(code) for code in codes if normalize_stock_code(code)}
    items: list[StockQuoteItem] = []
    for day in dates:
        ymd = day.replace("-", "")
        url, content = _fetch_package(ymd)
        try:
            rows = _parse_package(content, ymd)
        except (zipfile.BadZipFile, EOFError, zlib.error) as exc:
            raise RuntimeError(f"invalid TDX ZIP {url}: {type(exc).__name__}") from exc
        items.extend(_to_items([row for row in rows if not wanted or row["code"] in wanted], adjust or "none"))
        if count and len(items) >= count:
            return items[:count]
    return items


