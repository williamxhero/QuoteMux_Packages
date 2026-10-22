from __future__ import annotations

import io
import math
import os
import re
import struct
import tempfile
import urllib.error
import urllib.request
import zipfile
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import zlib

from platform_models import StockQuoteItem

TDX_PACKAGE_URL = "https://www.tdx.com.cn/products/data/data/g4day/{ymd}.zip"
TDX_BJ_FIRST_DAY = "20220506"
TDX_MIN_STOCK_ROWS = {"sh": 1800, "sz": 2400, "bj": 150}
# Backward-compatible test/config name retained after clarifying the contract.
TDX_MIN_PRICED = TDX_MIN_STOCK_ROWS
_MARKETS = ("sh", "sz", "bj")


def normalize_stock_code(value: object) -> str:
    text = str(value).strip().upper()
    suffix_market = {".SH": "sh", ".XSHG": "sh", ".SZ": "sz", ".XSHE": "sz", ".BJ": "bj"}
    declared = None
    for suffix, market in suffix_market.items():
        if text.endswith(suffix):
            declared = market
            text = text[:-len(suffix)]
            break
    prefix_market = next((m for prefix, m in (("SH", "sh"), ("SZ", "sz"), ("BJ", "bj")) if text.startswith(prefix)), None)
    if prefix_market:
        if declared and declared != prefix_market:
            raise ValueError(f"conflicting stock market in code: {value!r}")
        declared = prefix_market
        text = text[2:]
    if not re.fullmatch(r"[0-9]{6}", text):
        raise ValueError(f"invalid A-share stock code: {value!r}")
    market = _market_for_code(text)
    if market is None or declared and declared != market:
        raise ValueError(f"stock market conflicts with code: {value!r}")
    return text


def _market_for_code(code: str) -> str | None:
    if code.startswith(("4", "8", "920")):
        return "bj"
    if code[0] == "6" or code == "888880":
        return "sh"
    if code[0] in "03" and not code.startswith(("395", "399")):
        return "sz"
    return None


def _stock_market(market: str, code: str) -> bool:
    return _market_for_code(code) == market


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
                if not _stock_market(market, code):
                    continue
                block = md1[seq * 512 : (seq + 1) * 512]
                if len(block) != 512:
                    raise RuntimeError(f"{market}{code} quote block out of range")
                prev_close, open_, high, low, close = struct.unpack("<5d", block[4:44])
                volume = struct.unpack("<Q", block[56:64])[0]
                amount = struct.unpack("<d", block[72:80])[0]
                if not all(math.isfinite(v) for v in (prev_close, open_, high, low, close, amount)):
                    raise RuntimeError(f"{market}{code} contains a non-finite value")
                if open_ == high == low == 0 and volume == 0 and amount == 0 and close in (0, prev_close):
                    continue
                if min(prev_close, open_, high, low, close) <= 0:
                    raise RuntimeError(f"{market}{code} contains invalid zero/negative OHLC values")
                if not low <= min(open_, close) <= max(open_, close) <= high:
                    raise RuntimeError(f"{market}{code} contains invalid OHLC values")
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
            if len(rows) - before < TDX_MIN_STOCK_ROWS[market]:
                raise RuntimeError(f"{market} has too few priced stock rows: {len(rows) - before}")
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


@lru_cache(maxsize=64)
def _cached_package(ymd: str) -> tuple[str, bytes]:
    url = TDX_PACKAGE_URL.format(ymd=ymd)
    cache_dir = Path(os.getenv("QUOTEMUX_A_STOCK_DATA_CACHE", Path.home() / ".cache" / "quotemux" / "a_stock_data"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{ymd}.zip"
    if target.is_file():
        return url, target.read_bytes()
    _url, content = _fetch_package(ymd)
    fd, temp_name = tempfile.mkstemp(prefix=f".{ymd}.", suffix=".tmp", dir=cache_dir)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temp_name, target)
        except FileExistsError:
            pass
        return url, target.read_bytes()
    finally:
        Path(temp_name).unlink(missing_ok=True)


def _urllib_get(url: str) -> tuple[int, bytes]:
    request = urllib.request.Request(url, headers={"User-Agent": "QuoteMux/a-stock-data"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return int(response.status), response.read()


def _is_index_row(row: dict[str, Any]) -> bool:
    """Return true for rows whose exchange/code pair is not a stock identity."""
    return not _stock_market(str(row.get("market", "")), str(row.get("code", "")))


@lru_cache(maxsize=8)
def _parsed_day(ymd: str, content_length: int, content_crc: int) -> dict[str, dict[str, Any]]:
    """Parse one immutable daily archive once per process."""
    _url, content = _cached_package(ymd)
    if len(content) != content_length or zlib.crc32(content) != content_crc:
        raise RuntimeError(f"TDX package changed while cached: {ymd}")
    return {row["code"]: row for row in _parse_package(content, ymd)}


def _day_rows(ymd: str) -> dict[str, dict[str, Any]]:
    _url, content = _cached_package(ymd)
    return _parsed_day(ymd, len(content), zlib.crc32(content))


def get_stock_daily_snapshot_full(trade_date: str | date | datetime) -> list[StockQuoteItem]:
    """Return all stock rows from the official TDX package for one exact date."""
    day = _normalize_date(trade_date)
    ymd = day.replace("-", "")
    url, content = _cached_package(ymd)
    index = _parsed_day(ymd, len(content), zlib.crc32(content))
    try:
        rows = list(index.values())
    except (zipfile.BadZipFile, EOFError, zlib.error) as exc:
        raise RuntimeError(f"invalid TDX ZIP {url}: {type(exc).__name__}") from exc
    return _to_items(rows, "none")


def _to_items(rows: list[dict[str, Any]], adjust: str = "none") -> list[StockQuoteItem]:
    return [StockQuoteItem(
        code=normalize_stock_code(row["code"]), trade_time=row["trade_date"], freq="1d",
        open=row["open"], high=row["high"], low=row["low"], close=row["close"],
        pre_close=row["pre_close"], change=round(row["close"] - row["pre_close"], 4),
        pct_chg=round((row["close"] - row["pre_close"]) / row["pre_close"] * 100, 4) if row["pre_close"] else None,
        volume=float(row["volume"]), amount=float(row["amount"]), adjust=adjust,
        is_suspended=False, is_st=bool(re.match(r"^(?:\*?ST|S\*ST)", str(row["name"]).strip(), re.IGNORECASE)),
    ) for row in rows if not _is_index_row(row)]


def get_stock_quotes(
    codes: list[str], freq: str, trade_date: str, start_date: str, end_date: str,
    start_time: str, end_time: str, count: int | None, adjust: str,
) -> list[StockQuoteItem]:
    """Fetch exact-date full-market TDX daily bars and filter requested codes."""
    del start_time, end_time
    if freq != "1d":
        return []
    if adjust not in {"", "none"}:
        raise ValueError("a_stock_data supports unadjusted daily quotes only")
    if not codes:
        return []
    requested_date = _normalize_date(trade_date) if trade_date else ""
    if requested_date:
        if (start_date and _normalize_date(start_date) != requested_date) or (end_date and _normalize_date(end_date) != requested_date):
            raise ValueError("a_stock_data accepts one exact trade date only")
        dates = [requested_date]
    else:
        if not start_date or not end_date:
            raise ValueError("a_stock_data requires one exact trade date")
        start, end = _normalize_date(start_date), _normalize_date(end_date)
        if start != end:
            raise ValueError("a_stock_data accepts one exact trade date only")
        dates = [start]
    wanted = {normalize_stock_code(code) for code in codes}
    items: list[StockQuoteItem] = []
    for day in dates:
        ymd = day.replace("-", "")
        url, _content = _cached_package(ymd)
        try:
            rows = _day_rows(ymd)
        except (zipfile.BadZipFile, EOFError, zlib.error) as exc:
            raise RuntimeError(f"invalid TDX ZIP {url}: {type(exc).__name__}") from exc
        items.extend(_to_items([rows[code] for code in wanted if code in rows], adjust or "none"))
    if count is not None:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("count must be a non-negative integer")
        return items[:count]
    return items


