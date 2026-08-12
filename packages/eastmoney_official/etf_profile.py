from __future__ import annotations

import html
import re

from platform_models.migration_contracts import (
    EtfProfileData,
    EtfProfileRequest,
    MigrationPage,
    MigrationRecord,
)
from quotemux_packages.eastmoney_official.errors import P0ProviderError
from quotemux_packages.eastmoney_official.migration_common import raw_hash, utc_now_text


SOURCE = "eastmoney_fundf10_profile"
_TABLE_VALUE = r"<th[^>]*>\s*{label}\s*</th>\s*<td[^>]*>(.*?)</td>"
_FOUND_LABEL = r"成立日期\s*[：:]\s*<span[^>]*>(.*?)</span>"


def query_etf_profile(request: EtfProfileRequest) -> MigrationPage[EtfProfileData]:
    from quotemux_packages.eastmoney_official.http_text import get_text
    from quotemux_packages.eastmoney_official.policies import (
        CONNECT_TIMEOUT_SECONDS,
        ETF_PROFILE_MAX_RESPONSE_BYTES,
        REQUEST_TIMEOUT_SECONDS,
    )

    url = f"https://fundf10.eastmoney.com/jbgk_{request.code}.html"
    text = get_text(
        url,
        connect_timeout=CONNECT_TIMEOUT_SECONDS,
        request_timeout=REQUEST_TIMEOUT_SECONDS,
        max_bytes=ETF_PROFILE_MAX_RESPONSE_BYTES,
    )
    found_text = _extract(_FOUND_LABEL, text) or _first_slash(_table(text, "成立日期/规模"))
    found_date = _fund_date(found_text)
    if found_date == "":
        return _page(request, [], True)
    name = _table(text, "基金简称")
    full_name = _table(text, "基金全称")
    fund_type = _table(text, "基金类型")
    listing_date = _table(text, "上市日期")
    if listing_date:
        listing_date = _fund_date(listing_date)
    flags = [field for field, value in (("missing_name", name), ("missing_full_name", full_name), ("missing_fund_type", fund_type), ("missing_listing_date", listing_date)) if value == ""]
    projection: dict[str, object] = {
        "source_url": url,
        "fund_name": name,
        "fund_full_name": full_name,
        "fund_type": fund_type,
        "found_date_text": found_text,
        "listing_date_text": listing_date,
    }
    record = MigrationRecord[EtfProfileData](
        source_event_id=f"{request.market}.{request.code}:fund_detail_profile",
        raw_hash=raw_hash(projection),
        raw_projection=projection,
        data=EtfProfileData(
            code=request.code,
            market=request.market,
            security_code=f"{request.market}.{request.code}",
            name=name,
            full_name=full_name,
            fund_type=fund_type,
            found_date=found_date,
            listing_date=listing_date,
            field_quality_flags=flags,
        ),
    )
    return _page(request, [record], False)


def _page(request, records, confirmed_empty: bool):
    return MigrationPage[EtfProfileData](
        capability_id=request.capability_id,
        data_version=request.data_version,
        provider=request.provider,
        source=SOURCE,
        source_version=request.source_version,
        fetched_at=utc_now_text(),
        confirmed_empty=confirmed_empty,
        next_cursor="",
        records=records,
    )


def _extract(pattern: str, text: str) -> str:
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return _clean(match.group(1)) if match else ""


def _table(text: str, label: str) -> str:
    return _extract(_TABLE_VALUE.format(label=re.escape(label)), text)


def _clean(value: str) -> str:
    value = re.sub(r"<[^>]+>", "", value)
    return html.unescape(value).strip()


def _first_slash(value: str) -> str:
    return value.split("/", 1)[0].strip() if value else ""


def _fund_date(value: str) -> str:
    normalized = value.strip().replace("年", "-").replace("月", "-").replace("日", "")
    match = re.search(r"(\d{4})[-/.]?(\d{1,2})[-/.]?(\d{1,2})", normalized)
    if not match:
        return ""
    year, month, day = map(int, match.groups())
    try:
        from datetime import date

        return date(year, month, day).isoformat()
    except ValueError as exc:
        raise P0ProviderError("parse_error", "基金日期非法") from exc
