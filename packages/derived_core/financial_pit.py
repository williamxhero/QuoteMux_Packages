from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from datetime import date
import math

from platform_models import StockFinancialPitFactorItem, StockFinancialPitRawItem
from quotemux.infra.common import parse_date_text


FORMULA_VERSION = "derived_market_hub_v1"


def _quarter_period(value: str, offset: int = 0) -> str:
    parsed = parse_date_text(value)
    if parsed is None:
        return ""
    quarter_index = parsed.year * 4 + (parsed.month - 1) // 3 + offset
    year, quarter_zero = divmod(quarter_index, 4)
    month_day = ((3, 31), (6, 30), (9, 30), (12, 31))[quarter_zero]
    return date(year, month_day[0], month_day[1]).isoformat()


def _single_quarter_value(state: dict[str, StockFinancialPitRawItem], period: str, field_name: str) -> float | None:
    current = state.get(period)
    if current is None:
        return None
    current_value = getattr(current, field_name)
    if current_value is None or not math.isfinite(float(current_value)):
        return None
    parsed = parse_date_text(period)
    if parsed is None:
        return None
    if parsed.month == 3:
        return current_value
    previous = state.get(_quarter_period(period, -1))
    if previous is None:
        return None
    previous_value = getattr(previous, field_name)
    if previous_value is None or not math.isfinite(float(previous_value)):
        return None
    return current_value - previous_value


def _rolling_quarters(state: dict[str, StockFinancialPitRawItem], period: str, field_name: str, count: int, offset: int = 0) -> float | None:
    values = [_single_quarter_value(state, _quarter_period(period, offset - index), field_name) for index in range(count)]
    if any(value is None for value in values):
        return None
    return float(sum(value for value in values if value is not None))


def _positive_base_growth(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None or not math.isfinite(float(current)) or not math.isfinite(float(previous)) or previous <= 0:
        return None
    return (current / previous - 1) * 100


def _consecutive_three_quarter_minimum(
    values: dict[str, float | None],
    report_period: str,
) -> float | None:
    trailing = [values.get(_quarter_period(report_period, -offset)) for offset in range(3)]
    if any(value is None for value in trailing):
        return None
    return min(float(value) for value in trailing if value is not None)


def _share_lookup(share_rows: list[dict[str, object]]) -> dict[str, tuple[list[str], list[float]]]:
    grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in share_rows:
        code = str(row.get("code", ""))
        trade_date = str(row.get("trade_date", ""))
        total_share = row.get("total_share")
        if code != "" and trade_date != "" and total_share is not None:
            grouped[code].append((trade_date, float(total_share) * 10000.0))
    return {
        code: ([item[0] for item in sorted(values)], [item[1] for item in sorted(values)])
        for code, values in grouped.items()
    }


def _latest_shares(shares: dict[str, tuple[list[str], list[float]]], code: str, announcement_date: str) -> float | None:
    dates, values = shares.get(code, ([], []))
    index = bisect_right(dates, announcement_date) - 1
    if index < 0:
        return None
    value = values[index]
    return value if value > 0 else None


def _finite_count(item: StockFinancialPitRawItem) -> int:
    """选择同一报告期内财务字段最完整的已披露版本。"""
    fields = (
        "gross_profit_cumulative_cny",
        "total_operating_revenue_cumulative_cny",
        "ebit_cumulative_cny",
        "attributable_net_profit_cumulative_cny",
        "basic_eps_cumulative_cny_per_share",
        "deducted_net_profit_single_quarter_cny",
    )
    return sum(
        value is not None and math.isfinite(float(value))
        for value in (getattr(item, field_name) for field_name in fields)
    )


def build_financial_pit_factors(
    raw_items: list[StockFinancialPitRawItem],
    share_rows: list[dict[str, object]],
) -> list[StockFinancialPitFactorItem]:
    shares = _share_lookup(share_rows)
    by_code: dict[str, list[StockFinancialPitRawItem]] = defaultdict(list)
    for item in raw_items:
        by_code[item.code].append(item)
    factors: list[StockFinancialPitFactorItem] = []
    for code, events in by_code.items():
        state: dict[str, StockFinancialPitRawItem] = {}
        per_share_by_period: dict[str, float | None] = {}
        ebit_ps_by_period: dict[str, float | None] = {}
        eps_qoq_by_period: dict[str, float | None] = {}
        deducted_qoq_by_period: dict[str, float | None] = {}
        grouped_by_period: dict[str, list[StockFinancialPitRawItem]] = defaultdict(list)
        for item in events:
            grouped_by_period[item.report_period].append(item)
        ordered = sorted(
            (
                max(
                    period_events,
                    key=lambda item: (
                        _finite_count(item),
                        item.actual_announcement_date or item.announcement_date,
                        item.update_flag,
                    ),
                )
                for period_events in grouped_by_period.values()
            ),
            key=lambda item: (item.actual_announcement_date or item.announcement_date, item.report_period, item.update_flag),
        )
        for item in ordered:
            available_date = item.actual_announcement_date or item.announcement_date
            state[item.report_period] = item
            gross_ttm = _rolling_quarters(state, item.report_period, "gross_profit_cumulative_cny", 4)
            previous_gross_ttm = _rolling_quarters(state, item.report_period, "gross_profit_cumulative_cny", 4, -4)
            revenue_ttm = _rolling_quarters(state, item.report_period, "total_operating_revenue_cumulative_cny", 4)
            previous_revenue_ttm = _rolling_quarters(state, item.report_period, "total_operating_revenue_cumulative_cny", 4, -4)
            ebit_quarter = _single_quarter_value(state, item.report_period, "ebit_cumulative_cny")
            attributed_quarter = _single_quarter_value(state, item.report_period, "attributable_net_profit_cumulative_cny")
            total_shares = _latest_shares(shares, code, available_date)
            ebit_per_share = None if ebit_quarter is None or total_shares is None else ebit_quarter / total_shares
            attributed_per_share = None if attributed_quarter is None or total_shares is None else attributed_quarter / total_shares
            previous_per_share = per_share_by_period.get(_quarter_period(item.report_period, -1))
            deducted_ttm = _rolling_quarters(state, item.report_period, "deducted_net_profit_single_quarter_cny", 4)
            previous_deducted_ttm = _rolling_quarters(state, item.report_period, "deducted_net_profit_single_quarter_cny", 4, -1)
            eps_quarter = _single_quarter_value(state, item.report_period, "basic_eps_cumulative_cny_per_share")
            previous_eps_quarter = _single_quarter_value(state, _quarter_period(item.report_period, -1), "basic_eps_cumulative_cny_per_share")
            eps_qoq = _positive_base_growth(eps_quarter, previous_eps_quarter)
            deducted_qoq = _positive_base_growth(deducted_ttm, previous_deducted_ttm)
            per_share_by_period[item.report_period] = attributed_per_share
            ebit_ps_by_period[item.report_period] = ebit_per_share
            eps_qoq_by_period[item.report_period] = eps_qoq
            deducted_qoq_by_period[item.report_period] = deducted_qoq
            factors.append(
                StockFinancialPitFactorItem(
                    code=code,
                    announcement_date=available_date,
                    report_period=item.report_period,
                    formula_version=FORMULA_VERSION,
                    revision_source=f"tushare.fina_indicator_vip+income_vip:update_flag={item.update_flag}",
                    gross_profit_ttm_cny=gross_ttm,
                    gross_profit_ttm_yoy_pct=_positive_base_growth(gross_ttm, previous_gross_ttm),
                    total_operating_revenue_ttm_cny=revenue_ttm,
                    total_operating_revenue_ttm_yoy_pct=_positive_base_growth(revenue_ttm, previous_revenue_ttm),
                    ebit_latest_quarter_cny=ebit_quarter,
                    total_shares_latest=total_shares,
                    ebit_per_share_latest_quarter_cny=ebit_per_share,
                    attributable_net_profit_per_latest_share_cny=attributed_per_share,
                    attributable_net_profit_per_latest_share_qoq_pct=_positive_base_growth(attributed_per_share, previous_per_share),
                    deducted_net_profit_ttm_cny=deducted_ttm,
                    deducted_net_profit_ttm_qoq_pct=deducted_qoq,
                    ebit_ps_lf_consec_min_3q=_consecutive_three_quarter_minimum(ebit_ps_by_period, item.report_period),
                    basic_eps_latest_capital_lf_qoq_consec_min_3q=_consecutive_three_quarter_minimum(eps_qoq_by_period, item.report_period),
                    net_profit_deducted_ttm_qoq_consec_min_3q=_consecutive_three_quarter_minimum(deducted_qoq_by_period, item.report_period),
                )
            )
    return sorted(factors, key=lambda item: (item.announcement_date, item.code, item.report_period))
