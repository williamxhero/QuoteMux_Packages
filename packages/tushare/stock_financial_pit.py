from __future__ import annotations

import pandas as pd
import math

from platform_models import StockFinancialPitRawItem
from quotemux.infra.common import format_date_value, normalize_stock_code

from .helpers import query_frame
from .rate_limit import call_tushare_api


FINA_FIELDS = "ts_code,ann_date,end_date,eps,gross_margin,q_dtprofit,ebit,update_flag"
INCOME_FIELDS = "ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,total_revenue,oper_cost,ebit,n_income_attr_p,update_flag"


def _finite_float(value: object) -> float | None:
    """将 Tushare 的 NaN/无穷值统一转换为真正的缺失值。"""
    if value is None or not pd.notna(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _query_financial_frame(primary_api: str, fallback_api: str, **kwargs: object) -> pd.DataFrame:
    from .source import get_ts_pro

    pro = get_ts_pro()
    if pro is None:
        raise RuntimeError("Tushare provider 未配置有效 api_key")
    errors: list[str] = []
    for api_name in (primary_api, fallback_api):
        fetcher = getattr(pro, api_name, None)
        if not callable(fetcher):
            errors.append(f"{api_name}: 接口不可用")
            continue
        try:
            frame = call_tushare_api(api_name, fetcher, **kwargs)
        except Exception as exc:
            errors.append(f"{api_name}: {type(exc).__name__}: {exc}")
            continue
        if frame is not None and not frame.empty:
            return frame.copy()
    raise RuntimeError("Tushare 财务接口查询失败: " + "; ".join(errors))


def _normalize_financial_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    work = frame.copy()
    work["code"] = work["ts_code"].map(normalize_stock_code)
    work["announcement_date"] = work["ann_date"].map(format_date_value)
    work["report_period"] = work["end_date"].map(format_date_value)
    work["update_flag"] = work["update_flag"].fillna("").astype(str)
    return work


def get_stock_financial_pit_period(report_period: str, strict: bool = False) -> list[StockFinancialPitRawItem]:
    actual_period = format_date_value(report_period)
    if actual_period == "":
        raise ValueError("report_period 必须是有效日期")
    period_key = actual_period.replace("-", "")
    if strict:
        indicator_frame = _normalize_financial_frame(
            _query_financial_frame("fina_indicator_vip", "fina_indicator", period=period_key, fields=FINA_FIELDS)
        )
        income_frame = _normalize_financial_frame(
            _query_financial_frame("income_vip", "income", period=period_key, fields=INCOME_FIELDS)
        )
    else:
        indicator_frame = _normalize_financial_frame(query_frame("fina_indicator_vip", period=period_key, fields=FINA_FIELDS))
        income_frame = _normalize_financial_frame(query_frame("income_vip", period=period_key, fields=INCOME_FIELDS))
    if not income_frame.empty:
        income_frame = income_frame[income_frame["report_type"].fillna("").astype(str) == "1"].copy()
        income_frame["actual_announcement_date"] = income_frame["f_ann_date"].map(format_date_value)
        income_frame = income_frame.sort_values(["code", "report_period", "announcement_date", "update_flag", "actual_announcement_date"])
        income_frame = income_frame.drop_duplicates(["code", "report_period", "announcement_date", "update_flag"], keep="last")
    join_fields = ["code", "report_period", "announcement_date", "update_flag"]
    if indicator_frame.empty and income_frame.empty:
        return []
    if indicator_frame.empty:
        merged = income_frame.copy()
    elif income_frame.empty:
        merged = indicator_frame.copy()
    else:
        merged = indicator_frame.merge(income_frame, on=join_fields, how="outer", suffixes=("_indicator", "_income"))
    items: list[StockFinancialPitRawItem] = []
    for _, row in merged.sort_values(["code", "report_period", "announcement_date", "update_flag"]).iterrows():
        indicator_ebit = row.get("ebit_indicator", row.get("ebit"))
        income_ebit = row.get("ebit_income")
        items.append(
            StockFinancialPitRawItem(
                code=str(row["code"]),
                report_period=str(row["report_period"]),
                announcement_date=str(row["announcement_date"]),
                actual_announcement_date=str(row.get("actual_announcement_date", "")) if pd.notna(row.get("actual_announcement_date", "")) else "",
                report_type=str(row.get("report_type", "1")) if pd.notna(row.get("report_type", "1")) else "1",
                company_type=str(row.get("comp_type", "")) if pd.notna(row.get("comp_type", "")) else "",
                update_flag=str(row["update_flag"]),
                gross_profit_cumulative_cny=(
                    _finite_float(row.get("total_revenue")) - _finite_float(row.get("oper_cost"))
                    if _finite_float(row.get("total_revenue")) is not None and _finite_float(row.get("oper_cost")) is not None
                    else _finite_float(row.get("total_revenue")) * _finite_float(row.get("gross_margin")) / 100.0
                    if _finite_float(row.get("total_revenue")) is not None and _finite_float(row.get("gross_margin")) is not None
                    else None
                ),
                total_operating_revenue_cumulative_cny=_finite_float(row.get("total_revenue")),
                ebit_cumulative_cny=_finite_float(income_ebit) if _finite_float(income_ebit) is not None else _finite_float(indicator_ebit),
                attributable_net_profit_cumulative_cny=_finite_float(row.get("n_income_attr_p")),
                basic_eps_cumulative_cny_per_share=_finite_float(row.get("eps")),
                deducted_net_profit_single_quarter_cny=_finite_float(row.get("q_dtprofit")),
            )
        )
    return items
