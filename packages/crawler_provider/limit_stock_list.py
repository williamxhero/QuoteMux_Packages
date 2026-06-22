# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import Literal, TypedDict

from playwright.sync_api import Browser, Page, TimeoutError as PlaywrightTimeoutError, sync_playwright


class LimitStock(TypedDict):
    code: str
    name: str


class LimitStockListResult(TypedDict):
    limit_up: list[LimitStock]
    limit_down: list[LimitStock]


LimitStockType = Literal["limit_up", "limit_down"]

LIMIT_UP_URL = "https://quote.eastmoney.com/ztb/detail"
LIMIT_DOWN_URL = "https://quote.eastmoney.com/ztb/detail#type=dtgc"


def crawl_limit_stock_list() -> LimitStockListResult:
    """
    爬取东方财富当天涨停股和跌停股列表。
    """
    # ------------------ 1. 预处理：准备浏览器 ------------------
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            # ------------------ 2. 核心执行：分别读取涨停池和跌停池 ------------------
            limit_up = _crawl_one_pool(browser, LIMIT_UP_URL, "limit_up")
            limit_down = _crawl_one_pool(browser, LIMIT_DOWN_URL, "limit_down")
        finally:
            browser.close()

    # ------------------ 3. 后处理：返回强类型候选列表 ------------------
    return {"limit_up": limit_up, "limit_down": limit_down}


def _route_page_resource(route) -> None:
    url = route.request.url
    blocked_keywords = (
        "websitecaptcha",
        "anonflow",
        "bdstatics.eastmoney.com",
        "huaxiang.eastmoney.com",
    )
    if any(keyword in url for keyword in blocked_keywords):
        route.abort()
        return
    route.continue_()


def _crawl_one_pool(browser: Browser, url: str, stock_type: LimitStockType) -> list[LimitStock]:
    page = browser.new_page(viewport={"width": 1365, "height": 900})
    page.route("**/*", _route_page_resource)
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_selector("table tbody tr", timeout=30000)
        _load_all_rows(page, stock_type)
        return _extract_stock_rows(page)
    finally:
        page.close()


def _load_all_rows(page: Page, stock_type: LimitStockType) -> None:
    max_click_count = 50
    for _ in range(max_click_count):
        foot_text = _get_table_foot_text(page)
        if "无更多数据" in foot_text:
            return
        if "点击加载更多" not in foot_text:
            page.wait_for_timeout(500)
            foot_text = _get_table_foot_text(page)
            if "无更多数据" in foot_text:
                return
            if "点击加载更多" not in foot_text:
                raise RuntimeError(f"东方财富{_type_name(stock_type)}列表缺少加载状态文本")
        before_count = _get_row_count(page)
        page.locator("table tfoot").click(timeout=10000)
        try:
            page.wait_for_function(
                """count => {
                    const rows = document.querySelectorAll("table tbody tr").length;
                    const foot = document.querySelector("table tfoot");
                    const text = foot ? foot.innerText : "";
                    return rows > count || text.includes("无更多数据");
                }""",
                arg=before_count,
                timeout=15000,
            )
        except PlaywrightTimeoutError as exc:
            raise RuntimeError(f"东方财富{_type_name(stock_type)}列表加载更多超时") from exc
    raise RuntimeError(f"东方财富{_type_name(stock_type)}列表加载次数超过上限")


def _extract_stock_rows(page: Page) -> list[LimitStock]:
    header_cells = page.locator("table thead.originhead th").evaluate_all(
        """cells => cells.map(cell => cell.innerText.trim())"""
    )
    code_index = header_cells.index("代码")
    name_index = header_cells.index("名称")
    rows = page.locator("table tbody tr").evaluate_all(
        """(rows, indexes) => rows.map(row => {
            const cells = Array.from(row.querySelectorAll("td")).map(cell => cell.innerText.trim());
            return {code: cells[indexes.codeIndex] || "", name: cells[indexes.nameIndex] || ""};
        })""",
        {"codeIndex": code_index, "nameIndex": name_index},
    )
    result: list[LimitStock] = []
    for row in rows:
        code = str(row["code"]).strip()
        name = str(row["name"]).strip()
        if not re.fullmatch(r"\d{6}", code):
            raise RuntimeError(f"东方财富涨跌停列表返回了无效股票代码: {code}")
        if name == "":
            raise RuntimeError(f"东方财富涨跌停列表返回了空股票名称: {code}")
        result.append({"code": code, "name": name})
    return result


def _get_table_foot_text(page: Page) -> str:
    return page.locator("table tfoot").inner_text(timeout=10000).strip()


def _get_row_count(page: Page) -> int:
    return page.locator("table tbody tr").count()


def _type_name(stock_type: LimitStockType) -> str:
    if stock_type == "limit_up":
        return "涨停股"
    return "跌停股"
