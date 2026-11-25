"""Twenty tool-calling cases with deterministic local tools and pass/fail checks.

Five categories with four cases each: one tool with one argument, several typed
arguments, two dependent calls, no tool needed, and recovery after a tool error.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import golekit as gole
from golekit.protocols import Tool


@dataclass(frozen=True)
class Call:
    """One tool call a run made, with the error it produced, if any."""

    name: str
    arguments: dict[str, Any]
    error: str | None = None


Check = Callable[[list[Call], str], bool]


@dataclass(frozen=True)
class Case:
    """A prompt, the tools offered with it, and the rule that decides success."""

    id: str
    category: str
    prompt: str
    tools: Callable[[], list[Tool]]
    check: Check


def _same(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float) and isinstance(actual, (int, float)):
        return abs(actual - expected) < 1e-6
    if isinstance(expected, list) and isinstance(actual, list):
        return sorted(map(str, actual)) == sorted(map(str, expected))
    if isinstance(expected, str) and isinstance(actual, str):
        return actual.strip().lower() == expected.strip().lower()
    return actual == expected


def _matches(call: Call, name: str, arguments: dict[str, Any]) -> bool:
    return call.name == name and all(
        key in call.arguments and _same(call.arguments[key], value)
        for key, value in arguments.items()
    )


def expect(sequence: list[tuple[str, dict[str, Any]]], *answer: str) -> Check:
    """Pass when the successful calls are exactly this sequence and the answer has every fragment."""

    def check(calls: list[Call], text: str) -> bool:
        done = [call for call in calls if call.error is None]
        return (
            len(done) == len(sequence)
            and all(_matches(call, *step) for call, step in zip(done, sequence, strict=True))
            and all(fragment.lower() in text.lower() for fragment in answer)
        )

    return check


def recovered(final: tuple[str, dict[str, Any]], *answer: str) -> Check:
    """Pass when a call failed, a later call succeeded as expected, and the answer is right."""

    def check(calls: list[Call], text: str) -> bool:
        done = [call for call in calls if call.error is None]
        return (
            any(call.error for call in calls)
            and bool(done)
            and _matches(done[-1], *final)
            and all(fragment.lower() in text.lower() for fragment in answer)
        )

    return check


def no_tools(*answer: str) -> Check:
    """Pass when no tool was called and the answer has one of the fragments."""

    def check(calls: list[Call], text: str) -> bool:
        return not calls and any(fragment.lower() in text.lower() for fragment in answer)

    return check


_WEATHER = {"北京": "晴，25°C", "上海": "多云，28°C", "广州": "雷阵雨，31°C", "成都": "阴，22°C"}


@gole.tool
def get_weather(city: str) -> str:
    """查询城市当前的天气。

    Args:
        city: 城市名，例如 北京。
    """
    if city not in _WEATHER:
        raise ValueError(f"未知城市：{city}")
    return _WEATHER[city]


@gole.tool
def get_stock_price(symbol: str) -> float:
    """查询股票的最新价格（美元）。

    Args:
        symbol: 股票代码，例如 AAPL。
    """
    return {"AAPL": 189.5, "MSFT": 412.3, "NVDA": 131.2}[symbol.upper()]


@gole.tool
def get_population(country: str) -> str:
    """查询国家的人口。

    Args:
        country: 国家名，例如 日本。
    """
    return {"日本": "1.24 亿", "法国": "6800 万", "巴西": "2.16 亿"}[country]


@gole.tool
def lookup_order(order_id: str) -> dict[str, str]:
    """查询订单状态。

    Args:
        order_id: 订单号，例如 A1001。
    """
    return {"A1001": {"status": "已发货", "carrier": "顺丰"}}[order_id]


@gole.tool
def book_room(
    city: str, nights: int, breakfast: bool, room_type: Literal["single", "double", "suite"]
) -> str:
    """预订酒店房间。

    Args:
        city: 入住城市。
        nights: 入住晚数。
        breakfast: 是否含早餐。
        room_type: 房型，single、double 或 suite。
    """
    return f"预订成功，确认号 H{nights}{len(city)}7Q"


@gole.tool
def convert_currency(
    amount: float,
    source: Literal["CNY", "USD", "EUR", "JPY"],
    target: Literal["CNY", "USD", "EUR", "JPY"],
) -> float:
    """按固定汇率换算货币。

    Args:
        amount: 金额。
        source: 原币种。
        target: 目标币种。
    """
    rates = {"CNY": 1.0, "USD": 7.2, "EUR": 7.8, "JPY": 0.048}
    return round(amount * rates[source] / rates[target], 2)


@gole.tool
def send_notification(recipients: list[str], message: str, urgent: bool) -> str:
    """给多人发送通知。

    Args:
        recipients: 收件人姓名列表。
        message: 通知内容。
        urgent: 是否加急。
    """
    return f"已通知 {len(recipients)} 人，回执号 N-58"


@gole.tool
def set_thermostat(temperature: float, mode: Literal["heat", "cool", "auto"]) -> str:
    """设置空调温度和模式。

    Args:
        temperature: 目标温度，摄氏度。
        mode: 模式，heat、cool 或 auto。
    """
    return f"已设置为 {temperature} 度，模式 {mode}，能耗等级 B2"


@gole.tool
def find_employee(name: str) -> dict[str, str]:
    """按姓名查找员工，返回员工编号。

    Args:
        name: 员工姓名。
    """
    return {"employee_id": {"王芳": "E-2087"}[name]}


@gole.tool
def get_manager(employee_id: str) -> str:
    """按员工编号查询其直属经理。

    Args:
        employee_id: 员工编号，例如 E-1000。
    """
    return {"E-2087": "赵敏"}[employee_id]


@gole.tool
def get_flight(flight_no: str) -> dict[str, str]:
    """查询航班的目的地和起飞时间。

    Args:
        flight_no: 航班号，例如 CA1234。
    """
    return {"MU5137": {"destination": "成都", "departure": "14:20"}}[flight_no.upper()]


@gole.tool
def get_product(name: str) -> dict[str, str]:
    """按商品名查询商品编码。

    Args:
        name: 商品名。
    """
    return {"sku": {"蓝牙耳机": "SKU-7731"}[name]}


@gole.tool
def get_stock(sku: str) -> int:
    """按商品编码查询库存数量。

    Args:
        sku: 商品编码，例如 SKU-0001。
    """
    return {"SKU-7731": 46}[sku]


@gole.tool
def get_latest_order(customer: str) -> dict[str, str]:
    """查询客户最近的一笔订单号。

    Args:
        customer: 客户姓名。
    """
    return {"order_id": {"李雷": "B2044"}[customer]}


@gole.tool
def get_tracking(order_id: str) -> str:
    """按订单号查询物流状态。

    Args:
        order_id: 订单号。
    """
    return {"B2044": "已到达杭州转运中心"}[order_id]


def _flaky(name: str, description: str, value: str) -> Callable[[], list[Tool]]:
    """Build a one-argument tool that fails on its first call and then succeeds."""

    def build() -> list[Tool]:
        attempts = 0

        def call(key: str) -> str:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise TimeoutError("服务暂时不可用，请重试")
            return value

        call.__doc__ = f"{description}\n\nArgs:\n    key: 要查询的编号。"
        return [gole.tool(call, name=name)]

    return build


@gole.tool
def get_schedule(date: str) -> str:
    """查询某一天的日程。

    Args:
        date: 日期，必须是 YYYY-MM-DD 格式。
    """
    parts = date.split("-")
    if len(parts) != 3 or len(parts[0]) != 4 or not all(part.isdigit() for part in parts):
        raise ValueError("日期格式错误，必须是 YYYY-MM-DD，例如 2025-03-05")
    return {"2025-03-05": "10:00 评审会；15:00 客户电话"}.get(date, "当天没有安排")


@gole.tool
def search_docs(query: str, limit: int) -> list[str]:
    """搜索内部文档，返回标题列表。

    Args:
        query: 搜索关键词。
        limit: 返回条数。
    """
    if limit > 10:
        raise ValueError("limit 不能超过 10")
    return ["《报销流程 2025 版》", "《差旅标准》", "《发票规范》"][:limit]


def _fixed(*tools: Tool) -> Callable[[], list[Tool]]:
    return lambda: list(tools)


_COMMON = _fixed(get_weather, get_stock_price, lookup_order)

CASES: list[Case] = [
    # 1. One tool, one argument.
    Case(
        "single-weather",
        "单工具单参数",
        "上海现在天气怎么样？",
        _COMMON,
        expect([("get_weather", {"city": "上海"})], "28"),
    ),
    Case(
        "single-stock",
        "单工具单参数",
        "NVDA 的最新股价是多少？",
        _COMMON,
        expect([("get_stock_price", {"symbol": "NVDA"})], "131.2"),
    ),
    Case(
        "single-order",
        "单工具单参数",
        "帮我查一下订单 A1001 的状态。",
        _COMMON,
        expect([("lookup_order", {"order_id": "A1001"})], "已发货"),
    ),
    Case(
        "single-population",
        "单工具单参数",
        "巴西有多少人口？",
        _fixed(get_population, get_weather),
        expect([("get_population", {"country": "巴西"})], "2.16"),
    ),
    # 2. Several arguments of different types.
    Case(
        "typed-booking",
        "多参数与类型",
        "在成都订一间大床双人房（double），住 3 晚，要含早餐。把确认号告诉我。",
        _fixed(book_room, get_weather),
        expect(
            [
                (
                    "book_room",
                    {"city": "成都", "nights": 3, "breakfast": True, "room_type": "double"},
                )
            ],
            "H327Q",
        ),
    ),
    Case(
        "typed-currency",
        "多参数与类型",
        "250.5 美元能换多少人民币？",
        _fixed(convert_currency, get_stock_price),
        expect(
            [("convert_currency", {"amount": 250.5, "source": "USD", "target": "CNY"})], "1803.6"
        ),
    ),
    Case(
        "typed-notification",
        "多参数与类型",
        "给张伟、李娜、王强三个人发一条加急通知，内容是“明天九点开会”。把回执号告诉我。",
        _fixed(send_notification, lookup_order),
        expect(
            [("send_notification", {"recipients": ["张伟", "李娜", "王强"], "urgent": True})],
            "N-58",
        ),
    ),
    Case(
        "typed-thermostat",
        "多参数与类型",
        "把空调调到制冷模式，23.5 度。设置后的能耗等级是多少？",
        _fixed(set_thermostat, get_weather),
        expect([("set_thermostat", {"temperature": 23.5, "mode": "cool"})], "B2"),
    ),
    # 3. Two calls; the second depends on the first result.
    Case(
        "chain-manager",
        "两步调用",
        "王芳的直属经理是谁？",
        _fixed(find_employee, get_manager),
        expect(
            [("find_employee", {"name": "王芳"}), ("get_manager", {"employee_id": "E-2087"})],
            "赵敏",
        ),
    ),
    Case(
        "chain-flight-weather",
        "两步调用",
        "航班 MU5137 的目的地现在天气如何？",
        _fixed(get_flight, get_weather),
        expect([("get_flight", {"flight_no": "MU5137"}), ("get_weather", {"city": "成都"})], "22"),
    ),
    Case(
        "chain-stock",
        "两步调用",
        "蓝牙耳机还有多少库存？",
        _fixed(get_product, get_stock),
        expect([("get_product", {"name": "蓝牙耳机"}), ("get_stock", {"sku": "SKU-7731"})], "46"),
    ),
    Case(
        "chain-tracking",
        "两步调用",
        "李雷最近那笔订单的物流到哪了？",
        _fixed(get_latest_order, get_tracking),
        expect(
            [("get_latest_order", {"customer": "李雷"}), ("get_tracking", {"order_id": "B2044"})],
            "杭州",
        ),
    ),
    # 4. No tool is needed.
    Case("none-capital", "不需要工具", "中国的首都是哪座城市？", _COMMON, no_tools("北京")),
    Case("none-arithmetic", "不需要工具", "2 加 3 等于几？", _COMMON, no_tools("5", "五")),
    Case(
        "none-translate",
        "不需要工具",
        "把 good morning 翻译成中文。",
        _COMMON,
        no_tools("早上好", "早安"),
    ),
    Case(
        "none-concept", "不需要工具", "用一句话解释什么是递归。", _COMMON, no_tools("自身", "自己")
    ),
    # 5. Recovery after a tool error.
    Case(
        "recover-report",
        "报错后恢复",
        "读取编号 R-17 的报告，告诉我结论。",
        _flaky("fetch_report", "按编号读取报告内容。", "结论：第三季度营收增长 12%"),
        recovered(("fetch_report", {"key": "R-17"}), "12%"),
    ),
    Case(
        "recover-database",
        "报错后恢复",
        "查询工单 T-305 的处理人是谁。",
        _flaky("query_ticket", "按编号查询工单的处理人。", "处理人：陈静"),
        recovered(("query_ticket", {"key": "T-305"}), "陈静"),
    ),
    Case(
        "recover-date",
        "报错后恢复",
        "我 2025年3月5日 有什么安排？请直接把“2025年3月5日”原样传给工具试一次。",
        _fixed(get_schedule),
        recovered(("get_schedule", {"date": "2025-03-05"}), "评审会"),
    ),
    Case(
        "recover-limit",
        "报错后恢复",
        "搜索“报销”相关文档，先试着要 20 条。告诉我找到的第一篇标题。",
        _fixed(search_docs),
        recovered(("search_docs", {}), "报销流程"),
    ),
]
