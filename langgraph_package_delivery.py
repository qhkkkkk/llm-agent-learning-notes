"""用 LangGraph 实现一个带条件路由的包裹配送流程。"""

import operator
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph


class PackageState(TypedDict):
    """所有节点共享的包裹状态。"""

    package_id: str
    origin: str
    destination: str
    status: str
    # 普通字段默认被新值覆盖；添加 reducer 后，新值会与旧值合并。
    history: Annotated[list[str], operator.add]
    total_distance: Annotated[int, operator.add]
    priority: str


def receive_package(state: PackageState):
    """揽收站：记录包裹已从始发地揽收。"""
    return {
        "status": "已揽收",
        "history": [f"在{state['origin']}揽收"],
    }


def sort_package(state: PackageState):
    """分拣中心：根据目的地选择分拣中心。"""
    destination = state["destination"]

    if "北京" in destination:
        sorting_center = "北京分拣中心"
    elif "上海" in destination:
        sorting_center = "上海分拣中心"
    else:
        sorting_center = "其他地区分拣中心"

    return {
        "status": "已分拣",
        "history": [f"分拣至{sorting_center}"],
    }


def standard_delivery(state: PackageState):
    """标准配送：使用普通陆运。"""
    return {
        "status": "运输中",
        "history": ["标准陆运"],
        "total_distance": 500,
    }


def express_delivery(state: PackageState):
    """加急配送：使用空运。"""
    return {
        "status": "加急运输",
        "history": ["空运加急"],
        "total_distance": 800,
    }


def final_delivery(state: PackageState):
    """最终站点：将包裹标记为已签收。"""
    return {
        "status": "已签收",
        "history": [f"已送达{state['destination']}"],
    }


def select_delivery(state: PackageState) -> Literal["备注加急", "无备注"]:
    """路由函数：返回路径标签，而不是直接调用下一个节点。"""
    if state["priority"] == "加急":
        return "备注加急"
    return "无备注"


def build_delivery_system():
    """组装并编译配送状态图。"""
    delivery = StateGraph(PackageState)

    delivery.add_node("揽收站", receive_package)
    delivery.add_node("分拣中心", sort_package)
    delivery.add_node("标准配送", standard_delivery)
    delivery.add_node("加急配送", express_delivery)
    delivery.add_node("最终站点", final_delivery)

    delivery.add_edge(START, "揽收站")
    delivery.add_edge("揽收站", "分拣中心")
    delivery.add_conditional_edges(
        "分拣中心",
        select_delivery,
        {
            "备注加急": "加急配送",
            "无备注": "标准配送",
        },
    )
    delivery.add_edge("标准配送", "最终站点")
    delivery.add_edge("加急配送", "最终站点")
    delivery.add_edge("最终站点", END)

    return delivery.compile()


TEST_PACKAGES: list[PackageState] = [
    {
        "package_id": "P001",
        "origin": "北京",
        "destination": "上海",
        "status": "待揽收",
        "priority": "普通",
        "history": [],
        "total_distance": 0,
    },
    {
        "package_id": "P002",
        "origin": "广州",
        "destination": "乌鲁木齐",
        "status": "待揽收",
        "priority": "加急",
        "history": [],
        "total_distance": 0,
    },
]


def run_demo() -> None:
    """分别执行普通件和加急件，打印最终状态。"""
    delivery_system = build_delivery_system()

    for package in TEST_PACKAGES:
        result = delivery_system.invoke(package)
        print(f"\n配送包裹: {package['package_id']}")
        print("最终状态:", result["status"])
        print("配送历史:", result["history"])
        print("总里程:", result["total_distance"])


if __name__ == "__main__":
    run_demo()

