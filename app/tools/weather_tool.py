from __future__ import annotations

from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气摘要。

    当前为离线模式，适合开发联调和工具调用演示。
    支持城市：北京、上海、广州、深圳、杭州、成都等。
    """
    weather_data = {
        "北京": {"condition": "晴", "temperature_c": 23, "humidity": 35, "wind": "东南风 2 级"},
        "上海": {"condition": "多云转阴", "temperature_c": 18, "humidity": 68, "wind": "东北风 3 级"},
        "广州": {"condition": "雷阵雨", "temperature_c": 28, "humidity": 82, "wind": "南风 2 级"},
        "深圳": {"condition": "大雨", "temperature_c": 26, "humidity": 91, "wind": "东南风 4 级"},
        "杭州": {"condition": "阴", "temperature_c": 19, "humidity": 55, "wind": "东风 1 级"},
        "成都": {"condition": "阴转多云", "temperature_c": 22, "humidity": 48, "wind": "西南风 2 级"},
    }
    data = weather_data.get(city)
    if not data:
        return (
            f"【天气查询结果】\n城市：{city}\n状态：暂无离线天气数据\n"
            "说明：当前工具运行在离线模式，请接入真实天气 API 后替换实现。"
        )
    return (
        "【天气查询结果】\n"
        f"城市：{city}\n"
        f"天气：{data['condition']}\n"
        f"气温：{data['temperature_c']}°C\n"
        f"湿度：{data['humidity']}%\n"
        f"风力：{data['wind']}\n"
        "数据来源：本地离线样例数据"
    )
