from __future__ import annotations

from datetime import datetime

from langchain_core.tools import tool


@tool
def get_current_time() -> str:
    """获取当前本地时间，适合回答“现在几点”“今天几号”等问题。"""
    now = datetime.now().astimezone()
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")
