from __future__ import annotations

from datetime import datetime

from langchain_core.tools import tool


@tool
def get_current_time() -> str:
    now = datetime.now().astimezone()
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")
