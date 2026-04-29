"""最小 Ollama 探针。完整服务建议使用 `uvicorn backend.src.slothbearflow_backend.main:app` 启动。"""

import sys

from langchain_ollama import ChatOllama

if __name__ == "__main__":
    llm = ChatOllama(model="deepseek-r1:7b")
    print(sys.version)
    print(llm.invoke("你好， 请介绍一下你自己"))
