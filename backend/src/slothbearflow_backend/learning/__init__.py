"""后台复盘学习层（Hermes Background Review 范式）。

主链路回答完成后，由后台 review agent 复盘本轮对话，把用户偏好/经验沉淀为
Memory、可复用做法沉淀为 Skills，落盘为 Markdown 文件（真相源）+ SQLite 索引（派生）。
review 只经 LearningStore 写入学习命名空间，绝不污染主会话与对话历史。
"""
