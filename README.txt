Coding Agent

Git 仓库地址：
https://github.com/code-lab153/coding-agent

一、运行方式

建议使用 Python 3.11。安装依赖后，设置以下环境变量：

MODEL_API_KEY：模型 API Key
MODEL_NAME：模型名称
MODEL_BASE_URL：OpenAI-compatible API 地址（可选）

运行命令：

python main.py "你的编程任务"

Agent 默认在 workspace/ 目录中进行文件读取、代码修改和命令执行。API Key 不应写入仓库。

二、特色功能

本项目未使用 LangChain、OpenAI Agents SDK、AutoGen 等 Agent 框架，自行实现完整 Coding Agent Loop。LLM 根据当前上下文生成结构化 ToolCall，Controller 在本地执行工具，并将 ToolResult 返回模型形成“决策—执行—反馈”的循环。

Agent 提供 list_files、read_file、search_text、write_file、edit_file、run_command 六种工具，支持项目浏览、代码检索、文件创建、精确修改以及测试和命令执行。

项目进一步实现 TaskState 任务状态管理、修改后的 Verification Gate、重复工具调用检测、AGENTS.md 项目级指令、轻量 Command Policy 和结构感知 Context Manager。Canonical History 保留完整历史，模型输入对较旧的大型工具结果进行安全压缩，以支持更长的编程任务。

三、演示任务

demo/csv_showcase/ 中提供一个只有需求、验收测试和样例 CSV 数据的编程任务，不包含实现代码。将其复制到 workspace/ 后运行 Agent，可复现 Agent 从零构建 Tkinter CSV Data Analyzer、执行测试并根据反馈完成修改和验证的全过程。