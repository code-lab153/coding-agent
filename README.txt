Coding Agent

Git 仓库地址
https://github.com/code-lab153/coding-agent

一、如何运行

项目需要 Python 3.11，模型服务须兼容 OpenAI Responses API。

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"

在 PowerShell 中设置环境变量：

$env:MODEL_API_KEY="你的API Key"
$env:MODEL_NAME="模型名称"
$env:MODEL_BASE_URL="模型服务地址"

运行：

python main.py --trace compact "你的编程任务"

也可省略任务参数，启动后再输入任务；使用 --trace full 可查看完整工具输出。Agent 只在 workspace/ 中操作，请勿把真实密钥写入仓库。

二、特色功能

本项目未使用 Agent 框架，自行实现“模型决策—工具执行—结果反馈”的循环。Agent 提供 list_files、read_file、search_text、write_file、edit_file、run_command 六种工具，可以浏览项目、检索和修改代码并运行测试。

系统支持 AGENTS.md 项目规则、TaskState 状态记录、修改后的验证门、重复调用检测、命令安全策略和上下文压缩。完整历史单独保留，模型输入会精简较旧的大型工具输出，在控制上下文长度的同时保留关键证据。

三、其他说明

tests/ 包含主项目自动化测试；demo/csv_showcase/ 提供由任务说明、验收测试和样例数据组成的演示项目。将演示内容复制到 workspace/ 后运行 Agent，可观察它从零实现 CSV 分析程序并通过测试的过程。
