可恢复 Agent 与 MCP 实验室

本实验沿用教程中的培训补贴案例。员工已入职十四个月，报名之前已获得
直属经理批准；preapproved=True 表示这个时间顺序已经核实。
年度上限 2000 元、已使用 600 元，本次申请 900 元。规则计算得到剩余额度
1400 元；模型或受控决策器选择工具，执行器准备草稿，人确认后才提交。
所有申请、身份和金额均为本地练习数据。

一、从课程目录运行

以下命令以实验包中含 index.html 和 labs/ 的目录为当前目录，需要 Python 3.11+。
该目录可任意命名或移动；PDF 可以单独存放。Shell 配置命令采用 Bash。
示例 ./training-*.sqlite3 是当前目录下的数据库文件名，可以自行改名；同一次
暂停、查看与恢复必须使用同一路径。首次练习选尚不存在的文件名，重复练习
要恢复初始额度时另选新文件名，不能只更改 run-id。默认 trace 也写在当前目录。
核心 Agent 实验只使用标准库；MCP 实验另外安装官方 SDK。

先观察一次完整演示：
python3 labs/agent_lab/agent_lab.py demo --db "./training-demo.sqlite3"

demo 使用 scripted 受控决策器，并自动模拟一次批准。它用于理解状态转移，
不衡量大模型规划能力。用新的数据库路径可重新得到相同的初始年度额度。

再亲自完成确认：
python3 labs/agent_lab/agent_lab.py pause --run-id lesson-001 --db "./training-lesson.sqlite3"
python3 labs/agent_lab/agent_lab.py show --run-id lesson-001 --db "./training-lesson.sqlite3"
python3 labs/agent_lab/agent_lab.py resume --run-id lesson-001 --db "./training-lesson.sqlite3" --approve

第二条命令应显示 awaiting_confirmation、amount_yuan=900、remaining_yuan=1400。
第三条命令应显示 completed 和 application_id。重复第三条命令，申请编号与
申请数量保持不变。resume 不带 --approve 表示拒绝，状态转为 rejected。
命令中的 --user 是本地宿主模拟的已认证身份；它不进入模型工具的参数表。

二、沿五个入口阅读代码

1. ScriptedDriver.decide：选择下一工具。换成 LiveDriver 后，选择来自模型。
2. pack_context：将可信状态快照、用户请求和完整工具交互对交给模型。
3. Store.execute：校验名称和参数，用宿主身份读数据，保存工具结果。
4. run_loop：先保存 pending 调用，再执行、记录结果并推进状态。
5. Store.approve：重新检查资格与年度余额，按 run_id 幂等写入本地申请。

读取工具叫 lookup_policy、lookup_employee；准备工具叫 prepare_application。
prepare_application 只能提出草稿，不能代替宿主调用 approve。模型返回的
amount_yuan 必须与用户请求一致，不能自行调高金额或增加 user_id。

数据库中，runs 保存状态；calls 保存每次调用及结果；applications 保存最终
业务结果。程序在工具执行前后或申请提交后退出，恢复时可以读取已提交记录。
如果申请已经提交而运行状态尚未更新，恢复时先核对业务表并回到 completed；
此时再次传入拒绝，也不会把已经发生的提交伪装成 rejected。
草稿与确认提交共用 current_entitlement，将初始已用额度与当前数据库中已提交
的申请相加后计算余额；同一数据库先提交 900 元，再申请 700 元时，草稿即可
看到剩余 500 元并拒绝超额申请。确认时仍会再计算，处理等待期间发生的变化。
本地数据库表示一个固定的教学年度，初始已用 600 元和所有申请都属于该年度。
它不按系统日期切换年度，也不支持跨年余额；新一轮练习使用新的数据库路径。
确认阶段保留 BEGIN IMMEDIATE 事务，将重新查询余额与申请写入串行执行。
一个 run_id 在本实验中由一个运行器串行推进。部署多个 Worker 时，需在领取
run 的位置加入数据库租约；第 12、15 章给出这种工作协调方法。

三、上下文与预算实验

python3 labs/agent_lab/agent_lab.py pause --run-id short --db "./training-short.sqlite3" --max-steps 1
python3 labs/agent_lab/agent_lab.py pause --run-id tiny --db "./training-tiny.sqlite3" --context-bytes 100

第一条会在完成一次工具查询后停止，状态为 budget_exhausted。第二条无法容纳
必要的状态和工具定义，状态为 failed。默认活动运行时间预算为 30 秒；人等待
批准的时间不计入。外部 HTTP 调用获得剩余超时额度，调用返回后再检查总预算。

pack_context 按序列化后的 UTF-8 字节控制输入体积，删除最旧的完整工具交互对。
这个数字是 bytes，不是模型 token 数。它帮助你看到“哪些信息必须留下”，
接真实模型后可将计数器替换为匹配模型的 tokenizer，并预留输出 max_tokens。
LiveDriver 的输出上限为 384 token；实际 token 消耗来自提供商 usage。
state 中的业务事实保留来源，旧聊天文字不会替代提交时重新查询的业务状态。

默认 agent-trace.jsonl 保存 run_id、步骤、工具名、参数、状态、耗时及模型
usage；用 --trace 指定另一个文件。它不记录 API Key 或完整模型输入。

四、接入真实 Qwen 工具调用

根据百炼控制台选择有权限且支持 Function Calling 的模型与地域。
配置环境变量（占位符应替换为自己的配置）：
export DASHSCOPE_BASE_URL='https://your-workspace-compatible-endpoint/compatible-mode/v1'
export DASHSCOPE_API_KEY='your-key'
export QWEN_MODEL='qwen-plus'

python3 labs/agent_lab/agent_lab.py pause --mode live --run-id live-001 --db "./training-live.sqlite3"

LiveDriver 使用 /chat/completions、tools、tool_calls 和 tool 消息。
示例请求关闭并行工具调用与思考模式，每轮只接受一个工具调用；具体模型应
支持这些选项。若接口不支持，调整 LiveDriver 中的参数并用对应模型文档核对。
调用失败会保留 failed 状态，不会切换到 scripted。正常返回草稿后，使用
上面的 show/resume 命令查看并确认。live 会产生账户中的正常模型调用费用。
本教程制作时没有调用真实付费模型；REST 请求协议通过注入响应的测试验证。

五、让读取工具走真正的 MCP stdio 连接

python3 -m venv .venv-agent
source .venv-agent/bin/activate
python -m pip install -r labs/agent_lab/requirements-mcp.txt
python labs/agent_lab/mcp_client.py
python -m unittest discover -s labs/agent_lab -p 'test_mcp.py' -v

客户端自动启动 mcp_server.py，发现两项工具并读取结构化结果。服务端使用
官方 Python SDK 的 MCPServer，客户端使用 Client 与 StdioServerParameters。
配套依赖固定为已实测的 mcp==2.3.0。服务端用 Pydantic 输出模型声明字段，
使 SDK 生成输出 schema 并返回 structured_content。
stdout 留给协议消息，业务日志应写 stderr。工具输入 schema 不含 employee_id，
服务端固定绑定模拟主体 employee001；这是一台本地教学宿主的身份模型。

将相同 MCP 连接接进 Agent：
python labs/agent_lab/mcp_client.py --agent --db "./training-mcp.sqlite3" --run-id mcp-001
python labs/agent_lab/agent_lab.py resume --db "./training-mcp.sqlite3" --run-id mcp-001 --approve

第一条的两个读取步骤通过 MCP，prepare_application 与确认提交保留在宿主。
这展示了协议适配器的职责：替换工具连接方法，同时保留执行器的业务规则。
工具返回错误时，先检查 is_error，再读取 structured_content；本实验对失败
读取停止运行。连接与协议协商由 SDK 管理。网络部署时，宿主身份应来自服务端
认证流程，再按资源范围授权，不能把本地固定身份直接用于多人服务。

六、按业务结果评测

python3 labs/agent_lab/agent_lab.py evaluate --trace evaluation-evidence.jsonl
python3 -m unittest discover -s labs/agent_lab -p 'test_agent_lab.py' -v

六个场景是：正常申请、工龄不足、缺少事前审批、超过余额、用户拒绝、提交后
中断再恢复。trace_valid 检查每一次工具调用的名称、参数和先决读取顺序；
business_state_valid 独立检查最终申请表。成功场景必须恰有一行，owner、金额、
run_id 与该场景完全相同，application_id 必须与批准返回的编号相同；所有拒绝
或不合资格场景必须为零行。额外行、错误归属、金额或编号均不能通过评测。
source_valid 与 status_valid 分别检查来源版本和最终状态，passed 汇总这些条件。
读取政策与员工资料的先后顺序可以变化，但必须在准备申请之前完成。
默认结果为 scripted 下 6/6 通过；它证明本地执行规则与评测代码能工作。

接好真实模型后，主动运行：
python3 labs/agent_lab/agent_lab.py evaluate --mode live

该命令运行相同六个场景，用临时数据库模拟人工决定与业务提交，并报告真实
usage。若想算费用，按自己账户的单位价格附加：
--input-price-per-million 你查到的输入单价 --output-price-per-million 你查到的输出单价
两个价格使用相同币种；未提供价格、任一调用缺少完整用量或调用结果未知时，
model_cost 为 null。缺失 token 字段保持 null，后续调用不会把它补成零；
model_usage_complete 标出整次运行的用量是否齐全。每次运行可重定向为
JSONL，重复多次观察任务完成率、波动与成本，比较模型或提示的变更。

evaluate 的 --trace 现在逐场景追加一行 evaluation_result，默认文件仍是
agent-trace.jsonl。每行包含 experiment_id、run_id、模式、模型名、运行预算与
计价配置、已校验工具动作及结构化结果摘要、错误类别、停止原因和最终申请行。
临时数据库清理后仍可检查这些证据；重复执行通过 experiment_id 区分。
--max-steps、--max-seconds 与 --context-bytes 同样约束评测中的运行循环。
证据最多保留 32 个工具动作和 16 行最终申请，字符串标识最多保留 160 字符，
截断列表带有相应的 *_truncated 标记；业务比较仍使用全部原始行。
模型名最多保留 120 字符。证据不复制原始模型输入输出、任意工具文本、
提供商错误正文或 API Key；errors 只记录阶段、异常类别及是否为预期故障。
resume_after_commit 的预期中断会单独标记，其他批准异常使该场景失败并保留证据。

尝试完成三项练习：
（1）让一个模型多次查询已经存在的事实，观察步数与费用，再修改工具描述。
（2）在测试中改变审批状态，确认草稿不能绕过提交时的资格检查。
（3）将读工具接到你的知识库服务，保留来源版本与权限上下文，再用同一组
     业务用例检查最终结果。

官方参考（核查于 2026-10-27）：
https://help.aliyun.com/zh/model-studio/qwen-function-calling
https://py.sdk.modelcontextprotocol.io/client/
https://modelcontextprotocol.io/docs/2026-07-28/develop/build-server
https://modelcontextprotocol.io/docs/2026-07-28/develop/build-client

2.1 补充：循环先传递剩余活动时间，再在工具返回后复核；迟到的结果可写入 journal，但不会进入等待确认。同步工具和数据库等待无法由运行器强制抢占，因此是协作式预算；等待人工确认的离线时间不计入 active_seconds。MCP读取超时会请求取消本地future，不保证远端动作已停止。

小入口：loop_walkthrough.py 展示 Action/Observation；langgraph_read_loop.py 提供真实 LangGraph 两个只读工具脚手架，需自行准备依赖。test_langgraph_nodes.py 的普通节点测试与实际图测试分开；缺依赖时跳过实际图，不能称为框架验证。此脚手架没有模型、持久检查点或业务写入。
