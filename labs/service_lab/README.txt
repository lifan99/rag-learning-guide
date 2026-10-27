真实 RAG 服务实验室
对应教程第 19、20、23 章；命令均从 index.html 所在目录执行。
路径约定：从实验包中含 index.html 和 labs/ 的目录运行，目录名和存放位置不限；PDF 可另存。保留包内相对结构。Shell 配置与激活命令采用 Bash。

你将在同一条调用链中看到：原文 → 分块 → 向量化与入库 → 密集召回/BM25 → RRF → 可选重排 → 证据上下文 → JSON 回答 → 引用校验 → 逐题评测。

1. 先跑通可检查的接口

创建独立环境并安装 Flask：
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r labs/service_lab/requirements.txt

Ubuntu 如果提示缺少 ensurepip，先安装发行版的 python3-venv 包，再创建环境。

入库、启动：
    python labs/service_lab/service.py --mode fake ingest
    python labs/service_lab/service.py --mode fake serve --port 8000

另开终端查询：
    curl http://127.0.0.1:8000/health
    curl -X POST http://127.0.0.1:8000/ask -H 'Content-Type: application/json' -d '{"question":"培训补贴需要哪些条件？","top_k":1}'

fake 模式用确定性的哈希词项向量和证据摘录，回答会标出“离线抽取演示”，不会发网络模型请求。它用于观察 HTTP、数据契约、排序、引用和错误处理。health 的 alive 表示进程可响应，indexed 表示存在索引清单，不代表上游模型已经可用。

打开 service.py，按 main → Engine.ingest → Engine.ask → create_app 的顺序读。先找每个函数的输入输出，再看内部细节。Engine 接受可替换 provider，测试因此可以独立控制模型结果。

2. 换成真实模型与 Milvus

    python -m pip install -r labs/service_lab/requirements-live.txt

从百炼控制台取得与你的业务空间、地域和密钥匹配的 API Host。以下地址采用北京业务空间形式；将 YOUR_WORKSPACE 替换为实际空间 ID。若控制台给出其他入口，使用该入口，不混用地域。

    export DASHSCOPE_BASE_URL='https://YOUR_WORKSPACE.cn-beijing.maas.aliyuncs.com/compatible-mode/v1'
    export DASHSCOPE_RERANK_URL='https://YOUR_WORKSPACE.cn-beijing.maas.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank'
    read -rsp 'DASHSCOPE_API_KEY: ' DASHSCOPE_API_KEY
    export DASHSCOPE_API_KEY
    export RAG_EMBEDDING_MODEL=text-embedding-v4
    export RAG_EMBEDDING_DIM=1024
    export RAG_CHAT_MODEL=qwen-plus
    python labs/service_lab/service.py --mode live ingest
    python labs/service_lab/service.py --mode live serve --port 8000

停掉之前的 fake 服务后再使用同一端口。live 会调用所配置的模型接口，按你的账户计费；默认不启用重排。每个模型请求的网络超时由 RAG_TIMEOUT 控制，默认 20 秒。该示例不自动重试，也不会在上游失败后转成 fake。

默认 Milvus Lite 将数据库保存为 labs/service_lab/.state/milvus.db。已有 Milvus 服务时，可设置 MILVUS_URI=http://127.0.0.1:19530。当前示例面向本地、无需认证的实验实例；带认证的远程部署需在 Engine.client 中明确接入认证配置。

向量服务使用兼容 API 的 /embeddings；chat 使用 /chat/completions，并启用 JSON Object、非思考模式。JSON Object 约束 JSON 语法，validate_answer 继续验证字段、状态和本轮引用 ID。引用是否真正支持结论要在评测样例中检查。换模型时先查是否支持相同调用参数，再做小批回归。

入库只处理当前服务端教学身份可见且 active 的文档。清单保存语料、模型、维度与入口的签名；参数或资料变化后会要求重新 ingest。Milvus 集合名取自该签名，重复入库按稳定 chunk ID upsert，不会每次清空其他集合。

3. 做一次明确的对照实验

先用已有的四道题观察报告结构：
    python labs/service_lab/service.py --mode fake evaluate --top-k 1 --output service-k1-results.json
    python labs/service_lab/service.py --mode fake evaluate --top-k 3 --output service-k3-results.json

在已配置 live 的同一个环境中，比较密集召回与混合召回：
    RAG_RETRIEVAL=dense python labs/service_lab/service.py --mode live evaluate --top-k 3 --output service-dense-results.json
    RAG_RETRIEVAL=hybrid python labs/service_lab/service.py --mode live evaluate --top-k 3 --output service-hybrid-results.json

再单独打开重排：
    RAG_RETRIEVAL=hybrid RAG_RERANK=1 python labs/service_lab/service.py --mode live evaluate --top-k 3 --output service-rerank-results.json

RAG_RETRIEVAL 还支持 bm25。live 重排调用 qwen3-rerank；可用 RAG_RERANK_MODEL 更换当前账户支持的模型。fake 下打开重排是按词法分数重新排序的模拟，用来检查流程，不用于推断真实 reranker 的效果。

报告保存每题 question、gold_document_ids、三路候选 ID、最终文档、实际上下文、答案、引用、各阶段耗时、provider usage 和关键配置。summary 包含 Hit@K、Recall@K、截断 RR 均值（mrr）、服务成功率、结构与引用 ID 校验通过率，以及成功请求的平均/p95耗时。这里的 latency 是成功 Engine.ask 的业务链路耗时，包含本地资料检查、检索、可选重排、生成和校验，不含客户端到 Flask 的 HTTP 传输，也不代表并发压力下的响应时间。服务失败按检索 0 分计入整体均值，并保留错误类别。

top_k_chunks 是候选片段窗口上限：先取前 K 个候选，再按上下文预算保留完整块，实际进入上下文的块可能更少。Hit/Recall 与 rr/mrr 都按 final_document_ids 评价最终入选片段；rr 使用该列表中的位置，保留重复文档占据的名次，不能当作预算筛选前的全量候选排名指标。Recall 的 gold 是不同文档 ID，同一文档的重复片段只计一次覆盖。四题只用于检查程序和口径，p95 用最近秩方法，扩充题集与重复次数后再讨论性能。结构与引用 ID 通过率不等于事实正确率。

成本先按 stage/model 汇总 usage 中实际返回的 token，再使用该地域、模型、日期的输入输出单价换算；embedding、rerank 与 generation 分开计算。保留单价日期和计费单位，和平台账单核对。示例不硬编码会变化的价格，fake 的 usage 为空。

4. 把自己的数据接进来

文档文件采用与 labs/fixtures/documents.json 相同的 JSON 数组，每条有 id、version、title、text、allowed_groups、active。题目文件采用 questions.json 格式，每条有 id、question、gold_document_ids。

    python labs/service_lab/service.py --mode fake --documents "my-documents.json" ingest
    python labs/service_lab/service.py --mode fake --documents "my-documents.json" evaluate --questions "my-questions.json" --output my-results.json

my-documents.json 与 my-questions.json 只是示例文件名，请替换为实际路径；文件可任意命名或放在课程目录外，带空格的路径保留引号。--output 指定评测结果路径；RAG_STATE_DIR 可指定索引状态目录，默认在脚本旁的 .state 下。更改文档位置后按实际配置重新 ingest。

可先把 CRUD-RAG 中十个事件整理成这两个契约。gold_document_ids 由你做参考新闻到语料的映射，参考答案保存在评测侧。不要把题目、答案字段拼进知识原文。更换文档参数时，serve 与 evaluate 要使用相同 --documents。

5. 上下文和访问范围如何进入实现

RAG_CONTEXT_CHARS 控制证据块的字符预算，默认 5000。系统按排序保留完整块，不从金额或资格条件中间截断。它是便于观察的字符控制；准确 token 预算需要接入与模型匹配的 tokenizer，并把系统消息、问题、历史、工具结构和预留输出都纳入计算。第 20 章据此进一步完成上下文实验。

服务监听 127.0.0.1，使用服务端固定的 employee 教学身份，不提供登录功能。/ask 只接受 question 与 top_k，客户端不能提交 groups 来扩大权限。查询前重读源文件，Milvus 以允许的 chunk ID 预过滤，外部重排和生成前再次复核签名。将它扩成多用户服务时，把固定身份替换成认证中间件产生的可信访问上下文；规模增大后，用索引中的租户/权限字段取代长 ID 列表。

6. 验证和面试练习

    python -m unittest discover -s labs/service_lab -p test_service.py -v

请求追踪练习：Flask 在每次请求入口生成 trace_id，Engine.ask 的直接调用在业务入口生成；均早于参数校验、检索和模型调用。HTTP /ask 成功、400 参数错误和 503 业务失败的 JSON 都带 trace_id，X-Trace-ID 响应头使用同一值；/health、404、413 也可用该响应头与 JSON 关联。客户端提交的 X-Trace-ID 不作为服务器追踪 ID。评测的成功行与 result 共用 trace_id，参数/业务失败行同样保留它。

用命令行启动时，rag_http 与 rag_ask 关联日志会写到标准错误流；测试用 assertLogs 捕获。日志只记录 trace_id、固定路由名、状态码、运行模式和异常类型，不记录问题、原文、答案、请求 URL、密钥或上游异常正文。评测 JSON 则按实验设计保存问题、上下文和答案，应按其数据权限保存，不能当成可公开的运行日志。这里只实现本进程的关联 ID，没有把它透传为跨服务分布式追踪。新增测试检查成功/失败 ID 一致、每次请求 ID 独立、评测失败关联，以及关联日志不包含私有负载。

测试包括 Flask 请求、拒绝伪造组信息、私有/旧版本不入库、资料撤权后拒绝旧索引、无效引用、RRF 去重、评测记录、live 失败不降级、本地 HTTP 模型替身的请求/响应契约，以及装好依赖后的真实 Milvus Lite 入库/查询。测试密钥为固定字符串 local-test-only，不读取你的真实密钥、不产生模型账单。

完成后回答三道追问：
（1）为什么模型返回了合法 JSON，仍然需要引用与语义检查？
（2）为什么变更 embedding 模型后不能直接复用旧向量？
（3）增加重排让平均指标变好，但 p95 和成本都增加，你如何决定是否上线？

官方资料（核对日期：2026-10-27）
https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api
https://help.aliyun.com/zh/model-studio/text-rerank-api
https://help.aliyun.com/en/model-studio/qwen-structured-output
https://milvus.io/docs/quickstart.md
https://flask.palletsprojects.com/en/stable/testing/

2.1 补充：本地服务没有登录与多用户认证；groups 为固定教学配置。生成后返回前再校验资料/权限快照，变化时返回错误；这不等于跨系统原子授权。/health 的 indexed 只检查清单存在，不是完整就绪检查。网络与进程基础见教材第10、11、13、15章短桥。
