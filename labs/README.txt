RAG 系统化课程：离线检索实验室

路径约定：从实验包中含 index.html 和 labs/ 的目录运行，目录名和存放位置不限；PDF 可另存。保留包内相对结构。Shell 配置与激活命令采用 Bash。
环境：Python 3.11+，只使用标准库。
在课程目录运行：
  python3 labs/rag_lab.py demo
  python3 labs/rag_lab.py evaluate --top-k 1
  python3 labs/rag_lab.py evaluate --top-k 3
  python3 -m unittest discover -s labs -p 'test_lab.py' -v

demo：查看候选、来源、版本、词项相似度与引用上下文。
evaluate：在四道固定问题上计算文档级 Hit@k 与 Recall@k。
inspect-crud PATH：扫描指定 JSONL，检查 questions/answers/news1 字段并报告行号。

正文统一采用虚构企业制度，适合研究切分、检索、权限和版本变化。
教学检索器使用词项计数向量；第 6—10 章先解释表示、检索与服务契约；按核心路线到第 19 章接通真实 Embedding、Milvus 和生成服务后，再返回第 6 章完成真实向量对照。
详细步骤见 index.html 的“配套实验室”。

新版进阶实验（第19—24章）：
service_lab/README.txt：HTTP服务、真实模型与Milvus。
context_lab/README.txt：上下文选择、计量和结构化结果。
agent_lab/README.txt：可恢复工具流程、MCP与业务评测。
各目录有独立依赖文件，按 index.html 的核心路线与当前单元入口运行，不要求从第 1 章顺读到第 24 章；先区分离线机制验证与真实模型结果。
