上下文工程实验：先选择完整证据，再使用剩余预算放入历史

路径约定：从实验包中含 index.html 和 labs/ 的目录运行，目录名和存放位置不限；PDF 可另存。保留包内相对结构。Shell 配置与激活命令采用 Bash。
在课程目录运行（Python 3.11+，默认仅标准库）：
python3 labs/context_lab/context_lab.py demo
python3 labs/context_lab/context_lab.py evaluate
python3 -m unittest discover -s labs/context_lab -p 'test_context.py' -v

阅读顺序：Evidence → Counter → assemble → validate_answer。
assemble 每次加入完整证据或完整对话轮次后重新计数；超预算整块跳过。
当前问题与系统要求属于必须保留的部分，容纳不了时明确报错。
confirmed_facts 只接纳明确确认的用户陈述，服务端办理资格仍查询业务记录。
检索证据先按授权/版本过滤，再按来源与正文去重，保留来源和引用标识。
同一文档的不同条款可以同时保留；超预算的块不会挡住后续可用条款。

默认 unit=utf8_bytes_proxy：计算序列化消息的 UTF-8 字节，用于可重复的离线预算实验。
这不是模型 token 计数，不对应模型最大上下文，也不作为费用依据。
--history all 从旧到新尝试完整轮次，recent 从新到旧尝试，最终均按时间顺序输出。
“all”仍受预算约束；比较不同预算下选择了哪些轮次，以及关键条件是否完整。

精确分词扩展：在实验虚拟环境执行
python -m pip install -r labs/context_lab/requirements-tokenizer.txt
然后准备与部署模型匹配的
本地 tokenizer 文件及 chat_template；使用：
python3 labs/context_lab/context_lab.py demo --tokenizer "/path/to/local/tokenizer" --budget 4096 --reserve 512
引号中的 /path/to/local/tokenizer 是占位路径，替换为自己的本地目录；
可以在课程包外或包含空格、中文，不要求特定目录名。
此模式 local_files_only=True、trust_remote_code=False，不自动下载模型或运行自定义代码。
apply_chat_template 将角色标记、正文和生成起始标记一起编码；代码指定
return_dict=False 取得 token ID 列表，结果单位为本地模板 tokens。
只有模型与模板匹配时，计数才适用于对应部署。云服务可能额外包装输入，应以其
接口限制与返回 usage 校准，并留出裕量。本实验未调用云模型。

validate_answer 检查 JSON 字段、状态和引用归属；引用存在不代表结论被原文支持。
练习：构造一份字段合法但金额错误的回答，说明为什么仍需事实/条件评分。
evaluate 的两项结果检查预算和培训证据保留，衡量的是上下文构造，不是生成准确率。

对应教材第20章；官方参考：
https://huggingface.co/docs/transformers/main/en/chat_templating
