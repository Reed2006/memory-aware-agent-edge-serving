# Phase 1 本机设计与需求审读

审读日期：2026-10-25（America/Los_Angeles）。这是一份设计与可行性分析；已检查本机配置、已安装程序的帮助信息、官方文档和对应版本源码，尚未运行模型、收集轨迹或完成性能实验。以下配置与实验规模是待预实验校准的建议，不能当作结果。

**项目究竟在研究什么**

我们要让多个会反复调用模型和工具的 Agent 共用一台内存有限的机器。Agent 等工具时不需要推理，但之前算出的 KV 状态可能对下一步有用。系统需要在“给其他任务腾出执行容量”和“避免返回的任务重新计算历史”之间取舍；如果选择保留，还要让返回后的任务有机会及时使用它。

Phase 1 实现并测量固定并发 FIFO，判断这种冲突是否真实存在、代价多大。Phase 2 才根据证据决定是否联合调整执行顺序和缓存保留。当前不能预设重算一定是主瓶颈，也不能预设联合策略一定优于调优后的 FIFO。

**5 份文件分别提供了什么**

| 文件 | 内容与约束 | 对设计的影响 |
| --- | --- | --- |
| `Requirements/CSE 291A Fall 2026 Course Projects.md` | 通用评分、计算资源、复现要求；Phase 1 截止已改至 2026-11-04，Phase 2 为 2026-12-10 | 分离开发与评测 workload，固定版本，保留失败与超时，报告重复实验波动；交源码、数据、指令、日志和简洁报告 |
| `Requirements/CSE 291A Fall 2026 Course Projects (1).md` | Option 1 的具体要求 | FIFO + 固定并发；覆盖短/长输入、短/长输出、多步和受控工具等待；同时测 request 与完整 task；解释瓶颈 |
| `Requirements/Professor feedback.md` | 课堂反馈文字稿，非针对本 memo 的单独评语 | 不能只有人工参数实验；需要真实应用依据。改变 FIFO 要测公平性。压缩、分层、迁移等是深化建议，不是 Phase 1 必做项 |
| `Requirements/edge_agent_queue_cache_memo.pdf` | 两页 Phase 2 设计：continuation 优先、工具感知保留、pin/save/evict、联合消融 | 研究方向合理，但保留、清除、保存能否控制真正稀缺的资源必须先验证；Phase 1 不直接实现整套策略 |
| `Requirements/ChatGPT-Phase 1实施方案-20261025-1703.md` | 2082 行讨论，从基本 FIFO 到 Retail 轨迹、Mac 开发与 AWS 评测 | 保留 request/task 分层、真实任务依据、负面结果意识；修正缓存控制、回放前缀、工具延迟真实性和本机可行性的细节 |

通用说明提到 Option 1 默认 AWS，但 Option 1 专项说明明确允许 laptop。就现有材料而言，本机可作为完整实验平台，没有“正式结果必须来自 AWS”的专项要求。AWS 可以是后续严格物理内存限制或跨平台复核的选择。

**本机实际情况**

| 项目 | 检查结果 | 含义 |
| --- | --- | --- |
| 设备 | MacBook Air，Apple M4；10 核 CPU（4 性能 + 6 能效），10 核 GPU | 优先原生 Metal；主比较保持同一后端 |
| 内存 | 16 GB 统一内存 | CPU/GPU 共用物理内存，不能把 RSS 与 Metal buffer 简单相加 |
| 系统 | macOS 15.3，arm64 | 不把 Linux cgroup 方法直接套在 macOS 上 |
| 存储 | `df -h .` 显示约 40 GiB 可用 | 比 GPT 对话截图中的 25 GB 新；仅下载一个主模型并限制 checkpoint 数量 |
| llama.cpp | 已安装；`0.7.0`，build `11669`，commit `620d2225c` | 先复用并冻结此版本；源码核查使用同一 commit |
| Ollama | 0.18.2；已有 `gemma3:4b` 与 embedding 模型 | 可以辅助检查环境；主实验直接使用 llama-server，方便观察与控制缓存 |
| Python | 默认 3.14.3；另有 `/opt/homebrew/bin/python3.13` 3.13.12 | τ-bench v1.0.1 与当前主线要求 `>=3.12,<3.14`，使用独立 3.13 环境 |
| 其他 | uv、Homebrew、Command Line Tools 已有 | 不必先安装 Docker、Ray 或完整推理框架 |
| 当前内存状态 | swap 约 7836 MiB / 8192 MiB；内存压缩占用明显 | 这只是已有占用，不代表当前换页速率；正式实验要看运行区间增量 |

开始正式实验前减少无关后台负载，固定电源状态与实验姿态，记录启动前内存状态。MacBook Air 长时间负载下还要防止温度与频率变化混入策略差异：交错/随机化配置顺序、保持相同预热，并观察同一控制请求的速度漂移。若出现系统压力，记录该轮现象和停止原因；按预先规则区分机器背景干扰与被测负载导致的失败，不能事后挑选数据。

不需要为了研究内存而把整台机器推到系统 OOM。固定 KV 池容量、固定额外缓存预算，已经可以研究容量与重算。macOS 上的监控退出只是软保护，不能宣称实现了任意 GPU 分配下的进程物理内存硬上限。

**对原 GPT 讨论和 memo 的关键修正**

1. 固定并发必须说清楚限制什么。定义 `K` 为正在调用模型的请求数，`P` 为服务端 slot 数，`J` 为尚未完成的 Agent task 数。工具等待释放 `K` 许可，但 task 仍存在。主实验固定 `P` 与 context/cache 配置，只改变 `K <= P`。任务总数不能直接称为模型并发数。
2. FIFO 应按“某一步已准备好并进入队列”的顺序 dispatch。工具没返回的旧任务不能阻塞已经 ready 的新请求。因而 FIFO step order 不等于按最初 task arrival 一直优先。continuation 返回后排在新请求后面，也不自动违反原任务到达顺序意义上的同一种公平性；必须定义评测口径。
3. 有 ready backlog 且工具等待已释放许可时，工作守恒的 FIFO 不会仅仅因为一个 Agent 等工具就让推理完全空闲。出现空闲时应分辨没有 ready work、`K` 太小、slot 被策略保留、内部限制还是实现错误，不能直接归因于 FIFO。
4. KV 逻辑状态、预分配 buffer、进程驻留和系统总压力是不同对象。Q4 权重量化也不意味着 KV 是 4 bit；本方案固定 KV 为 f16。清除一个 slot 不保证 RSS 下降，保存状态也不自动完成清除。
5. 本机运行时另有 host prompt cache，`--cache-ram` 默认上限 8192 MiB。原讨论对这一层控制不足。必须显式设置；否则可能既多用统一内存，又遮住 slot 被覆盖后的重算现象。
6. `id_slot` 是请求路由，不是持久 pin。真正保留需要外部调度器不让其他请求覆盖该 slot，并协调 save/erase/restore 的生命周期。代价可能是减少可计算的 slots，而不是多占一块按任务动态分配的 RAM。
7. 时间变长不等于缓存自动失效。若没有 TTL 或竞争请求，单纯等更久可能完全不改变复用。分别控制墙钟等待和期间插入的请求数/长度，再测固定到达率下二者的共同影响。
8. Retail 工具是本地业务逻辑，可能非常快，不自带真实数据库网络延迟。人为 sleep 0.5/2/5 秒只能叫延迟敏感性分析；记录本地实测延迟作为默认，不称为生产网络 trace。
9. 固定旧 prompt、让模型重新生成、下一步又使用旧轨迹中的回复，会在回复不同处破坏前缀。固定 token 上限也不固定输出长度。性能回放必须验证输入/输出 token 工作量与前缀关系，不能只保存几个长度数字。
10. 长任务、已执行多步的任务不一定更接近结束。Phase 2 不使用测试轨迹的未来剩余步数/真实工具完成时间作在线决策。工具类型统计、历史 prefill 成本可以使用，未来信息只能是单独标注的 oracle 上界。
11. 保存文件后马上恢复可能主要命中文件页缓存；在统一内存上 CPU 与 GPU 也不是独立物理内存层。不能仅凭一次 API 成功就宣称“GPU 到 RAM 分层”或“磁盘卸载节省了物理内存”。
12. 小规模预实验不能支撑稳定的 P99。先看样本数、ECDF、P50/P95 和每轮波动；重复同一批轨迹也不等于新增同样多种独立真实任务。

**推荐的本地运行方式**

采用原生 `llama-server + Metal`、Python 3.13 的异步 workload runner、一个 ready-step FIFO 队列及独立采集程序。Agent 在进程内保存文本/工具状态；模型权重由一个 server 共享。无需为每个 Agent 起一个模型进程。

主模型候选选官方 **Qwen3-4B-GGUF / Q4_K_M**，使用 non-thinking 模式，先做工具调用格式与任务长度验证。它的完整注意力结构较适合解释 KV 成本；选择它不是因为已经证明能完成 Retail。若当前内存或推理耗时不合适，可先用 Qwen3-1.7B 做环境预验，但主比较一旦选定模型，就固定文件、SHA256、模板、权重和 KV 精度。不要把两种模型的数据拼成一条并发曲线。[官方模型卡](https://huggingface.co/Qwen/Qwen3-4B-GGUF)

建议最初设 `P=2`，总 context `C=16384`，明确关闭 unified KV，以启动日志实际确认每 slot 容量（目标约 8192）。先测 `K=1,2`。真实模板、全部 tool schema、历史和最大输出都需要计入容量；如果真实任务超出，不得静默截断或删掉重要政策。优先在开发集上调整 context/模型配置，再冻结。`P=4` 仅在预算与耗时允许后作为第二组容量实验，不能与 `P=2` 混为只改变 `K`。

按 Qwen3-4B 官方配置，36 层、8 个 KV heads、head_dim=128，f16 KV 的理论量级为：

`2 × 36 × 8 × 128 × 2 = 147456 bytes/token = 144 KiB/token`。

因此 16384 个 KV token 约 2.25 GiB，32768 约 4.5 GiB；还需权重、计算 buffer、Python 和系统开销。这是估算，最终以启动日志分配和运行采样为准，不能只拿模型文件大小估计实验预算。[官方 config](https://huggingface.co/Qwen/Qwen3-4B/blob/main/config.json)

可采用的两种固定缓存环境：

| 环境 | 设置思路 | 用途 |
| --- | --- | --- |
| A：仅 slot 保留 | prompt caching 开启；`--cache-ram 0`；明确关闭 idle-slot 额外保存 | 解释 slot 竞争和重复 prefill；属于显式受限配置，不称为默认生产行为 |
| B：简单有界主存缓存 | 同一 `P/C`，额外 `--cache-ram 512` MiB，明确 idle-slot 设置 | 验证现成简单机制是否已足够，避免为弱 baseline 设计复杂策略 |

先冻结其他参数。不要在主比较中同时开关 unified KV、换 KV 精度、改 context、换模型。容量扫描是单独实验。

本机帮助信息已确认以下开关存在：`--parallel`、`--ctx-size`、`--no-kv-unified`、`--cache-prompt`、`--cache-ram`、`--no-cache-idle-slots`、`--no-context-shift`、`--metrics`、`--slots`、`--slot-save-path`、`--fit off`、`--jinja`、`--chat-template-kwargs`。正式启动后还要保存实际生效配置，禁止自动调整悄悄改变实验容量。

**先做运行时能力验证，再跑完整 Agent**

这是 Phase 1 的第一个交付里程碑，不是直接实现 Phase 2。下面实验都使用同一模型、模板和采样规则，每种情形前明确缓存初始状态。

| 检查 | 操作 | 必须得到的证据 |
| --- | --- | --- |
| 正常复用 | A 在 slot 0 执行第一步，再提交保留原历史的 continuation | 请求级 reused tokens 上升、新处理 token 与 prefill 时间变化；记录实际 token 前缀 |
| 空等对照 | A 等不同时间，其间不提交竞争请求 | 判断运行时是否仅因时间流逝清除缓存 |
| 覆盖干扰 | A 等待期间让 B/C 使用 slots，再让 A 返回 | 自动路由与强制同 slot 覆盖分别检查；区分 slot 指派与 token 容量问题 |
| 显式清除 | idle slot 执行 erase，再返回 A | 复用减少、重新 prefill；同时测 RSS 与运行时逻辑容量，不能只看 API 返回 |
| 保存恢复 | idle A：save → erase → B 使用该 slot → 清理/确认目标 slot 可恢复 → restore A → continuation | 保存/恢复 token 与 bytes、完整客户端耗时、恢复后复用、输出/状态一致性检查 |

源码中的 save 分支写文件并返回，没有自动 erase；erase 分支调用 prompt_clear。slot 忙时，操作可能被延后。因此未来 cache manager 必须管理互斥与 reservation，并把等待/操作开销计入端到端时间。接口存在只证明实现入口存在，不能替代运行验证。[本机对应源码](https://github.com/ggml-org/llama.cpp/blob/620d2225c/tools/server/server-context.cpp)

保存路径只用于本地检查，设总量上限并逐轮清理本实验生成的文件。对比应包括 save + erase + restore 的生命周期成本以及阻塞其他请求的代价，不能只比 restore_ms 与 re-prefill。文件页缓存冷热状态单独标注；本机预算下不通过制造巨大文件或系统换页来伪造“冷磁盘”。

若这一步发现缓存几乎无收益、恢复不可靠或 FIFO 已足够，照实记录；主线仍可完成 baseline 和容量研究，Phase 2 收缩为有证据支持的机制。

**具体采用什么 Agent workload**

使用 Sierra 官方 `tau2-bench` 仓库的 **v1.0.1 Retail 文本环境**作为任务来源，冻结到解析后的 commit 和数据 hash。选择旧发布版是为了稳定项目范围；当前主线已是 2.0，任务、提示和评判有变化，不能把它们的得分混用。仅安装 text core，不引入 voice extras。[版本差异说明](https://github.com/sierra-research/tau2-bench#comparing-with-v101)、[v1.0.1 依赖](https://github.com/sierra-research/tau2-bench/blob/v1.0.1/pyproject.toml)

Retail 任务涵盖查询、取消/修改订单、退换货等，业务工具是本地 Python 状态操作。每个并发 task 使用独立初始数据库副本，避免不同客户任务相互修改状态形成额外实验因素。必须保留 benchmark 的身份确认与操作确认等语义；不能用任意三步脚本替代后继续称为原版 τ-bench。[工具实现](https://github.com/sierra-research/tau2-bench/blob/v1.0.1/src/tau2/domains/retail/tools.py)、[业务规则](https://github.com/sierra-research/tau2-bench/blob/v1.0.1/data/tau2/domains/retail/policy.md)

重要限制：该版 Retail 明确不支持 solo_mode。完整运行需要用户侧交互，而不只是 Agent 和工具。用户模拟模型、可能需要的评估模型必须明确配置，不能依赖默认云端调用。[环境实现](https://github.com/sierra-research/tau2-bench/blob/v1.0.1/src/tau2/domains/retail/environment.py)

本地分三层使用：

- **采集执行过程**：先对约 5 个开发任务做 pilot，本地模型分别扮演 agent 和 user，顺序采集，不把此过程当正式 serving 曲线。记录失败，不只保留成功任务。如果本地 user 或 agent 不可靠，可按课程要求用 TritonAI 离线收集，凭据通过环境变量提供；这是后备路线，不是本次已执行操作。
- **正式系统比较**：从冻结的任务/执行过程构建依赖正确的性能回放，测本机模型推理、FIFO 和缓存，不在测量期间调用额外 user LLM 或外部 API。明确标为 benchmark-derived replay，不声称有生产到达 trace。
- **语义核查**：在一个开发集选定的配置上做少量完整 live Agent 运行，使用真实工具和原版可用 evaluator，报告实际通过的校验项。若本地 user 也共享此 server，其推理必须进入同一受控队列并单独标记角色、计入资源负载；不能当隐形流量，也不能直接拿它与 agent-only replay 的绝对吞吐等同。固定用户脚本只能称为派生应用测试，不能冒充完整原版得分。

开发/评测任务按冻结版本提供的 split 划分，再预先固定子集和选择种子，例如约 10 个开发任务、20 个评测任务；任务不足或成本过高就明确缩小，不按成功与否挑选。任务类型、步数、最长上下文和实际输出长度的分布全部报告。超过当前容量的任务记为适配失败；若采用长度可容纳的子集，披露筛选规则、数量及对外推的限制。

本地工具耗时作为默认工具延迟。另加受控延迟只研究“工具慢时会发生什么”，不是测出了真实零售后端性能。用户等待与工具等待分开记录。

**回放如何避免做了不同的工作**

本项目不需要移植整个 TensorRT-LLM，但它的官方方法说明提供了一个参考：记录任务的多步结构、prefix identity、输出长度与工具等待，用 job 级指标评价系统。不能只保留 ISL/OSL，也不能把所有前缀替换成互相无关的随机 token。[官方方法说明](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog27_Evaluating_Agentic_Serving_with_Trace_Replay_and_Job_Level_Metrics.md)

第一版采用线性 task、固定步骤图与固定工具结果的受控回放，保存原始 messages、模板展开后的 token IDs、每次生成数量、工具/用户间隔和上下文保留关系。性能回放通过原生 completion 接口提交 token IDs；先验证该接口与真实 Agent 模板展开后的 prefill/复用行为相符。

对保留全部历史的连续两步，用 `P_(s+1) = P_s || G_s || Δ_(s+1)` 构造输入：`G_s` 是本轮刚实际生成的 token，`Δ` 是固定的后续角色标记、工具结果等。这样缓存中的已计算前缀与下一步输入一致。若原轨迹有模板重写、删除 reasoning、裁剪或其他非追加行为，按其真实前缀关系显式表达；第一版不悄悄把这种轨迹改成纯追加后声称完全等价。

性能回放可用固定 `n_predict`、ignore_eos 和空 stop 集合控制输出工作量，逐请求核查 `predicted_n`。这只用于系统负载，生成文本不驱动工具决策，不能拿其最终文本计语义成功。各策略的生成 token 内容可能不同，但总长度、步骤依赖和本任务的保留前缀关系必须匹配；对共享前缀另作验证。

若只做“固定每步旧 prompt 再自由生成”，应明确那是另一种 prompt 回放，不能声称完整复用了本轮生成的历史。若以后要做到跨策略逐 token 完全相同，还需独立验证强制输出/teacher forcing 实现及其开销，本 Phase 1 不假设现有 API 已提供这种能力。

回放结束只计 **replay completed jobs/s**；只有完整语义执行并通过指定 evaluator 的任务才计 **validated successful jobs/s**。两者不混用 goodput 名称。

**FIFO 的实现边界**

task 到达后，第一步进入一个 `asyncio.Queue`；严格按 ready enqueue 序号 dispatch，最多 `K` 个模型请求在途；同一个 task 的下一步必须等上一步和相应工具/用户事件完成，才重新排队。工具等待期间释放模型许可。不持有覆盖整个 Agent 生命周期的 semaphore。

使用一个 dispatcher 维护序号、在途数和 idle slots，避免多 worker 取队后实际提交次序反转。主 baseline 允许运行时自动选 slot，记录其选择；固定 affinity、reservation、pin 等属于单独对照或 Phase 2。若为测量而改变路由，要明确写成不同 baseline，不能暗中强化/弱化 FIFO。

HTTP 连接池容量至少覆盖 `K`，防止客户端连接池成为隐藏队列；设置任务/请求超时和 worker 异常清理，取消后确认服务端请求已停止才回收在途许可。显式记录 HTTP 错误、上下文超限、超时、取消及重试；主实验关闭自动重试，或把所有尝试和成本完整计入。

小范围测试应覆盖：ready FIFO 顺序、在途数不超过 `K`、工具等待释放许可、步骤依赖、失败/取消后不泄漏许可、回放长度与前缀关系。现有目录尚无代码和测试，因此这些是后续实现要求，不是已完成验证。

**最小实验矩阵：避免全排列**

| 实验 | 变量与规模建议 | 控制项 | 回答的问题 |
| --- | --- | --- | --- |
| E0：接口与复用 | 上述 A/B 干扰、erase、save/restore；至少 3 次 | 同一模型、slot/context 配置与明确的初始缓存状态 | Python 是否真的有必要的可见性和控制能力？ |
| E1：单次成本 | 输入 512/2048/4096，输出 64/256；串行；先确认完整 prompt 不超容量 | 缓存关闭或清空，固定采样与实际生成量 | prefill 与 decode 各占多少？长历史重算有多贵？ |
| E2：等待与竞争 | 一次只变一个因素：等待 0/0.5/2 秒；期间竞争任务 0/1/4；短/长前缀 | 先单因素，再取少量交叉；缓存 A/B 分开 | 是时间、竞争次数、slot 覆盖还是长度造成复用损失？ |
| E3：完整任务并发 | `P=2` 固定；`K=1,2`；同一批 10–20 个评测任务；3 个种子 | 模型、数据、容量、输出工作量固定 | 并发收益、排队代价、完整任务尾延迟如何变化？ |
| E4：到达压力 | 使用开发集定标的约 0.5/0.9/1.2 倍参考吞吐；共享同一 arrival schedule | 约 60–120 个到达任务实例/轮，按预算调整 | 从空闲到过载，队列/失败/完成量如何变化？ |
| E5：缓存简单对照 | 在 E3/E4 的代表配置比较 `cache-ram=0/512` | 固定其他参数与相同 KV 池 | 简单现成功能是否已足够解决问题？ |

E4 的参考速率必须来自开发集，在评测前锁定；不能每种 `K` 用自己的吞吐重新定标后称为同负载。Poisson arrival 是受控假设；有限 batch 是另一类实验，分别命名。发生器按绝对单调时钟调度，不等上一任务完成才发新任务；记录 planned/actual arrival 与 lag，避免把闭环误当开环。

每轮先预热计算路径，再建立规定的缓存初态；cold model load 独立记录。同一配置至少 3 次，报告每轮点和变异，不只是把所有样本池化。短期先跑 E0/E1/E3，只有时间允许再扩展 E2/E4/E5 的代表点。

有限 batch 以首个到达至末个终态的 makespan 计完成速度。开环实验冻结 arrival window，并继续 drain 或到预设 deadline；同时报告窗口内完成量、最终完成量、未完成/失败数和 drain 时间。超时不是零延迟，不能只在成功样本上报告一个漂亮 P95 而隐藏完成率。声明 completion 分位数只对应完成样本，并展示未完成比例。

E2 的受控插入请求属于机制实验；真实到达负载下，工具等待会改变这段时间里可能插入多少竞争工作，这是另一个端到端因素。两者不能合并成“等待时间直接让 cache 过期”的解释。

**记录哪些数据，才能解释结果**

| 对象 | 最低记录内容 | 解释边界 |
| --- | --- | --- |
| task | ID、原任务 ID/复制实例 ID、类型、到达/首 dispatch/终态、步数、状态、评价结果 | latency 包含全部步骤、排队、工具和用户等待 |
| request | task/step/role、enqueue/dispatch/首 token/完成时间、slot、实际输入/输出、错误 | `done-dispatch` 是客户端 round trip，可能含内部排队，不能直接称模型 service time |
| 模型内部 | `timings.cache_n/prompt_n/prompt_ms/predicted_n/predicted_ms`，原始响应与服务端日志 | 先实际核验字段含义；不把全局指标当逐请求值 |
| 缓存操作 | action、slot、开始/结束、bytes、saved/restored/erased tokens、失败和等待 | 不能用 save 成功代替释放成功 |
| 资源 | server/runner RSS，必要时系统 footprint 观测；启动时权重/KV/compute buffer；系统 pressure、swap/pageout 增量 | 统一内存不重复相加；sampled peak 不是绝对瞬时峰值 |
| 队列 | ready 长度、在途数、ready 但无 dispatch 的时间、运行时 deferred/processing 状态 | 分开外部队列、内部队列与没有 ready work |

本机版本的 chat 响应文档已有 `timings.cache_n` 等字段；原生 completion 还有 `tokens_cached/tokens_evaluated`。正式跑之前用小请求确认各自语义并保存原始字段。`/metrics` 提供服务端累计量，`/slots` 提供 slot 状态；它们不自动等于完整的逐任务 KV 字节占用。[固定版本接口说明](https://github.com/ggml-org/llama.cpp/blob/620d2225c/tools/server/README.md)

优先依赖 request response 与事件日志；资源采样约 200–500 ms，较重的 `/slots` 与 `/metrics` 轮询降到约 1 Hz，单独验证 collector 开销。若需请求级 server-start，使用带请求关联 ID 的日志或小范围补充 instrumentation；不能用“总耗时减 prefill/decode”精确反推内部排队，因为仍有其他开销。

公平性从 Phase 1 开始保存：首次 dispatch 等待、新任务与 continuation 的等待分布、短/长任务的 P95、最大 ready wait、超时/未开始比例。Phase 2 再计算同任务相对 FIFO 的 slowdown。可另用同任务隔离执行时间归一化；工具等待占比高会掩盖调度退化，因而同时报告 ready waiting。不要靠一个 Jain index 代替全部公平性分析。

**Phase 1 应交出的结论形态**

至少有：一套可运行 FIFO 与日志；冻结 workload 与版本清单；复现实验指令；原始失败/成功记录；吞吐/完成时延随并发与负载的曲线；cache reuse、prefill 与内存证据；简洁报告解释机制。主要看完整任务完成，token/s 辅助解释。

| 观察到的证据 | 合理的 Phase 2 方向 | 不应作出的结论 |
| --- | --- | --- |
| cache 丢失显著增加 prefill，竞争越强端到端影响越大 | continuation 与缓存保留的联合实验；有界 reservation/aging | “任何内存受限模型都需要同样策略” |
| save/restore 可靠且总生命周期成本有优势，slot 容量确实可再利用 | 单一 checkpoint 机制 + 调度，做 queue-only/cache-only/joint 消融 | “写了文件就减少了同量物理 RAM” |
| 简单 512 MiB cache 已基本消除问题 | 把它作为强 baseline，研究更紧预算或收缩主张 | 跳过简单对照，只展示弱 FIFO |
| decode 占主导、reuse 很高、内存压力弱 | 收缩到并发/容量研究，或在合法长上下文 workload 上重新验证 | 强行宣称发生了 cache thrashing |
| 本地小模型无法可靠驱动任务 | 报告质量失败；改用离线有效轨迹作系统研究并限制语义主张 | 把脚本步骤跑完称为客服问题解决 |

memo 中的保留评分也应在 Phase 2 简化：保留收益取决于“工具返回 + 再排队到实际执行”的间隔，以及期间是否被覆盖，不只看 tool duration。内存成本要与节省时间统一单位，例如容量占用乘影子价格；不提前调一套未经数据支持的复杂权重。若把所有 slots 保留给尚未 ready 的任务，要有超时释放/保底可运行容量，防止饥饿与停滞。

**按当前截止时间安排实施**

| 日期（本地） | 工作与验收点 |
| --- | --- |
| 10/26 | 整理背景负载，冻结 server 与 Python 环境，选模型；E0 确认复用、覆盖、保存/恢复是否成立 |
| 10/27–10/28 | 实现严格定义的 FIFO、事件记录、失败清理与必要测试；完成 Retail 5-task pilot 与容量筛查 |
| 10/29–10/30 | 冻结开发/评测子集，完成回放；验证每步 token 工作量和前缀关系；完成 E1/E3 小样本 |
| 10/31–11/01 | 运行代表性负载、竞争与简单缓存对照，至少 3 次；记录真实耗时后决定扩大规模 |
| 11/02 | 分析性能/公平性和资源证据；补少量 live 核查与必要反例 |
| 11/03–11/04 | 整理报告、原始数据、失败记录与可复现入口，完成干净环境复跑和提交 |

本机 Metal 的完整 Phase 1 可以先独立成立。仅当需要严格 Linux 内存上限、背景干扰无法控制或希望验证另一后端时再迁移 AWS；重定标到达速率并在新平台上完整重跑对照，不把 Mac 与 AWS 绝对速度混在一起。

本次只新增本设计文档。没有修改原始 Requirements，没有安装依赖、下载模型、调用付费模型、启动推理或生成实验结果。
