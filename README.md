# Memory-aware Agent Edge Serving — Phase 1

本地版本：一个共享的 llama-server、ready-step FIFO、真实 Retail Agent 执行、受控 microbench 和多任务性能回放。实验结果及局限见 `results/REPORT.md`；原始材料保留在 `Requirements/`。

`configs/local.json` 是本次运行的配置快照。启动参数由 CLI 控制，实验矩阵在 `phase1/bench.py`；修改该 JSON 不会自动改变执行配置。

## 目录

| 路径 | 内容 |
| --- | --- |
| `phase1/core.py` | FIFO、HTTP 客户端、请求事件与 token 前缀工具 |
| `phase1/adapters.py` | 严格工具 JSON 格式兼容，不补全工具决策 |
| `phase1/capture.py` | 官方 Retail orchestrator + 本地 Qwen agent + DeepSeek user；程序化 ENV 评价 |
| `phase1/bench.py` | micro、cache probe、完整多步回放与资源采样 |
| `phase1/analyze.py` | 从原始数据生成 CSV、PNG、PDF |
| `phase1/server.py` | 本地 Metal 或 AWS CPU 后端启动 |
| `phase1/check_restore.py`、`phase1/verify.py`、`phase1/report.py` | 恢复正确性检查、工作量审计、中文结果报告 |
| `phase1/package.py` | 生成迁移压缩包，排除凭据、权重与临时 KV 文件，并检查凭据未泄露 |
| `results/manifest.json` | 实际版本、模型与数据 hash、启动命令 |
| `results/raw/` | 原始 live 轨迹、请求、任务、运行与资源日志 |
| `results/diagnostics/` | 修复前的诊断失败记录；不混入正式比较 |
| `results/summary/` | 图表使用的导出数据 |
| `figures/` | 独立的高清 PNG 与矢量 PDF 图表 |

## 安装与复现

在项目根目录执行。模型文件约 2.3 GiB；凭据不进入代码或实验配置。

```bash
uv venv --python 3.13 .venv
git clone --depth 1 --branch v1.0.1 https://github.com/sierra-research/tau2-bench third_party/tau2-bench
uv pip install --python .venv/bin/python -r requirements-lock.txt
mkdir -p models results/cache results/logs
curl -fL --retry 3 --continue-at - \
  -o models/Qwen3-4B-Q4_K_M.gguf \
  https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/bc640142c66e1fdd12af0bd68f40445458f3869b/Qwen3-4B-Q4_K_M.gguf
.venv/bin/python -m pytest tests -q
.venv/bin/python -m phase1.server
curl http://127.0.0.1:18080/health
```

使用已安装的 llama.cpp build 11669 / commit `620d2225c`。其他版本不能保证参数和 cache 行为一致。启动器后台运行 server；停止时只向 `results/server.json` 中本实验的 PID 发送 SIGTERM。

DeepSeek 仅用于 live 用户模拟，不在正式回放期间调用。优先读取 `DEEPSEEK_API_KEY` 环境变量；也支持用户明确指定的本地单行 `API.md`。文件已忽略，不要将其打包提交。不要把 key 放进命令行参数、配置或报告。

```bash
.venv/bin/python -m phase1.capture --split train --count 2 --max-steps 40
.venv/bin/python -m phase1.capture --split test --count 8 --max-steps 40
.venv/bin/python -m phase1.bench micro
.venv/bin/python -m phase1.bench cache
.venv/bin/python -m phase1.bench macro
.venv/bin/python -m phase1.check_restore
.venv/bin/python -m phase1.verify
.venv/bin/python -m phase1.analyze
```

依次运行，禁止采集与性能实验同时占用推理服务。live 采集按现有 task 文件跳过已完成记录。性能文件采用 append-only；重新复现实验前使用新的结果目录或归档原结果，不要把两轮完整复跑追加后当成额外独立重复。当前程序以项目根目录为工作目录。

## 实验口径

`K` 限制正在进行模型调用的请求数；`P=2` 是固定服务端 slot 数。工具等待不占 `K`。FIFO 按每一步 ready 后的入队顺序 dispatch，不能把它理解为对完整 task 的永久优先权。

micro 使用精确 token 数与固定生成数量，关闭 prompt cache。cache probe 固定原前缀并追加本轮实际生成的 token，比较保留、空等、覆盖、清除、保存恢复。

macro 的来源为预先固定的官方 test 子集，包括失败运行。为控制本机运行时间，主 pilot 使用冻结随机顺序中的前 4 个完整任务（不按成功与否筛选），`K=1/2` 各重复 3 次；全部 8 个 live 记录均保留。回放保持记录的步骤数和输出 token 数，以真实输入增长量标定新观察长度，并保留本轮生成的历史。这是 **tau-derived append-only workload**；原始聊天模板重写被转换为追加式，不是逐字节忠实 trace，也不是重新执行客服语义。默认 `recorded` 间隔含真实采集中的用户 API 和工具等待，不能统称为生产工具延迟；可用 `--gap-mode zero` 或 `--gap-mode both` 扩展，零间隔仅作敏感性对照。

吞吐是完成所有定义步骤的 replay jobs/min，不能称为语义成功 goodput。live ENV 得分只验证数据库状态和环境断言，没有评估 Retail 的全部自然语言断言，因此不是官方 leaderboard 得分。

服务端 `predicted_ms` 是生成阶段的墙钟区间，并发 prefill 等活动可能使其暂停或变长；不把它解释为独占的 GPU decode 计算时间。请求 round trip 还可能含运行时内部等待，因此不以总时间减两个 timings 字段精确推算内部排队。

各配置固定相同的任务、步骤、生成量、KV 精度、容量、slots 与缓存策略。使用有限 batch 与到达抖动，尚不代表生产 open-loop 到达过程。P95 是小样本描述性分位数；图中的 SD 表示重复运行波动，不是总体置信区间。失败和未完成任务与完成时延分开报告。

## 转到 AWS CPU

保持同一模型 SHA256、workload 文件、runtime commit、KV 精度、`P/C` 与实验代码。在 Linux 上构建同一源码版本，将 `llama-server` 加入 PATH：

```bash
git clone https://github.com/ggml-org/llama.cpp
git -C llama.cpp checkout 620d2225c
cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build --config Release -j --target llama-server
export PATH="$PWD/llama.cpp/build/bin:$PATH"
# 在新 AWS 工作副本中归档 Mac 结果，避免追加混入 CPU 测量。
mkdir -p reference/mac
mv results reference/mac/results
mv figures reference/mac/figures
mkdir -p results/raw results/logs results/cache results/summary
.venv/bin/python -m phase1.server --cpu --threads 8
.venv/bin/python -m phase1.bench micro
.venv/bin/python -m phase1.bench cache
.venv/bin/python -m phase1.bench macro --workload reference/mac/results/workload.json
```

线程数按实际实例 vCPU 校准，不能照搬 laptop 的吞吐或到达率。只做回放无须再次调用 DeepSeek；原始来源文件仍应归档以便审计。CPU 侧需要另记实例型号、runtime 编译配置与模型 hash；本机的 `manifest.json` 只是 Mac 实验的证据。Mac 与 AWS 分别完成同机对照，绝对速度不混合统计。
