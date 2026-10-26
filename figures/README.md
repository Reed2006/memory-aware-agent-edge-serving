# Phase 1 图表

每张图提供高清 PNG 和矢量 PDF；原始数据在 `results/raw/`，导出表在 `results/summary/`。重新生成：项目根目录运行 `.venv/bin/python -m phase1.analyze`。

| 图 | 内容 | 阅读重点 |
| --- | --- | --- |
| `01_micro_cost` | 输入处理、生成区间、总输出速率 | 对比短/长上下文以及 K=1/2；生成区间并非纯 GPU decode 时间 |
| `02_cache_reuse` | 保留、空等、覆盖、清除、保存恢复 | 对比 continuation 的 prefill 成本与复用比例 |
| `03_macro_performance` | 完整多步回放的完成速度与 P50/P95 | 固定同一任务与 token 工作量，比较 K=1/2 的重复运行 |
| `04_macro_distribution` | 完成时延 ECDF、首次请求与 continuation 等待 | 看尾部和等待代价，不能把重复任务当成独立真实任务 |
| `05_execution_timeline` | 一轮任务的执行时间线 | 蓝绿色是模型 round trip，灰色是 FIFO 等待，空白是步骤间间隔 |
| `06_memory` | RSS、系统可用内存、swap-out 增量 | 统一内存下 RSS 不等于整体物理 footprint；有预分配 buffer |
| `07_live_validation` | 8 个完整 Retail 任务的实际执行 | ENV 检查通过不等于通过所有自然语言断言 |
| `08_checkpoint_cost` | 重建历史与保存/恢复机制成本 | 对数纵轴；温热文件缓存，尚不能称为调度器端到端加速 |
| `09_macro_cache` | 全部输入复用与 continuation 完整历史复用 | 公共 system/tool 前缀可能掩盖具体 Agent 历史的丢失；后者按前一步 token 数推导 |

误差棒为重复运行的标准差。micro 每个配置与缓存条件各重复 3 次；macro 主 pilot 每批 4 个完整来源任务，K=1/2 各重复 3 次。P95 只作小样本描述，不作总体置信区间或生产服务承诺。
