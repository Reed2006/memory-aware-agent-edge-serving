"""Write a Chinese results report from the verified exported measurements."""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def main():
    root = Path('results')
    verified = json.loads((root / 'verification.json').read_text())
    restore_check = json.loads((root / 'restore_validation.json').read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    micro = pd.read_csv(root / 'summary/micro_runs.csv')
    cache = pd.read_csv(root / 'summary/cache.csv')
    macro = pd.read_csv(root / 'summary/macro_summary.csv')
    requests = pd.read_csv(root / 'summary/macro_requests.csv')
    jobs = pd.read_csv(root / 'summary/macro_jobs.csv')
    cache_macro = pd.read_csv(root / 'summary/macro_cache.csv')
    live = pd.read_csv(root / 'summary/live_quality.csv')
    checkpoint = pd.read_csv(root / 'summary/checkpoint_cost.csv')
    resources = pd.read_csv(root / 'summary/resources.csv')
    workload = json.loads((root / 'workload.json').read_text())
    best_k = int(macro.groupby('k').jobs_per_minute.mean().idxmax())
    pass_count = int((live.env_reward == 1).sum())
    now = datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(timespec='seconds')

    micro_rows = []
    for length in [512, 2048, 4096]:
        for k in [1, 2]:
            x = micro[(micro.length == length) & (micro.output == 64) & (micro.k == k)]
            micro_rows.append([f'{length:,}', k, f'{x.prefill_ms.mean()/1000:.2f}',
                               f'{x.decode_ms.mean()/1000:.2f}', f'{x.generated_tokens_s.mean():.2f}'])
    cache_rows = []
    for length in [2048, 4096]:
        for mode in ['retain', 'wait', 'overwrite', 'erase', 'save_restore']:
            x = cache[(cache.length == length) & (cache['mode'] == mode)]
            cache_rows.append([f'{length:,}', mode, f'{x.prompt_ms.mean()/1000:.3f}',
                               f'{x.prompt_ms.std()/1000:.3f}', f'{100*x.reuse_fraction.mean():.2f}%'])
    macro_rows = []
    for k in [1, 2]:
        x = macro[macro.k == k]
        cm = cache_macro[cache_macro.k == k]
        macro_rows.append([k, len(x), f'{int(x.completed.sum())}/{int(x.tasks.sum())}',
            f'{x.jobs_per_minute.mean():.3f} ± {x.jobs_per_minute.std():.3f}',
            f'{x.p50_s.mean():.1f}', f'{x.p95_s.mean():.1f}',
            f'{100*cm.input_reuse_fraction.mean():.1f}%',
            f'{100*cm.continuation_history_hit.mean():.1f}%'])
    job_rows = []
    source = {t['task_id']: t for t in workload['tasks']}
    for task_id in workload['selected_ids']:
        x = jobs[jobs.task_id.astype(str) == task_id]
        job_rows.append([task_id, len(source[task_id]['steps']),
                         f'{x[x.k == 1].latency_s.mean():.1f}',
                         f'{x[x.k == 2].latency_s.mean():.1f}'])
    live_rows = [[str(row.task_id), int(row.agent_calls), int(row.generated_tokens),
                  int(row.max_prompt), '通过' if row.env_reward == 1 else '未通过',
                  row.termination] for row in live.itertuples()]
    resource_rows = []
    for suite in ['micro', 'cache', 'macro']:
        x = resources[resources.label.str.startswith(suite)]
        resource_rows.append([suite, f'{x.peak_server_rss.max()/2**30:.2f}',
                              f'{x.min_system_available.min()/2**30:.2f}'])
    checkpoint_rows = []
    for length in [2048, 4096]:
        x = checkpoint[checkpoint.length == length]
        checkpoint_rows.append([f'{length:,}', f'{x.saved_mib.mean():.1f}',
                               f'{x.save_roundtrip_s.mean():.3f}', f'{x.restore_roundtrip_s.mean():.3f}',
                               f'{x.handling_and_prefill_s.mean():.3f}'])
    pair = jobs.pivot(index=['repeat', 'task_id'], columns='k', values='latency_s')
    slowdown = pair[2] / pair[1]
    timing_rows = []
    mean_recorded_gap = sum(sum(s['gap_s'] for s in t['steps']) for t in workload['tasks'])/len(workload['tasks'])
    for k in [1, 2]:
        rq = requests[requests.k == k]
        totals = rq.groupby(['label', 'task_id'])[['queue_s', 'roundtrip_s']].sum()
        timing_rows.append([k, f'{totals.queue_s.mean():.1f}', f'{totals.roundtrip_s.mean():.1f}',
                            f'{mean_recorded_gap:.1f}', f'{jobs[jobs.k == k].latency_s.mean():.1f}'])
    serial = requests[requests.k == 1]
    serial_prefill_fraction = serial.prompt_ms.sum()/(serial.prompt_ms.sum()+serial.predicted_ms.sum())
    raw_resources = [json.loads(s) for s in (root / 'raw/resources.jsonl').read_text().splitlines()]
    paging_delta = (raw_resources[-1]['swap_out'] - raw_resources[0]['swap_out']) / 2**30
    sources = [json.loads(p.read_text()) for p in (root / 'raw/live_test').glob('task_*.json')]
    tool_times = [t['elapsed_s'] for r in sources for t in r.get('tool_events', [])]
    remote = [json.loads(s) for s in (root / 'raw/live_test/remote_usage.jsonl').read_text().splitlines()]
    prompt_usage = sum(r['usage'].get('prompt_tokens', 0) for r in remote)
    completion_usage = sum(r['usage'].get('completion_tokens', 0) for r in remote)

    content = f'''# Phase 1 本机实跑报告

生成时间：{now}（America/Los_Angeles）。所有数值来自本项目保存的实际运行记录。

**完成内容与研究口径**

本机已完成 FIFO 基线、microbench、缓存机制实验、真实 Retail 执行和多步骤 macro 回放。macro pilot 中观测到平均完成速度较高的是 **K={best_k}**；这是本机探索结果，不能当作 AWS CPU 的最优参数，也不是 Phase 2 已在独立开发集选定的最终 baseline。

micro 共 12 种配置，各重复 3 次，共 36 轮、54 个请求；缓存实验共 10 种条件，各重复 3 次，共 30 轮；macro 共 6 轮、24 个任务实例、{verified['macro_requests']} 个模型请求。每个 macro batch 均执行 {verified['steps_per_macro_batch']} 步、生成 {verified['generated_tokens_per_macro_batch']:,} token。全部正式性能请求和任务均正常完成，输出配额、输入长度、前缀关系与 FIFO dispatch 顺序通过程序审计，详见 `verification.json`。这不代表全部客服任务语义成功。

**实际环境**

M4 MacBook Air，16 GB 统一内存，macOS 15.3；原生 Metal；Qwen3-4B Q4_K_M，f16 KV；llama.cpp build 11669 / commit `620d2225c`。固定 P=2，总 context=24,576，每 slot=12,288，关闭 unified KV，额外 host prompt cache=0，continuous batching 保持启用。权重、runtime、数据 hash 与完整启动命令见 `manifest.json`。

模型 SHA256：`{manifest['model_sha256']}`。

机器接入 AC 电源。OS 电源接口快照未记录 thermal/performance warning，但没有锁定 CPU/GPU 频率、隔离后台应用或获得完整温度轨迹，因此不能宣称消除了所有硬件漂移。

**micro：长历史的重复处理成本很明显**

下表固定每次生成 64 token，均为 3 次重复的均值。K=2 每轮同时提交两个请求；总输出速率为本轮生成 token 总量除以 batch 完成时间，包含输入处理。

{table(['输入 token', 'K', 'Prefill 秒', '生成区间秒/请求', '总输出 token/s'], micro_rows)}

关闭缓存的全部请求 reused tokens=0，实际输入和生成量与设定一致。短输入下并发可能提高输出速率，长输入下并发并未带来同等收益。不能把并发曲线变化自动归因于物理内存不足。

`predicted_ms` 是运行时的生成阶段墙钟区间，可能包含被其他请求 prefill 阻塞的时间。部分并发请求的客户端 round trip 也明显长于其两个 timings 字段之和，说明还存在运行时等待或其他开销；本报告不以简单相减精确估计内部排队。图 01 使用“generation interval”避免把它等同为纯 GPU decode 计算。

**缓存：竞争覆盖比短暂空等更重要**

每个条件重复 3 次。A 执行第一步后，下一步输入实际保留本轮生成的 token；覆盖实验显式指定同一个 slot，用于验证机制，不代表自动路由一定以同样频率丢失缓存。

{table(['初始输入', '条件', 'Continuation prefill 均值秒', 'SD 秒', '输入复用比例'], cache_rows)}

空等 0.5 秒没有导致观察到的前缀失效；覆盖或清除则需要重新处理历史。保持缓存与重建缓存的差距，足以支持继续研究跨步骤状态管理。但这尚未证明任何调度策略的端到端收益。

保存恢复也实际跑通了：save → erase → 其他请求使用 slot → 清理目标 slot → restore → continuation。以下“机制总成本”是客户端 save、erase、restore 耗时加恢复后 prefill，排除共同的 decode 和插入请求本身的执行成本。

{table(['初始输入', '保存大小 MiB', 'Save 秒', 'Restore 秒', '机制总成本秒'], checkpoint_rows)}

文件缓存处于温热状态，save 未额外强制持久化刷盘，不能把这些数值叫作冷 SSD 读写性能。保存文件不自动清除 slot；清除状态也不证明 KV buffer 归还给 OS。本实验说明可以保留并恢复状态、让 slot 用于其他工作，尚未实现或证明进程物理内存动态缩容。

性能测量结束后，另做了一次独立恢复正确性检查：同一 4,096-token 初始输入保存状态，在执行其他请求覆盖后恢复，以同样采样配置生成 64 token，与直接保留状态的输出逐 token 相同。原始状态文件与恢复后再次序列化文件的 SHA256 是否相同：**{restore_check['serialized_state_hash_equal']}**。证据保存在 `restore_validation.json`；该检查未混入性能比较。

4,096-token 试验保存状态约 580 MiB；真实 Retail 初始输入已约五千 token。因此原设计里提到的 512 MiB host cache 只是候选预算，不能先验认为足以保存一个完整 Retail 状态。下一阶段要按实际 bytes 确定预算，并对各策略保持相同总内存条件。

**真实任务：执行链跑通，但小模型质量有限**

使用原版 `tau2-bench v1.0.1` Retail 环境、工具、orchestrator 及程序化 ENV evaluator。DeepSeek `deepseek-flash` 只扮演用户；正式性能回放期间不调用外部 API。固定 test split，seed=291，随机选择 8 个任务，最多 40 个对话步骤，保留全部失败。

{table(['Task', 'Agent 调用', '输出 token', '最长输入', 'ENV 检查', '终止原因'], live_rows)}

ENV 检查通过 **{pass_count}/8**。这只评价数据库状态和环境断言，没有执行全部自然语言断言，不能称为官方 τ-bench 总成功率。`max_steps`、错误过多、过早转人工都作为质量失败保留。合法工具名和参数对象组成的 JSON 文本会被严格转换为结构化 tool_calls；未补全任何工具或参数，原始响应及转换标记均保留。修复前的失败轨迹单独存放在 `diagnostics/`，不混入正式比较。

live 采集对本地 Agent 每次输出上限为 512 token，用户模型为 256；回放使用每一步实际生成量而非统一改短。全部 110 个 live Agent 请求的本地 token 计数与服务端 usage 一致。本次按用户指定使用 DeepSeek，尚未适配课程说明中提到的 TritonAI。

本地工具耗时中位数约 {pd.Series(tool_times).median()*1000:.3f} ms。记录中的步骤间间隔还包括用户 API 等待，不能称为真实生产数据库网络延迟。最终 test 采集调用用户 API {len(remote)} 次，记录输入 {prompt_usage:,}、输出 {completion_usage:,} token；该用量不含开发/诊断和最初接口探测。

**macro：比较完整任务完成，而不是只看 token/s**

本机 pilot 使用冻结随机顺序中的前四个完整任务：{', '.join(workload['selected_ids'])}，包含语义失败任务，不按成功与否筛选。K=1/2 各重复 3 次，使用同一 repeat 的相同到达抖动和同一记录间隔；固定 slots、容量、模型和精度。

回放保留来源的全部步骤数和生成数量，以实际输入增长标定新的观察长度，并在下一步保留本轮真正生成的前缀。原版聊天模板的重写被转换为 append-only；这是 **benchmark-derived workload**，不是逐字节忠实 trace，也没有再次执行客服语义。受控回放的“完成”只指执行完所有定义步骤。

本次 58 个步骤的回放输入长度与原始 live 轨迹逐步一致，长度偏差为零；这不表示 token 内容和历史重写方式逐字节相同。

{table(['K', '重复', '完成/提交', '任务/分钟 均值±SD', 'P50 秒均值', 'P95 秒均值', '输入 token 复用', '完整历史复用'], macro_rows)}

输入 token 复用按请求 token 数加权；完整历史复用是一个推导指标：continuation 的 `cache_n` 至少覆盖前一步输入长度加生成长度减一，才算完整历史命中。减一对应最后一个生成 token 通常还未被计算入 KV。公共 system/tool 前缀的复用不自动等于保住了某个 Agent 自己的历史。

该完整历史阈值专门针对本次追加式回放；原版 chat 模板可能重写末尾历史，不能直接套用同一个阈值。当前观察到的是设定 P=2 和缓存预算下的 slot 历史覆盖，不等于已经证明物理 RAM 或总 KV token 池耗尽。固定 slot 分区本身也会限制可驻留的任务状态数量。

各任务完成时延如下，均为三轮均值：

{table(['Task', '完整步数', 'K=1 秒', 'K=2 秒'], job_rows)}

按相同 repeat、相同 task 配对，K=2/K=1 完成时延比值均值为 {slowdown.mean():.3f}，范围 {slowdown.min():.3f}–{slowdown.max():.3f}；其中 {(slowdown>1).sum()}/{len(slowdown)} 个配对实例在 K=2 下更慢。这是并发设置的代价比较，尚未评价 continuation-priority 策略的公平性。首次请求、continuation 等待和执行时间线见图 04/05。

完整任务平均时间的分解如下。模型 round trip 包含服务端内部等待，不能直接当成纯计算时间；记录间隔是回放定义值，异步等待及记录本身还会有少量额外开销。

{table(['K', '累计外部排队秒/任务', '累计模型 round trip 秒/任务', '定义的间隔秒/任务', '端到端秒/任务'], timing_rows)}

K=1 下，服务端累计 prefill 在 prefill+generation interval 中约占 {serial_prefill_fraction*100:.1f}%。K=1 没有本实验内其他并发模型请求，这个比例可用于理解串行服务成本；K=2 的时段可能重叠或暂停，因此不作同样的墙钟占比分解。

每轮只有 4 个来源任务，P95 只是小样本描述性插值；重复 3 次不等于获得了 12 种独立任务。当前为有限 batch，尚未进行多到达率 open-loop 稳态与长期饥饿实验。

模型 HTTP 请求超时为 300 秒，macro 没有额外业务级任务 deadline；“全部完成”不代表满足某个生产延迟 SLO。

**资源观察与解释边界**

{table(['阶段', '采样峰值 server RSS GiB', '系统可用内存最低 GiB'], resource_rows)}

从首个到末个资源样本，系统 swap-out 计数增加约 {paging_delta:.3f} GiB；这是系统级观测，不能全部归因于 llama-server。资源约每 250 ms 采样，RSS 不等于 Metal 与进程的完整物理 footprint；不能把 RSS 和所有 GPU buffer 简单相加，也不能把采样峰值称为绝对瞬时峰值。

固定 f16 KV 池的理论量级约 3.375 GiB，另有模型和计算 buffer。本阶段有明确的缓存容量控制，但没有 macOS 进程物理内存硬上限。物理内存释放、硬预算保证、冷磁盘卸载收益都尚未被证明。

**对下一步的实际建议**

这些结果支持把 Phase 2 的问题收窄为：在有限的状态容量里，减少有价值历史被覆盖后的重复 prefill，并检验保存恢复是否值得。先建立预算相同的简单 host-cache / 固定保留 baseline，再比较 queue-only、cache-only 和联合机制；不要仅凭机制成本图宣称联合策略更快。

还应增加一个便宜但强的对照：固定同时存活的 Agent task 数 J=1/2/4，而不只调请求并发 K。本次主实验固定 J=4；如果简单限制 J 就能减少状态竞争，复杂联合策略必须在这个对照之上证明收益。限制 J 时，未获准启动的任务等待也必须计入端到端时间。

AWS CPU 首先复现相同模型、runtime、workload 与固定并发基线，再扩大任务数量、context/到达率和反例条件。最终最佳固定 K 必须在独立开发 workload 上选择；本机结果不能直接作为 CPU 平台的最佳参数。CPU 侧也应做少量 live 核查，不能把性能回放解释为语义质量保证。

本次 micro 的独立输入扫描到 4,096 token；macro 的更长历史直接由服务端 timings 测量。下一版应补 8K/10K 成本点，并分开控制 slot 数和总 KV 预算，区分分区组织限制与真正的 token 容量压力。

迁移步骤见项目 `README.md`。冻结的 `workload.json` 可以直接回放，无需再次调用 DeepSeek，也不需要把 API key 或本机 KV checkpoint 复制到 AWS。

**文件与复现**

- `raw/`：完整请求、任务、运行、资源和实际 Retail 轨迹。
- `summary/`：生成图表的 CSV；`verification.json`：工作量和日志一致性审计。
- `../figures/`：9 张图，每张有 PNG 和 PDF，含独立图表说明。
- `../phase1/`、`../tests/`、`../configs/`：实现、测试与实验配置。
- `manifest.json`、`model.sha256`、`../requirements-lock.txt`：版本与可复现依赖。

重新生成：`.venv/bin/python -m phase1.verify`、`.venv/bin/python -m phase1.analyze`、`.venv/bin/python -m phase1.report`。性能脚本追加原始日志，完整复跑前应归档已有结果，不能直接追加混成更多独立重复。
'''
    (root / 'REPORT.md').write_text(content)
    print('Wrote results/REPORT.md')


if __name__ == '__main__':
    main()
