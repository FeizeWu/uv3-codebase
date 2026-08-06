# UV3 供给驱动的联合分桶调度方案

日期：2026-08-07
状态：核心实现与关键验收已完成（2026-08-07）；四节点正式启动仍由用户执行
适用代码：`uv3/train/fsdp2_trainer.py::OnlineJointBucketBatcher`
目标：在保留五档静态文本长度和多分辨率 batch 的前提下，消除由数据分布漂移造成的 descriptor buffer 填满中断，并量化动态调度对速度的影响。

## 1. 结论与保证范围

采用“长期权重优先 + 最短公共 ready 桶 fallback”后，可以保证：

- 不会因为固定文本桶目标暂时或长期缺少供给而把 descriptor buffer 填满。
- 不需要在 buffer 异常时长时间切换为全 1024。
- 所有 rank 每步仍使用相同的 text target。
- 每个 rank 仍可选择自己的图像宽高比。
- scheduler 不丢样本；短 caption 只会被提升到更长 target 并增加 padding。

保证成立需要满足：

1. 数据源能继续提供可 tokenization 的有效 descriptor。
2. descriptor 能被归入配置中的一个分辨率桶。
3. 每个 rank 最终能获得至少一个完整本地 batch 的有效样本。
4. 所有 rank 按相同顺序执行 scheduler 控制同步。

本方案不保证处理以下独立故障：

- 整个候选 batch 全部图片损坏且没有有效 donor。
- Parquet/Tar/OSS 持续读取失败。
- tokenizer、CUDA、NCCL 或进程异常。
- 数据源有限且不足一个完整 batch。

因此这里的“不会停”特指：不会再因为联合桶供需不匹配触发 `online joint bucket descriptor buffer is full`。

## 2. 为什么可以保证 buffer 有界

生产配置有五个分辨率桶，本地 batch size 为 12。对 text target=1024 来说，所有 native text bucket 都可用，因此只需要某一个分辨率累计 12 条样本。

鸽巢上界为：

```text
5 × (12 - 1) + 1 = 56
```

即：任意 rank 只要缓存 56 个有效 descriptor，至少一个分辨率必然拥有 12 条样本，所以该 rank 的 1024 target 必然 ready。

对每个 rank，ready target 具有单调性：

```text
512 ready  => 640/768/896/1024 都 ready
640 ready  => 768/896/1024 都 ready
...
```

各 rank 的 ready 集合都是某个 target 开始直到 1024 的后缀集合，因此所有 rank 的 ready 集合交集必然非空，最坏包含 1024。

只要 scheduler 在达到 hard limit 前停止为不可用的目标继续读取，并从公共 ready 集合中选择目标，就始终能消费一个 batch，buffer 不会因为目标不可用而无限增长。

## 3. 桶、rank 与队列

四节点八卡训练共有 32 个 rank。每个 rank 是一个独立 GPU 进程，拥有自己的数据分片、DataLoader 和 CPU descriptor 队列。

每个 rank 内部有：

```text
5 resolution buckets × 5 native text buckets = 25 queues
```

队列 key 为：

```text
(resolution_name, native_text_bucket)
```

descriptor 缓存在 CPU 内存中，包含 tar path、offset、caption、token IDs、resolution metadata 和 resume cursor。图像在组成 batch 之后才解码，buffer 不是 GPU 显存。

每个训练 step：

- 所有 rank 使用同一个 text target。
- 各 rank 从自己的队列中选择一个本地 resolution。
- 各 rank 输出本地 BS12。
- 32 个 rank 构成 global batch 384。

## 4. 调度算法

### 4.1 长期基准权重

初始使用当前验证过的权重：

```text
[20, 18, 18, 14, 30]
```

统计单位必须是“每 rank 至少 50,000 个新 tokenized descriptors”，不是全局合计 50,000。

每个 rank 只维护五个 native text bucket 计数。所有 rank 都达到 50,000 后：

1. 计算每个 rank 的四个 prefix CDF。
2. 对每个 prefix 取所有 rank 的最小值。
3. 减去固定 2 个百分点安全余量。
4. 对累计需求向下取整，保证 rounding 后仍不越过安全 CDF。
5. 转换为五个整数权重，总和为 100。
6. rank 0 生成 schedule version 并广播。
7. 在统一 schedule 周期边界安装新权重。
8. 清零本窗口计数，开始下一个 50,000-descriptor 窗口。

公式：

```text
safe_cdf[k] = min_rank(observed_cdf[rank, k]) - 0.02
demand_cdf[k] = floor(100 * safe_cdf[k]) / 100
weights = diff([0, demand_cdf..., 1]) * 100
```

不根据全局平均调整，因为一个偏长 rank 足以让整个分布式任务失败。

50,000/rank 下，比例约 20% 的桶标准误差约 0.18 个百分点；2 个百分点余量足以覆盖统计噪声。当前 BS12 下首次更新约在 4,167 step。

### 4.2 正常路径

基准 schedule 给出本 step 的期望目标 `desired_target`。

每个 rank 首先检查自己的队列：是否在某一个 resolution 下有至少 BS12 条 native length 不超过 `desired_target` 的样本。

如果所有 rank 都 ready：

```text
selected_target = desired_target
```

此时行为与当前固定 schedule 一致，不引入额外 padding。

### 4.3 有界 lookahead

如果本 rank 的 desired target 尚未 ready，可以继续读取 descriptor，但读取必须有界：

```text
lookahead_per_slot = 512 descriptors
soft_buffer_limit = 6144 descriptors
hard_buffer_limit = 8192 descriptors
```

本 step 最多读取：

```text
min(lookahead_per_slot, soft_buffer_limit - current_buffer)
```

满足 desired target 后立即停止读取。达到 lookahead 或 soft limit 后仍不满足，则进入公共 ready fallback，不再继续为该目标读取。

`512` 是初始建议值，需要用真实数据 replay 衡量 fallback 率与 buffer 峰值；它不是正确性常量。正确性只要求 lookahead 在 hard limit 前有界，并允许形成至少一个本地 ready target。

### 4.4 最短公共 ready target

每个 rank 生成 5-bit ready mask：

```text
bit0=512, bit1=640, bit2=768, bit3=896, bit4=1024
```

通过固定顺序的控制 collective 求所有 rank 的 bitwise AND：

```text
common_ready_mask = AND(all_rank_ready_masks)
```

选择规则：

```text
if desired_target in common_ready_mask:
    selected_target = desired_target
else:
    selected_target = smallest common ready target greater than desired_target
```

由于 ready 集合单调，如果 desired 不在公共集合，则不可能存在更短的公共 ready target，因此上述选择总能得到唯一结果。

示例：

```text
desired = 512
rank0 最短 ready = 512
rank1 最短 ready = 640
rank2 最短 ready = 768
其他 rank 最短 ready <= 768

selected = 768
```

只有至少一个 rank 当前最短 ready target 为 1024 时，本 step 才使用 1024；不会把后续 schedule 整体改成全 1024。

### 4.5 控制通信

控制同步必须满足：

- 所有 rank 每个 emitted batch 调用次数一致。
- 不能与异步 FSDP NCCL collective 交叉改变调用顺序。
- 不应放进可能跨 step 预取的 CUDA/NCCL 路径。

推荐使用独立的 Gloo control process group，同步一个 ready bitmask 和必要的少量整数。通信量可以忽略，但必须实测它是否引入额外 rank barrier 延迟。

若最终证明 rank-local text target 在 FSDP2 + FlexAttention 下完全安全，也可以取消公共 target 通信；在完成多 rank golden test 之前不采用该简化。

### 4.6 样本选择

确定 selected target 后，每个 rank：

1. 在本地 ready resolutions 中优先选择 native-fit 样本最多的 resolution。
2. 在该 resolution 内优先消费最长的 eligible native bucket。
3. 同一 native queue 内保持 FIFO。
4. 短 caption 可以提升到 selected target。
5. 不允许为了 fallback 丢 descriptor。

这沿用当前 `_take_eligible()` 的最长 native 优先策略，可以优先排空真正造成压力的长 caption，并减少无意义 promotion。

## 5. Resume 状态

为了让中断恢复后的目标序列与不中断训练一致，需要保存：

- 当前 base weights 和 schedule version。
- smooth schedule 内容及 cursor。
- 当前 50,000 窗口的五桶计数和每 rank 已统计数量。
- 25 个 descriptor queues。
- pending decode specs。
- source cursor。
- desired/selected target 累计计数。
- fallback 累计计数。

恢复时必须校验 tokenizer、text buckets、resolution buckets、batch size 和 schedule version 未变化。

## 6. 最小遥测设计

目标是回答三个问题：

1. fallback 是否足够少。
2. fallback 是否经常被迫升到 1024。
3. buffer 是否真正保持有界。

### 6.1 常规聚合记录

只由 rank 0 每 100 step 写一条聚合记录。100K step 约 1,000 条，预计远低于 1 MB。

每条记录包含：

```text
window_start_step
window_end_step
base_weights[5]
desired_target_counts[5]
selected_target_counts[5]
fallback_counts_by_pair[desired][selected]  # 仅上三角非零项
fallback_steps
fallback_to_1024_steps
fallback_extra_text_tokens_sum
global_buffer_max
global_buffer_mean
global_buffer_peak_rank
lookahead_exhausted_steps
control_wait_ms_sum
control_wait_ms_max
```

不记录 caption、sample ID、逐队列逐 step 明细或32份rank日志。

### 6.2 权重更新事件

每次 50,000/rank 权重更新单独写一条事件：

```text
schedule_version
samples_per_rank_min/max
worst_rank_supply_cdf[4]
old_weights[5]
new_weights[5]
safety_margin
```

预计100K训练中约24次，存储可忽略。

### 6.3 异常诊断快照

仅在以下条件首次出现或距离上次快照超过 1,000 step 时记录：

- global buffer > 4096。
- 100-step 窗口 fallback rate > 5%。
- fallback_to_1024 rate > 1%。
- lookahead exhausted。

快照只记录峰值 rank 的 25 个 queue counts、ready mask 和当前 desired/selected target，不记录样本内容。

## 7. 初始适用性阈值

这些阈值用于判断策略是否值得保留，不作为正确性断言：

```text
正常生产数据 fallback rate            < 1%
fallback 到 1024 的 step 比例          < 0.1%
平均额外 text padding token            < 4 tokens/sample 等价量
global buffer p99                       < 4096
global buffer hard-limit hit            = 0
control collective 平均开销             < 单 step 时间的 0.2%
```

若 fallback 率持续超过 1%，优先检查长期权重是否与 worst-rank CDF 对齐；不直接增大 buffer。

若 1024 fallback 超过 0.1%，检查是否存在特定 rank/shard 的 caption 分布偏移，以及 lookahead 是否过短。

## 8. 测试计划

### 8.1 单元与性质测试

1. Ready mask 单调性。
2. 多 rank ready mask 交集必然包含一个 target。
3. `selected_target` 是不小于 desired 的最短公共 ready target。
4. 任意输入序列下 buffered descriptors 不超过 hard limit。
5. scheduler 不丢 descriptor，emitted + buffered + pending 与 source accounting 一致。
6. rounding 后长期 demand CDF 不超过 worst-rank supply CDF 减安全余量。

建议使用 property-based 或至少数千个随机分布/seed。

### 8.2 极端分布测试

1. 所有 caption 都属于 1024。
2. 所有 caption 都属于 512。
3. 正常分布运行后突然永久切换为全 1024。
4. 只有一个 rank 全 1024，其余 rank 正常。
5. 各 rank 分别只能形成 512/640/768/896/1024。
6. 文本长度与 resolution 强相关。
7. 某个 resolution 极少但持续出现。
8. 混入坏图并触发 replacement。

每项至少运行远超8192个descriptor；descriptor-only soak 建议达到每 rank 1M。

### 8.3 多 rank 训练测试

1. 两卡小模型：不同 rank 使用不同本地 resolution，但相同 selected text target。
2. FlexAttention + torch.compile forward/backward。
3. 触发512→640、512→768和896→1024 fallback。
4. 断言所有 rank 的 selected target 序列完全一致。
5. 检查无持续 recompile、collective mismatch、NCCL timeout或显存增长。

### 8.4 Resume 等价测试

比较：

```text
连续运行 N+M
vs
运行 N → checkpoint → 新进程 resume → 运行 M
```

必须逐项一致：

- 后续 sample IDs。
- desired target 序列。
- selected target 序列。
- fallback位置。
- base weights、统计窗口、schedule cursor。
- model/optimizer/teacher/projector状态。

### 8.5 真实数据验证

1. 重放此前事故 rank8 的真实 descriptor 顺序。
2. 覆盖至少32个rank的真实 shard分配。
3. 先跑10K-step descriptor-only。
4. 再跑100K-step descriptor-only。
5. 最后进行100–500 step四节点真实训练smoke，测控制通信开销。

## 9. 实施顺序

1. 将 target 选择逻辑抽成无 I/O 的纯 scheduler，先写性质测试。
2. 实现有界 lookahead 与 ready mask，不接入分布式训练。
3. 实现多 rank 公共 target 控制同步。
4. 接入长期 50,000/rank 权重更新。
5. 接入 checkpoint/resume state。
6. 接入最小聚合遥测。
7. 完成 descriptor-only adversarial soak。
8. 完成两卡 Flex/FSDP2测试。
9. 完成四节点短程真实数据smoke。
10. 所有门槛通过后才用于正式100K训练。

## 10. 不采用的方案

- 不在 buffer 升高后长时间全 1024 drain。
- 不为每个 rank 单独使用未验证的 text target。
- 不按当前 buffer 比例完全随机选桶；buffer 是历史调度后的有偏状态。
- 不通过增大8192来掩盖供需失配。
- 不静默drop descriptor。
- 不记录逐step、逐rank、逐queue大日志。

## 11. 待确认参数

初始建议：

```text
long_term_window_per_rank = 50000
long_term_safety_margin = 0.02
lookahead_per_slot = 512
soft_buffer_limit = 6144
hard_buffer_limit = 8192
telemetry_interval_steps = 100
diagnostic_rate_limit_steps = 1000
```

其中 50,000/rank 和 2% margin 有统计依据；lookahead=512、soft limit=6144 需要通过真实descriptor replay确定，不应在没有测量的情况下视为最终值。

## 12. 实施与验收记录（2026-08-07）

已完成：

- 新增纯状态机 `uv3/data/dynamic_bucket_scheduler.py`，实现单调 ready mask、最短公共 ready target、worst-rank prefix CDF、2% 安全余量、整数权重和周期边界安装。
- `OnlineJointBucketBatcher` 已接入有界 lookahead、独立 Gloo 控制组、rank-local resolution、最长 eligible native bucket/FIFO、不丢 descriptor 的动态路径。
- 生产配置启用推荐参数：`512 / 6144 / 8192 / 50000 / 0.02`；历史测速配置保持默认关闭。
- checkpoint schema v3 保存完整 scheduler、25 队列、pending specs、parquet cursor、统计窗口及样本计数；不再保存非确定性的 Gloo wall-clock 等待值。
- rank 0 每 100 step 写聚合遥测，权重更新单独记录，异常快照按 1000 step 限流。
- 启动脚本 preflight 强制检查动态调度器及生产参数，避免四节点配置漂移。

关键验收证据：

- 单元/性质/恢复相关测试 31 项通过；最终仓库全量 pytest 结果见本次提交记录。
- 六类 descriptor-only 极端分布各输出 10,000 样本，覆盖全长、全短、分布永久漂移、文本/分辨率强相关、稀有分辨率和坏图 replacement；hard buffer 无越界、样本守恒成立。
- 两卡 120-step 分布式测试通过：selected target 序列一致、resolution 保持 rank-local、长期权重在周期边界同步安装。
- 两卡真实 tar 数据的 FSDP2 + FlexAttention + `torch.compile` 前向/反向 3 step 通过，无 collective mismatch、NCCL timeout 或显存增长。
- 四卡真实 tar 数据完成连续 4 step 与 2+resume+2 step 对照：loss/样本目标序列一致，数据/RNG/checkpoint 调度状态完全一致；CUDA 数值状态仅有极小非确定性（模型最大绝对差 `1.62e-5`，teacher `1.86e-9`）。

按“不要过度测试”的最新要求，未继续执行每场景 1M descriptor、100K-step replay 或本机可替代性较低的重复长测。四节点 100–500 step smoke 留作用户用正式提交环境执行；当前实现不会自主启动正式大型训练。
