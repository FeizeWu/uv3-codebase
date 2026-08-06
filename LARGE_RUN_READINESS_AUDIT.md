# UV3 大型预训练上线前审计

日期：2026-08-06  
被审计代码：`/mnt/data/users/wfz/uv3-codebase`，基线 commit `b3b5631526f28092a2ca838b457b1654018899e1`  
直接 bug 修复 commit：`0b3c6b5`  
参考实现：`/Users/wufeize/uv3/UniWorld-Pretrain-main`  
目标规模：4 节点 × 8 GPU，纯单流约 3.86B MMDiT，Self-Flow，至少 100K step

## 审计边界与当前结论

- 按要求，Self-Flow 的“具体对齐层数/层数选择”不参与对齐判断。
- 本轮只直接修复已经导致训练中断的 online joint bucket buffer 问题，并把生产默认步数改为 100K。
- 下列其余发现均未修改生产实现；需要先由项目负责人确认，再逐项修复和验证。
- 当前结论：**不建议立即启动 100K 大实验**。至少 `P0-01`、`P0-02`、`P0-03` 需要先决策。

## 已直接修复：联合分桶 descriptor buffer 溢出

### 根因

旧 text bucket 权重为 `[23, 21, 20, 15, 21]`。在当前 Qwen3.5-4B tokenizer 和生产 caption 上，短桶累计需求超过真实供给，无法进入短桶的长 caption 会永久积压；`8192` 只是把必然崩溃延后。

50,000 条生产 caption 的 native bucket 分布为：

| Bucket | 512 | 640 | 768 | 896 | 1024 |
|---|---:|---:|---:|---:|---:|
| 比例 | 22.08% | 18.97% | 19.49% | 14.85% | 24.61% |
| 供给 CDF | 22.08% | 41.05% | 60.54% | 75.39% | 100% |
| 旧需求 CDF | 23% | 44% | 64% | 79% | 100% |
| 新需求 CDF | 20% | 38% | 56% | 70% | 100% |

事故 rank8 的 epoch-0 真实早期 120,000 条 caption 更偏长：供给 CDF 为 `[21.453, 39.657, 58.723, 73.583]%`，旧配置四个前缀全部不可行，与约 step 3190 崩溃一致。

### 已落地修改

- 配额改为 `[20, 18, 18, 14, 30]`。
- 保持所有 rank 的 text/joint target 调度一致；没有采用未验证的 rank-local target fallback。
- tokenization chunk 不再越过 descriptor hard limit。
- 坏图补样在 buffer 已满时，不再继续塞入无关 descriptor；改用当前 batch 中完整有效的 image-text pair 回填并计数。
- 生产 config 与多机 launcher 的默认 `MAX_STEPS` 改为 `100000`，launcher 拒绝小于 100K 的生产提交。
- launcher preflight 锁定新 bucket 权重和最小 buffer。

### 验证证据

- 全量测试：`44 passed`。
- 新增满 buffer + 坏图回填回归测试。
- 新增生产权重 CDF 可行性回归测试。
- 100K-step 队列 Monte Carlo，16 个 seed：
  - 旧权重：`0/16` 通过，step `8276–9172` 越过 8192。
  - 新权重：`16/16` 通过，peak buffer `516–624`。
- 事故 rank8 真实 shard/row-group/DataLoader 顺序重放 120,000 descriptors：
  - `9996` batches / `119952` emitted samples。
  - peak buffer `2692`，final buffer `48`，未越过 8192。
  - 末尾 48 条是 25 个 joint queues 中不足一个 batch 的尾料，下一 epoch 会继续填充。

## P0：大型实验阻断项

### P0-01：BF16 master/EMA/Adam 状态破坏 Self-Flow EMA 语义

证据：

- `uv3/train/fsdp2_trainer.py` 在 FSDP 包装前把 MMDiT 整体转为 BF16。
- teacher 从 BF16 student deepcopy；EMA 对 BF16 shard 直接执行 `lerp_(weight=1e-4)`。
- 当前真实 step-3000 checkpoint 中，student、teacher、Adam `exp_avg/exp_avg_sq` 全部是 BF16。
- UniWorld 参考实现保留 FP32 model/EMA，低精度只用于计算路径。

最小数值复现，`decay=0.9999` 连续更新 10,000 次：

| teacher → student | FP32 结果 | BF16 结果 |
|---|---:|---:|
| `0.01 → 0.02` | `0.0163214` | `0.0100098` |
| `1 → 2` | `1.63214` | `1.0` |

真实 checkpoint 证据：旧 10K run 从 step 2K 到 10K，抽样 teacher 元素有 `97.56%–99.70%` 完全不变，teacher MAD 仅 `0.8e-6–6.4e-6`；对应 student MAD 为 `8.1e-4–3.58e-3`。

风险：teacher 接近冻结，projector 可能主要在拟合一个静态 target；Self-Flow loss 快速下降不能证明 EMA 正确。

建议修复方向（尚未实施）：

- student、teacher、projector 和 Adam master/moments 保持 FP32 storage。
- FSDP `MixedPrecisionPolicy` 继续用 BF16 forward/reduce。
- 修复后先做显存、吞吐和数值 smoke，再决定是否允许 100K。

必须测试：

1. 构建、FSDP 包装和 optimizer 首步后检查 dtype contract。
2. 两卡 sharded EMA 与单卡 FP32 reference 连续更新对比。
3. `decay=.9999` 解析值 10K-step 回归测试。
4. 真实 3.86B 模型显存峰值和速度测试。

### P0-02：运行环境跨项目、跨 venv 污染，不可复现

`.venv-cu128/site-packages/uniworld_shared.pth` 注入了：

- `/mnt/data/users/lzj/Uniworld/.venv/lib/python3.12/site-packages`
- `/mnt/workspace/users/lzj/Uniworld/third_party/transfusion-pytorch`
- `/root/.local/lib/python3.12/site-packages`
- `/usr/local/lib/python3.12/dist-packages`
- `/usr/lib/python3/dist-packages`

当前实际 provenance：

| 模块 | 版本 | 实际来源 |
|---|---|---|
| torch | `2.10.0+cu128` | wfz `.venv-cu128` |
| transformers | `5.14.1` | lzj `Uniworld/.venv` |
| diffusers | `0.39.0` | lzj `Uniworld/.venv` |
| torchao | `0.14.0+git` | `/usr/local` |
| pyarrow | `22.0.0` | `/usr/local` |
| Pillow | `12.2.0` | wfz `.venv-cu128` |

同时，`pyproject.toml` 要求 torch ≥ 2.12，`uv.lock` 锁 torch 2.13 / pyarrow 25，与实际环境均不一致；torchao 未完整声明。合作者删除或升级其 venv 就能改变本项目行为。用户日志中 diffusers traceback 来自 lzj venv，是已发生的直接证据。

建议修复方向（尚未实施）：建立完全自包含、锁定版本的环境或镜像，删除跨 venv `.pth`。四节点 preflight 输出并比对 Python、模块路径/版本、torch/CUDA/NCCL、git SHA 和 config hash。

### P0-03：checkpoint resume 不是精确续训

当前恢复 model、optimizer、部分 torch RNG，但没有完整恢复：

- Python `random` 和 NumPy RNG。
- tar/source 的真实 cursor。
- online joint bucket 的 25 个队列、schedule cursor、buffer。
- DataLoader worker/prefetch/decode 状态。
- 下一批稳定 sample IDs。

descriptor 虽产生 `shard_pos/row_pos`，但 batch materialize 丢掉这些 ID；保存的 `data_status` 没有随训练推进更新。resume 实际从 epoch 头重新开始，会重复此前数据前缀。100K 中断后这不是小误差。

此外：

- trainer 在验证 checkpoint 前可能覆盖 run `config.yaml`。
- `RESUME_EXPECTED_STEP` 没有读取 checkpoint 验证真实 step。
- optimizer state `strict=False` 可能在布局变化时静默漏载。
- Self-Flow 开/关导致 checkpoint key 和 optimizer layout 不兼容。

必须测试：连续 `N+M` 与 `N → save → 新进程 resume → M` 对比 model、teacher、projector、optimizer、RNG、后续 sample IDs 和 loss；不能只验证曲线 step 连上。

## Self-Flow / Flow Matching 与 UniWorld 对齐结论

以下已确认一致或数学等价：

- FM 插值和 velocity target：两边 t 方向/符号相反，但组合后数学等价。
- 当前 `logit_normal μ=0, σ=1, no shift`。
- student 两个 timestep 独立同分布采样；teacher 使用 `min(t,s)`。
- 当前 `mask_ratio=0.25` 的 token latent/time 混合。
- projector 拓扑 `Linear(H,2H) → SiLU → Linear(2H,H)`。
- teacher detach + FP32 normalize 的 cosine feature loss。
- optimizer step 后更新 EMA 的时序。
- pure-single per-token path：2/4 层小模型 golden test 中，与 scalar timestep path 最大误差 `5.96e-7–7.15e-7`；带 padding mask 结果相同。

已知有意差异，不直接判定为 bug：

- 当前测速/训练使用 AdamW；UniWorld 实验使用 Muon+Adam。
- 当前 AdamW betas 为 `(0.9, 0.95)`，UniWorld 默认 AdamW 为 `(0.9, 0.999)`；当前每步还会做 gradient clipping 1.0，而参考实现没有 clipping。两项都应作为明确的训练超参决定，并通过短程 grad-norm/clip-rate 统计确认，而不能默认为与参考实现一致。
- 当前评估固定 student；UniWorld validation 使用 EMA。两套曲线不能直接横向比较。
- 当前 Self-Flow coeff 为 `0.8`；参考实际脚本为 `0.5`。需要作为超参决策，而不是默认为“已对齐”。
- 对齐层数按要求不纳入本审计。

## 核心模型 / VAE / Qwen 架构对齐结论

只读逐项对照暂未发现核心公式级 P0 错位：

- MMDiT 使用同一 diffusers Flux2 transformer 语义：packed input/output 128、head dim 128、四轴 RoPE `(32,32,32,32)`、theta 2000、无 guidance embedding。
- latent patchify/unpatchify 的 reshape/permute 操作与 UniWorld 一致。
- FLUX.2 VAE packed batch-normalization normalize/de-normalize 公式一致。
- Qwen3.5 都读取 `language_model(...).last_hidden_state`。
- flow interpolation、`target=noise-clean` 和 Euler `t=1→0` 逐步公式一致。
- FM loss 都在 FP32 中对所有 token/channel 做 mean。

已运行的正向测试：

- `tests/test_flow.py + tests/test_attn_mask.py`：`9/9` 通过。
- `tests/test_self_flow.py`：`8/8` 通过；包括 uniform token-t 时 custom per-token path 与原生 scalar path 等价。
- `tests/test_flexmask.py` GPU 测试：`3/3` 通过；padding-hole/document mask 与 dense SDPA 对齐。

仍缺的上线证据：

- 同权重完整 MMDiT 的 FlexAttention vs dense SDPA 端到端 golden，而不只是 kernel/mask 层测试。
- BF16、五种宽高比、五种 text bucket、真实 padding pattern 的组合矩阵。
- compiled student + compiled teacher + FSDP2 的多 rank backward/显存稳定性。

Qwen padding/joint attention mask 是 UV3 相对参考实现的有意增强，已有 padding inert / truncate 相关测试；VAE BF16 + `force_upcast=False` 是效率选择，公式一致，但仍应在最终锁定环境中重跑数值/质量 smoke。

## P1：确认后应修复或明确接受

### P1-01：Self-Flow feature shape 被静默截断

trainer 使用 `min(student_tokens, teacher_tokens)` 后截断两边；参考实现遇到 shape 不一致会失败。若 text/image 切片错位，当前实现可能继续训练错误 token pairing。

建议测试：5 个 resolution × 5 个 text bucket 下，断言 student/teacher feature shape 完全相同且 image token 数等于 `n_img`。

### P1-02：Self-Flow 配置与参考存在未显式化差异

- `projector_dim` 配置字段存在，但 trainer 未传入，非默认值静默无效。
- projector 初始化使用 PyTorch 默认 Kaiming/随机 bias；参考为 Xavier/bias=0。
- `mask_ratio=0` 语义不一致：参考为全部 paired token，当前特判为全部原始 t；当前 `.25` 不受影响。
- mask/ratio/EMA/coeff/projector_dim 缺少生产级范围校验。
- `SelfFlowConfig.enabled` 默认 True，配置遗漏时可能意外开启。
- patch-size 参数没有贯穿 Self-Flow 调用；当前 patch=2 安全，其他值有风险。

### P1-03：后期关闭 Self-Flow 尚不能安全 resume

启用 Self-Flow 的 checkpoint model 是 `ModuleList([mmdit, projector])`，key 带 `0.`/`1.`；关闭后目标是裸 MMDiT，optimizer layout 也不同。当前没有转换工具或同 run 的 off schedule，与“收敛后关闭 Self-Flow”计划冲突。

### P1-04：部分训练配置字段写了但未生效

- `warmup_steps`、`lr_schedule` 没有实际 scheduler 实现；当前 100K 将使用恒定 `1e-4`，除非另行修改。
- `mixed_precision`、`sharding_strategy`、`compile_dynamic` 等部分字段未真正控制 trainer，实际行为有硬编码。
- YAML 解析只提取 dataclass 已知字段，未知字段会被静默丢弃；配置键拼错时不会失败，而会悄悄使用默认值。UniWorld 对未知字段会在 dataclass 构造时抛错。
- `grad_accum>1` 时 step/checkpoint/eval 仍按 microbatch 计数，未使用 FSDP `no_sync`，checkpoint 可能落在累积中途且不会保存梯度。
- student 与 projector 分开 clip，不等价于统一 global norm。

当前 `grad_accum=1` 避开部分问题，但应增加配置约束，避免未来误用。建议增加 unknown-key fail-fast 测试，并逐个验证 production YAML 中声明的字段确实改变 effective config/runtime。

### P1-05：数据读取存在静默跳过与配置失效风险

- parquet row-group 任意异常会静默 `continue`；UniWorld/TuVAE 类实现通常重试后显式失败。存储抖动可能静默丢整组数据。
- `META_COLS` 固定默认 caption 字段；切换 config caption field 时可能不生效。
- `shuffle_shards` 配置没有完整传入 dataset。
- 当前监控的 buffer/drop/fallback 主要来自 rank0；本次真实事故发生在 rank8，rank0 指标无法预警。
- DataPrefetcher 没有转发 batcher 已生成的 decode-error/fallback 字段，因此现有监控中的坏图 drop/fallback 实际可能长期显示为 0。
- rank 样本数和坏图数不等时，缺少正式的跨 rank 计数/padding 合约。

建议测试：故障注入 row-group 读取、跨 rank 样本计数、caption field 切换、manifest shuffle、坏图比例不均、kill/resume sample-ID 连续性。

### P1-06：Palette/alpha 图片处理有数据语义问题

P 模式且带 byte transparency 的图片直接 `.convert("RGB")` 会触发用户看到的 warning，并把透明像素背后的 palette RGB 当成可见颜色。最小 fixture 中透明红像素会输出红色；TuVAE 参考先转 RGBA 再在白底合成，输出白色。

建议测试：P+transparency、RGBA、LA 三种 fixture，明确统一背景色和像素 golden value。该问题本轮未修改。

### P1-06b：源图保护与 crop 质量策略尚未明确

- descriptor 路径没有用 metadata 预先限制最大源图像素、非法 offset/size 和超大压缩成员。
- 抽样 200K 图中有 2 张超过 64MP，最大约 78.6MP；长跑时并行全尺寸解码可能产生明显 CPU RSS 峰值。
- 100K 图抽样中，约 `2.607%` 即使分配到最近宽高比桶仍会裁掉超过 30% 面积；约 `2.612%` 源图面积小于目标 bucket，需要放大。
- TuVAE 有 `max_image_pixels`、`min_crop_area_ratio` 和 source-area 过滤。是否采用属于数据质量决策，不能未经小规模抽检直接照搬。

建议测试：超大图/非法 metadata fixture、RSS soak、crop-area 分布与肉眼样本审查。

### P1-07：checkpoint 长跑可靠性不足

- 只有一个持续覆盖的 `ckpt.pt`，没有 2–3 份轮转恢复点。
- OSS/FUSE 上不能仅凭 rename 假定完整原子语义。
- 只检查文件大小和 step，没有 checksum/完整 tensor 校验。
- rank0 保存失败时，其他 rank 可能卡在 barrier。
- 最终 step 可能重复写一次约 29GB checkpoint。

建议做保存阶段故障注入，并保证任何阶段失败后至少一份旧 checkpoint 可恢复。

### P1-08：100K 自动评测与快照策略仍是 10K 时代设计

- `snapshot_checkpoints_for_eval.py`、`run_pending_fixed_evals.py` 默认仍为 10K。
- 多机 launcher 本身不会启动周期评测；依赖在另一台机器手工启动 watcher。
- 每 1K 保留约 29GB 完整快照，100K 约 2.9TB；FP32 master/teacher 修复后可能更大。
- 每 2K 的 10K FID 中间产物还会继续增长。
- 没有 quota preflight、retention 或成功评估后清理策略。

建议先明确：恢复 checkpoint 每 5K–10K，轮转 2–3 份；1K 样图使用临时快照，评估成功后自动清理；FID 周期和长期保留点单独配置。

### P1-09：FlexAttention / compile 缺少真实多机 shape 矩阵测试

现有验证主要是小模型或单机。需要覆盖 5 text bucket × 5 resolution、不同 rank 不同宽高比、compiled student+teacher、FSDP forward/backward，并检查：

- graph 数预热后稳定。
- 无持续重新编译/eager fallback。
- 显存不随 step/shape 增长。
- attention mask 与 padded alignment token 完全屏蔽。

### P1-10：NCCL 异常退出没有完整清理

trainer 缺少 `try/finally destroy_process_group()`；用户日志中的 warning 是异常退出的次生问题。还缺少真实多节点 rank crash、SIGTERM、网络故障的 bounded shutdown 测试。

### P1-11：监控首页的总 loss 不是全局 loss

trainer 将 rank0 的 `loss.item()` 直接写入 JSON，没有对总 loss 做跨 rank all-reduce；UniWorld 会先对 logged loss 做全局平均。UV3 的 FM/Self-Flow 分项会经 resolution 统计做全局归约，因此当前页面上的 `total loss` 与两个分项的统计口径并不一致。

这不改变反向传播，但会误导大型实验的趋势判断和跨 run 对比。建议测试：构造各 rank 不同的已知 loss，断言日志总 loss 等于全局加权平均，并明确分桶/总 loss 是否按样本数或 token 数加权。

## P2：可在阻断项后处理

- code fingerprint 未覆盖 `self_flow.py`、`flow.py`、noise scheduler、bucket sampler、FlexAttention/eval 关键文件，也未写入 checkpoint 强校验。
- 监控后端长期读取完整 telemetry JSONL；四节点、2 秒采样每天约 138 万行，100K 长跑会逐渐变慢。
- metrics 默认只返回最后 5000 点，samples/evaluations 也有固定尾部上限，100K 前半段会在 UI 中消失。
- 失败状态只检查 stdout 尾部，NCCL 输出可能把真正 traceback 挤掉。
- throughput 在未来 `grad_accum>1` 时可能口径错误。
- tar short read 当前直接按坏图处理；更稳妥的语义是关闭并重开 handle、有限重试，仍失败才计坏图，以区分 OSS 瞬时读取问题。
- 生产训练集首次 100K 预计尚未走到 epoch 尾部，但更长实验需要 rank 等长/padding 合约；manifest raw sample rank spread 抽样约 `1.17%`。
- `ModelConfig.head_dim`、`axes_dims_rope`、`in_channels`、`out_channels` 当前没有真正参与 MMDiT 构造；`patch_size` 的名字可配置，但 patchify 实现固定为 2×2。当前生产值恰好与实际实现一致，因此尚未触发错误，但未来改配置时应 fail-fast 或真正贯穿实现。

## 建议审批顺序

1. `P0-01`：批准 FP32 master/EMA/optimizer storage 修复方案及显存预算。
2. `P0-02`：批准建立自包含环境，确定最终 torch/transformers/diffusers/torchao 版本。
3. `P0-03`：决定 100K 是否要求严格 data-exact resume；建议要求。
4. 确认 Self-Flow coeff、projector 初始化、student/EMA 评估权重等有意差异。
5. 修复已批准问题后，执行 25-shape 多机 compile/Flex 测试和 save/resume 等价测试。
6. 完成 100–500 step 四节点真实数据 soak：坏图、checkpoint、resume、自动样图、FID、监控和故障退出全部覆盖。
7. 只在上述门槛通过后启动正式 100K。

## 本轮未做的事情

- 未修改 BF16/FP32 精度策略。
- 未修改 Self-Flow 公式、projector、coeff、mask semantics 或 feature shape 行为。
- 未修改环境或依赖。
- 未修改 resume、checkpoint 轮转、LR scheduler、评测 retention、监控和 NCCL 清理。
- 未启动新的正式训练。
