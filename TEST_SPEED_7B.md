# 7B 训练速度测试

## 目标与统一口径

- MMDiT：`hidden_size=4096`、4 个 double-stream block、24 个 single-stream block，旧实测标称约 7.30B 参数。
- 文本编码器：冻结的 Qwen3.5-9B，BF16、每卡复制；不使用 FP8。
- 8 张 H20 96GB，FSDP2 节点内 8 卡 FULL_SHARD；多机时固定 `num_shard=8`，节点间 replicate。
- AdamW，关闭 self-flow；图片 256×256，图片 token 为 256。
- Qwen 与 MMDiT 同步使用 `512/640/768/896/1024` 五个静态文本桶，`dynamic=False`，Flex Attention block size 128。
- tokenize once + pinned token batch，VAE `_encode` compile。
- 前 20 step 作为编译/预热期；使用 step 30→40 的累计 `spd` 差分计算严格 wall-clock，包含数据与 CPU 开销。

## 旧配置参考（不可与新 A/B 直接混用）

旧 `bench_real_7b_8gpu_20260803` 使用 Muon + self-flow、固定 1024、BS5，稳态约 0.2077 step/s，即 8.31 样本/秒，峰值显存约 83.61 GiB/卡。它用于说明 7B 能运行，不作为新优化组合的 baseline。

## 测试矩阵

| 配置 | 单卡 BS | `reshard_after_forward` | 目的 | 状态 |
|---|---:|---:|---|---|
| 7B 五桶 BF16 baseline | 5 | `True` | 测量标准 FSDP2 路径 | 完成 |
| 7B 五桶 BF16 no-reshard | 5 | `False` | 判断更大模型能否从省掉 backward all-gather 获得更高收益 | 完成，进入基线 |
| 7B 五桶 BF16 no-reshard | 6 | `False` | 验证额外显存能否转化为吞吐 | 完成，推荐 |
| 7B 五桶 BF16 no-reshard | 7 | `False` | 确认 batch 平台上限 | 完成，不推荐 |
| 上一推荐项 + `max-autotune-no-cudagraphs` | 6 | `False` | 测试长训内核搜索收益 | 完成，长训推荐 |

## 实测结果

每项均使用相同真实图文数据、五桶调度、前 20 step 预热，并以 step 30→40 的累计 `spd` 差分计算严格 wall-clock：

| 配置 | wall-clock/step | 单机吞吐 | 相对上一推荐项 | 峰值显存/卡 | step 40 loss |
|---|---:|---:|---:|---:|---:|
| BS5，`reshard_after_forward=True` | 2.0647 秒 | 19.373 样本/秒 | — | 75.75 GiB | 1.404876 |
| BS5，`reshard_after_forward=False` | 1.9856 秒 | 20.145 样本/秒 | **+3.98%** | 78.83 GiB | 1.404889 |
| **BS6，`reshard_after_forward=False`** | **2.2894 秒** | **20.966 样本/秒** | **+4.08%** | **84.22 GiB** | 1.424735¹ |
| BS7，`reshard_after_forward=False`（不推荐） | 2.6760 秒 | 20.926 样本/秒 | **-0.19%** | 89.03 GiB | 1.446184¹ |
| **BS6 + `max-autotune-no-cudagraphs`** | **2.2610 秒** | **21.230 样本/秒** | **+1.26%** | **84.45 GiB** | 1.424743 |

¹ BS 不同导致每步样本与 loss 对位不同，只用于检查 finite/下降趋势。

7B 上关闭 reshard 的收益明显高于 3B（+3.98% vs +0.97%）：模型参数更多，省掉 backward 前的参数 all-gather 更有价值，而显存只增加约 3.08 GiB。FSDP 节省出的空间也确实能转化为更大 batch：BS5→BS6 再提升 4.08%；BS7 已进入平台期且峰值达到 93.6%，因此停止 BS8。默认编译下推荐 BS6。

BS6 的 MMDiT `max-autotune-no-cudagraphs` 再提升 1.26%，与 3B 的 +1.22% 接近。首次五桶搜索启用 PyTorch 2.10 的 `TORCHINDUCTOR_DISTRIBUTED_MAX_AUTOTUNE_GEMM=1`，让 8 个 rank 分担 GEMM 算子搜索并同步结果；复用已生成的 BS6 Qwen/VAE 缓存、只冷编译 MMDiT max-autotune 图时，完整 50 step 为 28 分 26 秒。该成本对 1B 长训可忽略，但短实验保持 `compile_mode=default`。

7B 无法用同样 batch 做“不开 FSDP”的公平速度对照：仅 7.30B 训练参数、梯度和 AdamW 状态，再叠加每卡约 18GB 的冻结 Qwen 与激活，就会超过 96GB。FSDP2 在这里首先是可运行性条件。可严格隔离的 FSDP 策略收益是：关闭 reshard +3.98%，利用释放空间从 BS5 提到 BS6 再 +4.08%；叠加 max-autotune 后，相对标准 reshard BS5 累计提升 **9.58%**。

## 1B 图文样本时间

多机估算沿用 1/2/4/8/16/32 台分别 100%/99%/98%/97%/96%/95% 的扩展效率：

| 机器数 | GPU 数 | 旧 Muon+self-flow | BS5 标准 reshard | BS5 no-reshard | BS6 default | BS7 | **BS6 max-autotune** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 1393.14 天 | 597.43 天 | 574.54 天 | 552.03 天 | 553.08 天 | **545.18 天** |
| 2 | 16 | 703.60 天 | 301.73 天 | 290.17 天 | 278.80 天 | 279.34 天 | **275.35 天** |
| 4 | 32 | 355.39 天 | 152.40 天 | 146.57 天 | 140.82 天 | 141.09 天 | **139.08 天** |
| 8 | 64 | 179.53 天 | 76.99 天 | 74.04 天 | 71.14 天 | 71.27 天 | **70.26 天** |
| 16 | 128 | 90.70 天 | 38.89 天 | 37.40 天 | 35.94 天 | 36.01 天 | **35.49 天** |
| 32 | 256 | 45.83 天 | 19.65 天 | 18.90 天 | 18.16 天 | 18.19 天 | **17.93 天** |

当前 32 台推荐稳态估算为 17.93 天；加上 28 分 26 秒冷启动约为 **17.95 天**，仍基于 95% 扩展效率。真实 2/4 台 HSDP 效率需要多机测速校准。7B 的 FSDP/no-reshard 与 batch 收益高于 3B，但 7.3B MMDiT 的计算量也显著更大，因此仅靠这些等价优化不足以把 1B 样本压到 10 天以内。

相对旧 Muon+self-flow、固定 1024 的 8.31 样本/秒，新 BF16 推荐组合达到 21.23 样本/秒，吞吐约为 **2.56 倍**；32 台估算从 45.83 天降到 17.93 天，缩短约 **60.9%**。
