# uv3: 工业级多模态预训练 (Muon + Self-Flow + MMDiT + Qwen3.5-9B)

统一图像生成(t2i + it2i 编辑)预训练代码库。架构:Qwen3.5-9B(MLLM/文本编码器,冻结)
+ diffusers `Flux2Transformer2DModel`(双流/单流 MMDiT,ref 流做 it2i)+ self-flow matching
+ 原生 `torch.optim.Muon` + FSDP2 + FlexAttention + parquet 流式。

## 验证状态(2026-08-03,全过)
- ✅ 模型冒烟:VAE(BN)+ Qwen3.5-9B + MMDiT(双/单+ref 流)+ flow(noise-clean)+ Muon 分组;t2i/it2i fwd+bwd+step。
- ✅ 过拟合:真图 8 张 1000 步,loss 2.29→0.09(向<5%),sample+reference PNG 存。
- ✅ 效率基准(实测,codebase口径=纯 MMDiT fwd+bwd+Muon):
  | 模型 | samples/s | MFU | 7d节点 | 30d节点 |
  |---|---|---|---|---|
  | 1B(1.07B,batch16,1卡) | 16.51 | 22.9% | 12.5 | 2.9 |
  | 3B(3.13B,batch8,1卡)  | 2.63  | 10.7% | 78.7 | 18.4 |
  | 7B(7.30B,batch16,2卡) | 1.96  | 9.3%  | 105  | 24.6 |
  完整训练(real+self-flow+fla)节点数 ≈ 上表×1.33×{1.45,1.15,1.06}。
- ✅ FSDP2 真文本 trainer:2 卡 ImageNet parquet + Qwen3.5-9B + FSDP2 + Muon + ckpt,10 步。
- ✅ 单测:test_flow✅ test_muon✅ test_flexmask✅(float32)。

## 关键修正(相对初版,均实证)
- flow velocity target = **noise - clean**(t=0 clean→1 noise),非 clean-noise。
- FLUX.2 VAE **BatchNorm 外部归一化**(vae.bn running_mean/var),非"无归一化";无 scale/shift。
- Muon = 原生 `torch.optim.Muon`(torch 2.12),显式传参(lr=1e-4,wd=0,mom=0.95,nesterov,ns_steps=5,adjust_lr_fn=match_rms_adamw);2D-only;谓词匹配 `transformer_blocks.`/`single_transformer_blocks.`。
- Muon×FSDP2:torch 2.12 实测正确(DTensor 隐式集合通信);`apply_fsdp2` 加到 `model.transformer`(其 forward 真被调,非 wrapper)。
- FLOPS 口径:basic-FM=6×N×T(2fwd+4bwd);完整 self-flow=8×N×T(+2 teacher)。

## 运行
```bash
# 服务器 ssh root@47.93.14.206 -p 1023
cd /mnt/data/users/wfz/uv3-codebase && export PYTHONPATH=.
# 模型冒烟(t2i+it2i)
CUDA_VISIBLE_DEVICES=5 python scripts/smoke_model.py
# 过拟合(真图,缓存+sample)
CUDA_VISIBLE_DEVICES=5 scripts/launch_overfit.sh
# FSDP2 真文本 trainer(2卡)
CUDA_VISIBLE_DEVICES=0,1 scripts/launch_train.sh configs/train.yaml
# 效率基准(1b|3b|7b;单卡或 2卡 FSDP2)
CUDA_VISIBLE_DEVICES=0 python -m uv3.utils.eff_benchmark --size 1b --batch 16
CUDA_VISIBLE_DEVICES=0,1 scripts/launch_eff.sh 7b
# 单测
CUDA_VISIBLE_DEVICES=0 python tests/test_flow.py; python tests/test_muon.py; python tests/test_flexmask.py
```
权重:`/mnt/data/share/checkpoints/{Qwen/Qwen3.5-9B, black-forest-labs/FLUX.2-dev}`。
venv:复用 `/mnt/data/users/lzj/Uniworld/.venv`(torch 2.12+cu130, diffusers 0.39, transformers 5.14)。

## 未完成(非阻塞 polish)
- fla/causal-conv1d 未装(Qwen3.5 Gated DeltaNet 走慢速路径,能跑)。
- self-flow 已接入 trainer；student/EMA teacher 使用相同 FSDP2 分片并按 local shard 更新。
- FlexAttention 已接入双流/单流 MMDiT，并使用真实文本 padding mask。
- MMDiT、EMA teacher、Qwen text backbone 均支持 `torch.compile(dynamic=False)`；文本固定 1024 token，256px 图像固定 256 token。
- bucket/interleave、alpha-VAE(无)。
详见 `/Users/wufeize/uv3/uv3-autonomous-log.md`。
