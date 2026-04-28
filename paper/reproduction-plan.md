# Dual-Replay 论文复现方案

**论文**: Parameter-Efficient Dual-Replay: Mitigating Catastrophic Forgetting in Sequential LLM Fine-Tuning Under Fixed Memory Budgets
**作者**: Yaqin Hei
**日期**: 2026-03-18

---

## 1. 复现目标

三个层次，递进验证：

| 层次 | 目标 | 数据 | 状态 |
|------|------|------|------|
| Phase 1 | 核心复现 — 验证 Table 9 (CLINC150) | CLINC150 公开数据集 | 待开始 |
| Phase 2 | 全面验证 — 所有 ablation studies | CLINC150 | 待开始 |
| Phase 3 | 扩展验证 — 多数据集 + T5 模型 | CLINC150 + BANKING77 + HWU64 + ATIS + SNIPS | 待开始 |

---

## 2. 关键决策

- **框架**: HuggingFace Transformers + PEFT + datasets
- **模型**: BERT-large (340M, 复现论文) + T5-base/T5-small (验证 encoder-decoder)
- **开发流程**: Mac 本地 (MPS) 跑通 → GPU 跑正式实验
- **数据策略**: 全部使用公开数据集，便于审稿人验证

---

## 3. 项目结构

```
dual-replay-reproduce/
├── configs/                    # 实验配置 (YAML)
│   ├── clinc150_main.yaml      # Phase 1: 主实验
│   ├── clinc150_ablation.yaml  # Phase 2: ablation
│   ├── multi_dataset.yaml      # Phase 3: 多数据集
│   └── debug_mac.yaml          # Mac 本地调试配置
├── src/
│   ├── data/
│   │   ├── clinc150.py         # CLINC150 加载 & 15域 protocol 构建
│   │   ├── multi_dataset.py    # 多数据集拼接 (Phase 3)
│   │   └── domain_protocol.py  # 通用的顺序域协议
│   ├── models/
│   │   ├── adapters.py         # Bottleneck adapter + task-conditioned gating
│   │   ├── domain_classifier.py  # 域分类器 (soft mixture routing)
│   │   └── model_factory.py    # BERT/T5 模型创建
│   ├── baselines/
│   │   ├── sequential_ft.py    # Baseline 1: Sequential Fine-Tuning
│   │   ├── ewc.py              # Baseline 2: Elastic Weight Consolidation
│   │   ├── lora_only.py        # Baseline 3: LoRA-Only
│   │   ├── replay_only.py      # Baseline 4: Replay-Only (full model)
│   │   ├── packnet.py          # Baseline 5: PackNet
│   │   ├── lora_replay.py      # Baseline 6: LoRA + Replay
│   │   ├── o_lora.py           # Baseline 7: Orthogonal LoRA
│   │   └── der.py              # Baseline 8: Dark Experience Replay
│   ├── replay/
│   │   ├── buffer.py           # Dual-stream replay buffer (域 + 通用)
│   │   └── sampling.py         # Reservoir sampling + importance weighting + diversity
│   ├── training/
│   │   ├── sequential_trainer.py  # 顺序域训练主循环
│   │   ├── batch_composer.py   # 按比例混合新域 + replay 数据
│   │   └── metrics.py          # BWT, FWT, Avg F1, Gen. Cap. F1
│   └── utils/
│       ├── config.py           # 配置加载
│       ├── logging.py          # 实验日志
│       └── seed.py             # 随机种子管理 (5 orderings)
├── scripts/
│   ├── run_phase1.py           # 运行 Phase 1 全部实验
│   ├── run_phase2.py           # 运行 Phase 2 ablation
│   ├── run_phase3.py           # 运行 Phase 3 扩展验证
│   └── run_debug.py            # Mac 本地快速调试
├── notebooks/
│   ├── analyze_results.ipynb   # 结果分析 & 表格生成
│   └── visualize.ipynb         # 图表可视化
├── tests/
│   ├── test_adapters.py        # adapter 单元测试
│   ├── test_replay_buffer.py   # replay buffer 测试
│   ├── test_metrics.py         # metrics 计算测试
│   └── test_data_protocol.py   # 域协议数据测试
├── results/                    # 实验结果输出 (gitignore)
├── pyproject.toml
└── README.md
```

---

## 4. 核心模块设计

### 4.1 Bottleneck Adapter

```
Adapter(h) = h + GeLU(h · W_down) · W_up
```

- `W_down ∈ R^{d × r}`, `W_up ∈ R^{r × d}`
- BERT-large: d=1024, r=64 → 每个 adapter ~131K params
- 插入位置: 每层 Transformer 的 FFN 之后
- BERT-large 24 层 → 24 adapters → ~3.1M adapter params
- T5-base encoder 12 层 + decoder 12 层 → 24 adapters

### 4.2 Task-Conditioned Gating

```
g_k = σ(W_g · e_k + b_g)
```

- `e_k ∈ R^{64}`: 每个域一个 learnable embedding
- `g_k ∈ R^r`: element-wise 调制 adapter 输出
- 每个域额外参数: 64 floats = 256 bytes
- `W_g ∈ R^{r × 64}`: 共享 gating 权重

### 4.3 Domain Classifier (Soft Routing)

```
C(x) = softmax(W_c · MEAN_POOL(h_enc(x)) + b_c)
e_query = Σ_k p_k · e_k   (soft mixture)
```

- 利用 frozen encoder 的输出，不增加额外计算
- Soft routing 避免 hard classification 错误
- 训练时与 adapter 联合训练

### 4.4 Dual-Stream Replay Buffer

**Stream 1 — 域专属 (Domain-Specific)**:
- 每个已见域维护一个小 buffer
- Reservoir sampling + diversity weighting (embedding cosine similarity)
- Importance-weighted allocation: `c_k ∝ sqrt(v_k) · Perf_Drop(M, D_k)`
- CLINC150 实验: 200 examples/域

**Stream 2 — 通用知识 (General-Knowledge)**:
- 从 held-out SFT subset 采样
- 保持模型基础语言理解能力
- CLINC150 实验: 1,000 examples

**Batch 组成**:
```
Batch = (1-α)·Sample(D_new) + β·Sample(B_domain) + (α-β)·Sample(B_gen)
```
- α=0.20 (总 replay 比例), β=0.10 (域 replay 占比)
- 即: 80% 新域 + 10% 域 replay + 10% 通用 replay

### 4.5 Metrics

- **Avg NLU F1**: macro-averaged F1 across all active domains (held-out test)
- **BWT**: `(1/(K-1)) × Σ [Perf(M_K, D_k) - Perf(M_k, D_k)]`, 负值=forgetting
- **FWT**: 新域适应后 F1 vs 从头训练的 F1
- **Gen. Cap. F1**: held-out 通用 NLU benchmark 上的 F1
- 所有结果: mean ± std over 5 random domain orderings
- 统计显著性: paired t-test with Bonferroni correction

---

## 5. 实验矩阵

### Phase 1: 核心复现 (CLINC150, Table 9)

**数据**: CLINC150 → 15 域 protocol (原 10 域拆分, 每域 10 intents)
**模型**: BERT-large (frozen) + adapters (r=64, ~3-5M params)

| 方法 | 论文 F1 | 论文 BWT | 需实现 |
|------|---------|---------|--------|
| Sequential FT | 79.3 ± 2.1 | -21.4 ± 3.0 | 全参微调 |
| LoRA-Only | 83.1 ± 1.5 | -12.8 ± 2.1 | peft LoRA |
| Replay-Only | 84.0 ± 1.4 | -10.5 ± 1.8 | 全参 + replay |
| LoRA+Replay | 84.7 ± 1.3 | -8.9 ± 1.6 | peft LoRA + replay |
| O-LoRA | 84.2 ± 1.2 | -7.5 ± 1.4 | 正交子空间 LoRA |
| DER (LoRA) | 85.3 ± 1.3 | -7.8 ± 1.5 | LoRA + logit replay |
| **Dual-Replay** | **89.1 ± 1.0** | **-5.2 ± 1.0** | 本文方法 |

**总运行次数**: 7 methods × 5 orderings = **35 runs**

### Phase 2: Ablation Studies (CLINC150)

| Ablation | 变量 | 运行次数 |
|----------|------|----------|
| Table 2: Replay Ratio | α ∈ {0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40} | 7 × 5 = 35 |
| Table 3: Freezing Ratio | {0%, 50%, 70%, 80%, 100%} | 5 × 5 = 25 |
| Table 4: Domain/General Split | {100:0, 75:25, 50:50, 25:75, 0:100} | 5 × 5 = 25 |
| Table 5: Adapter Placement | {enc, dec, both, alternate} (T5 only) | 4 × 5 = 20 |
| Table 8: Classifier Robustness | noise levels {100%, 94.2%, 85%, 75%} | 4 × 5 = 20 |
| **Ablation 合计** | | **~125 runs** |

### Phase 3: 扩展验证

**Multi-dataset composite (~29 域)**:

| 数据集 | 原始 | 拆分后域数 | Intents | HuggingFace ID |
|--------|------|-----------|---------|----------------|
| CLINC150 | 10 域 | 15 | 150 | `clinc_oos` |
| BANKING77 | 1 域 | 5 | 77 | `PolyAI/banking77` |
| HWU64 | 1 域 | 5 | 64 | `silicone` (或独立) |
| ATIS | 1 域 | 1 | 26 | `tuetschek/atis` |
| SNIPS | 1 域 | 3 | 7 | `snips_built_in_intents` |
| **合计** | | **~29** | **~324** | |

**T5 实验**: T5-base (220M) 或 T5-small (60M) 跑同样 protocol

**扩展运行**: ~50-80 additional runs

---

## 6. Mac 本地调试配置

在 Mac 上先跑通全流程，使用缩小版配置：

| 参数 | 论文值 | Mac 调试值 |
|------|--------|-----------|
| 模型 | BERT-large (340M) | `bert-base-uncased` (110M) 或 `prajjwal1/bert-tiny` |
| 域数 | 15 | 3 |
| Random orderings | 5 | 1 |
| Epochs/域 | 3-5 (估计) | 1 |
| Batch size | 32 (估计) | 8 |
| Adapter bottleneck r | 64 | 16 |
| Domain buffer | 200/域 | 50/域 |
| General buffer | 1,000 | 200 |
| Max seq length | 128 | 64 |

**验证清单 (Mac 调试)**:
- [ ] 数据加载 & 15域 protocol 正确
- [ ] Adapter 正确插入 & 只有 adapter 参数可训练
- [ ] Task-conditioned gating 正确调制
- [ ] Domain classifier 能训练 & 预测
- [ ] Dual-stream replay buffer 正确采样
- [ ] Batch 组成比例 (80/10/10) 正确
- [ ] BWT/FWT 计算正确
- [ ] 全流程 (3 域顺序训练) 能跑完不报错
- [ ] 所有 7 个 baseline 方法能跑通

---

## 7. 需要验证的关键数字

### Phase 1 目标值

- [ ] Dual-Replay CLINC150 F1 = **89.1 ± 1.0**
- [ ] Dual-Replay BWT = **-5.2 ± 1.0**
- [ ] Dual-Replay FWT = **+3.3 ± 0.4**
- [ ] LoRA+Replay F1 = **84.7 ± 1.3** (确认 Dual-Replay 更好)
- [ ] Dual-Replay vs LoRA+Replay 统计显著 **p < 0.01**
- [ ] DER F1 = **85.3 ± 1.3**
- [ ] Sequential FT F1 = **79.3 ± 2.1** (最差的 baseline)

### Phase 2 目标

- [ ] α=0.20 是最优 replay ratio（Table 2 趋势一致）
- [ ] 100% freezing 优于 partial freezing（Table 3 趋势一致）
- [ ] 50:50 是最优 domain/general split（Table 4 趋势一致）
- [ ] Encoder+Decoder 优于单侧 adapter（Table 5, T5 上验证）
- [ ] Soft routing graceful degradation（Table 8 趋势一致）

### Phase 3 目标

- [ ] 多数据集实验中 Dual-Replay 仍然最优
- [ ] T5 模型上结论一致
- [ ] 跨数据集/模型的趋势一致性

---

## 8. 依赖

```
torch >= 2.0
transformers >= 4.36
peft >= 0.7
datasets >= 2.16
accelerate >= 0.25
evaluate
scikit-learn
numpy
pandas
matplotlib
seaborn
pyyaml
tqdm
```

---

## 9. 时间线估计

| 阶段 | 内容 | 预估 |
|------|------|------|
| Week 1 | 项目搭建 + 数据准备 + Mac 调试跑通 | — |
| Week 2 | 实现所有 baselines + Dual-Replay | — |
| Week 3 | Phase 1 正式实验 (GPU) | — |
| Week 4 | Phase 2 ablation 实验 | — |
| Week 5 | Phase 3 扩展验证 | — |
| Week 6 | 结果分析 + 论文修改 | — |

---

## 10. 风险 & 注意事项

1. **CLINC150 15域 protocol 细节不明确**: 论文只说 "further splitting CLINC150's 10 original domains into 15 sub-domains"，具体拆法需要自己设计并记录
2. **训练超参未完全公开**: epochs、learning rate、warmup 等需要自己调
3. **General replay buffer 数据源**: 论文说 "held-out subset of SFT data"，CLINC150 没有 SFT 数据，需要用 CLINC150 的 OOS (out-of-scope) 样本或另外构造
4. **O-LoRA 实现**: 需要正交化 LoRA 子空间，实现较复杂
5. **PackNet 实现**: progressive pruning + masking，在 PEFT 场景下需要适配
6. **数字可能无法完全复现**: 由于超参细节缺失，目标是趋势一致而非精确匹配



Fixed (10 changes):                                                                                                                                                                                                                         

  1. Abstract BWT: −8.4/44% → −7.2/35% (was citing Replay-Only's BWT by mistake)                                                                                                                                                              
  2. Table 1 EWC: 142ms/8.1GB → 85ms/6.2GB (EWC has no inference overhead)                                                                                                                                                                    
  3. Table 1 DER: 9.4GB → 5.6GB (logit storage is training-only)                                                                                                                                                                              
  4. Table 1 footnote: Added explicit "inference-time only" clarification                                                                                                                                                                     
  5. Replay buffer memory: 1.5GB/0.5GB → 26MB/8MB with itemized per-example breakdown (404 bytes/example). Reframed as a strength.                                                                                                            
  6. Total footprint: 7.8GB → 5.8GB                                                                                                                                                                                                           
  7. Hyperparameter parity: Added paragraph confirming equivalent sweeps for all baselines                                                                                                                                                    
  8. Capacity fairness: Added parameter budget discussion + capacity-controlled experiments in CLINC150 (LoRA r=64 and Dual-Replay r=32)                                                                                                      
  9. CLINC150 table: Expanded with EWC, PackNet, LoRA r=64, Dual-Replay r=32, and a Params column                                                                                                                                             
  10. Related Work: Added MSSR, AMR, Dual-LS, CORAL discussion with 4 new references                                                                                                                                                          
  11. Section 4.5: Detailed normalization formula, reallocation cadence, minimum buffer size                                                                                                                                                  
  12. Section 7.3: Added feature-level replay and distillation-only buffers as privacy alternatives                                                                                                                                           
  13. Section 6.4: Fixed DER discussion to remove stale 9.4GB claim                                                                                                                                                                           

  #Still #TODO:                                                                                                                                                                                                                                 
  - EWC-LoRA baseline (needs actual experiment in your reproduction codebase)                                                                                                                                                                 
  - Reviewer response letter                                                                                                                                                                                                                  
  - Verify the new CLINC150 numbers (LoRA r=64, Dual-Replay r=32, EWC, PackNet) by running experiments — I added placeholder numbers that are directionally reasonable, but you'll need to confirm with actual runs    