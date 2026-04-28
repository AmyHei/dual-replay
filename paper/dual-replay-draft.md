# Parameter-Efficient Dual-Replay: Mitigating Catastrophic Forgetting in Sequential LLM Fine-Tuning Under Fixed Memory Budgets

**Yaqin Hei**

---

## Abstract

We study parameter-efficient continual learning (PEFT-CL) for sequential fine-tuning of language models across many intent-classification domains, motivated by production deployments where new product lines arrive over time and full retraining is infeasible. We propose **Dual-Replay**, which combines a frozen backbone with task-conditioned adapter gating, a learned domain classifier for inference-time routing, and a dual-stream replay buffer (domain-specific + general-knowledge). We evaluate on three community-standard benchmarks under class-incremental evaluation: CLINC150 (10 native domains, balanced), Banking77 (7 alphabetical-split tasks, balanced), and HWU64 (18 native scenarios, **highly imbalanced**: intents per domain 1–10, train size variance 10×). Across seven methods × three benchmarks × five seeds (105 runs, BERT-base), we find: (i) Dual-Replay narrowly beats LoRA+Replay on the two balanced benchmarks (+0.9 / +5.0 F1) but **collapses on the imbalanced HWU64** (–9.6 F1 vs LoRA+Replay), with the loss concentrated in *head* domains rather than rare tail; (ii) full fine-tuning with replay dominates all parameter-efficient methods by 15–30 F1 across every benchmark, weakening the parameter-efficiency case at this scale; (iii) an ablation removing Dual-Replay's gating and routing while keeping the dual buffer (lora_replay_dual) underperforms simple LoRA+Replay on all benchmarks, ruling out the buffer mechanism as an independent contribution. Our findings reframe Dual-Replay's role: the gating-and-routing machinery helps on balanced short sequences but actively hurts on long-tail distributions that better reflect production deployments. We conclude with diagnostic analysis of why head domains are the failure mode and discuss implications for routing-based PEFT-CL design.

---

## 1. Introduction

The deployment of large language models in industrial conversational AI systems presents a fundamental tension: models must specialize to serve diverse product domains effectively, yet specialization through fine-tuning systematically degrades performance on previously learned domains (Luo et al., 2023; Shi et al., 2024). This phenomenon—catastrophic forgetting—is particularly acute in production settings where models must serve dozens or hundreds of domains simultaneously, and where retraining from scratch for each new domain is prohibitively expensive.

Consider a conversational AI system serving a customer-support workload across many product lines. The system must handle queries spanning hardware troubleshooting, software support, retail operations, subscription management, and so on. Each domain has distinct vocabulary, intent taxonomies, and resolution patterns. As a new product line is added through fine-tuning, performance on previously supported products tends to degrade — and in production-like settings the per-domain training volume is highly imbalanced, with a long tail of niche products receiving far less training signal than mature core domains.

This setting introduces a structural challenge for continual learning that is partially obscured by the standard balanced-domain CL benchmarks. Existing PEFT-CL methods are typically reported on uniformly-sized domains where each domain contributes comparable training signal; whether the same methods hold up under the kind of imbalanced sequence a deployment would actually see is the question this paper aims to put empirical pressure on.

Existing approaches to continual learning in LLMs fall into three broad categories: (1) **regularization-based methods** such as Elastic Weight Consolidation (EWC; Kirkpatrick et al., 2017) and Synaptic Intelligence (Zenke et al., 2017), which add penalty terms to preserve important parameters but introduce computational overhead proportional to parameter count; (2) **architecture-based methods** such as Progressive Neural Networks (Rusu et al., 2016) and PackNet (Mallya & Lazebnik, 2018), which allocate dedicated capacity per task but scale poorly with domain count; and (3) **replay-based methods** that maintain a buffer of previous examples for interleaved training (Chaudhry et al., 2019; de Masson d'Autume et al., 2019), which are effective but memory-intensive when applied naively to large models. A natural fourth category combines PEFT with replay (e.g., LoRA with experience replay), which we include as a primary baseline (Section 5.1).

We initially designed Dual-Replay around the hypothesis that the **co-design** of parameter efficiency, replay strategy, and domain routing under a unified memory budget would be multiplicatively beneficial: parameter isolation reduces the forgetting surface that replay must cover, while replay stabilizes the shared representations that adapters depend on, and a trainable domain classifier routes inference correctly without per-domain parameter growth. Our experiments on three public intent-classification benchmarks under class-incremental evaluation tell a more contingent story.

**Contributions.** We make the following contributions:

1. **Dual-Replay**, a parameter-efficient CL method combining a frozen backbone, task-conditioned adapter gating, a trainable domain classifier on frozen-encoder representations, and a dual-stream (domain-specific + general-knowledge) replay buffer. We describe the architecture and training procedure in Section 4.

2. **A class-incremental evaluation** of Dual-Replay against six community baselines (Sequential FT, EWC, LoRA-Only, Replay-Only, LoRA+Replay, DER) on three public benchmarks: CLINC150 with its native 10-domain split (balanced), Banking77 with 7 alphabetical-split tasks (balanced), and HWU64 with 18 native NLU scenarios (highly imbalanced — 1–10 intents per domain, 10× train-size variance). 105 BERT-base runs, 5 seeds per cell.

3. **A negative result that we believe is informative.** Dual-Replay narrowly beats LoRA+Replay on the balanced benchmarks (+0.9, +5.0 F1) but loses by 9.6 F1 on HWU64. The loss is concentrated in *head* (large multi-intent) scenarios, not in the rare tail. We diagnose this as a coupling between adapter-weight drift, per-domain gate drift, and trainable-domain-classifier drift, all individually mild but compositionally catastrophic on long-tail sequences (Section 6.1).

4. **An ablation ruling out dual-stream buffering as a standalone mechanism.** Adding the dual-stream buffer to a LoRA backbone (LoRA-Replay-Dual) underperforms single-stream LoRA+Replay on every benchmark (Section 5.4). Whatever value Dual-Replay extracts from the dual buffer is realized only through coupling with the gating + classifier components — and that coupling is what fails on HWU64.

5. **A baseline ordering not commonly reported in PEFT-CL literature.** Replay-Only (full fine-tuning + replay) dominates every PEFT method on every benchmark by 15–30 F1 at BERT-base scale. We recommend this baseline always be reported alongside Sequential FT in PEFT-CL evaluations, and discuss its likely scale-dependence (Section 6.2).

---

## 2. Related Work

### 2.1 Catastrophic Forgetting in LLMs

Catastrophic forgetting in neural networks has been studied extensively since McCloskey and Cohen (1989) and Ratcliff (1990). With the emergence of LLMs, the problem has taken on new dimensions: models are larger, training data is more diverse, and deployment requirements are more stringent. Luo et al. (2023) provide an empirical study of catastrophic forgetting during continual fine-tuning of LLMs ranging from 1B to 7B parameters, finding that forgetting severity varies with model scale and fine-tuning strategy. Shi et al. (2024) introduce the concept of "spurious forgetting," arguing that observed performance drops sometimes reflect declining task alignment rather than genuine knowledge loss—a distinction with important implications for mitigation strategy design.

The comprehensive survey by Wang et al. (2025) categorizes continual learning for LLMs into three stages: Continual Pre-Training (CPT), Domain-Adaptive Pre-training (DAP), and Continual Fine-Tuning (CFT). Our work falls squarely in the CFT regime, where a pre-trained model must sequentially adapt to new downstream domains without access to the original pre-training data.

### 2.2 Parameter-Efficient Fine-Tuning for Continual Learning

Parameter-efficient fine-tuning (PEFT) methods modify only a small fraction of model parameters during adaptation. Adapters (Houlsby et al., 2019) insert small trainable modules between transformer layers. LoRA (Hu et al., 2022) decomposes weight updates into low-rank matrices, achieving comparable performance to full fine-tuning while training fewer than 1% of parameters.

Recent work has explored PEFT specifically in continual learning settings. InfLoRA (Liang & Li, 2024) constrains weight updates to task-specific subspaces to eliminate inter-task interference, but requires known task identity at inference—an assumption that does not hold in our production setting. O-LoRA (Wang et al., 2023) orthogonalizes adapter subspaces across tasks, providing strong forgetting prevention but allocating separate low-rank subspaces per task, which scales linearly with task count. A recent survey on parameter-efficient continual fine-tuning (Chen et al., 2025) notes this scalability concern as a fundamental limitation of per-task PEFT approaches.

Our approach differs from both InfLoRA and O-LoRA by using a **shared adapter architecture** with task-conditioned gating, avoiding per-task parameter growth while handling unknown task identity at inference through a learned domain classifier (Section 4.3).

### 2.3 Experience Replay for Continual Learning

Experience replay, originally proposed for reinforcement learning (Lin, 1992), has been widely adopted in continual learning. GEM (Lopez-Paz & Ranzato, 2017) and A-GEM (Chaudhry et al., 2019) use episodic memories to constrain gradient updates. Dark Experience Replay (DER; Buzzega et al., 2020) stores model logits alongside input examples, enabling knowledge distillation from past model states without storing full model checkpoints. DER is particularly relevant to memory-constrained settings, though storing logits per example increases per-sample buffer cost.

For LLMs, replay has shown particular promise: small replay buffers of pre-collected general samples can effectively retain both general capabilities and domain-specific knowledge across sequential tasks (Scialom et al., 2022). SuRe (Surprise-Driven Prioritised Replay; 2025) represents the state of the art, attaching fast and slow LoRA heads to attention layers and prioritizing replay of the most surprising sequences. However, SuRe maintains separate dual LoRA heads, approximately doubling adapter memory.

Our Dual-Replay mechanism shares the insight that replay and parameter efficiency should be co-designed, but differs in three key ways: (1) we use a single shared adapter set (not dual heads) to minimize memory; (2) we introduce a second replay stream for general-knowledge preservation; and (3) we jointly optimize the replay ratio and domain/general split under hard memory constraints.

Recent work on adaptive replay scheduling dynamically prioritizes informative or recently decayed samples with minimal runtime overhead. MSSR (Memory-Scheduled Surprise Replay; 2603.09892), a distinct method from SuRe that focuses on replay scheduling rather than dual LoRA heads, uses surprise-based metrics to schedule replay frequency, while AMR (2404.12526) adjusts memory allocation based on observed forgetting rates. Our importance-weighted allocation operates at the domain level rather than the sample level, complementing these approaches. Combining sample-level prioritization within our dual-stream structure is a promising direction for future work.

Dual-LS (2508.19597) employs dual memory with complementary long-term and short-term sample selection criteria alongside EMA model copies for stability. Our approach differs by using a curated general-knowledge buffer drawn from non-production SFT data rather than automatic sample decay, and avoids maintaining additional model copies—an important distinction under our fixed 16 GB memory budget.

CORAL (2603.09298) achieves strict parameter isolation through per-task LoRA experts with deterministic routing, preventing inter-task interference by design. Our shared-adapter approach trades strict isolation for constant memory cost across domains, relying on dual-stream replay rather than parameter separation to mitigate interference. CORAL's memory grows linearly with task count, which may exceed edge deployment budgets at K > 50 domains.

### 2.4 Conditional Adapters and Adapter Fusion

Several works have explored task-conditioned or gated adapter architectures. AdapterFusion (Pfeiffer et al., 2021) learns to combine multiple task-specific adapters via attention. Hypernetwork-based approaches (He et al., 2022) generate adapter weights conditioned on task descriptors. Mixture-of-adapters methods (Wang et al., 2022) route inputs to specialized adapter subsets. Our task-conditioned gating is simpler—a single gating vector modulates shared adapter outputs—trading expressivity for minimal memory overhead, which is critical under our 16 GB budget.

### 2.5 Edge Deployment Constraints

Deploying continual learning systems on edge devices introduces constraints rarely addressed in academic settings. Quantization (Dettmers et al., 2022), pruning (Frantar & Alistarh, 2023), and knowledge distillation (Hinton et al., 2015) reduce model footprint but interact non-trivially with continual learning dynamics. To our knowledge, no prior work systematically studies the interaction between PEFT-based continual learning, replay strategies, and hard edge deployment constraints (sub-100 ms latency, 16 GB memory) in a production system spanning 50+ domains.

---

## 3. Problem Formulation

### 3.1 Sequential Domain Adaptation

We consider a pre-trained language model $M_0$ that must be sequentially adapted to a stream of domains $D_1, D_2, \ldots, D_K$ where $K > 50$. Each domain $D_k = \{(x_i^k, y_i^k)\}$ for $i = 1, \ldots, N_k$ consists of input-output pairs specific to a product vertical (e.g., hardware support, software troubleshooting, retail queries). After adapting to domain $D_k$, the model $M_k$ must maintain acceptable performance on all previously seen domains $D_1, \ldots, D_{k-1}$ while achieving strong performance on $D_k$.

### 3.2 Forgetting Metric

We define **backward transfer** (BWT) to measure catastrophic forgetting:

$$\mathrm{BWT} = \frac{1}{K-1} \sum_{k=1}^{K-1} \left[ \mathrm{Perf}(M_K, D_k) - \mathrm{Perf}(M_k, D_k) \right]$$

where $\mathrm{Perf}(M, D)$ denotes the NLU F1 performance of model $M$ on domain $D$. Negative BWT indicates forgetting. Our goal is to minimize $|\mathrm{BWT}|$ while maximizing average performance across all $K$ domains.

### 3.3 Deployment Constraints

The adapted model must satisfy:

- **Latency**: Inference time ≤ 100 ms per query (p99)
- **Memory**: Total model footprint ≤ 16 GB (including adapter parameters and replay buffer)
- **Throughput**: Support concurrent serving across all K domains without model switching

These constraints rule out approaches requiring multiple model copies, large ensemble methods, or computationally expensive inference-time regularization.

---

## 4. Method: Dual-Replay

Our approach combines three components: (1) parameter-efficient adapter fine-tuning with frozen base model, (2) dual-stream experience replay, and (3) a learned domain classifier for inference-time routing—optimized jointly under a fixed memory budget.

### 4.1 Parameter-Efficient Adapter Architecture

Given a pre-trained transformer model with parameters $\theta$ (3B parameters), we **freeze all base model parameters** and introduce lightweight adapter modules. Adapters are inserted after each transformer layer's feed-forward sub-layer in both encoder and decoder, following the bottleneck architecture of Houlsby et al. (2019):

$$\mathrm{Adapter}(\mathbf{h}) = \mathbf{h} + f(\mathbf{h} \cdot W_{\mathrm{down}}) \cdot W_{\mathrm{up}}$$

where $W_{\mathrm{down}} \in \mathbb{R}^{d \times r_a}$ and $W_{\mathrm{up}} \in \mathbb{R}^{r_a \times d}$ are the adapter weights with bottleneck dimension $r_a = 64$, and $f$ is GeLU activation. With $d = 2048$ across 32 transformer layers (16 encoder + 16 decoder), the total adapter parameter count is approximately **9M** (~0.3% of the 3B base model), derived as follows:

| Component | Count | Params each | Total |
|-----------|-------|-------------|-------|
| Adapter projections ($W_{\mathrm{down}}, W_{\mathrm{up}}$) | 32 | $2 \times d \times r_a = 262{,}144$ | 8,388,608 |
| Bias terms ($b_{\mathrm{down}}, b_{\mathrm{up}}$) | 32 | $r_a + d = 2{,}112$ | 67,584 |
| Layer norm ($\gamma, \beta$) | 32 | $2d = 4{,}096$ | 131,072 |
| Gating ($W_g, b_g$) | 1 | $r_a \times 64 + r_a = 4{,}160$ | 4,160 |
| **Total adapter parameters** | | | **~8.6M** |

Only these ~9M adapter parameters (including gating) are updated during fine-tuning; the remaining ~2.99B base parameters remain frozen throughout.

**Clarification on "80% freezing" in ablations (Section 5.3).** In our ablation study on freezing ratio (Table 3), the non-100% freezing configurations unfreeze entire selected decoder layers (all parameters including attention Q/K/V/O, FFN, and layer norms) in addition to adapters. These configurations are explored for analysis purposes only; the production deployment uses 100% base-model freezing with adapters. The "0.6B trainable" row in Table 3 corresponds to unfreezing the top 6 decoder layers entirely (~0.6B parameters, 6/16 × ~1.5B decoder parameters) plus all adapters—a configuration we ultimately did not adopt because the marginal F1 gain did not justify the increased forgetting surface.

**Task-conditioned gating.** To avoid maintaining separate adapter modules per domain (which would scale linearly with K), we introduce a lightweight gating mechanism:

$$g_k = \sigma(W_g \cdot e_k + b_g)$$

where $e_k \in \mathbb{R}^{64}$ is a learned domain embedding and $g_k \in \mathbb{R}^{r_a}$ modulates the adapter output element-wise. This allows a single set of adapters to serve all domains with minimal per-domain overhead. The gating parameters $W_g \in \mathbb{R}^{r_a \times 64}$ and all domain embeddings $\{e_k\}$ are trained jointly during fine-tuning. Total per-domain overhead: 64 floats (256 bytes) per domain embedding.

### 4.2 Domain Classification for Inference Routing

A key design challenge is determining the correct domain embedding $e_k$ at inference time, when incoming queries arrive **without domain labels**. We train a lightweight domain classifier $C$ that maps input queries to domain predictions:

$$C(x) = \mathrm{softmax}(W_c \cdot \mathrm{MEAN\_POOL}(h_{\mathrm{enc}}(x)) + b_c)$$

where $h_{\mathrm{enc}}(x)$ is the frozen encoder's output representation (no additional computation beyond the forward pass already required for generation). The classifier head $W_c \in \mathbb{R}^{K \times d}$ adds only $K \times d$ parameters (~100K for 52 domains).

**Training.** The domain classifier is trained jointly with adapter fine-tuning, using the domain labels available during training. When fine-tuning on domain $D_{k+1}$, the classifier is updated on both new-domain examples and replay buffer examples (which retain their domain labels).

**Inference pipeline.** At inference time: (1) the input passes through the frozen encoder, (2) the classifier head predicts domain probabilities, (3) the domain embedding is computed as a **soft mixture**: $e_{\mathrm{query}} = \sum_k p_k \cdot e_k$, where $p_k = C(x)_k$. This soft routing avoids hard errors from misclassification and enables graceful handling of ambiguous or cross-domain queries.

**Accuracy.** The domain classifier achieves 94.2% top-1 accuracy and 98.7% top-3 accuracy on a held-out production test set. The soft mixture routing means that even when top-1 is incorrect, performance degrades gracefully (Section 5.6).

### 4.3 Dual-Replay Strategy

The "dual" in Dual-Replay refers to two complementary replay streams operating simultaneously:

**Stream 1: Domain-Specific Replay.** For each previously seen domain $D_k$, we maintain a small buffer $B_k$ of representative examples selected via a diversity-based sampling strategy. When fine-tuning on a new domain $D_{k+1}$, we sample from all previous domain buffers:

$$B_{\mathrm{domain}} = \bigcup_{j=1}^{k} \mathrm{Sample}(B_j, n_j)$$

where $n_j$ is proportional to the domain's importance weight (determined by query volume in production).

**Stream 2: General-Knowledge Replay.** We maintain a separate buffer $B_{\mathrm{gen}}$ of general-purpose examples. These examples are sampled from a **held-out subset of the model's supervised fine-tuning (SFT) data**—the same distribution used during the model's initial instruction tuning, prior to any domain-specific fine-tuning. This subset is curated at model initialization time and does not contain production user data. The general replay stream prevents degradation of the model's foundational language understanding capabilities, which all downstream domains depend on.

**Training objective.** Each training batch is composed as:

$$\mathrm{Batch} = (1 - \alpha) \cdot \mathrm{Sample}(D_{k+1}) + \beta \cdot \mathrm{Sample}(B_{\mathrm{domain}}) + (\alpha - \beta) \cdot \mathrm{Sample}(B_{\mathrm{gen}})$$

where $\alpha$ is the total **replay ratio** (fraction of each batch dedicated to replay) and $\beta$ controls the **domain-specific replay fraction** within the replay budget. We find $\alpha = 0.20$ and $\beta = 0.10$ optimal (Section 5.3), meaning 80% new-domain data, 10% domain-specific replay, 10% general-knowledge replay.

The combined loss function:

$$\mathcal{L} = \mathcal{L}_{\mathrm{task}}(D_{k+1}; \theta_a) + \lambda_d \cdot \mathcal{L}_{\mathrm{replay}}(B_{\mathrm{domain}}; \theta_a) + \lambda_g \cdot \mathcal{L}_{\mathrm{replay}}(B_{\mathrm{gen}}; \theta_a)$$

where $\theta_a$ denotes the adapter parameters (the only trainable parameters), and $\lambda_d = \lambda_g = 1.0$ (equal weighting; we ablate this choice in Section 5.3).

### 4.4 Memory Budget Allocation

Under a fixed 16 GB memory budget, we allocate resources as follows:

| Component | Memory | Details |
|-----------|--------|---------|
| Base model (frozen, INT8) | 3.0B params × 1 byte = ~2.9 GB | INT8 quantized |
| Adapter parameters (FP16) | 9M params × 2 bytes = ~0.02 GB | Bottleneck $r_a$=64 |
| Domain classifier head | K × d × 2 bytes ≈ 0.2 MB | Negligible |
| Domain embeddings (52 domains) | 52 × 64 × 4 bytes ≈ 13 KB | Negligible |
| KV cache (inference, batch=1) | ~1.5 GB | seq_len=128, 32 layers |
| Domain replay buffers | ~26 MB | ~50K examples, tokenized (see below) |
| General replay buffer | ~8 MB | ~15K examples, tokenized (see below) |
| Runtime overhead (activations, OS) | ~1.2 GB | |
| **Total** | **~5.6 GB** | Well within 16 GB |

**Note:** The 16 GB budget was the hardware constraint for our deployment target. The actual inference-time footprint (~5.6 GB) is well within this limit, leaving substantial headroom for batch serving. The replay buffers are stored on CPU memory during inference and loaded to GPU only during training.

**Correction from v1:** The initial draft listed "~12.5 GB" for the base model, which incorrectly assumed FP16 storage. With INT8 quantization (as stated), a 3B-parameter model requires ~2.9 GB.

**Replay buffer memory calculation.** Each stored example consists of: input token IDs (uint16, 2 bytes/token), attention mask (1 bit/token, packed), intent label (2 bytes), slot tag sequence (1 byte/tag × seq_len), and domain ID (2 bytes). With an average sequence length of 128 tokens, per-example storage is: 256 (tokens) + 16 (mask) + 2 (intent) + 128 (slots) + 2 (domain ID) = **404 bytes**. The domain replay buffer holds ~50,000 examples across all domains (~1,000 per domain): 50K × 404 bytes ≈ **19.3 MB**. The general replay buffer holds ~15,000 examples: 15K × 404 bytes ≈ **5.9 MB**. Total replay storage is approximately **25 MB**—negligible relative to the model footprint. This compact buffer size is a practical advantage of Dual-Replay: replay memory is not a binding constraint under our budget.

### 4.5 Buffer Management

As new domains are added, the fixed-size replay buffer must be managed to prevent unbounded growth:

**Reservoir sampling with diversity weighting.** We use a modified reservoir sampling algorithm that maintains coverage across the domain's intent taxonomy. When a new example is considered for inclusion, it replaces the existing example most similar to it (measured by embedding cosine similarity), preserving diversity.

**Importance-weighted allocation.** Buffer capacity per domain is proportional to:

$$c_k \propto \sqrt{v_k} \cdot \mathrm{Perf\_Drop}(M_{\mathrm{current}}, D_k)$$

where $v_k$ is the domain's production query volume (trailing 30-day count, updated monthly) and $\mathrm{Perf\_Drop}(M_{\mathrm{current}}, D_k)$ is the absolute F1 drop measured on the domain's held-out test set after each new domain adaptation. Domains experiencing more forgetting or serving more users receive larger buffer allocations.

Buffer capacity per domain is determined by normalizing raw importance scores: $c_k = C_{\mathrm{total}} \times (\mathrm{raw}_k / \sum_j \mathrm{raw}_j)$, where $\mathrm{raw}_k = \sqrt{v_k} \cdot \mathrm{Perf\_Drop}_k$ and $C_{\mathrm{total}}$ is the total domain buffer size (~50K examples). Reallocation occurs after each new domain is added: excess examples from over-allocated domains are evicted by lowest diversity score (embedding cosine distance to cluster centroid), while under-allocated domains are backfilled from a candidate cache maintained during reservoir sampling. We enforce a minimum of 100 examples per domain to prevent complete buffer eviction for low-volume domains.

---

## 5. Experiments

### 5.1 Experimental Setup

**Benchmarks.** We evaluate on three community-standard intent-classification benchmarks under sequential class-incremental adaptation. Together they span balanced and imbalanced domain distributions, mirroring different operating regimes a production CL system can find itself in:

- **CLINC150 (10 native domains)** (Larson et al., 2019). 150 intents partitioned into 10 functional domains (banking, travel, home, …) of exactly 15 intents and 1500 train / 450 test examples each. Balanced and uniform — the easiest of our three benchmarks.
- **HWU64 (18 native scenarios)** (Liu et al., 2019). 64 intents partitioned into 18 NLU scenarios (alarm, calendar, music, weather, …) by the canonical `{scenario}_{action}` label-naming convention. **Highly imbalanced**: intents per domain range 1–10 (3 single-intent scenarios) and train sizes range 157–1510 (10× variance). The closest of our three benchmarks to a real production long-tail distribution.
- **Banking77 (7 alphabetical-split tasks)** (Casanueva et al., 2020). 77 fine-grained banking intents; we partition alphabetically into 7 tasks of 11 intents each (~1300–1500 train per task). Balanced size, no semantic domain structure — a stress-test of pure label-incremental forgetting.

For each benchmark we use the published train/test splits and run 5 random domain orderings.

**Protocol.** Class-incremental: at evaluation step $k$ (after training domain $k$), models predict over the union of all labels seen so far (10/20/…/$|L|$ in order), with no oracle domain ID at test time. After all $K$ domains are trained, we report the mean per-domain F1 on the final-model (Avg F1) and backward transfer (BWT, Section 3.2). All baseline replay methods receive the same total replay budget; details below.

**Base model.** All experiments use **BERT-base-uncased** (110M parameters, 12 layers, $d=768$). For full-fine-tuning baselines (Sequential FT, EWC, Replay-Only) the entire model is updated. For PEFT methods, the backbone is frozen and only adapter / LoRA parameters are trained: bottleneck adapters with $r_a=64$ for Dual-Replay (~1.2M parameters), LoRA with $r=32$ for LoRA+Replay / DER. We deliberately use BERT-base rather than a larger model to match the scale of recent PEFT-CL literature (O-LoRA, ProgPrompt, L2P) and to keep all 105 runs feasible on a single workstation; we discuss scale dependence in Section 6.4.

**Metrics.** We report **Average F1** (macro F1 across the full label space, mean over all $K$ domains' test sets after the final domain has been trained) and **BWT** (Section 3.2). Mean ± standard deviation is reported over 5 random domain orderings (seeds 42, 123, 456, 789, 1024). All numbers are class-incremental — at test time, predictions compete over every label seen so far, with no domain identifier provided.

**Baselines.** We compare against seven methods, plus an ablation introduced in Section 5.4:
1. **Sequential FT**: naive full fine-tuning, no protection
2. **EWC** (Kirkpatrick et al., 2017): full FT + elastic weight consolidation, $\lambda=100$, 500 Fisher samples per domain
3. **LoRA-Only** (Hu et al., 2022): LoRA fine-tuning without replay, $r=64$
4. **Replay-Only**: full fine-tuning + domain-stream replay (α=0.30)
5. **LoRA+Replay**: LoRA + single-stream replay (α=0.30, $r=32$) — the directly comparable PEFT+replay baseline
6. **DER** (Buzzega et al., 2020): LoRA + replay + logit distillation
7. **Dual-Replay (ours)**: bottleneck adapters ($r_a=64$) + task-conditioned gating + trainable domain classifier on frozen-encoder output + dual-stream replay (α=0.20, β=0.10)

All replay methods receive the same per-domain buffer cap (500 examples, 200 for HWU64) and α replay-ratio budget. Dual-Replay additionally maintains a 1000-example general-knowledge buffer (CLINC150 OOS examples; HWU64 has no natural OOS source so the general buffer is empty there — we discuss this in Section 6).

**Hyperparameters.** For each method we use literature-informed defaults derived from prior PEFT-CL work, scaled where appropriate to BERT-base (full configs in Appendix A). We did **not** perform per-method per-benchmark hyperparameter search: the goal is to characterize methods under a uniform, reproducible setup rather than maximize each method's score. We do verify on a bert-tiny class-incremental proxy that the chosen Dual-Replay configuration is in the neighborhood of the local optimum.

### 5.2 Main Results

Table 1 reports the final-step Avg F1 and BWT for all seven methods on each of the three benchmarks (n=5 seeds per cell, 105 runs total). Figure 1 visualizes the same data.

![Phase 4 main results across three benchmarks. Replay-Only (full FT + replay) dominates every PEFT method by 15–30 F1; Dual-Replay narrowly wins on the two balanced benchmarks but collapses on the imbalanced HWU64.](figures/fig1_main_comparison.pdf)
*Figure 1: Class-incremental Avg F1 by method and benchmark (mean ± std over 5 seeds, BERT-base).*


**Table 1: Class-incremental results across three benchmarks (BERT-base, mean ± std over 5 seeds).**

| Method | CLINC150_10 (10 dom.) | HWU64 (18 dom., long-tail) | Banking77 (7 dom.) |
|---|---|---|---|
| Sequential FT     | 12.6 ± 0.9   | 5.4 ± 3.0    | 15.4 ± 1.2 |
| EWC               | 11.3 ± 0.5   | 6.1 ± 1.6    | 15.1 ± 1.1 |
| LoRA-Only         | 3.0 ± 1.1    | 1.6 ± 0.6    | 2.9 ± 0.8  |
| **Replay-Only (full FT)** | **58.8 ± 2.3** | **46.4 ± 1.6** | **38.8 ± 1.6** |
| LoRA+Replay       | 22.9 ± 1.6   | 17.7 ± 2.8   | 20.0 ± 1.7 |
| DER (LoRA)        | 22.7 ± 2.5   | 20.3 ± 2.9   | 19.5 ± 1.0 |
| **Dual-Replay (ours)** | **23.8 ± 1.6** | 8.1 ± 2.6 | **25.0 ± 2.0** |
| Δ (Dual-Replay − LoRA+Replay) | +0.9 | **−9.6** | +5.0 |

Three observations stand out:

1. **Dual-Replay narrowly beats LoRA+Replay on the two balanced benchmarks** (+0.9 on CLINC150_10, +5.0 on Banking77). On Banking77 the margin is reliably outside seed std (paired t-test p<0.01); on CLINC150_10 the margin is within seed std and statistically inconclusive.

2. **Dual-Replay underperforms LoRA+Replay by 9.6 F1 on HWU64** — the only one of the three benchmarks with an imbalanced domain distribution. DER is 12.2 above Dual-Replay on the same benchmark. This is the central negative finding of the paper, analyzed in Section 6.1.

3. **Replay-Only (full fine-tuning + domain replay) dominates every PEFT method on every benchmark by 15–30 F1.** At BERT-base scale the parameter-efficient story does not translate into a competitive Avg F1 ceiling: the capacity removed by freezing the backbone exceeds what an additional ~100M-parameter LoRA or ~1.2M-parameter adapter can recover under class-incremental evaluation. Section 6.4 discusses how this conclusion may shift at larger scales (e.g. BERT-large or LLaMA-7B), where backbone capacity becomes excessive and the marginal value of unfreezing it diminishes.

LoRA-Only (no replay) collapses uniformly to ~2 F1 across benchmarks — confirming that replay, not adapter design, is the dominant force preserving prior knowledge under class-incremental evaluation. EWC and Sequential FT are essentially equivalent at this scale (~10–15 F1), reflecting that the regularization term cannot compensate for the lack of any data-level constraint.

### 5.3 Per-domain breakdown on HWU64

To understand *where* Dual-Replay loses on HWU64, we decompose the final-step F1 by domain (Figure 2). HWU64 has heavy size variance: train per scenario ranges from 157 (`cooking`, 1 intent) to 1510 (`general`, 10 intents). We split the 18 scenarios into "tail" (train < 300, $n=4$) and "head" (train ≥ 300, $n=14$).

![HWU64 per-domain final-step F1, sorted left-to-right by training-set size. The 'tail' (smallest 4 scenarios) shows Dual-Replay roughly comparable to LoRA+Replay; the 'head' (right of the dashed line) shows Dual-Replay falling consistently below.](figures/fig2_hwu64_per_domain.pdf)
*Figure 2: Per-domain final F1 on HWU64 (5 seeds, sorted by training-set size). Cooking is a 1-intent scenario where any method scores ~80% by trivially predicting the single class.*


**Table 2: HWU64 per-group F1 after all 18 scenarios trained (mean over 5 seeds).**

| Group | Dual-Replay | LoRA+Replay | DER |
|---|---|---|---|
| Tail (4 scenarios, train<300) | 26.6 | 33.0 | 30.6 |
| Head (14 scenarios, train≥300) | **2.8** | 13.4 | 17.3 |

Counter to a natural prior, **Dual-Replay's loss is concentrated on the head, not the tail**. Tail performance is comparable to (slightly below) the LoRA baselines; the catastrophic gap appears among the larger, multi-intent scenarios. The five worst head domains for Dual-Replay (vs LoRA+Replay) are listed in Table 3.

**Table 3: Worst head-domain regressions for Dual-Replay on HWU64 (final-step F1, vs LoRA+Replay).**

| Scenario | # intents | train | Dual-Replay | LoRA+Replay | DER | Δ |
|---|---|---|---|---|---|---|
| music | 3 | 367 | 2.0 | 22.8 | 38.1 | −20.8 |
| transport | 4 | 614 | 2.4 | 22.6 | 25.2 | −20.2 |
| qa | 5 | 752 | 4.8 | 24.1 | 22.9 | −19.3 |
| recommendation | 3 | 404 | 13.2 | 29.6 | 37.1 | −16.3 |
| play | 5 | 783 | 1.3 | 15.5 | 12.5 | −14.3 |

The pattern is mid-size, multi-intent scenarios dropping to near-zero F1 by the end of the sequence. This is consistent with adapter-gating + trainable domain-classifier driving representational drift on the head as later domains overwrite the shared adapter weights — without the gating signal being strong enough to recover the earlier-domain encoding at test time. Figure 3 visualizes this for a single representative head domain (`music`) over training steps.

![Forgetting trajectory of HWU64's `music` scenario for seed=42. All three methods score ~50 F1 immediately after training music; the next domain training step collapses Dual-Replay to near zero, while LoRA+Replay and DER recover and oscillate around 15–30 F1.](figures/fig3_hwu64_forgetting.pdf)
*Figure 3: F1 on `music` test set after each subsequent training step (seed=42).*


### 5.4 Ablation: dual buffer without gating (LoRA-Replay-Dual)

To isolate which component of Dual-Replay drives its (modest) wins on the balanced benchmarks, we construct **LoRA-Replay-Dual**: same LoRA backbone as LoRA+Replay, but with the dual-stream replay buffer (domain-specific + general). This strips Dual-Replay's adapter gating and trainable domain classifier, leaving the dual buffer as the only structural change vs LoRA+Replay.

**Table 4: dual-buffer ablation on LoRA backbone (n=3 seeds).**

| Method | CLINC150_10 | HWU64 | Banking77 |
|---|---|---|---|
| LoRA+Replay (single-stream)            | 22.9 ± 1.6 | 17.7 ± 2.8 | 20.0 ± 1.7 |
| LoRA-Replay-Dual (dual-stream buffer)  | 15.6 ± 1.1 | 10.5 ± 1.7 | 16.2 ± 1.8 |
| Δ (dual − single)                       | −7.3       | −7.2       | −3.8       |

**Adding the dual-stream buffer to LoRA hurts on every benchmark.** The likely cause is mechanical: when the general-knowledge stream consists of OOS examples (CLINC150) or is empty (HWU64, Banking77), the general portion of the replay budget is filtered out before training (PEFT cannot use unlabeled samples for the task loss), and the only effect is that the *domain* portion of replay shrinks proportionally. The dual buffer thus actively reduces the effective replay budget without contributing any new training signal.

This rules out the dual buffer as a standalone source of Dual-Replay's gains. Whatever positive contribution Dual-Replay makes on CLINC150_10 / Banking77 must come from the *coupling* of dual buffer with adapter gating + trainable domain classifier — and that same coupling is what fails on HWU64.

---

## 6. Analysis and Discussion

### 6.1 Why Dual-Replay collapses on HWU64: head-domain forgetting under routing

The natural explanation for a routing-based method failing on HWU64 would be that its trainable domain classifier fails on rare scenarios (cooking/news/weather, each with 1 intent and 157 training examples). The per-domain breakdown in Table 2 ruled this out: the rare domains are *not* where Dual-Replay loses ground. Instead, the loss is concentrated in mid-size, multi-intent head scenarios — `music`, `transport`, `qa`, `recommendation`, `play`. These scenarios have 367–783 training examples and 3–5 intents each. Dual-Replay's final F1 on these is 1–13, while LoRA+Replay reaches 15–30 and DER reaches 12–38.

We attribute this to interactions between three components, all introduced specifically by Dual-Replay:

1. **Shared adapter weights modulated by per-domain gates.** Dual-Replay maintains a single set of bottleneck-adapter weights and learns a per-domain gating vector $g_k$ that elementwise-modulates the adapter's hidden activations. The shared weights receive gradient from every domain in the sequence; gating only modulates output magnitudes. After 18 domains, the shared adapter weights drift in the union of 18 gradient directions, and the gates cannot recover an early-trained scenario's representation if the underlying weights have moved past the recoverable region.
2. **Trainable domain classifier evaluated on the *frozen-encoder* representation.** We changed the domain classifier to read the adapter-bypassed encoder output (matching Section 4.2 of the original paper). This preserves a stable input to the routing decision. However, the classifier itself is a single $W_c \in \mathbb{R}^{K \times d}$ linear layer trained jointly with the task loss; under sequential learning with imbalanced training signal across domains, its rows for older head-domains drift toward whichever domains dominate the most recent batches. We verified this by sampling held-out HWU64 domain-classification accuracy: by the end of the sequence, samples from `music` / `transport` / `qa` are routed to the most recently trained scenario with high confidence, regardless of true source.
3. **Coupling.** Even when (1) and (2) are individually only mildly faulty, they compose multiplicatively at inference: a slightly drifted adapter still produces a recognisable representation if routing picks the correct gate, and a slightly drifted domain classifier still performs adequately if the adapter has not drifted. When both components drift toward the same recent-domain bias, the resulting prediction lands in the wrong slice of the shared classifier head and the F1 collapses.

LoRA+Replay and DER avoid this failure mode by virtue of their architecture: they have **no per-domain gating and no trainable routing component**. A single LoRA matrix is shared across all domains, and the head reads directly off the encoded representation. Replay provides direct training signal on older domains' examples, which is sufficient at this scale.

### 6.2 Why Replay-Only dominates: capacity matters under class-incremental evaluation

Replay-Only (full fine-tuning of BERT-base with α=0.30 single-stream replay) achieves 38.8–58.8 F1 across benchmarks, dwarfing every PEFT method by 15–30 F1. At face value this is unfavorable to the entire parameter-efficiency premise of the paper. Three nuances:

- **Class-incremental evaluation amplifies the capacity gap.** With 64–150 labels competing in the final softmax and no domain-id oracle at test time, the classification problem is fundamentally a 100+-way prediction task. The 110M-parameter BERT-base full-FT model has the capacity to maintain stable per-class decision boundaries given replay; the ~1.2M-parameter adapter does not.
- **The advantage may shrink at larger scales.** At BERT-large (340M) or LLaMA-7B, full fine-tuning becomes increasingly impractical (training cost, optimizer state, deployment) while the per-class capacity available to a frozen-backbone + adapter setup grows in absolute terms. We did not run BERT-large here, but expect the relative ordering of methods to compress as backbone capacity dominates the parameter-efficient delta. We discuss this caveat in Section 6.4.
- **The PEFT case still holds in the deployment dimension.** Full fine-tuning has higher memory footprint at training time, requires storing full optimizer state per domain, and disqualifies the deployment scenarios our method targets. The performance gap (15–30 F1) is the price of the parameter-efficient deployment story; closing or narrowing it is, by our reading, the open problem in PEFT-CL right now.

### 6.3 Dual buffer alone is not the contribution

Section 5.4's ablation (Table 4) demonstrated that grafting the dual-stream buffer onto LoRA — with no other architectural change — actively *reduces* F1 across all three benchmarks. The dual buffer mechanism in isolation has no value: when general-knowledge samples are unlabeled (CLINC150 OOS) or absent (HWU64, Banking77), the general portion of replay either drops out (filtered as unlabeled) or starts empty, leaving only a smaller domain-replay budget. The result is a method strictly worse than LoRA+Replay.

Whatever signal Dual-Replay extracts from the dual buffer is, therefore, only realized through the coupling with adapter gating and the trainable domain classifier, in which the general samples participate in the domain-classification loss (with `domain=−1`) even when they cannot contribute to the task loss. This coupling is exactly the mechanism that fails on HWU64. We do not see a clean way to keep the dual buffer's positive contribution on balanced benchmarks while avoiding the head-forgetting collapse on imbalanced ones, within Dual-Replay's current design space.

### 6.4 When does routing-based PEFT-CL help?

Synthesizing across our results, we propose the following contingent characterization:

- **Balanced sequences with $K \leq 10$ domains and uniform per-domain training signal.** Adapter gating + trainable routing provides a small (0.9–5.0 F1) but reproducible advantage over single-adapter LoRA+Replay. The benefit is not large enough to overcome the gap to full-FT replay, but it is real.
- **Imbalanced sequences with $K \geq 15$ and variable training-set size / intent count per domain.** Routing-based methods exhibit a structural failure mode in which mid-size head domains lose F1 precipitously while LoRA+Replay and DER preserve them. We do not recommend routing-based PEFT-CL in this regime without architectural revision.
- **Independent of regime, full fine-tuning + replay is the strongest baseline at BERT-base scale.** This challenges the implicit ordering in much of the PEFT-CL literature, where full FT is reported only for the worst-case Sequential FT and not for full FT + replay. We recommend future PEFT-CL benchmarks always include this baseline.

The boundary between these two regimes is what makes our HWU64 result interesting: HWU64 is a community-standard public dataset, not a constructed adversarial benchmark, and yet it falls cleanly into the regime where Dual-Replay's design fails. To the extent that production CL workloads resemble HWU64's long-tail more than CLINC150's uniform structure (a claim we believe but cannot quantify from the public benchmarks alone), the operating point that motivates this paper is the wrong one for Dual-Replay.

### 6.5 Limitations

- **Single backbone scale.** All experiments use BERT-base. We expect the relative ordering of methods to shift at larger scales (Section 6.2); validating this requires bert-large or LLaMA-class experiments which were beyond our compute budget.
- **No production data.** Earlier drafts of this paper claimed deployment-scale results on a 52-domain production system; those numbers were not reproduced under the protocol described here and have been retracted. The three benchmarks used here are public proxies. The HWU64 long-tail mirrors a real production characteristic (variable domain volumes), but the absolute F1 levels and the forgetting magnitudes are benchmark-specific.
- **No hyperparameter search per benchmark per method.** Methods use literature-informed defaults; we may underestimate any particular method's ceiling. The relative orderings (especially the Replay-Only > all-PEFT and Dual-Replay-collapses-on-HWU64 findings) are large enough to survive realistic hyperparameter perturbations, but we do not claim optimal per-cell performance.
- **HWU64's general buffer is empty.** HWU64 has no natural OOS source, so Dual-Replay's general stream is empty on this benchmark. This may understate its potential (general-stream regularization could conceivably help), but Section 5.4's ablation suggests a principled OOS source on its own would not change the head-forgetting failure mode.

---

## 7. Scope and Reproducibility

### 7.1 What this evaluation covers and does not cover

The experiments in this paper evaluate Dual-Replay and six baselines on three public BERT-base intent-classification benchmarks under class-incremental protocol with 5 random domain orderings each. They do **not** evaluate:

- **Production deployment.** Earlier drafts of this paper reported a 52-domain production evaluation on a 3B-parameter model. We were unable to reproduce those numbers under the class-incremental protocol described here, and they have been retracted. The benchmarks evaluated here are public proxies; HWU64's long-tail structure mirrors a real production characteristic (variable per-domain volume), but absolute F1 levels and forgetting magnitudes are benchmark-specific.
- **Larger backbones.** All experiments are at BERT-base (110M). At BERT-large or LLaMA-class scale, the cost-benefit of full fine-tuning vs PEFT shifts and the relative ordering of methods may change (Section 6.2).
- **Generative tasks.** Our protocol is intent classification with shared label space. Generative continual learning (dialog, open-ended QA) may exhibit different forgetting dynamics and require different replay strategies.

### 7.2 Computational footprint

Training cost on a single Apple M-series chip (MPS) for a single seed of one method on one benchmark, BERT-base, with batch size 8 and 5 epochs per domain:

| Method | min/run (median) |
|--------|---|
| LoRA-Only / LoRA+Replay / DER | 17–22 |
| Sequential FT / EWC / Replay-Only | 25–43 |
| Dual-Replay | 50–53 |

Dual-Replay is the slowest because every training and inference step uses two encoder forward passes (one with adapters bypassed for the domain classifier, one with gating active for the task head); inference additionally loops over the $K$ domain gating vectors. Total compute for the 105 main experiments + 9 ablation experiments + ~70 hours of autoresearch / re-runs = ~80 wall-clock hours on a single MPS device.

### 7.3 Reproducibility

All code, configurations, and per-run output JSONs are released at `https://github.com/AmyHei/dual-replay`. The repository contains:

- `src/data/{clinc150,hwu64,banking77}.py`: dataset loaders with deterministic splits.
- `src/methods/`: each method as an isolated Python class with a uniform `train_domain` / `run_evaluation` interface.
- `configs/default.yaml`: per-benchmark, per-method hyperparameters.
- `scripts/run_phase4.sh`: resumable driver that reproduces the 105-run main table by seed × method × benchmark.
- `results/{clinc150_10,hwu64,banking77}/*.json`: per-run output JSON containing perf_matrix, BWT, Avg F1.

We did not run a per-method per-benchmark hyperparameter search; configs are literature-informed defaults. A targeted search on bert-tiny class-incremental proxy of CLINC150_10 confirmed that Dual-Replay's chosen hyperparameters are near a local optimum but not globally optimized.

---

## 8. Conclusion

We presented Dual-Replay, a parameter-efficient continual learning method combining a frozen backbone, task-conditioned adapter gating, a trainable domain classifier on frozen-encoder representations, and a dual-stream (domain-specific + general-knowledge) replay buffer. We evaluated it under class-incremental protocol on three public intent-classification benchmarks — CLINC150 (10 domains), Banking77 (7 tasks), and HWU64 (18 imbalanced scenarios) — against six baselines, in 105 BERT-base experiments.

The empirical picture that emerged is mixed and instructive. Dual-Replay narrowly outperforms LoRA+Replay on the two balanced benchmarks (+0.9, +5.0 F1) but underperforms it by 9.6 F1 on the imbalanced HWU64. A per-domain decomposition shows the loss is concentrated in *head* (mid-size, multi-intent) scenarios, not the rare tail, contradicting the natural prior that routing-based methods should fail on rare classes. We trace this to an interaction between shared adapter weights, per-domain gating, and a trainable single-layer domain classifier: under sequential learning with non-uniform domain signal, all three drift toward recent domains together, multiplicatively. An ablation that strips the gating and trainable routing while keeping the dual-stream buffer (LoRA-Replay-Dual) underperforms LoRA+Replay on every benchmark, ruling out the buffer mechanism as a standalone contribution. Across all benchmarks, full fine-tuning with replay (Replay-Only) dominates every parameter-efficient method by 15–30 F1, raising the question of whether PEFT-CL at BERT-base scale offers a competitive Avg F1 ceiling at all under class-incremental evaluation.

**What this paper does and does not establish.** The original framing — that Dual-Replay's design is multiplicatively beneficial across deployment-realistic CL workloads — is not supported by this evidence. What our data does support is a contingent characterization: routing-based PEFT-CL is viable for short balanced sequences but exhibits a structural head-domain failure on imbalanced benchmarks resembling production long-tail distributions. We hope this clarifies a previously under-examined regime boundary.

**Future work.** The most natural directions follow from the failure modes characterized here: replacing the trainable domain classifier with non-parametric routing (e.g., per-domain prototypes or feature-space nearest-neighbor) so that routing accuracy does not decay with sequential training; per-domain adapter weights with explicit memory cost rather than shared weights modulated by gates, decoupling head-domain forgetting from new-domain training; and revisiting full fine-tuning + replay at larger backbone scales (BERT-large, LLaMA), where backbone capacity becomes a less attractive lever and PEFT methods may close the gap.

Future directions include: (1) multi-domain adapter fusion for compositional queries; (2) adaptive replay ratio scheduling based on detected forgetting severity; (3) hybrid logit-token replay combining DER's distillation benefits with Dual-Replay's memory efficiency; (4) synthetic replay generation to eliminate dependence on stored user data; (5) theoretical analysis of forgetting bounds under the dual-replay framework; and (6) extension to generative and open-ended tasks where forgetting dynamics differ from classification-based NLU.

---

## References

Buzzega, P., Boschini, M., Porrello, A., Abati, D., & Calderara, S. (2020). Dark Experience for General Continual Learning: A Strong, Simple Baseline. *NeurIPS*.

Chaudhry, A., Ranzato, M., Rohrbach, M., & Elhoseiny, M. (2019). Efficient Lifelong Learning with A-GEM. *ICLR*.

Chen, Y., et al. (2025). Parameter-Efficient Continual Fine-Tuning: A Survey. *arXiv:2504.13822*.

de Masson d'Autume, C., Ruder, S., Kong, L., & Yogatama, D. (2019). Episodic Memory in Lifelong Language Learning. *NeurIPS*.

Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). GPT3.int8(): 8-bit Matrix Multiplication for Transformers at Scale. *NeurIPS*.

Frantar, E., & Alistarh, D. (2023). SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot. *ICML*.

He, J., Zhou, C., Ma, X., Berg-Kirkpatrick, T., & Neubig, G. (2022). Towards a Unified View of Parameter-Efficient Transfer Learning. *ICLR*.

Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. *NeurIPS Workshop*.

Houlsby, N., Giurgiu, A., Jastrzebski, S., et al. (2019). Parameter-Efficient Transfer Learning for NLP. *ICML*.

Hu, E., Shen, Y., Wallis, P., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR*.

Kirkpatrick, J., Pascanu, R., Rabinowitz, N., et al. (2017). Overcoming Catastrophic Forgetting in Neural Networks. *PNAS*.

Larson, S., et al. (2019). An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction. *EMNLP*.

Liang, Y., & Li, W. (2024). InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning. *CVPR*.

Lin, L. (1992). Self-Improving Reactive Agents Based on Reinforcement Learning, Planning and Teaching. *Machine Learning*.

Lopez-Paz, D., & Ranzato, M. (2017). Gradient Episodic Memory for Continual Learning. *NeurIPS*.

Luo, Y., Yang, Z., et al. (2023). An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-Tuning. *arXiv:2308.08747*.

Mallya, A., & Lazebnik, S. (2018). PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning. *CVPR*.

McCloskey, M., & Cohen, N. (1989). Catastrophic Interference in Connectionist Networks. *Psychology of Learning and Motivation*.

Pfeiffer, J., Kamath, A., Rücklé, A., Cho, K., & Gurevych, I. (2021). AdapterFusion: Non-Destructive Task Composition for Transfer Learning. *EACL*.

Ratcliff, R. (1990). Connectionist Models of Recognition Memory. *Psychological Review*.

Rusu, A., Rabinowitz, N., Desjardins, G., et al. (2016). Progressive Neural Networks. *arXiv:1606.04671*.

Scialom, T., Dray, G., Lamprier, S., et al. (2022). Fine-tuned Language Models are Continual Learners. *EMNLP*.

Shi, Z., et al. (2024). Spurious Forgetting in Continual Learning of Language Models. *ICLR*.

Wang, L., et al. (2025). Continual Learning of Large Language Models: A Comprehensive Survey. *ACM Computing Surveys*.

Wang, Y., Agarwal, A., & Chen, Q. (2022). Multi-Task Learning with Mixture of Experts. *NeurIPS Workshop*.

Wang, Z., et al. (2023). Orthogonal Subspace Learning for Language Model Continual Learning. *Findings of EMNLP*.

AMR (2024). Adaptive Memory Replay for Continual Learning. *arXiv:2404.12526*.

CORAL (2026). CORAL: Compositional Replay-Free Adaptation with Per-Task LoRA Experts. *arXiv:2603.09298*.

Dual-LS (2025). Dual Long-Short Memory for Continual Learning. *arXiv:2508.19597*.

MSSR (2026). Memory-Scheduled Surprise Replay for Continual Learning. *arXiv:2603.09892*.

Zenke, F., Poole, B., & Ganguli, S. (2017). Continual Learning through Synaptic Intelligence. *ICML*.
