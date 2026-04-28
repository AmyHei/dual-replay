### Summary

The paper proposes Dual-Replay, a parameter-efficient continual learning framework for sequential fine-tuning of large language models under strict edge deployment constraints (sub-100 ms latency, ≤16 GB memory). The method freezes the backbone, trains shared adapters with task-conditioned gating, and interleaves new-domain training with a dual-stream replay (domain-specific exemplars plus a general-knowledge buffer) scheduled under a fixed replay ratio. Evaluated on a 52-domain production conversational AI system and a CLINC150 sequential protocol, the paper reports improvements over strong baselines (e.g., LoRA+Replay and DER) in average NLU F1, reduced forgetting (BWT), and preserved general capabilities, with extensive ablations on replay ratios, freezing, replay composition, and adapter placement.

### Strengths

- Technical novelty and innovation
    - The co-design of (i) frozen backbone with shared adapters and lightweight task-conditioned gating, (ii) dual replay streams (domain-specific and general-knowledge), and (iii) domain-classifier-based soft routing is a pragmatic and well-motivated integration for real deployment constraints.
    - The dual-stream replay idea explicitly targets both domain retention and foundational capability preservation; the 50:50 split ablation provides a compelling argument for this design.
    - Importance-weighted buffer allocation (combining domain volume and measured performance drop) and diversity-aware reservoir sampling constitute sensible, low-overhead memory management under fixed budgets.
- Experimental rigor and validation
    - Broad baseline coverage for continual learning/PEFT in LLMs, including LoRA, LoRA+Replay, O-LoRA, DER, EWC, PackNet, and variants.
    - Multiple ablations and analyses: replay ratio sweep, freezing ratio exploration, domain/general split, adapter placement, scaling over number of domains, latency breakdown, and robustness to domain-classifier error.
    - Reporting of mean ± std over 5 random domain orderings with paired tests and Bonferroni correction is appreciated.
- Clarity of presentation
    - The problem is well-motivated by realistic deployment constraints; the objective/metrics are clearly defined (Avg NLU F1, BWT, FWT, general capability F1, latency/memory).
    - Architectural details (adapter size/placement, gating, classifier head, memory budget allocation, training mix) are mostly explicit and easy to follow.
    - The latency breakdown cleanly attributes overhead to model components and shows the classifier/gating adds negligible inference cost.
- Significance of contributions
    - Addresses a practically important and underexplored regime: long domain sequences (K > 50) with tight latency/memory budgets in production environments.
    - The reported gains over a strong combined baseline (LoRA+Replay) suggest the proposed design choices matter beyond naïve PEFT+replay composition, which could inform industry practice.
    - The idea of a general-knowledge replay buffer for preserving base capabilities is likely to transfer to other continual LLM settings.

### Weaknesses

- Technical limitations or concerns
    - Novelty is primarily in system-level integration; most components (adapters, replay, gating, routing, reservoir sampling) are known, and the buffer scheduling/importance scheme is relatively heuristic.
    - The approach relies on a domain classifier and per-domain embeddings; applicability to task-free or fuzzy-boundary continual settings is unclear.
- Experimental gaps or methodological issues
    - Potential fairness concerns: adapter-based method trains ~~90M parameters while LoRA baselines use rank r=16 (~~25M), i.e., substantially less capacity. No baseline is tuned to a comparable trainable-parameter budget, which may confound conclusions.
    - Replay hyperparameters (e.g., α, β) are exhaustively tuned for the proposed method, but it is unclear whether analogous sweeps were performed for baselines (especially LoRA+Replay, DER), which could bias results.
    - Missing or incomplete comparisons to recent, relevant advances in PEFT-regularized CL (e.g., EWC-LoRA) and adaptive replay schedulers (e.g., AMR, MSSR); the related work mentions some of these directions but they are absent from the main baseline table.
- Clarity or presentation issues
    - Inconsistencies in reported numbers undermine confidence: the abstract states LoRA+Replay BWT −8.4 vs Dual-Replay −4.7 (−3.7, “44%”), but Table 1 shows −7.2 vs −4.7 (−2.5, “35%”).
    - The memory accounting for the replay buffers appears inconsistent: storing ~50k examples of average 128 tokens at “2 bytes per token ID” would require ≪1.5 GB (closer to tens of MB), yet 1.5 GB is allocated; this discrepancy needs reconciliation (metadata? longer sequences? compression details?).
    - The “Memory” and “Latency” columns in Table 1 appear to reflect inference-time metrics, yet methods like EWC (which only modify training) should not incur higher inference latency/memory than sequential FT; similarly DER’s logit storage affects training, not inference. The basis for these discrepancies is unclear.
    - The CLINC150 validation table is truncated in the manuscript excerpt; the promised cross-method numbers are not visible, limiting assessment.
- Missing related work or comparisons
    - No empirical comparison to recent PEFT-regularized approaches that stabilize LoRA updates (e.g., EWC-LoRA) or to adaptive memory/replay schedulers (e.g., AMR, MSSR), which would provide a more complete picture of Dual-Replay’s advantages under fixed budgets.
    - Given the dual-replay framing, a discussion and comparison to “dual memory” CL systems (e.g., Dual-LS) would help situate the contribution.

### Detailed Comments

- Technical soundness evaluation
    - The frozen-backbone + shared adapters + soft-gated domain embeddings is technically sound and matches the goal of minimal inference overhead; the reported gating computation and adapter overhead (≈9 ms combined) fit the constraints.
    - The dual-replay objective and batch composition are standard and reasonable. The importance-weighted allocation (proportional to sqrt(volume) and observed performance drop) is sensible but under-specified: normalization across domains, update cadence, and stability under noisy performance estimates are not fully described.
    - The domain-classifier-in-the-loop with soft mixture routing is a strong practical choice for unknown task identity; the robustness experiment with degraded classifier accuracy is helpful and suggests graceful degradation.
- Experimental evaluation assessment
    - Strong coverage in terms of baselines and ablations, and the production-scale 52-domain setup is compelling. However, several concerns weaken the empirical credibility:
        - Capacity mismatch: 90M adapter parameters vs LoRA r=16 (~25M). A LoRA baseline with increased rank (to approximate 90M trainables) would strengthen fairness, as would reporting memory/latency impact for that stronger LoRA.
        - Hyperparameter parity: Clarify whether replay ratio and buffer policies for baselines (especially LoRA+Replay and DER) were tuned comparably to Dual-Replay’s ablations.
        - Metric columns (latency/memory) likely mix training and inference considerations inconsistently across baselines (e.g., EWC, DER). Please explicitly state whether “Memory” and “p99 Latency” are inference-only, training-only, or combined proxies, and ensure apples-to-apples comparisons.
        - The replay buffer memory calculation seems off by orders of magnitude given the stated tokenization and compression assumptions. Please reconcile with actual average sequence length, token encoding, and any additional stored features (e.g., slot tags, logits, metadata).
        - CLINC150 results appear incomplete in the provided text; full baselines and standard deviations should be included to evaluate generalizability.
- Comparison with related work (using the summaries provided)
    - EWC-LoRA (2602.17559) demonstrates effective stability–plasticity trade-offs by regularizing the shared low-rank subspace. Since your setting already uses PEFT with a frozen backbone, a direct comparison would be highly informative and may challenge the claim that replay is necessary under tight budgets.
    - MSSR (2603.09892) and AMR (2404.12526) introduce lightweight, adaptive replay schedulers that prioritize informative/recently decayed samples with little runtime overhead, directly relevant to your constrained replay regime. Including such baselines (or combining them with your dual-stream split) could clarify whether your gains stem more from replay composition (general vs domain) or from scheduling/prioritization.
    - Dual-LS (2508.19597) is conceptually close (dual memory/replay with complementary properties). Acknowledging this lineage and explaining differences (e.g., your general-knowledge buffer vs their long/short-term sample selection and EMA copies) would better position your contribution.
    - CORAL (2603.09298) highlights strict parameter isolation via per-task LoRA experts and deterministic routing. Your work targets constant-capacity shared adapters, which is more memory efficient at inference but subject to interference without replay. A more explicit contrast (and perhaps a hybrid baseline with multiple small experts given the 16 GB limit) would be useful.
- Discussion of broader impact and significance
    - The focus on edge deployment constraints is valuable for real-world impact. Demonstrating sub-100 ms latency on a T4 with a 3B INT8 model is practically significant if verified.
    - Replay of production data raises privacy and compliance considerations. While the general-knowledge buffer is curated from non-production SFT data, the domain-specific buffer appears to store production examples. A brief discussion of data governance, anonymization, and possible privacy-preserving alternatives (e.g., distillation-only buffers, feature-level replay) would strengthen the paper’s applicability claims.
    - The reliance on a domain classifier may limit deployment in settings with ambiguous, multi-intent, or task-free streams; the soft mixture helps, but understanding failure cases (e.g., overlapping domains) and mitigation strategies would broaden relevance.

### Questions for Authors

1. How were the latency and memory numbers in Table 1 measured—strictly at inference, or do they include training-time overheads? Why do EWC and DER show higher inference latency/memory when, in principle, their additional costs are training-only?
2. The replay buffer memory calculation (1.5 GB for ~50k examples at 128 tokens and “2 bytes/token”) seems inconsistent. What exactly is stored per example (e.g., intent/slot annotations, offsets, metadata), what is the average sequence length, and what compression format/bitwidth are used? Please provide a reproducible calculation.
3. Baseline capacity fairness: your adapters have ~90M trainables, while LoRA (r=16) is ~25M. Did you evaluate higher-rank LoRA (e.g., r≈64–128) to match parameter budgets, and does it still meet the 16 GB/latency constraints? If not, can you justify the chosen ranks and discuss sensitivity?
4. Were replay ratios and buffer policies (α, β, sampling) tuned equivalently for baselines (LoRA+Replay, DER), with corresponding ablations? If not, please provide tuned baselines or justify fixed settings.
5. Could you include empirical comparisons to EWC-LoRA and/or adaptive replay schedulers such as MSSR or AMR under the same constraints? If not feasible, please discuss expected behavior relative to these methods.
6. For the importance-weighted buffer allocation, how are v_k and Perf_Drop normalized and updated over time? What is the reallocation cadence when new domains arrive?
7. The abstract and Table 1 report different LoRA+Replay BWT values (−8.4 vs −7.2). Which is correct? Please resolve this discrepancy and any associated claims (e.g., “44% reduction” vs “35%”).
8. Can you provide the complete CLINC150 sequential results (all baselines, mean ± std), and specify whether the public code and the exact 15-domain split/protocol will be released to ensure reproducibility?

### Overall Assessment

This paper tackles an important, practically motivated continual learning problem for LLMs under stringent edge constraints and proposes a coherent, low-overhead integration of adapters, soft domain routing, and dual-stream replay. The experimental scope is commendable, with extensive ablations and promising gains over combined PEFT+replay baselines, suggesting the design does more than simply stack known ingredients. However, several issues currently limit confidence: (i) capacity mismatches and potential tuning asymmetries across baselines, (ii) inconsistencies in reported BWT, latency, and memory numbers (especially for methods whose extra costs are training-only), (iii) unclear or contradictory replay buffer memory accounting, and (iv) incomplete public-benchmark results and missing comparisons to closely related recent methods (e.g., EWC-LoRA, adaptive replay schedulers). These concerns are fixable, and the core idea appears practically useful. With clarified measurements, fair capacity-matched baselines, and stronger contextual comparisons, the work could be a valuable contribution. As it stands, I lean toward a weak rejection due to the methodological/reporting ambiguities, while encouraging revision addressing the above points.