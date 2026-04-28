### Summary

The paper proposes Dual-Replay, a parameter-efficient continual learning framework for sequential LLM fine-tuning under strict deployment constraints (sub-100 ms latency, ≤16 GB memory). The approach combines selective parameter freezing with shared, task-conditioned adapters and a dual replay mechanism that interleaves domain-specific and general-knowledge samples under a fixed memory budget. Evaluated over 52 sequential domains in a production conversational AI system, the method reports a 10% relative NLU F1 improvement and a 23% reduction in catastrophic forgetting over the strongest single-mechanism baseline, while meeting the stated latency and memory constraints.

### Strengths

- Technical novelty and innovation
    - The co-design of parameter isolation (via selective freezing and shared adapters with task-conditioned gating) and dual replay (domain-specific plus general) under a hard memory budget is a thoughtful, deployment-oriented contribution.
    - The explicit treatment and empirical exploration of the interplay between freezing ratio and replay ratio provide practical insights seldom documented in continual LLM settings.
    - Importance-weighted buffer allocation that accounts for both query volume and measured forgetting is a sensible and pragmatic heuristics-driven policy.
- Experimental rigor and validation
    - The evaluation spans 52 domains in a real production environment and tracks latency and memory usage alongside continual learning metrics (F1, BWT, FWT), which is valuable for practitioners.
    - Ablation studies on replay ratio, freezing ratio, and contribution of dual vs single replay streams help isolate design choices and show useful trade-offs.
    - A scaling analysis over increasing domain count supports the claim that benefits accumulate with longer task sequences.
- Clarity of presentation
    - The paper is generally clearly written and well-structured; the motivation, problem formulation, and method are easy to follow at a high level.
    - The deployment constraints and memory budget breakdown are explicitly stated, helping to ground the method’s practicality.
- Significance of contributions
    - Addressing catastrophic forgetting for LLMs under edge-like constraints is important and of broad interest; the presented approach and empirical results could inform industry practice.
    - The paper surfaces failure modes and operational considerations (e.g., buffer staleness, ordering sensitivity), which are under-discussed in research papers but crucial in production.

### Weaknesses

- Technical limitations or concerns
    - Ambiguity in parameter-freezing vs. trainable proportions: the method is described as freezing 80% of parameters while training adapters, but tables imply 20% of base weights are also trainable (0.6B params), which conflicts with the memory budget table that lists only frozen base-model memory plus small adapter memory at inference. The precise configuration (which layers are unfrozen, quantization state, inference-time parameter set) needs clarification.
    - The task-conditioned gating mechanism requires a domain embedding at inference, yet the paper asserts unknown task identity at inference. How the correct domain embedding is obtained or mixed per query is unclear; the gating design, routing mechanism, and training objective for the gate are under-specified.
- Experimental gaps or methodological issues
    - A critical missing baseline is PEFT + replay (e.g., adapters or LoRA with replay) under the same memory/latency budget. Comparing only “LoRA-only” and “Replay-only” leaves the central claim of synergy insufficiently tested against the most relevant combined baseline.
    - Evaluation is reported solely on proprietary, production datasets. The lack of results on public continual-learning benchmarks for NLU/dialogue (e.g., CLINC150, MultiWOZ-based sequences, or established CL benchmarks adapted to LLMs) hinders reproducibility and independent verification.
    - Baseline coverage omits several recent parameter-efficient continual-learning methods specifically designed for LLMs (e.g., O-LoRA/orthogonal subspaces, InfLoRA, prompt-based continual learning). Some are cited but not compared experimentally.
    - Absence of statistical significance reporting and limited details on variability across random seeds/orderings beyond a brief note on 2.1 F1 variance.
- Clarity or presentation issues
    - Details on the general-knowledge replay buffer’s source and curation are sparse; how “representative pretraining distribution” examples are obtained at fine-tuning time is not explained.
    - The training objective and sampling split between domain vs general buffers are fixed at 50/50 in replay without analysis of alternative allocations or adaptive scheduling.
    - Metrics definitions (e.g., exact NLU tasks composing “General Capability F1”) and evaluation protocols (splits, number of runs) are insufficiently documented to allow replication.
- Missing related work or comparisons
    - Limited discussion of conditional/soft-gated adapters, hypernetwork-based conditioning, and adapter fusion/mixture approaches that share parameters across tasks while avoiding per-task growth.
    - Replay with distillation targets/logits (e.g., DER) is cited but not included as a baseline; given the tight memory budget, dark experience replay could be highly relevant.

### Detailed Comments

- Technical soundness evaluation
    - The high-level idea—reduce forgetting surface via freezing/PEFT and stabilize via replay—aligns with known continual learning intuitions and is technically sound. The dual replay streams (domain-specific, general) are a reasonable design to preserve both task-specific and foundational capabilities.
    - However, the core mechanism for task-conditioned gating is under-specified. If task identity is unknown, how is e_k obtained? Is there a learned domain router, a soft mixture over domain embeddings, or per-query retrieval to choose g_k? Without this, it is hard to assess the inference-time behavior, potential routing errors, and their impact on performance and latency.
    - The reported memory allocation is plausible for INT8 3B parameters, but the indicated trainable parameter counts at certain freezing ratios appear inconsistent with the stated inference memory. If 20% of base parameters are updated and remain unfrozen at inference, that storage should be reflected. Alternatively, if during training some are unfrozen but a final frozen composite is deployed, please clarify the process and memory implications.
- Experimental evaluation assessment
    - The ablations are useful: replay ratio sweeps demonstrate a practical Pareto frontier, freezing ratio analysis indicates the interplay between capacity and interference, and the “dual vs single stream” breakdown shows that both replays contribute differently.
    - The main missing element is a fair, controlled comparison against the closest baseline: adapters (or LoRA) + replay under the same constraints. This is necessary to substantiate that the proposed buffer management, gating, and specific architectural choices (e.g., module placement) are the source of gains rather than the generic combination of PEFT and ER.
    - The production deployment claims are valuable but not independently verifiable; including a public benchmark evaluation would materially strengthen the paper. Reporting confidence intervals/standard deviations, and more detailed sensitivity to domain orderings beyond a single number, would improve robustness claims.
    - The latency and memory results are helpful; still, the hardware environment, batching constraints, and sequence length distributions should be provided to contextualize the 82 ms p99 claim for a 3B model.
- Comparison with related work (using the summaries provided)
    - The paper correctly situates itself relative to EWC/SI (regularization), PackNet/Progressive (architecture isolation), and ER/GEM families. It also references parameter-efficient continual learning (InfLoRA, O-LoRA) and recent replay prioritization (SuRe). The novelty lies in a pragmatic co-design under strict deployment constraints and the use of a dual replay stream plus shared, gated adapters to avoid linear parameter growth with domains.
    - Nevertheless, comparisons to O-LoRA/orthogonal subspaces, InfLoRA, or prompt-based continual learning with replay would be informative, especially given their parameter efficiency and potential to meet the same constraints.
    - For replay, a DER-like approach storing logits or features could be more memory-efficient than raw token sequences; an ablation including logit-based replay would help quantify trade-offs within the 16 GB budget.
- Discussion of broader impact and significance
    - The work is strongly motivated by real-world needs and provides concrete guidance on buffer allocation, memory budgeting, and operational workflows (A/B testing, online buffer updates). This is likely impactful for practitioners deploying LLMs across many domains with constrained resources.
    - However, storing and replaying production user data raises privacy and compliance concerns. The paper should discuss data governance, anonymization, and retention policies, as well as alternatives such as synthetic replay or distilled targets to reduce sensitive data storage.
    - The identified failure modes—OOD multi-intent queries and buffer staleness—are important in production. The proposed future directions (adapter fusion, adaptive replay scheduling, KD for buffer compression) are reasonable and align with known remedies.

### Questions for Authors

1. How is the domain embedding e_k determined at inference when task identity is unknown? Do you use a learned domain router, a soft mixture over domains, or per-query retrieval? Please detail the routing/gating pipeline and its training objective.
2. Please reconcile the apparent inconsistency between “freezing 80%” with 0.6B trainable parameters and the inference memory budget that treats the base model as frozen INT8 plus small adapter memory. Which parameters remain trainable and which are ultimately deployed? Are any base weights updated and then re-quantized/frozen for deployment?
3. Can you add a baseline of PEFT + replay (e.g., LoRA or adapter-based replay) under the same memory/latency constraints to isolate the benefit of your specific Dual-Replay design versus the generic combination?
4. What is the source of the general-knowledge replay buffer, and how do you ensure it approximates the pretraining distribution without violating licensing or privacy constraints? Have you tried knowledge distillation/logit replay to reduce memory?
5. Could you report results on at least one public continual-learning NLU/dialogue benchmark (with domain sequences), including mean and variance across multiple orderings, to improve reproducibility?
6. How sensitive are the results to the 50/50 split between domain-specific and general replay within α? Did you explore adaptive schedules for α and for the domain-vs-general split?
7. What hardware and batching configuration yields the reported 82 ms p99 latency? Please include typical sequence lengths, batch sizes, and GPU model.
8. Are adapters placed in both encoder and decoder blocks? What are the bottleneck ranks used, and are there layer-wise differences? An ablation on adapter placement could strengthen the architectural choices.

### Overall Assessment

This paper tackles an important and timely problem: mitigating catastrophic forgetting for LLMs in multi-domain, resource-constrained deployments. The pragmatic co-design of parameter efficiency and replay, the dual replay streams, and the attention to real operational constraints are commendable. The results across 52 production domains, latency/memory accounting, and ablations provide useful guidance for practitioners and suggest that the approach is effective in the described environment.

However, there are notable gaps that limit confidence in the central claims. Most importantly, the absence of a direct PEFT+replay baseline under the same constraints undermines the argument for the specific synergy claimed by Dual-Replay. The gating mechanism and the freezing/training configuration require clarification, and the lack of public-benchmark validation and statistical rigor reduces reproducibility and generality. While I find the work practically relevant and promising, I recommend a borderline reject in its current form; addressing the baseline, clarity, and reproducibility issues would substantially strengthen its suitability for NeurIPS.