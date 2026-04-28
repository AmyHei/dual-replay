### Summary

The paper proposes Dual-Replay, a parameter-efficient continual learning framework for sequential domain adaptation of LLMs under strict latency and memory budgets. The method freezes the base model and trains shared bottleneck adapters with task-conditioned gating, while interleaving new-domain data with a dual replay stream: (i) domain-specific exemplars and (ii) general-knowledge examples drawn from held-out SFT data. Evaluated on 52 production domains and on CLINC150, Dual-Replay achieves higher average NLU F1, reduced catastrophic forgetting (BWT), and meets a sub-100 ms, ≤16 GB deployment constraint, with ablations analyzing replay ratio, freezing, and replay split.

### Strengths

- Technical novelty and innovation
    - The co-design of frozen backbone adapters with lightweight task-conditioned gating and a dual replay stream (domain-specific + general-knowledge) is a thoughtful, practical combination under tight resource budgets.
    - Importance-weighted domain buffer allocation (√volume × performance drop) is a principled heuristic that adapts memory to user impact and observed forgetting.
    - Soft mixture routing via a learned domain classifier reduces the brittleness of hard task routing without incurring significant inference overhead.
- Experimental rigor and validation
    - Large-scale evaluation over 52 sequential domains with five random orderings and paired tests with Bonferroni correction, plus a public benchmark (CLINC150), strengthens empirical claims.
    - Deployment-centric metrics (p99 latency, peak GPU memory) and a concrete memory budget breakdown demonstrate practical viability on a T4 target.
    - Ablations on replay ratio, freezing ratio, and replay split provide insight into stability–plasticity trade-offs and design choices.
- Clarity of presentation
    - The method is described with clear architectural details (adapter design and placement, gating, routing) and explicit training/inference workflows.
    - Memory accounting and buffer storage format are transparently presented, including a correction about quantization assumptions.
- Significance of contributions
    - Addresses a high-impact, pervasive challenge—catastrophic forgetting for many-domain LLMs under deployment constraints—delivering measurable gains with low overhead.
    - The findings and practical guidance (e.g., α=0.20, 50:50 replay split, full freezing) are likely to transfer to applied settings.

### Weaknesses

- Technical limitations or concerns
    - Reliance on a domain classifier and learned domain embeddings presumes a relatively stable, known domain set and may struggle with OOD or multi-intent queries (acknowledged but not deeply evaluated).
    - The method targets NLU-style tasks (intent/slot), leaving open questions about transfer to more generative or open-ended tasks where replay and routing dynamics may differ.
- Experimental gaps or methodological issues
    - Capacity mismatch in the main results: adapters (~~90M trainable) vs LoRA baselines (~~25M). Although capacity-matched comparisons are claimed on CLINC150 (r=64, ~100M), those details are not shown in the main results and are not visible in the provided text.
    - Dual-Replay benefits from a general-knowledge replay set drawn from SFT data; it is unclear whether baselines receive the same general buffer (fairness of comparison), or only domain exemplars. This could partially explain the advantage in “General Capability F1”.
    - Some state-of-the-art continual PEFT/replay scheduling baselines (e.g., SuRe; CORAL under constrained K; more recent prioritization methods) are discussed but not included in main comparisons or are excluded due to expected memory growth without a thorough empirical check under K=52 and the given budget.
    - Only five orderings are used; while understandable, more permutations would increase statistical robustness for highly order-sensitive protocols.
- Clarity or presentation issues
    - Minor rendering artifacts in tables impede precise reading of a few numbers; some key results (e.g., CLINC150 capacity-matched comparisons) are referred to but not visible in the excerpt.
    - Limited quantitative analysis of routing errors and the specific benefit of soft vs hard routing in terms of per-domain performance and confusion patterns.
- Missing related work or comparisons
    - While related work coverage is broad, the empirical comparison set omits some strong recent PEFT-CL or replay-scheduling approaches or does not fully normalize for access to general replay data, making it harder to isolate which components drive the gains.

### Detailed Comments

- Technical soundness evaluation
    - The architectural choices are reasonable: full freezing reduces interference; adapters provide plasticity; gating modulates shared capacity with minimal per-domain overhead; and dual replay stabilizes both domain-specific and general capabilities.
    - Importance-weighted buffer allocation using √volume × performance drop is a sound heuristic aligning with user impact and forgetting severity; however, the per-domain dependence on held-out test metrics may be hard to reproduce in academic settings and may couple evaluation and training decisions.
    - Soft mixture routing over domain embeddings is technically sound and likely more robust than hard routing, but it would benefit from a more granular analysis of error modes (e.g., cross-domain overlap, multi-intent inputs).
- Experimental evaluation assessment
    - The 52-domain production evaluation and clear deployment constraints are valuable; reporting p99 latency and detailed memory footprints is a strong practical contribution.
    - The ablations convincingly show the benefits of α≈0.20 and a 50:50 domain/general split; they also suggest full freezing is an effective choice under the given capacity.
    - Concerns: (i) Capacity mismatch with LoRA baselines in main tables may inflate relative gains; (ii) fairness around access to general SFT replay data for baselines is unclear; (iii) broader baselines (SuRe, CORAL variants within the same memory headroom) would strengthen claims; (iv) training-time costs (e.g., DER’s logit storage, EWC Fisher) are acknowledged but not comprehensively reported, limiting total-cost comparisons.
- Comparison with related work (using the summaries provided)
    - Relative to HyperFormer/HyperPrompt-style conditional adapters/prompts (task-conditioned, often requiring task IDs), the proposed soft domain routing and shared adapters echo the spirit of conditional parameterization but aim for deployment simplicity with minimal per-domain overhead.
    - Compared to O-LoRA and CORAL-style expert isolation, Dual-Replay opts for constant memory (shared adapters) and uses replay rather than hard parameter isolation, trading off some interference for practical scalability at K>50; this positioning is reasonable given the 16 GB constraint.
    - Replay mechanisms like DER and prioritization (MSSR/AMR) are related; Dual-Replay’s novelty lies in a dual-stream buffer (including general-knowledge retention) and domain-level importance weighting under a single budget. Including sample-level prioritization within the dual stream would be a natural extension.
    - The broader PECFT surveys emphasize the challenge of scaling per-task modules; Dual-Replay’s shared adapters with gating directly address that, providing a complementary direction to hypernetwork-based approaches.
- Discussion of broader impact and significance
    - The work has strong applied impact: improving many-domain robustness under modest hardware while avoiding user-facing regressions is highly valuable for production conversational AI.
    - Replay of production-like data raises typical privacy/memorization questions; the paper states the general buffer excludes production data, and domain buffers are tokenized exemplars—still, a brief discussion of privacy controls, retention policies, and compliance would be beneficial.
    - The method’s practicality and the clear recipe (α, split, freezing) could make it a go-to baseline for industrial CL under resource constraints.

### Questions for Authors

1. Do all replay-based baselines (e.g., LoRA+Replay, DER) have access to the same general-knowledge replay buffer as Dual-Replay? If not, can you report results where they do, to isolate the effect of the dual-stream design rather than access to additional general data?
2. In the main 52-domain results, can you include capacity-matched baselines (e.g., LoRA r≈64, ~100M params) and also a reduced-adapter variant (r≈32, ~22M) to disentangle algorithmic effects from capacity?
3. Can you provide a detailed analysis of domain-classifier errors and the benefits of soft vs hard routing (e.g., per-domain F1 under top-1 vs soft mixtures, performance on ambiguous/multi-intent examples)?
4. How sensitive are results to the total replay buffer size (e.g., 10K, 25K, 100K examples) under the same memory budget? Is the 50:50 split still optimal across sizes?
5. Could you compare against at least one strong sample-prioritized replay scheduler (e.g., MSSR/AMR-like) embedded within your dual-stream setup to assess whether sample-level prioritization adds further gains?
6. Beyond NLU (intent/slot), have you tested Dual-Replay on more generative tasks (e.g., dialog generation or QA) where forgetting manifests differently? If so, how do α and the domain/general split behave?
7. What safeguards exist for privacy and memorization in the domain buffers, and how would the approach adapt to stricter data retention policies (e.g., differential privacy or short-lived buffers)?
8. Could you clarify training-time costs across methods (e.g., wall-clock, GPU-hours) to complement the inference-centric resource reporting?

### Overall Assessment

This paper addresses an important and practical problem—continual domain adaptation of LLMs under stringent deployment constraints—and presents a well-motivated, effective combination of shared adapters, soft domain routing, and dual-stream replay with thoughtful buffer management. The large-scale production evaluation and deployment-focused metrics are compelling, and the ablations provide useful guidance. However, some comparative fairness concerns (general replay access for baselines, capacity mismatches in main results) and missing stronger baselines temper the strength of the claims. Clarifying these points and adding capacity-matched and general-replay-normalized baselines would substantially strengthen the case. Overall, I find the contribution valuable to the NeurIPS community, particularly for practitioners, with moderate originality but strong applied significance.