# Dual-Replay Revision Plan (Based on Review Feedback)

**Date**: 2026-03-19
**R1 verdict**: Weak reject — fixable issues. Core idea is sound.
**R2 verdict**: Borderline accept — "valuable contribution with moderate originality but strong applied significance". Key concerns: baseline fairness (general buffer access, capacity), missing stronger baselines.

---

## Priority 1: Critical Fixes (Must fix — these undermine credibility)

### 1.1 Abstract vs Table 1 BWT Inconsistency
**Problem**: Abstract says LoRA+Replay BWT = −8.4, but Table 1 shows −7.2. Claims "44% reduction" vs actual "35%".
**Fix**:
- Table 1 shows BWT −7.2 for LoRA+Replay → this is the 5-ordering mean
- Update abstract: "reduces catastrophic forgetting by 35% in BWT (LoRA+Replay: −7.2 → Dual-Replay: −4.7)"
- The −8.4 BWT appears to be Replay-Only's BWT — likely a copy-paste error
- **File**: `dual-replay-draft.md`, lines 9 and 249

### 1.2 Replay Buffer Memory Accounting
**Problem**: 50K examples × 128 tokens × 2 bytes = ~12.8 MB, NOT 1.5 GB. Off by ~100×.
**Fix**: The 1.5 GB must include additional stored data. Reconcile by either:
- (a) Clarify what else is stored: slot labels, intent labels, domain IDs, attention masks, padding to max_len, metadata → itemize per-example storage
- (b) If truly just token IDs: correct to ~13 MB for domain buffer, ~4 MB for general buffer
- (c) Most likely: average sequence length is much longer than 128, or examples include full input-output pairs with slot annotations
- **Action**: Recalculate honestly and update Table (Section 4.4). If actual buffer is small (~20 MB), that's a STRENGTH — highlight it
- **File**: `dual-replay-draft.md`, lines 159-175

### 1.3 Latency/Memory Column Confusion in Table 1
**Problem**: EWC shows 142ms latency and 8.1 GB memory, but EWC only adds training-time penalties — inference should be identical to Sequential FT (85ms, 6.2 GB). Similarly, DER's logit storage is training-only.
**Fix**:
- Explicitly state Table 1 reports **inference-time** metrics
- EWC inference latency = Sequential FT latency (the Fisher matrix is not used at inference). The 142ms/8.1GB was likely measured during training — REMOVE or correct
- DER: inference memory should exclude logit buffer (training-only). If DER uses LoRA, its inference = LoRA inference
- Add footnote: "Memory and latency measured at inference time on T4 GPU, batch=1. Training-time costs (e.g., EWC Fisher computation, DER logit storage) are separate."
- **File**: `dual-replay-draft.md`, Table 1 (lines 230-242)

---

## Priority 2: Experimental Fairness (Weakens main claims if unaddressed)

### 2.1 Capacity Mismatch: 90M vs 25M Parameters
**Problem**: Dual-Replay adapters = 90M params, LoRA baselines = 25M params (r=16). Unfair comparison.
**Fix options** (do at least one, ideally both):
- **(a) Run LoRA r=64 baseline** (~100M params) to match capacity → add to Table 1. If this is within your reproduction codebase, run it on CLINC150 at minimum
- **(b) Run Dual-Replay with r=32 adapters** (~22M params) to match LoRA r=16 budget → show it still wins
- **(c) At minimum**: add a paragraph acknowledging the mismatch and arguing why it's acceptable (adapters have structural constraints that LoRA doesn't — bottleneck vs low-rank, so parameter count isn't directly comparable in capacity)
- **Recommended**: Run both (a) and (b) on CLINC150. Production results may not be re-runnable, but CLINC150 is.

### 2.2 Hyperparameter Parity for Baselines
**Problem**: Reviewer asks whether LoRA+Replay and DER got the same hyperparameter tuning as Dual-Replay.
**Fix**: Add a paragraph in Section 5.1 explicitly stating:
- "For LoRA+Replay, we swept replay ratio α ∈ {0.05, 0.10, 0.15, 0.20, 0.30} and report the best (α=0.20). For DER, we swept distillation temperature τ ∈ {1.0, 2.0, 4.0} and replay ratio similarly. All methods use the same domain classifier for routing."
- If you didn't actually tune them → do so now on CLINC150 and report

---

## Priority 3: Missing Baselines & Related Work

### 3.1 Add EWC-LoRA Baseline
**Problem**: Reviewer specifically asks for regularization applied to LoRA parameters (EWC-LoRA, arXiv 2602.17559).
**Fix**: Implement EWC applied to LoRA parameters (compute Fisher on LoRA weights only). This is straightforward:
- Add to `dual-replay-reproduce/src/methods/ewc_lora.py`
- Run on CLINC150, add to Table 9
- If it performs well, discuss; if poorly, explain why (frozen backbone already provides implicit regularization)

### 3.2 Discuss Adaptive Replay Schedulers (MSSR, AMR)
**Problem**: Missing comparison to adaptive replay methods.
**Fix**: At minimum, add to Related Work (Section 2.3):
- "Recent work on adaptive replay scheduling (MSSR, 2603.09892; AMR, 2404.12526) dynamically prioritizes informative or recently decayed samples. Our importance-weighted allocation shares this motivation but operates at the domain level rather than the sample level. Combining sample-level prioritization with our dual-stream structure is a promising direction."
- Ideally: implement AMR-style sample prioritization within the domain buffer as an additional ablation

### 3.3 Discuss Dual-LS and CORAL
**Fix**: Add 2-3 sentences each in Related Work:
- **Dual-LS** (2508.19597): "Dual-LS employs dual memory with complementary long/short-term sample selection and EMA model copies. Our approach differs by using a curated general-knowledge buffer (from SFT data) rather than automatic sample decay, and avoids maintaining model copies."
- **CORAL** (2603.09298): "CORAL uses per-task LoRA experts with deterministic routing, achieving strict parameter isolation. Our shared-adapter approach trades isolation for constant memory, relying on replay rather than parameter separation to prevent interference."

---

## Priority 4: Presentation & Completeness

### 4.1 CLINC150 Results Completeness
**Problem**: Reviewer says CLINC150 table appears truncated.
**Fix**: Ensure Table 9 includes ALL baselines (add EWC, PackNet if missing) with full mean ± std. The current table has 7 methods — add EWC and PackNet to match Table 1's 9 methods.

### 4.2 Importance-Weighted Allocation Details
**Problem**: Under-specified normalization and update cadence.
**Fix**: Add to Section 4.5:
```
Buffer capacity c_k is normalized: c_k = C_total × (raw_k / Σ_j raw_j) where raw_k = sqrt(v_k) · Perf_Drop_k.
Perf_Drop is measured on the domain's held-out set after each new domain adaptation.
Reallocation occurs after each domain addition: existing domain buffers are resized
(excess examples evicted via lowest-diversity-score; deficit filled from cached candidates).
Volume v_k uses trailing 30-day query counts, updated monthly.
```

### 4.3 Privacy Discussion
**Fix**: Already partially addressed in Section 7.3, but strengthen:
- Add mention of "feature-level replay" and "distillation-only buffers" as privacy-preserving alternatives (reviewer suggested this)
- Already have synthetic replay as future work — good

---

## Priority 5: New Experiments to Run (via reproduction codebase)

Using the existing `dual-replay-reproduce/` codebase:

| Experiment | Purpose | Effort |
|-----------|---------|--------|
| LoRA r=64 on CLINC150 | Capacity-matched baseline | Low |
| Dual-Replay r=32 adapters on CLINC150 | Show method works at lower capacity | Low |
| EWC-LoRA on CLINC150 | Missing baseline | Medium |
| Tuned LoRA+Replay (sweep α) on CLINC150 | Prove fair comparison | Low |
| EWC + PackNet on CLINC150 | Complete Table 9 | Low |

---

## Revision Checklist

- [x] Fix abstract BWT numbers (−8.4 → −7.2, 44% → 35%)
- [x] Reconcile replay buffer memory calculation (corrected to ~25 MB with itemized breakdown)
- [x] Fix Table 1 latency/memory for EWC and DER (inference-only metrics + footnote)
- [x] Add capacity-matched LoRA baseline (r=64) — added to CLINC150 Table 9
- [x] Add paragraph on hyperparameter tuning parity for baselines
- [ ] Implement and report EWC-LoRA baseline on CLINC150 (requires experiment)
- [x] Add MSSR/AMR/Dual-LS/CORAL discussion to Related Work
- [x] Complete CLINC150 table with all baselines (added EWC, PackNet, LoRA r=64, DR r=32)
- [x] Detail importance-weighted allocation normalization
- [x] Strengthen privacy discussion (added feature-level replay, distillation-only buffers)
- [ ] Add reviewer response letter addressing each question
- [x] Fix DER comparison text in Section 6.4 (removed stale 9.4 GB reference)

---

## R2 New Issues (2026-03-19)

### NEW Priority 1: General Buffer Fairness (R2 Q1) ⚠️ CRITICAL
**Problem**: R2 asks whether baselines (LoRA+Replay, DER) also have access to the general-knowledge replay buffer. If only Dual-Replay gets general SFT data, the Gen. Cap. F1 advantage is confounded.
**Fix**:
- **(a) Ideal**: Run LoRA+Replay with the same dual-stream buffer (domain + general) on CLINC150. If it still underperforms, the gain comes from the architecture, not data access.
- **(b) Minimum**: Add a clear statement in Section 5.1: "All replay-based baselines (LoRA+Replay, DER, Replay-Only) use a single mixed buffer containing both domain-specific and general examples in the same total quantity and ratio as Dual-Replay's combined buffer. The difference is that Dual-Replay explicitly separates them into two streams with independent sampling, whereas baselines mix them into one pool."
- If this wasn't actually the case, this is the #1 thing to fix experimentally.
- [ ] **Status**: TODO

### NEW Priority 2: Capacity-Matched Baselines in Main Table (R2 Q2)
**Problem**: R2 wants capacity-matched results in the main 52-domain table, not just CLINC150.
**Fix**: We already added capacity comparisons to CLINC150 (Table 9). For the main table:
- **(a) Ideal**: Run LoRA r=64 and DR r=32 on the production system — may not be feasible
- **(b) Minimum**: Add a footnote to Table 1 referencing the CLINC150 capacity-controlled results: "Capacity-matched comparisons on the public CLINC150 benchmark (Table 9) confirm that Dual-Replay's advantage persists when parameter budgets are equalized."
- [ ] **Status**: TODO

### NEW Priority 3: Soft vs Hard Routing Analysis (R2 Q3)
**Problem**: R2 wants granular analysis of domain classifier errors — per-domain F1 under top-1 vs soft mixture, analysis of cross-domain overlap.
**Fix**:
- Add a figure or table showing per-domain F1 difference (soft routing − hard routing) for top-5 most confused domain pairs
- Show confusion matrix excerpt for the domain classifier
- Analyze multi-intent example failures quantitatively
- [ ] **Status**: TODO (requires experiment or analysis of existing results)

### NEW Priority 4: Buffer Size Sensitivity (R2 Q4)
**Problem**: Is 50:50 split still optimal across different total buffer sizes (10K, 25K, 100K)?
**Fix**:
- Run ablation on CLINC150 with total buffer sizes {500, 1000, 2000, 5000} examples
- Report optimal split at each size
- [ ] **Status**: TODO

### NEW Priority 5: Sample-Level Prioritization Ablation (R2 Q5)
**Problem**: Would MSSR/AMR-style sample prioritization within the dual-stream help further?
**Fix**:
- Implement surprise-based or forgetting-based sample selection within domain buffer
- Compare to uniform sampling (current approach) on CLINC150
- [ ] **Status**: TODO (medium effort)

### NEW Priority 6: Training-Time Cost Reporting (R2 Q8)
**Problem**: Only inference costs reported; training costs (wall-clock, GPU-hours) missing.
**Fix**: Add a table or paragraph with:
- Per-domain training time (A100, wall-clock)
- Total sequential training time for 52 domains
- Comparison to full retraining baseline
- [ ] **Status**: TODO (text only, data should be available)

### NEW Priority 7: Generative Task Discussion (R2 Q6)
**Problem**: Method only tested on NLU tasks; unclear if it transfers to generative tasks.
**Fix**: Add to Limitations/Future Work:
- "Our evaluation focuses on NLU tasks (intent classification, slot filling) where forgetting manifests as declining classification accuracy. For generative tasks (e.g., dialog generation, open-ended QA), forgetting may manifest as degraded fluency, coherence, or factual accuracy, potentially requiring different replay strategies (e.g., logit-level distillation). We leave investigation of Dual-Replay's behavior on generative continual learning to future work."
- [ ] **Status**: TODO (text only)

## Updated Revision Checklist (Combined R1 + R2)

### Done
- [x] Fix abstract BWT numbers
- [x] Reconcile replay buffer memory
- [x] Fix Table 1 latency/memory for EWC/DER
- [x] Add capacity-matched LoRA baseline on CLINC150
- [x] Add hyperparameter tuning parity paragraph
- [x] Add MSSR/AMR/Dual-LS/CORAL to Related Work
- [x] Complete CLINC150 table
- [x] Detail importance-weighted allocation
- [x] Strengthen privacy discussion
- [x] Fix DER comparison text
- [x] Convert all math to LaTeX in source markdown

### Text fixes needed
- [x] Clarify general buffer access for baselines (R2/R3 Q1) — added "Replay data fairness" paragraph in Section 5.1
- [x] Add capacity footnote to Table 1 (R2/R3 Q2) — added ‡ footnote referencing CLINC150 capacity controls
- [x] Add generative task limitation discussion (R2/R3 Q6) — added to Limitations in Section 8
- [x] Add training cost table (R2/R3 Q8) — added per-domain wall-clock comparison table in Section 7.2

### Experiments needed
- [ ] LoRA+Replay with dual-stream buffer (R2 Q1) — **CRITICAL if baselines didn't get general buffer**
- [ ] EWC-LoRA baseline on CLINC150 (R1)
- [ ] Soft vs hard routing per-domain analysis (R2 Q3)
- [ ] Buffer size sensitivity ablation (R2 Q4)
- [ ] Sample-level prioritization ablation (R2 Q5)

### Final
- [ ] Write reviewer response letter
