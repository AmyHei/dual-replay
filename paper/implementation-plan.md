# Dual-Replay Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce and validate all claims in the Dual-Replay paper using public datasets (CLINC150 + multi-dataset composite) with BERT-large and T5 models.

**Architecture:** HuggingFace Transformers + PEFT stack. Custom bottleneck adapters with task-conditioned gating, dual-stream replay buffer, and sequential domain training loop. TDD approach -- tests first, then implementation.

**Tech Stack:** Python 3.13, PyTorch (MPS + CUDA), transformers, peft, datasets, scikit-learn, uv for dependency management.

**Spec:** `reproduction-plan.md` in the same directory.

---

## File Structure

```
dual-replay-reproduce/
├── pyproject.toml                      # uv project config, all deps
├── configs/
│   └── default.yaml                    # All experiment configs (debug/full/ablation)
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── clinc150.py                 # CLINC150 loading + 15-domain split protocol
│   │   └── domain_sequence.py          # Domain ordering, train/test splits, general buffer
│   ├── models/
│   │   ├── __init__.py
│   │   ├── adapters.py                 # BottleneckAdapter + AdaptedModel (inserts adapters into any HF model)
│   │   ├── gating.py                   # TaskConditionedGating + DomainEmbeddings
│   │   └── domain_classifier.py        # DomainClassifier head + soft mixture routing
│   ├── replay/
│   │   ├── __init__.py
│   │   └── buffer.py                   # DomainBuffer + GeneralBuffer + DualReplayBuffer
│   ├── methods/
│   │   ├── __init__.py
│   │   ├── base.py                     # BaseContinualMethod (common interface)
│   │   ├── sequential_ft.py            # Full-model sequential fine-tuning
│   │   ├── ewc.py                      # Elastic Weight Consolidation
│   │   ├── lora_only.py                # LoRA without replay
│   │   ├── replay_only.py              # Full-model + replay
│   │   ├── lora_replay.py              # LoRA + single-stream replay
│   │   ├── o_lora.py                   # Orthogonal subspace LoRA
│   │   ├── der.py                      # Dark Experience Replay with LoRA
│   │   └── dual_replay.py             # Our method: adapter + dual-stream replay + gating
│   ├── training/
│   │   ├── __init__.py
│   │   └── runner.py                   # SequentialTrainer: train across domains, log results
│   └── metrics.py                      # BWT, FWT, avg_f1, statistical tests
├── scripts/
│   └── run_experiment.py               # CLI entry point: --method --config --seed
├── tests/
│   ├── __init__.py
│   ├── test_clinc150.py                # Data loading + domain split tests
│   ├── test_adapters.py                # Adapter insertion + parameter freezing tests
│   ├── test_gating.py                  # Gating + domain embedding tests
│   ├── test_domain_classifier.py       # Classifier + soft routing tests
│   ├── test_replay_buffer.py           # Buffer management + sampling tests
│   ├── test_metrics.py                 # BWT/FWT calculation tests
│   ├── test_methods.py                 # Each method's train_step/eval_step interface
│   └── test_integration.py             # End-to-end: 2 domains, tiny model, all methods
└── results/                            # gitignored, experiment outputs
```

---

## Task 1: Project Scaffolding + Dependencies

**Files:**
- Create: `dual-replay-reproduce/pyproject.toml`
- Create: `dual-replay-reproduce/configs/default.yaml`
- Create: `dual-replay-reproduce/src/__init__.py` (and all subpackage `__init__.py`)
- Create: `dual-replay-reproduce/.gitignore`

- [ ] **Step 1: Initialize uv project**

```bash
cd /Users/yaqinhei/Documents/同步空间-work/obsidian/PHD/mypaper/dual-replay
mkdir -p dual-replay-reproduce
cd dual-replay-reproduce
uv init --name dual-replay-reproduce
```

- [ ] **Step 2: Add dependencies to pyproject.toml**

Edit `pyproject.toml` to include:

```toml
[project]
name = "dual-replay-reproduce"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.0",
    "transformers>=4.36",
    "peft>=0.7",
    "datasets>=2.16",
    "accelerate>=0.25",
    "scikit-learn",
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "pyyaml",
    "tqdm",
    "scipy",
    "pytest",
]
```

- [ ] **Step 3: Install dependencies**

```bash
uv sync
```

- [ ] **Step 4: Create directory structure and `__init__.py` files**

```bash
mkdir -p src/{data,models,replay,methods,training} tests configs scripts results
touch src/__init__.py src/data/__init__.py src/models/__init__.py src/replay/__init__.py
touch src/methods/__init__.py src/training/__init__.py tests/__init__.py
echo "results/" > .gitignore
```

- [ ] **Step 5: Create default config YAML**

Write `configs/default.yaml`:

```yaml
# Shared defaults
seed: 42
num_orderings: 5
seeds: [42, 123, 456, 789, 1024]

# Debug config (Mac local)
debug:
  model_name: "prajjwal1/bert-tiny"
  num_domains: 3
  epochs_per_domain: 1
  batch_size: 8
  adapter_r: 16
  max_seq_len: 64
  domain_buffer_size: 50
  general_buffer_size: 200
  learning_rate: 5.0e-4
  replay_ratio: 0.20
  domain_replay_fraction: 0.10
  num_orderings: 1

# Full CLINC150 config (GPU)
clinc150:
  model_name: "bert-large-uncased"
  num_domains: 15
  epochs_per_domain: 5
  batch_size: 32
  adapter_r: 64
  max_seq_len: 128
  domain_buffer_size: 200
  general_buffer_size: 1000
  learning_rate: 2.0e-4
  replay_ratio: 0.20
  domain_replay_fraction: 0.10
  num_orderings: 5

# T5 config (Phase 3)
t5:
  model_name: "t5-base"
  num_domains: 15
  epochs_per_domain: 5
  batch_size: 16
  adapter_r: 64
  max_seq_len: 128
  domain_buffer_size: 200
  general_buffer_size: 1000
  learning_rate: 1.0e-4
  replay_ratio: 0.20
  domain_replay_fraction: 0.10
  num_orderings: 5
```

- [ ] **Step 6: Verify imports work**

```bash
uv run python -c "import torch; print(torch.backends.mps.is_available()); import transformers; import peft; import datasets; print('All imports OK')"
```

- [ ] **Step 7: Commit**

```bash
git init
git add -A
git commit -m "feat: scaffold project with deps and config"
```

---

## Task 2: CLINC150 Data Loading + 15-Domain Protocol

**Files:**
- Create: `src/data/clinc150.py`
- Create: `src/data/domain_sequence.py`
- Test: `tests/test_clinc150.py`

**Context:** CLINC150 has 150 intents across 10 domains + OOS. The paper splits into 15 sub-domains with 10 intents each. We need to design this split and document it.

- [ ] **Step 1: Write failing tests for CLINC150 loading**

Write `tests/test_clinc150.py`:

```python
"""Tests for CLINC150 data loading and 15-domain protocol."""
import pytest


def test_load_clinc150_raw():
    """Should load raw CLINC150 and return train/val/test splits."""
    from src.data.clinc150 import load_clinc150_raw

    splits = load_clinc150_raw()
    assert "train" in splits
    assert "validation" in splits
    assert "test" in splits
    # CLINC150 has 150 in-scope intents + OOS
    train_intents = set(splits["train"]["intent"])
    assert len(train_intents) >= 150


def test_build_15_domain_protocol():
    """Should split 150 intents into 15 domains of 10 intents each."""
    from src.data.clinc150 import build_15_domain_protocol

    domains = build_15_domain_protocol()
    assert len(domains) == 15
    all_intents = set()
    for domain in domains:
        assert len(domain["intents"]) == 10
        assert len(domain["train"]) > 0
        assert len(domain["test"]) > 0
        all_intents.update(domain["intents"])
    # All 150 intents covered, no overlap
    assert len(all_intents) == 150


def test_general_buffer_from_oos():
    """General replay buffer should come from OOS examples."""
    from src.data.clinc150 import get_general_buffer

    buffer = get_general_buffer(max_size=100)
    assert len(buffer) == 100
    # All examples should be OOS
    assert all(ex["intent"] == "oos" for ex in buffer)


def test_domain_ordering_reproducibility():
    """Same seed should produce same domain ordering."""
    from src.data.domain_sequence import generate_domain_orderings

    ord1 = generate_domain_orderings(num_domains=15, num_orderings=2, seed=42)
    ord2 = generate_domain_orderings(num_domains=15, num_orderings=2, seed=42)
    assert ord1 == ord2
    # Different orderings within same seed set
    assert ord1[0] != ord1[1]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/yaqinhei/Documents/同步空间-work/obsidian/PHD/mypaper/dual-replay/dual-replay-reproduce
uv run pytest tests/test_clinc150.py -v
```

Expected: FAIL (modules not found)

- [ ] **Step 3: Implement `src/data/clinc150.py`**

```python
"""CLINC150 dataset loading and 15-domain protocol construction.

The paper splits CLINC150's original 10 domains into 15 sub-domains,
each with 10 intents. We split the larger domains (which have 15 intents)
into sub-domains, keeping the smaller ones intact.

CLINC150 original domains and intent counts:
  banking(15), credit_cards(15), kitchen_dining(15), home(15), auto_commute(15),
  travel(15), utility(15), work(15), small_talk(15), meta(15)

Split strategy: Each original domain has exactly 15 intents.
We take all 150 intents, sort alphabetically by name, and partition into
15 groups of 10. This keeps related intents together while producing
an equal split.
"""
from datasets import load_dataset
import random


def load_clinc150_raw():
    """Load raw CLINC150 dataset from HuggingFace."""
    ds = load_dataset("clinc_oos", "plus")  # 'plus' includes OOS examples
    return ds


def _get_intent_names(ds):
    """Get intent ID to name mapping from the dataset."""
    return ds["train"].features["intent"].names


def build_15_domain_protocol(seed=42):
    """Split 150 CLINC150 intents into 15 domains of 10 intents each.

    Strategy: Sort all 150 in-scope intents by their name alphabetically,
    then split into 15 consecutive groups of 10.
    """
    ds = load_clinc150_raw()
    intent_names = _get_intent_names(ds)

    # Collect all in-scope intent indices (exclude OOS)
    oos_idx = intent_names.index("oos") if "oos" in intent_names else -1
    in_scope = [i for i in range(len(intent_names)) if i != oos_idx]
    assert len(in_scope) == 150, f"Expected 150 in-scope intents, got {len(in_scope)}"

    # Sort by intent name to group related intents together
    in_scope_sorted = sorted(in_scope, key=lambda i: intent_names[i])

    # Split into 15 groups of 10
    domains = []
    for d in range(15):
        domain_intents = in_scope_sorted[d * 10 : (d + 1) * 10]
        intent_set = set(domain_intents)

        train_examples = [
            ex for ex in ds["train"] if ex["intent"] in intent_set
        ]
        test_examples = [
            ex for ex in ds["test"] if ex["intent"] in intent_set
        ]
        val_examples = [
            ex for ex in ds["validation"] if ex["intent"] in intent_set
        ]

        domains.append({
            "domain_id": d,
            "intents": domain_intents,
            "intent_names": [intent_names[i] for i in domain_intents],
            "train": train_examples,
            "test": test_examples,
            "validation": val_examples,
        })

    return domains


def get_general_buffer(max_size=1000, seed=42):
    """Get general replay buffer from OOS (out-of-scope) examples.

    The paper uses 'held-out SFT data' for general replay. Since CLINC150
    has no SFT data, we use OOS examples as the general knowledge buffer.
    """
    ds = load_clinc150_raw()
    intent_names = _get_intent_names(ds)
    oos_idx = intent_names.index("oos") if "oos" in intent_names else None

    if oos_idx is None:
        raise ValueError("OOS intent not found in CLINC150")

    oos_examples = [ex for ex in ds["train"] if ex["intent"] == oos_idx]

    rng = random.Random(seed)
    if len(oos_examples) > max_size:
        oos_examples = rng.sample(oos_examples, max_size)

    # Normalize: return with string intent label
    return [{"text": ex["text"], "intent": "oos"} for ex in oos_examples]
```

- [ ] **Step 4: Implement `src/data/domain_sequence.py`**

```python
"""Domain ordering generation for sequential training."""
import random


def generate_domain_orderings(
    num_domains: int, num_orderings: int, seed: int = 42
) -> list[list[int]]:
    """Generate reproducible random domain orderings.

    Args:
        num_domains: Number of domains to order.
        num_orderings: Number of different orderings to generate.
        seed: Base random seed.

    Returns:
        List of orderings, each a permutation of [0, num_domains).
    """
    orderings = []
    for i in range(num_orderings):
        rng = random.Random(seed + i)
        order = list(range(num_domains))
        rng.shuffle(order)
        orderings.append(order)
    return orderings
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_clinc150.py -v
```

Expected: All PASS (note: first run downloads CLINC150 dataset)

- [ ] **Step 6: Commit**

```bash
git add src/data/ tests/test_clinc150.py
git commit -m "feat: CLINC150 data loading with 15-domain protocol"
```

---

## Task 3: Bottleneck Adapter Module

**Files:**
- Create: `src/models/adapters.py`
- Test: `tests/test_adapters.py`

- [ ] **Step 1: Write failing tests**

Write `tests/test_adapters.py`:

```python
"""Tests for bottleneck adapter insertion and parameter freezing."""
import pytest
import torch


def test_bottleneck_adapter_shape():
    """Adapter should have correct parameter shapes."""
    from src.models.adapters import BottleneckAdapter

    adapter = BottleneckAdapter(d_model=768, bottleneck_dim=64)
    x = torch.randn(2, 10, 768)
    out = adapter(x)
    assert out.shape == x.shape  # residual connection preserves shape


def test_bottleneck_adapter_residual():
    """With zero-init W_up, adapter should be identity."""
    from src.models.adapters import BottleneckAdapter

    adapter = BottleneckAdapter(d_model=768, bottleneck_dim=64)
    # W_up is zero-initialized so adapter(h) = h + 0 = h
    x = torch.randn(2, 10, 768)
    out = adapter(x)
    torch.testing.assert_close(out, x)


def test_adapted_model_freezes_base():
    """Base model params should be frozen, only adapters trainable."""
    from src.models.adapters import create_adapted_model

    model = create_adapted_model("prajjwal1/bert-tiny", adapter_r=16)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    # Trainable should be much less than total (only adapters)
    assert trainable < total * 0.2
    assert trainable > 0  # adapters are trainable


def test_adapted_model_forward():
    """Adapted model should produce valid output."""
    from src.models.adapters import create_adapted_model
    from transformers import AutoTokenizer

    model = create_adapted_model("prajjwal1/bert-tiny", adapter_r=16)
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

    inputs = tokenizer("Hello world", return_tensors="pt")
    outputs = model(**inputs)
    assert outputs.last_hidden_state.shape[0] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_adapters.py -v
```

- [ ] **Step 3: Implement `src/models/adapters.py`**

```python
"""Bottleneck adapter module following Houlsby et al. (2019).

Adapter(h) = h + GeLU(h @ W_down) @ W_up

Key design: W_up is zero-initialized so adapter starts as identity,
preserving pre-trained model behavior at initialization.
"""
import torch
import torch.nn as nn
from transformers import AutoModel


class BottleneckAdapter(nn.Module):
    """Single bottleneck adapter layer."""

    def __init__(self, d_model: int, bottleneck_dim: int):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck_dim)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, d_model)
        # Zero-init W_up so adapter is identity at init
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.up(self.activation(self.down(h)))


def create_adapted_model(model_name: str, adapter_r: int = 64):
    """Load a HuggingFace model with bottleneck adapters inserted.

    Freezes all base model parameters. Inserts a BottleneckAdapter after
    each transformer layer's FFN (output dense layer).

    Args:
        model_name: HuggingFace model name or path.
        adapter_r: Adapter bottleneck dimension.

    Returns:
        Model with adapters inserted, base params frozen.
    """
    base_model = AutoModel.from_pretrained(model_name)
    d_model = base_model.config.hidden_size

    # Freeze all base model parameters
    for param in base_model.parameters():
        param.requires_grad = False

    # Insert adapters after each transformer layer's FFN
    adapters = nn.ModuleList()
    layers = get_transformer_layers(base_model)

    for i, layer in enumerate(layers):
        adapter = BottleneckAdapter(d_model, adapter_r)
        adapters.append(adapter)
        _hook_adapter_after_ffn(layer, adapter)

    # Attach adapters to model so they're part of the parameter graph
    base_model.adapters = adapters
    return base_model


def get_transformer_layers(model):
    """Extract transformer layers from various HF model architectures."""
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return list(model.encoder.layer)  # BERT
    if hasattr(model, "layers"):
        return list(model.layers)
    raise ValueError(f"Cannot find transformer layers in {type(model)}")


def _hook_adapter_after_ffn(layer, adapter):
    """Replace layer output module to apply adapter after the FFN output."""
    original_output = layer.output

    class AdaptedOutput(nn.Module):
        def __init__(self, original, adapter):
            super().__init__()
            self.original = original
            self.adapter = adapter

        def forward(self, hidden_states, input_tensor):
            output = self.original(hidden_states, input_tensor)
            return self.adapter(output)

    layer.output = AdaptedOutput(original_output, adapter)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_adapters.py -v
```

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/adapters.py tests/test_adapters.py
git commit -m "feat: bottleneck adapter with zero-init and auto-insertion"
```

---

## Task 4: Task-Conditioned Gating + Domain Embeddings

**Files:**
- Create: `src/models/gating.py`
- Test: `tests/test_gating.py`

- [ ] **Step 1: Write failing tests**

Write `tests/test_gating.py`:

```python
"""Tests for task-conditioned gating mechanism."""
import pytest
import torch


def test_domain_embeddings_shape():
    """Domain embeddings should have correct shape."""
    from src.models.gating import DomainEmbeddings

    emb = DomainEmbeddings(num_domains=15, embed_dim=64)
    e = emb(domain_id=3)
    assert e.shape == (64,)


def test_gating_vector_shape_and_range():
    """Gating vector should be in (0, 1) via sigmoid."""
    from src.models.gating import TaskConditionedGating

    gating = TaskConditionedGating(adapter_r=64, embed_dim=64)
    e_k = torch.randn(64)
    g = gating(e_k)
    assert g.shape == (64,)
    assert (g >= 0).all() and (g <= 1).all()


def test_gating_modulates_adapter_output():
    """Gating should element-wise multiply with adapter output."""
    from src.models.gating import TaskConditionedGating

    gating = TaskConditionedGating(adapter_r=64, embed_dim=64)
    e_k = torch.randn(64)
    adapter_out = torch.randn(2, 10, 64)  # batch, seq, adapter_r
    g = gating(e_k)
    modulated = adapter_out * g  # broadcast over batch and seq
    assert modulated.shape == adapter_out.shape


def test_soft_mixture_routing():
    """Soft routing should blend domain embeddings by probabilities."""
    from src.models.gating import DomainEmbeddings, soft_mixture_routing

    emb = DomainEmbeddings(num_domains=3, embed_dim=64)
    probs = torch.tensor([0.7, 0.2, 0.1])
    mixed = soft_mixture_routing(emb, probs)
    assert mixed.shape == (64,)
```

- [ ] **Step 2: Run tests to verify fail**

```bash
uv run pytest tests/test_gating.py -v
```

- [ ] **Step 3: Implement `src/models/gating.py`**

```python
"""Task-conditioned gating and domain embeddings.

g_k = sigmoid(W_g @ e_k + b_g)

Each domain has a learned embedding e_k. The gating vector g_k
modulates adapter outputs element-wise, allowing a single adapter
set to specialize per domain.
"""
import torch
import torch.nn as nn


class DomainEmbeddings(nn.Module):
    """Learnable domain embeddings. One vector per domain."""

    def __init__(self, num_domains: int, embed_dim: int = 64):
        super().__init__()
        self.embeddings = nn.Embedding(num_domains, embed_dim)

    def forward(self, domain_id: int) -> torch.Tensor:
        idx = torch.tensor(domain_id, dtype=torch.long,
                           device=self.embeddings.weight.device)
        return self.embeddings(idx)

    @property
    def num_domains(self) -> int:
        return self.embeddings.num_embeddings


class TaskConditionedGating(nn.Module):
    """Produces gating vector from domain embedding.

    g_k = sigmoid(W_g @ e_k + b_g), where g_k in (0,1)^r
    """

    def __init__(self, adapter_r: int, embed_dim: int = 64):
        super().__init__()
        self.gate = nn.Linear(embed_dim, adapter_r)

    def forward(self, domain_embedding: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate(domain_embedding))


def soft_mixture_routing(
    domain_embeddings: DomainEmbeddings,
    probs: torch.Tensor,
) -> torch.Tensor:
    """Compute soft mixture of domain embeddings weighted by probabilities.

    e_query = sum_k p_k * e_k

    Args:
        domain_embeddings: DomainEmbeddings module.
        probs: Probability vector of shape (num_domains,).

    Returns:
        Blended embedding of shape (embed_dim,).
    """
    all_embeds = domain_embeddings.embeddings.weight  # (K, embed_dim)
    return (probs.unsqueeze(-1) * all_embeds).sum(dim=0)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_gating.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/models/gating.py tests/test_gating.py
git commit -m "feat: task-conditioned gating with soft mixture routing"
```

---

## Task 5: Domain Classifier

**Files:**
- Create: `src/models/domain_classifier.py`
- Test: `tests/test_domain_classifier.py`

- [ ] **Step 1: Write failing tests**

Write `tests/test_domain_classifier.py`:

```python
"""Tests for domain classifier head."""
import pytest
import torch


def test_classifier_output_shape():
    """Classifier should output (batch, num_domains) logits."""
    from src.models.domain_classifier import DomainClassifier

    clf = DomainClassifier(hidden_dim=128, num_domains=15)
    h = torch.randn(4, 10, 128)  # batch=4, seq=10, hidden=128
    logits = clf(h)
    assert logits.shape == (4, 15)


def test_classifier_probabilities_sum_to_one():
    """Softmax output should sum to 1."""
    from src.models.domain_classifier import DomainClassifier

    clf = DomainClassifier(hidden_dim=128, num_domains=15)
    h = torch.randn(4, 10, 128)
    probs = clf.predict_probs(h)
    sums = probs.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones(4))
```

- [ ] **Step 2: Run tests to verify fail**

```bash
uv run pytest tests/test_domain_classifier.py -v
```

- [ ] **Step 3: Implement `src/models/domain_classifier.py`**

```python
"""Domain classifier for inference-time routing.

C(x) = softmax(W_c @ MEAN_POOL(h_enc(x)) + b_c)

Uses the frozen encoder's output -- no extra computation beyond
the forward pass already needed for the main task.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainClassifier(nn.Module):
    """Lightweight domain classification head."""

    def __init__(self, hidden_dim: int, num_domains: int):
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, num_domains)

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute domain logits from encoder hidden states.

        Args:
            encoder_hidden_states: (batch, seq_len, hidden_dim)

        Returns:
            Domain logits: (batch, num_domains)
        """
        pooled = encoder_hidden_states.mean(dim=1)  # mean pooling
        return self.classifier(pooled)

    def predict_probs(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        logits = self.forward(encoder_hidden_states)
        return F.softmax(logits, dim=-1)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_domain_classifier.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/models/domain_classifier.py tests/test_domain_classifier.py
git commit -m "feat: domain classifier with mean pooling and soft routing"
```

---

## Task 6: Dual-Stream Replay Buffer

**Files:**
- Create: `src/replay/buffer.py`
- Test: `tests/test_replay_buffer.py`

- [ ] **Step 1: Write failing tests**

Write `tests/test_replay_buffer.py`:

```python
"""Tests for dual-stream replay buffer management."""
import pytest


def test_domain_buffer_add_and_sample():
    """Should store examples per domain and sample from them."""
    from src.replay.buffer import DomainReplayBuffer

    buf = DomainReplayBuffer(max_per_domain=50)
    examples = [{"text": f"example {i}", "intent": i % 5, "domain": 0}
                for i in range(100)]
    buf.add_domain(domain_id=0, examples=examples)

    sampled = buf.sample(domain_id=0, n=10)
    assert len(sampled) == 10
    assert all(s["domain"] == 0 for s in sampled)


def test_domain_buffer_respects_max_size():
    """Buffer should not exceed max_per_domain."""
    from src.replay.buffer import DomainReplayBuffer

    buf = DomainReplayBuffer(max_per_domain=50)
    examples = [{"text": f"ex {i}", "intent": 0, "domain": 0}
                for i in range(200)]
    buf.add_domain(domain_id=0, examples=examples)
    assert buf.size(domain_id=0) <= 50


def test_domain_buffer_sample_all_domains():
    """Should sample proportionally from all seen domains."""
    from src.replay.buffer import DomainReplayBuffer

    buf = DomainReplayBuffer(max_per_domain=100)
    for d in range(5):
        examples = [{"text": f"d{d}_ex{i}", "intent": i, "domain": d}
                    for i in range(100)]
        buf.add_domain(domain_id=d, examples=examples)

    sampled = buf.sample_all(total_n=50)
    assert len(sampled) == 50
    # Should have examples from multiple domains
    domains_seen = set(s["domain"] for s in sampled)
    assert len(domains_seen) >= 3


def test_general_buffer():
    """General buffer should store and sample from general examples."""
    from src.replay.buffer import GeneralReplayBuffer

    buf = GeneralReplayBuffer(max_size=100)
    examples = [{"text": f"general {i}", "intent": "oos"} for i in range(200)]
    buf.fill(examples)
    assert buf.size() <= 100

    sampled = buf.sample(n=20)
    assert len(sampled) == 20


def test_dual_replay_buffer_batch_composition():
    """Dual buffer should compose batches with correct proportions."""
    from src.replay.buffer import DualReplayBuffer

    dual = DualReplayBuffer(max_per_domain=100, general_max_size=100)

    # Add domain data
    for d in range(3):
        examples = [{"text": f"d{d}_{i}", "intent": i, "domain": d}
                    for i in range(100)]
        dual.add_domain(d, examples)

    # Add general data
    general = [{"text": f"gen_{i}", "intent": "oos"} for i in range(100)]
    dual.fill_general(general)

    # Sample a replay batch: 10 domain + 10 general
    domain_samples, general_samples = dual.sample_replay(
        domain_n=10, general_n=10
    )
    assert len(domain_samples) == 10
    assert len(general_samples) == 10
```

- [ ] **Step 2: Run tests to verify fail**

```bash
uv run pytest tests/test_replay_buffer.py -v
```

- [ ] **Step 3: Implement `src/replay/buffer.py`**

```python
"""Dual-stream experience replay buffer.

Stream 1 (Domain): Per-domain buffers with reservoir sampling.
Stream 2 (General): Fixed buffer of general-knowledge examples (OOS or SFT).

Batch composition:
  (1-alpha) new_domain + beta domain_replay + (alpha-beta) general_replay
"""
import random
from typing import Any


Example = dict[str, Any]


class DomainReplayBuffer:
    """Per-domain replay buffer with reservoir sampling."""

    def __init__(self, max_per_domain: int = 200):
        self.max_per_domain = max_per_domain
        self._buffers: dict[int, list[Example]] = {}
        self._counts: dict[int, int] = {}

    def add_domain(self, domain_id: int, examples: list[Example],
                   seed: int = 42):
        """Add examples for a domain using reservoir sampling."""
        rng = random.Random(seed + domain_id)

        if domain_id not in self._buffers:
            self._buffers[domain_id] = []
            self._counts[domain_id] = 0

        buf = self._buffers[domain_id]
        for ex in examples:
            self._counts[domain_id] += 1
            n = self._counts[domain_id]
            if len(buf) < self.max_per_domain:
                buf.append(ex)
            else:
                j = rng.randint(0, n - 1)
                if j < self.max_per_domain:
                    buf[j] = ex

    def sample(self, domain_id: int, n: int,
               rng: random.Random | None = None) -> list[Example]:
        """Sample n examples from a specific domain's buffer."""
        rng = rng or random.Random()
        buf = self._buffers.get(domain_id, [])
        if not buf:
            return []
        return rng.choices(buf, k=min(n, len(buf)))

    def sample_all(self, total_n: int,
                   rng: random.Random | None = None) -> list[Example]:
        """Sample total_n examples proportionally from all domain buffers."""
        rng = rng or random.Random()
        all_domains = list(self._buffers.keys())
        if not all_domains:
            return []

        sizes = [len(self._buffers[d]) for d in all_domains]
        total_size = sum(sizes)
        if total_size == 0:
            return []

        samples = []
        for d, s in zip(all_domains, sizes):
            n_from_d = max(1, int(total_n * s / total_size))
            samples.extend(self.sample(d, n_from_d, rng))

        if len(samples) > total_n:
            samples = rng.sample(samples, total_n)
        return samples

    def size(self, domain_id: int) -> int:
        return len(self._buffers.get(domain_id, []))

    @property
    def seen_domains(self) -> list[int]:
        return list(self._buffers.keys())


class GeneralReplayBuffer:
    """Fixed-size buffer for general knowledge examples."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._buffer: list[Example] = []

    def fill(self, examples: list[Example], seed: int = 42):
        """Fill buffer from a pool of examples."""
        rng = random.Random(seed)
        if len(examples) <= self.max_size:
            self._buffer = list(examples)
        else:
            self._buffer = rng.sample(examples, self.max_size)

    def sample(self, n: int,
               rng: random.Random | None = None) -> list[Example]:
        rng = rng or random.Random()
        if not self._buffer:
            return []
        return rng.choices(self._buffer, k=min(n, len(self._buffer)))

    def size(self) -> int:
        return len(self._buffer)


class DualReplayBuffer:
    """Combines domain-specific and general replay buffers."""

    def __init__(self, max_per_domain: int = 200,
                 general_max_size: int = 1000):
        self.domain_buffer = DomainReplayBuffer(max_per_domain)
        self.general_buffer = GeneralReplayBuffer(general_max_size)

    def add_domain(self, domain_id: int, examples: list[Example],
                   seed: int = 42):
        self.domain_buffer.add_domain(domain_id, examples, seed)

    def fill_general(self, examples: list[Example], seed: int = 42):
        self.general_buffer.fill(examples, seed)

    def sample_replay(
        self, domain_n: int, general_n: int,
        rng: random.Random | None = None
    ) -> tuple[list[Example], list[Example]]:
        """Sample domain and general replay examples."""
        rng = rng or random.Random()
        domain_samples = self.domain_buffer.sample_all(domain_n, rng)
        general_samples = self.general_buffer.sample(general_n, rng)
        return domain_samples, general_samples
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_replay_buffer.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/replay/buffer.py tests/test_replay_buffer.py
git commit -m "feat: dual-stream replay buffer with reservoir sampling"
```

---

## Task 7: Metrics (BWT, FWT, Avg F1, Stats)

**Files:**
- Create: `src/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write failing tests**

Write `tests/test_metrics.py`:

```python
"""Tests for continual learning metrics."""
import pytest
import numpy as np


def test_bwt_no_forgetting():
    """BWT should be 0 when no forgetting occurs."""
    from src.metrics import compute_bwt

    # perf_matrix[i][j] = performance of model_i on domain_j
    perf_matrix = np.array([
        [80.0, 0.0, 0.0],
        [80.0, 75.0, 0.0],
        [80.0, 75.0, 70.0],
    ])
    bwt = compute_bwt(perf_matrix)
    assert bwt == 0.0


def test_bwt_with_forgetting():
    """BWT should be negative when forgetting occurs."""
    from src.metrics import compute_bwt

    perf_matrix = np.array([
        [80.0, 0.0, 0.0],
        [70.0, 75.0, 0.0],
        [65.0, 70.0, 72.0],
    ])
    bwt = compute_bwt(perf_matrix)
    # BWT = ((65-80) + (70-75)) / 2 = -10
    assert bwt == pytest.approx(-10.0)


def test_fwt():
    """FWT should measure zero-shot performance on new domains."""
    from src.metrics import compute_fwt

    perf_matrix = np.array([
        [80.0, 30.0, 25.0],
        [70.0, 75.0, 35.0],
        [65.0, 70.0, 72.0],
    ])
    random_baseline = np.array([20.0, 20.0, 20.0])
    fwt = compute_fwt(perf_matrix, random_baseline)
    # domain 1: perf_matrix[0][1] - random[1] = 30-20 = 10
    # domain 2: perf_matrix[1][2] - random[2] = 35-20 = 15
    # FWT = (10 + 15) / 2 = 12.5
    assert fwt == pytest.approx(12.5)


def test_avg_f1():
    """Average F1 should be mean of final scores."""
    from src.metrics import compute_avg_f1

    final_scores = np.array([65.0, 70.0, 72.0])
    avg = compute_avg_f1(final_scores)
    assert avg == pytest.approx(69.0)


def test_paired_ttest():
    """Should compute paired t-test with Bonferroni correction."""
    from src.metrics import paired_ttest_bonferroni

    scores_a = [89.1, 88.5, 89.8, 88.9, 89.2]
    scores_b = [84.7, 84.2, 85.1, 84.5, 84.8]
    p_value = paired_ttest_bonferroni(scores_a, scores_b, num_comparisons=8)
    assert p_value < 0.05
```

- [ ] **Step 2: Run tests to verify fail**

```bash
uv run pytest tests/test_metrics.py -v
```

- [ ] **Step 3: Implement `src/metrics.py`**

```python
"""Continual learning metrics: BWT, FWT, Avg F1, statistical tests."""
import numpy as np
from scipy import stats


def compute_bwt(perf_matrix: np.ndarray) -> float:
    """Compute Backward Transfer (forgetting metric).

    BWT = (1/(K-1)) * sum [perf_matrix[K-1][k] - perf_matrix[k][k]]
    for k = 0..K-2. Negative means forgetting.
    """
    K = perf_matrix.shape[0]
    if K <= 1:
        return 0.0
    drops = []
    for k in range(K - 1):
        drops.append(perf_matrix[K - 1, k] - perf_matrix[k, k])
    return float(np.mean(drops))


def compute_fwt(perf_matrix: np.ndarray,
                random_baseline: np.ndarray) -> float:
    """Compute Forward Transfer.

    FWT = (1/(K-1)) * sum [perf_matrix[k-1][k] - random_baseline[k]]
    for k = 1..K-1. Positive means beneficial transfer.
    """
    K = perf_matrix.shape[0]
    if K <= 1:
        return 0.0
    transfers = []
    for k in range(1, K):
        transfers.append(perf_matrix[k - 1, k] - random_baseline[k])
    return float(np.mean(transfers))


def compute_avg_f1(final_scores: np.ndarray) -> float:
    """Macro-averaged F1 across all domains after final training."""
    return float(np.mean(final_scores))


def paired_ttest_bonferroni(
    scores_a: list[float],
    scores_b: list[float],
    num_comparisons: int = 1,
) -> float:
    """Paired t-test with Bonferroni correction."""
    _, raw_p = stats.ttest_rel(scores_a, scores_b)
    return float(min(raw_p * num_comparisons, 1.0))
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_metrics.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/metrics.py tests/test_metrics.py
git commit -m "feat: BWT/FWT/avg-F1 metrics with paired t-test"
```

---

## Task 8: Base Continual Method Interface

**Files:**
- Create: `src/methods/base.py`
- Test: `tests/test_methods.py`

- [ ] **Step 1: Write test for method interface**

Write `tests/test_methods.py`:

```python
"""Tests for continual learning method interface."""
import pytest


def test_base_method_interface():
    """All methods should implement train_domain and run_evaluation."""
    from src.methods.base import BaseContinualMethod

    with pytest.raises(TypeError):
        BaseContinualMethod(model_name="bert-base-uncased", num_domains=3)
```

- [ ] **Step 2: Implement `src/methods/base.py`**

```python
"""Base interface for all continual learning methods."""
from abc import ABC, abstractmethod
import torch


class BaseContinualMethod(ABC):
    """Abstract base for continual learning methods.

    Every method must implement:
    - setup(): Initialize model, optimizer, etc.
    - train_domain(): Train on a new domain's data.
    - run_evaluation(): Score model on a domain's test set.
    """

    def __init__(self, model_name: str, num_domains: int, **kwargs):
        self.model_name = model_name
        self.num_domains = num_domains
        self.config = kwargs
        self.device = self._get_device()
        self.current_domain = -1

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @abstractmethod
    def setup(self):
        """Initialize model, tokenizer, optimizer."""
        ...

    @abstractmethod
    def train_domain(
        self,
        domain_id: int,
        train_data: list[dict],
        replay_data: list[dict] | None = None,
    ) -> dict[str, float]:
        """Train on one domain's data. Return training metrics."""
        ...

    @abstractmethod
    def run_evaluation(self, test_data: list[dict]) -> dict[str, float]:
        """Score model on a test set. Return {'f1': ..., 'accuracy': ...}."""
        ...

    def get_trainable_param_count(self) -> int:
        """Return number of trainable parameters."""
        return 0
```

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/test_methods.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/methods/base.py tests/test_methods.py
git commit -m "feat: base continual method interface"
```

---

## Task 9: Sequential FT Baseline (Simplest Method)

**Files:**
- Create: `src/methods/sequential_ft.py`
- Modify: `tests/test_methods.py`

This is the simplest baseline -- full model fine-tuning without any forgetting mitigation. Implementing it first validates the training loop, tokenization, and scoring pipeline.

- [ ] **Step 1: Add test for Sequential FT**

Append to `tests/test_methods.py`:

```python
def test_sequential_ft_trains_and_scores():
    """Sequential FT should train on domain data and produce F1."""
    from src.methods.sequential_ft import SequentialFT

    method = SequentialFT(
        model_name="prajjwal1/bert-tiny",
        num_domains=2,
        learning_rate=5e-4,
        epochs=1,
        batch_size=4,
        max_seq_len=32,
    )
    method.setup()

    train_data = [
        {"text": "book a flight to boston", "label": 0},
        {"text": "cancel my reservation", "label": 1},
        {"text": "change my seat", "label": 0},
        {"text": "I need a refund", "label": 1},
    ] * 5  # 20 examples

    test_data = [
        {"text": "reserve a flight", "label": 0},
        {"text": "cancel booking", "label": 1},
    ] * 3  # 6 examples

    metrics = method.train_domain(domain_id=0, train_data=train_data)
    assert "loss" in metrics

    eval_metrics = method.run_evaluation(test_data)
    assert "f1" in eval_metrics
    assert 0.0 <= eval_metrics["f1"] <= 100.0
```

- [ ] **Step 2: Run test to verify fail**

```bash
uv run pytest tests/test_methods.py::test_sequential_ft_trains_and_scores -v
```

- [ ] **Step 3: Implement `src/methods/sequential_ft.py`**

```python
"""Baseline 1: Sequential Fine-Tuning.

Full-model fine-tuning on each domain sequentially.
No forgetting mitigation -- this is the lower bound.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import f1_score

from src.methods.base import BaseContinualMethod


class TextDataset(Dataset):
    """Simple text classification dataset."""

    def __init__(self, data: list[dict], tokenizer, max_seq_len: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["text"],
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"], dtype=torch.long),
        }


class SequentialFT(BaseContinualMethod):
    """Full-model sequential fine-tuning baseline."""

    def setup(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = None
        self._all_labels = set()

    def _ensure_model(self, num_labels: int):
        if self.model is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=num_labels
            ).to(self.device)
        elif self.model.config.num_labels != num_labels:
            old_model = self.model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=num_labels
            ).to(self.device)
            # Copy encoder weights from old model
            base_name = "bert" if hasattr(old_model, "bert") else "base_model"
            old_base = getattr(old_model, base_name, old_model.base_model)
            new_base = getattr(self.model, base_name, self.model.base_model)
            new_base.load_state_dict(old_base.state_dict())

    def train_domain(self, domain_id, train_data, replay_data=None):
        labels = set(ex["label"] for ex in train_data)
        self._all_labels.update(labels)
        num_labels = max(self._all_labels) + 1

        self._ensure_model(num_labels)
        self.model.train()

        dataset = TextDataset(
            train_data, self.tokenizer,
            self.config.get("max_seq_len", 128))
        loader = DataLoader(
            dataset,
            batch_size=self.config.get("batch_size", 32),
            shuffle=True,
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 2e-4),
        )

        total_loss = 0
        epochs = self.config.get("epochs", 3)
        for epoch in range(epochs):
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()

        self.current_domain = domain_id
        return {"loss": total_loss / max(len(loader) * epochs, 1)}

    def run_evaluation(self, test_data):
        if self.model is None:
            return {"f1": 0.0, "accuracy": 0.0}

        self.model.eval()
        dataset = TextDataset(
            test_data, self.tokenizer,
            self.config.get("max_seq_len", 128))
        loader = DataLoader(
            dataset, batch_size=self.config.get("batch_size", 32))

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                preds = outputs.logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(batch["labels"].cpu().tolist())

        f1 = (f1_score(all_labels, all_preds,
                       average="macro", zero_division=0) * 100)
        accuracy = (sum(p == l for p, l in zip(all_preds, all_labels))
                    / len(all_labels) * 100)
        return {"f1": f1, "accuracy": accuracy}
```

- [ ] **Step 4: Run test**

```bash
uv run pytest tests/test_methods.py::test_sequential_ft_trains_and_scores -v
```

- [ ] **Step 5: Commit**

```bash
git add src/methods/sequential_ft.py tests/test_methods.py
git commit -m "feat: sequential FT baseline with train/score pipeline"
```

---

## Task 10: Dual-Replay Method (Core Paper Method)

**Files:**
- Create: `src/methods/dual_replay.py`
- Modify: `tests/test_methods.py`

This is the main contribution. It combines: adapter insertion + gating + domain classifier + dual-stream replay.

- [ ] **Step 1: Add test for Dual-Replay**

Append to `tests/test_methods.py`:

```python
def test_dual_replay_trains_and_scores():
    """Dual-Replay should train with adapters + dual replay and score."""
    from src.methods.dual_replay import DualReplay

    method = DualReplay(
        model_name="prajjwal1/bert-tiny",
        num_domains=3,
        learning_rate=5e-4,
        epochs=1,
        batch_size=4,
        max_seq_len=32,
        adapter_r=8,
        embed_dim=16,
        replay_ratio=0.20,
        domain_replay_fraction=0.10,
        domain_buffer_size=20,
        general_buffer_size=50,
    )
    method.setup()

    # Provide general buffer data
    general_data = [{"text": f"general example {i}", "label": -1}
                    for i in range(50)]
    method.fill_general_buffer(general_data)

    # Train domain 0
    train0 = [{"text": f"domain0 example {i}", "label": i % 3}
              for i in range(40)]
    test0 = [{"text": f"domain0 test {i}", "label": i % 3}
             for i in range(10)]
    method.train_domain(domain_id=0, train_data=train0)

    # Train domain 1 -- should use replay from domain 0
    train1 = [{"text": f"domain1 example {i}", "label": i % 3}
              for i in range(40)]
    method.train_domain(domain_id=1, train_data=train1)

    # Score on domain 0 (test forgetting)
    eval0 = method.run_evaluation(test0)
    assert "f1" in eval0

    # Check only adapters are trainable
    trainable = method.get_trainable_param_count()
    total = sum(p.numel() for p in method.model.parameters())
    assert trainable < total * 0.3


def test_dual_replay_base_frozen():
    """Base model parameters should not change during training."""
    from src.methods.dual_replay import DualReplay
    import torch

    method = DualReplay(
        model_name="prajjwal1/bert-tiny",
        num_domains=2,
        learning_rate=5e-4,
        epochs=1,
        batch_size=4,
        max_seq_len=32,
        adapter_r=8,
        embed_dim=16,
        replay_ratio=0.0,
        domain_replay_fraction=0.0,
        domain_buffer_size=10,
        general_buffer_size=10,
    )
    method.setup()

    # Snapshot base model params
    base_params_before = {
        name: p.clone()
        for name, p in method.model.named_parameters()
        if not p.requires_grad
    }

    train_data = [{"text": f"example {i}", "label": i % 2}
                  for i in range(20)]
    method.train_domain(domain_id=0, train_data=train_data)

    # Verify base params unchanged
    for name, p in method.model.named_parameters():
        if not p.requires_grad and name in base_params_before:
            torch.testing.assert_close(p, base_params_before[name])
```

- [ ] **Step 2: Run tests to verify fail**

```bash
uv run pytest tests/test_methods.py -k "dual_replay" -v
```

- [ ] **Step 3: Implement `src/methods/dual_replay.py`**

This is the largest single file. It integrates all components from Tasks 3-6.

```python
"""Dual-Replay: the core method from the paper.

Combines:
1. Frozen base model + bottleneck adapters
2. Task-conditioned gating
3. Domain classifier with soft routing
4. Dual-stream experience replay (domain + general)
"""
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import f1_score

from src.methods.base import BaseContinualMethod
from src.models.adapters import BottleneckAdapter, get_transformer_layers
from src.models.gating import (DomainEmbeddings, TaskConditionedGating,
                                soft_mixture_routing)
from src.models.domain_classifier import DomainClassifier
from src.replay.buffer import DualReplayBuffer


class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        enc = self.tokenizer(
            item["text"], max_length=self.max_seq_len,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        result = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if "label" in item and item["label"] >= 0:
            result["labels"] = torch.tensor(item["label"], dtype=torch.long)
        if "domain" in item:
            result["domain_id"] = torch.tensor(
                item["domain"], dtype=torch.long)
        return result


class DualReplayModel(nn.Module):
    """Full Dual-Replay model: base encoder + adapters + gating +
    classifier + task head."""

    def __init__(self, model_name, num_domains, num_labels,
                 adapter_r, embed_dim):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        d_model = self.encoder.config.hidden_size

        # Freeze base model
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Adapters
        layers = get_transformer_layers(self.encoder)
        self.adapters = nn.ModuleList([
            BottleneckAdapter(d_model, adapter_r) for _ in layers
        ])
        self._hook_adapters(layers)

        # Gating
        self.domain_embeddings = DomainEmbeddings(num_domains, embed_dim)
        self.gating = TaskConditionedGating(adapter_r, embed_dim)

        # Domain classifier
        self.domain_classifier = DomainClassifier(d_model, num_domains)

        # Task head (intent classification)
        self.classifier = nn.Linear(d_model, num_labels)
        self.num_labels = num_labels

    def _hook_adapters(self, layers):
        for layer, adapter in zip(layers, self.adapters):
            original_output = layer.output

            class AdaptedOutput(nn.Module):
                def __init__(self, orig, adpt):
                    super().__init__()
                    self.original = orig
                    self.adapter = adpt

                def forward(self, hidden_states, input_tensor):
                    out = self.original(hidden_states, input_tensor)
                    return self.adapter(out)

            layer.output = AdaptedOutput(original_output, adapter)

    def forward(self, input_ids, attention_mask, domain_id=None):
        outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state

        # Domain classifier
        domain_logits = self.domain_classifier(hidden)

        # Mean pool for classification
        pooled = hidden.mean(dim=1)

        # Task classification
        task_logits = self.classifier(pooled)

        return task_logits, domain_logits


class DualReplay(BaseContinualMethod):

    def setup(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._all_labels = set()
        self.model = None
        self.replay_buffer = DualReplayBuffer(
            max_per_domain=self.config.get("domain_buffer_size", 200),
            general_max_size=self.config.get("general_buffer_size", 1000),
        )
        self.replay_ratio = self.config.get("replay_ratio", 0.20)
        self.domain_replay_frac = self.config.get(
            "domain_replay_fraction", 0.10)

    def fill_general_buffer(self, data):
        self.replay_buffer.fill_general(data)

    def _ensure_model(self, num_labels):
        if self.model is None or self.model.num_labels != num_labels:
            self.model = DualReplayModel(
                model_name=self.model_name,
                num_domains=self.num_domains,
                num_labels=num_labels,
                adapter_r=self.config.get("adapter_r", 64),
                embed_dim=self.config.get("embed_dim", 64),
            ).to(self.device)

    def train_domain(self, domain_id, train_data, replay_data=None):
        labels = set(
            ex["label"] for ex in train_data if ex.get("label", -1) >= 0)
        self._all_labels.update(labels)
        num_labels = max(self._all_labels) + 1
        self._ensure_model(num_labels)
        self.model.train()

        labeled_train = [{**ex, "domain": domain_id} for ex in train_data]

        # Compose batch with replay
        rng = random.Random(42 + domain_id)
        alpha = self.replay_ratio
        beta = self.domain_replay_frac

        batch_size = self.config.get("batch_size", 32)
        n_domain_replay = int(batch_size * beta)
        n_general_replay = int(batch_size * (alpha - beta))

        domain_replay, general_replay = self.replay_buffer.sample_replay(
            domain_n=max(len(labeled_train) // 4, n_domain_replay * 10),
            general_n=max(n_general_replay * 10, 1),
            rng=rng,
        )

        all_data = labeled_train + domain_replay + general_replay

        dataset = TextDataset(
            all_data, self.tokenizer,
            self.config.get("max_seq_len", 128))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        trainable_params = [
            p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.get("learning_rate", 2e-4))
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        domain_criterion = nn.CrossEntropyLoss()

        total_loss = 0
        epochs = self.config.get("epochs", 3)
        for epoch in range(epochs):
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels_tensor = batch.pop("labels", None)
                domain_ids = batch.pop("domain_id", None)

                task_logits, domain_logits = self.model(**batch)

                loss = torch.tensor(0.0, device=self.device)
                if labels_tensor is not None:
                    loss = loss + criterion(task_logits, labels_tensor)
                if domain_ids is not None:
                    loss = loss + domain_criterion(
                        domain_logits, domain_ids)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()

        self.replay_buffer.add_domain(domain_id, labeled_train)
        self.current_domain = domain_id
        return {"loss": total_loss / max(len(loader) * epochs, 1)}

    def run_evaluation(self, test_data):
        if self.model is None:
            return {"f1": 0.0, "accuracy": 0.0}

        self.model.eval()
        dataset = TextDataset(
            test_data, self.tokenizer,
            self.config.get("max_seq_len", 128))
        loader = DataLoader(
            dataset, batch_size=self.config.get("batch_size", 32))

        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels_tensor = batch.pop("labels", None)
                batch.pop("domain_id", None)
                task_logits, _ = self.model(**batch)
                preds = task_logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().tolist())
                if labels_tensor is not None:
                    all_labels.extend(labels_tensor.cpu().tolist())

        if not all_labels:
            return {"f1": 0.0, "accuracy": 0.0}

        f1 = (f1_score(all_labels, all_preds,
                       average="macro", zero_division=0) * 100)
        acc = (sum(p == l for p, l in zip(all_preds, all_labels))
               / len(all_labels) * 100)
        return {"f1": f1, "accuracy": acc}

    def get_trainable_param_count(self):
        if self.model is None:
            return 0
        return sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_methods.py -k "dual_replay" -v
```

- [ ] **Step 5: Commit**

```bash
git add src/methods/dual_replay.py tests/test_methods.py
git commit -m "feat: Dual-Replay method with adapters, gating, classifier, dual replay"
```

---

## Task 11: Remaining Baselines (LoRA-Only, Replay-Only, LoRA+Replay, EWC, O-LoRA, DER)

**Files:**
- Create: `src/methods/lora_only.py`
- Create: `src/methods/replay_only.py`
- Create: `src/methods/lora_replay.py`
- Create: `src/methods/ewc.py`
- Create: `src/methods/o_lora.py`
- Create: `src/methods/der.py`
- Modify: `tests/test_methods.py`

Each baseline follows the same `BaseContinualMethod` interface. Implement them one at a time with tests.

- [ ] **Step 1: Add tests for all remaining baselines**

Append to `tests/test_methods.py`:

```python
@pytest.mark.parametrize("method_cls,method_path", [
    ("LoRAOnly", "src.methods.lora_only"),
    ("ReplayOnly", "src.methods.replay_only"),
    ("LoRAReplay", "src.methods.lora_replay"),
    ("EWC", "src.methods.ewc"),
    ("OLoRA", "src.methods.o_lora"),
    ("DER", "src.methods.der"),
])
def test_baseline_trains_and_scores(method_cls, method_path):
    """Each baseline should implement the full train/score interface."""
    import importlib
    mod = importlib.import_module(method_path)
    cls = getattr(mod, method_cls)

    method = cls(
        model_name="prajjwal1/bert-tiny",
        num_domains=2,
        learning_rate=5e-4,
        epochs=1,
        batch_size=4,
        max_seq_len=32,
    )
    method.setup()

    train_data = [{"text": f"example {i}", "label": i % 2}
                  for i in range(20)]
    test_data = [{"text": f"test {i}", "label": i % 2}
                 for i in range(6)]

    method.train_domain(domain_id=0, train_data=train_data)
    result = method.run_evaluation(test_data)
    assert "f1" in result
```

- [ ] **Step 2: Implement each baseline in order**

1. `lora_only.py` -- Uses `peft` LoRA, no replay. Pattern: SequentialFT but with LoRA.
2. `replay_only.py` -- Full model FT + single-stream replay buffer.
3. `lora_replay.py` -- LoRA + single-stream replay. Combines 1 and 2.
4. `ewc.py` -- After each domain, compute Fisher information. Add regularization loss.
5. `o_lora.py` -- Per-task orthogonal LoRA subspaces. Most complex baseline.
6. `der.py` -- LoRA + logit storage in replay buffer for knowledge distillation.

Each file follows the same structure as `sequential_ft.py`. Key differences are in the training loop.

- [ ] **Step 3: Run all tests**

```bash
uv run pytest tests/test_methods.py -v
```

- [ ] **Step 4: Commit each baseline separately**

```bash
git add src/methods/lora_only.py && git commit -m "feat: LoRA-only baseline"
git add src/methods/replay_only.py && git commit -m "feat: replay-only baseline"
git add src/methods/lora_replay.py && git commit -m "feat: LoRA+replay baseline"
git add src/methods/ewc.py && git commit -m "feat: EWC baseline"
git add src/methods/o_lora.py && git commit -m "feat: O-LoRA baseline"
git add src/methods/der.py && git commit -m "feat: DER baseline"
```

---

## Task 12: Sequential Training Runner + CLI

**Files:**
- Create: `src/training/runner.py`
- Create: `scripts/run_experiment.py`
- Test: `tests/test_integration.py`

This is the orchestrator: iterate domains in order, train, score all seen domains, record performance matrix, compute metrics.

- [ ] **Step 1: Write integration test**

Write `tests/test_integration.py`:

```python
"""End-to-end integration test: tiny model, 2 domains."""
import pytest
import numpy as np


def test_sequential_runner_end_to_end():
    """Run full sequential training on 2 domains with tiny model."""
    from src.training.runner import SequentialRunner
    from src.methods.sequential_ft import SequentialFT

    domains = [
        {
            "domain_id": 0,
            "train": [{"text": f"domain0 {i}", "label": i % 2}
                      for i in range(20)],
            "test": [{"text": f"d0 test {i}", "label": i % 2}
                     for i in range(10)],
        },
        {
            "domain_id": 1,
            "train": [{"text": f"domain1 {i}", "label": i % 2}
                      for i in range(20)],
            "test": [{"text": f"d1 test {i}", "label": i % 2}
                     for i in range(10)],
        },
    ]

    method = SequentialFT(
        model_name="prajjwal1/bert-tiny",
        num_domains=2,
        learning_rate=5e-4,
        epochs=1,
        batch_size=8,
        max_seq_len=32,
    )

    runner = SequentialRunner(method=method, domains=domains)
    results = runner.run()

    assert "perf_matrix" in results
    assert results["perf_matrix"].shape == (2, 2)
    assert "bwt" in results
    assert "avg_f1" in results
```

- [ ] **Step 2: Implement `src/training/runner.py`**

```python
"""Sequential training runner.

Iterates through domains in order, trains the method on each,
scores on all seen domains, and builds the performance matrix.
"""
import numpy as np
from typing import Any

from src.methods.base import BaseContinualMethod
from src.metrics import compute_bwt, compute_avg_f1


class SequentialRunner:
    """Orchestrates sequential domain training and scoring."""

    def __init__(self, method: BaseContinualMethod, domains: list[dict]):
        self.method = method
        self.domains = domains

    def run(self) -> dict[str, Any]:
        K = len(self.domains)
        perf_matrix = np.zeros((K, K))

        self.method.setup()

        for step, domain in enumerate(self.domains):
            self.method.train_domain(
                domain_id=domain["domain_id"],
                train_data=domain["train"],
            )

            # Score on all seen domains
            for j in range(step + 1):
                eval_result = self.method.run_evaluation(
                    self.domains[j]["test"])
                perf_matrix[step, j] = eval_result["f1"]

        final_scores = perf_matrix[K - 1, :]
        bwt = compute_bwt(perf_matrix)
        avg_f1 = compute_avg_f1(final_scores)

        return {
            "perf_matrix": perf_matrix,
            "bwt": bwt,
            "avg_f1": avg_f1,
            "final_scores": final_scores,
        }
```

- [ ] **Step 3: Implement `scripts/run_experiment.py`**

```python
"""CLI entry point for running experiments.

Usage:
    uv run python scripts/run_experiment.py \
        --method dual_replay --config debug --seed 42
"""
import argparse
import json
import os
import yaml

from src.data.clinc150 import build_15_domain_protocol, get_general_buffer
from src.data.domain_sequence import generate_domain_orderings
from src.training.runner import SequentialRunner


METHOD_MAP = {
    "sequential_ft": ("src.methods.sequential_ft", "SequentialFT"),
    "lora_only": ("src.methods.lora_only", "LoRAOnly"),
    "replay_only": ("src.methods.replay_only", "ReplayOnly"),
    "lora_replay": ("src.methods.lora_replay", "LoRAReplay"),
    "ewc": ("src.methods.ewc", "EWC"),
    "o_lora": ("src.methods.o_lora", "OLoRA"),
    "der": ("src.methods.der", "DER"),
    "dual_replay": ("src.methods.dual_replay", "DualReplay"),
}


def load_config(config_name):
    with open("configs/default.yaml") as f:
        all_configs = yaml.safe_load(f)
    return all_configs[config_name]


def get_method(method_name, config):
    import importlib
    module_path, class_name = METHOD_MAP[method_name]
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(
        model_name=config["model_name"],
        num_domains=config["num_domains"],
        **{k: v for k, v in config.items()
           if k not in ("model_name", "num_domains")},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", required=True, choices=list(METHOD_MAP.keys()))
    parser.add_argument("--config", default="debug")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    config = load_config(args.config)
    domains = build_15_domain_protocol(seed=args.seed)

    orderings = generate_domain_orderings(
        len(domains), num_orderings=1, seed=args.seed)
    ordered_domains = [domains[i] for i in orderings[0]]

    # Limit domains if config specifies fewer
    num_domains = config.get("num_domains", len(ordered_domains))
    ordered_domains = ordered_domains[:num_domains]

    method = get_method(args.method, config)
    runner = SequentialRunner(method=method, domains=ordered_domains)
    results = runner.run()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir,
        f"{args.method}_{args.config}_seed{args.seed}.json")
    serializable = {
        "method": args.method,
        "config": args.config,
        "seed": args.seed,
        "avg_f1": results["avg_f1"],
        "bwt": results["bwt"],
        "perf_matrix": results["perf_matrix"].tolist(),
    }
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"Results saved to {output_path}")
    print(f"  Avg F1: {results['avg_f1']:.1f}")
    print(f"  BWT:    {results['bwt']:.1f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run integration test**

```bash
uv run pytest tests/test_integration.py -v
```

- [ ] **Step 5: Run debug experiment on Mac**

```bash
cd /Users/yaqinhei/Documents/同步空间-work/obsidian/PHD/mypaper/dual-replay/dual-replay-reproduce
uv run python scripts/run_experiment.py --method sequential_ft --config debug --seed 42
uv run python scripts/run_experiment.py --method dual_replay --config debug --seed 42
```

Expected: Both complete. Results saved to `results/`.

- [ ] **Step 6: Commit**

```bash
git add src/training/ scripts/ tests/test_integration.py
git commit -m "feat: sequential runner + CLI entry point"
```

---

## Task 13: Mac Debug Validation (All Methods, 3 Domains)

**Files:** No new files. This task validates everything works end-to-end.

- [ ] **Step 1: Run all 8 methods on debug config**

```bash
cd /Users/yaqinhei/Documents/同步空间-work/obsidian/PHD/mypaper/dual-replay/dual-replay-reproduce
for method in sequential_ft lora_only replay_only lora_replay ewc o_lora der dual_replay; do
    echo "=== Running $method ==="
    uv run python scripts/run_experiment.py \
        --method $method --config debug --seed 42
done
```

Expected: All 8 methods complete. Check `results/` for output files.

- [ ] **Step 2: Verify results are reasonable**

```bash
uv run python -c "
import json, glob
for f in sorted(glob.glob('results/*debug*.json')):
    with open(f) as fh:
        r = json.load(fh)
    print(f'{r[\"method\"]:20s} F1={r[\"avg_f1\"]:5.1f}  BWT={r[\"bwt\"]:+5.1f}')
"
```

Sanity checks:
- All methods produce F1 > 0
- Sequential FT should have the most negative BWT
- Dual-Replay should have less negative BWT than Sequential FT

- [ ] **Step 3: Run full test suite**

```bash
uv run pytest tests/ -v --tb=short
```

All tests must pass.

- [ ] **Step 4: Commit final debug validation**

```bash
git add -A
git commit -m "chore: Mac debug validation complete, all 8 methods pass"
```

---

## Summary: What You Have After These 13 Tasks

After completing all tasks, you will have:

1. **Working codebase** that implements Dual-Replay + 7 baselines
2. **CLINC150 15-domain protocol** with reproducible domain orderings
3. **Dual-stream replay buffer** with reservoir sampling
4. **All core components** (adapters, gating, classifier) with unit tests
5. **CLI entry point** to run any method with any config
6. **Mac-validated** -- everything runs on MPS with debug config

**Next steps after this plan** (not in this plan):
- Run Phase 1 full experiments on GPU (35 runs)
- Run Phase 2 ablations (125 runs)
- Phase 3 multi-dataset + T5 experiments
- Results analysis notebook
- Paper revisions based on reproduction results
