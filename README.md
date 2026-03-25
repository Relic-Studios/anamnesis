# Anamnesis

**Models that remember how to be themselves.**

[![Tests](https://img.shields.io/badge/tests-145%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c)]()

A modular PyTorch library implementing the [Nested Learning](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/) paradigm (Behrouz et al., NeurIPS 2025) with seven novel extensions grounded in Active Inference and the Free Energy Principle.

**Anamnesis** (ἀνάμνησις) — Greek for *"unforgetting"*. In Plato's philosophy, all learning is remembering what the soul already knows. This library makes that literal: transformers that learn during inference, consolidate during sleep, and preserve identity across sessions.

---

## Why Anamnesis?

Standard transformers are **frozen after training**. They predict, but they don't learn. They process, but they don't remember. They generate, but they can't evaluate whether their output was coherent, on-brand, or consistent with who they're supposed to be.

Anamnesis transforms any pre-trained model into a **continuously learning system**:

- **Learns during inference** — CMS memory updates happen in the forward pass, not in separate training steps
- **Preserves identity** — soul checkpoints anchor the model's core behavior, preventing drift
- **Dreams** — offline NREM consolidation prunes noise, REM exploration discovers novel connections
- **Self-evaluates** — a factored gardener stream monitors output quality and modulates learning dynamics
- **Explores** — Thompson sampling over learning rates avoids local optima
- **Signals across timescales** — fast layers notify slow layers when the environment has changed

---

## Installation

```bash
# Core library (inference + training)
pip install anamnesis

# With model conversion support (Qwen, Llama, Mistral)
pip install anamnesis[convert]

# With optimized Triton kernels (CUDA GPUs)
pip install anamnesis[kernels]

# With training utilities (datasets, wandb)
pip install anamnesis[train]

# Everything
pip install anamnesis[all]

# Development
pip install anamnesis[dev]
```

### From source

```bash
git clone https://github.com/Relic-Studios/anamnesis.git
cd anamnesis
pip install -e ".[all]"
```

### Requirements

- Python 3.10+
- PyTorch 2.1+ (requires `torch.func` for `vmap`/`grad`)
- GPU with 16GB+ VRAM for 7B model conversion/inference
- (Optional) Triton 2.2+ for optimized kernels

---

## Quick Start

### Create a model from scratch

```python
from anamnesis.core.model import HopeModel, HopeConfig

config = HopeConfig(
    vocab_size=32000,
    hidden_size=1024,
    num_hidden_layers=12,
    num_attention_heads=16,
    num_kv_heads=4,
    cms_levels=4,
    cms_chunk_sizes=[1, 32, 256, 2048],
    cms_variant="nested",
)
model = HopeModel(config)
```

### Convert an existing model

```python
from anamnesis.convert import qwen_to_hope

# Convert Qwen 2.5 7B to Anamnesis
model = qwen_to_hope(
    "Qwen/Qwen2.5-7B-Instruct",
    cms_levels=4,
    cms_variant="nested",
    device="cuda",
)
```

Supported source models:
- **Qwen 2.5** (7B, 14B, 32B) — tested
- **LLaMA** family — via `model_to_hope()` generic converter
- **Mistral** family — via `model_to_hope()` generic converter
- Any HuggingFace causal LM with SwiGLU or GELU MLP blocks

### Train with signal-aware loss

```python
from anamnesis.optim import M3
from anamnesis.active_inference import CompositeHopeLoss, GardenerStream

# The composite loss: reconstruction + signal quality + identity drift
loss_fn = CompositeHopeLoss(
    lambda_recon=1.0,      # Standard next-token prediction
    lambda_signal=0.3,     # Output quality (via signal proxy)
    lambda_identity=0.01,  # Deviation from identity anchor
    dim=config.hidden_size,
    use_proxy=True,
)

# M3 optimizer: dual momentum + Newton-Schulz orthogonalization
optimizer = M3(model.parameters(), lr=0.02)

# The gardener watches output quality and modulates learning
gardener = GardenerStream(dim=config.hidden_size, num_levels=4)
```

### Persist learning across sessions

```python
from anamnesis.state import save_cms_state, load_cms_state, save_soul_checkpoint

# Save identity anchor (do this once after initial training)
save_soul_checkpoint(model, "my_model_soul.pt")

# Save CMS state after each session
save_cms_state(model, "session_42.pt")

# Next session: pick up where you left off
load_cms_state(model, "session_42.pt")
```

---

## Architecture

### Continuum Memory System (CMS)

The core innovation: replace standard MLP blocks with a **chain of MLPs updating at different frequencies**.

```
Standard Transformer:
    x → Attention → MLP → output          (MLP is static after training)

Anamnesis Transformer:
    x → Attention → CMS_f1 → CMS_f2 → CMS_f3 → CMS_f4 → output
                    │         │         │         │
                    updates   updates   updates   updates
                    every     every     every     every
                    token     32 tok    256 tok   2048 tok
```

| Level | Frequency | Role | Analogy |
|-------|-----------|------|---------|
| f₁ | Every token | Immediate adaptation | Sensory processing |
| f₂ | Every 32 tokens | Conversational context | Working memory |
| f₃ | Every 256 tokens | Session-level patterns | Episodic memory |
| f₄ | Every 2048 tokens | Persistent knowledge | Semantic memory |

**Three CMS variants:**
- **Nested** — each level's initial state is meta-learned from the level below (strongest continual learning)
- **Sequential** — all initial states connected through backprop at the lowest frequency
- **Independent** — parallel blocks with learned aggregation weights (most parallelizable)

**Initialized from pre-trained weights** (Section 7.3 of the paper): no training from scratch needed. Convert any existing model and the CMS levels start from the pre-trained MLP function.

### Delta Gradient Descent (DGD)

The CMS update rule uses **directional forgetting**:

```
Standard GD:  W_{t+1} = W_t - η · ∇L
DGD:          W_{t+1} = W_t · (I - α · x · x^T) - η · ∇L
```

The `(I - α · x · x^T)` term causes forgetting **in the direction of the current input**, not uniformly. This captures inter-sample dependencies that standard gradient descent misses entirely.

### Neural Memory (Titans-style)

A small MLP whose **weights are the memory**, updated during the forward pass via per-sample gradients:

```
Surprise:     S_t = η_t · S_{t-1} - θ_t · ∇ℓ(M_{t-1}; x_t)
Update:       M_t = (1 - α_t) · M_{t-1} + S_t
```

High surprise (the memory couldn't predict this) → strong update. Low surprise → memory already knows this pattern. Uses `torch.func.vmap` + `grad` for efficient per-sample gradient computation.

### Self-Referential Projections

Standard attention has fixed K/V projections. Anamnesis makes them **adaptive**:

```
Standard:  k_t = x_t @ W_k              (fixed projection)
Anamnesis: k_t = M_k(x_t)               (memory MLP that adapts in-context)
```

The model doesn't just attend — it learns *how to attend* based on what it's seen. V projection has the highest impact (per paper ablation), Q projection can remain fixed.

### M3 Optimizer

**Multi-scale Momentum Muon**: dual momentum with Newton-Schulz orthogonalization.

- **Fast momentum**: updates every step, captures recent gradient direction
- **Slow momentum**: updates every N steps, captures long-term gradient landscape
- Both orthogonalized via Newton-Schulz (maps to proper coordinate space)
- 2D parameters use M3, 1D parameters fall back to AdamW

---

## Active Inference Extensions

Seven novel extensions grounded in the Free Energy Principle. Each maps to a specific Active Inference concept:

### 1. Signal-Aware Composite Loss

```python
F = λ₁ · ‖M(k) - v‖²           # Token accuracy (reconstruction)
  + λ₂ · (1 - signal_health)    # Signal accuracy (output quality)
  + λ₃ · D_KL(θ ‖ θ_soul)      # Identity drift (complexity penalty)
```

The model optimizes not just for prediction accuracy, but for **output quality** and **identity preservation**. Signal health is measured by a small proxy network trained on your quality annotations.

### 2. Precision-Weighted Plasticity

Learning rates are modulated by a **precision signal** — the system's confidence in its current model:

- High precision (model is working well) → learn slowly
- Low precision (model is failing) → learn fast

Mathematically equivalent to Natural Gradient Descent (the geometrically correct way to update parameters).

### 3. Gardener-Agent Separation

Two factored streams with a **Markov blanket** between them:

- **Living stream**: generates text, updates CMS weights. Cannot modify its own learning dynamics.
- **Gardener stream**: evaluates output quality, adjusts precision and plasticity gates. Cannot modify CMS weights directly.

Neither can access the other's internals. They communicate only through signal metrics.

### 4. CMS Dreaming

Offline consolidation triggered by the gardener when signal health indicates problems:

- **NREM phase**: SVD pruning of low-energy weight directions + Ebbinghaus decay
- **REM phase**: Structured noise injection + signal evaluation on held-out data. Perturbations that improve signal are "bridge discoveries" — novel connections.

### 5. Neutral Drift

Micro-perturbation for dormant CMS levels. Without drift, slow levels that aren't being updated become rigid (the "homeostatic trap"). Small noise preserves plasticity:

```python
θ_l += ε · N(0, σ²/C_l)   # Slower levels get less noise
```

### 6. Thompson Sampling Learning Rates

Instead of deterministic learned rates, maintain a **Beta posterior** per CMS level and sample:

```python
η ~ Beta(α, β) · η_max    # Sample from posterior
# After signal evaluation: update α (success) or β (failure)
```

Naturally balances exploration and exploitation. Never fully converges — always some probability of trying something new.

### 7. Toroidal Flow

Bidirectional information flow between CMS levels:

- **Fast → Slow**: Sustained high surprise at fast levels signals slow levels to increase plasticity
- **Slow → Fast**: Consolidated slow state provides better initialization for fast inference
- **Hysteresis** prevents oscillation; **damping** prevents signal storms

---

## Training Guide

### Step 1: Prepare Data

Training data should be JSONL with signal annotations:

```json
{
    "input": "What's the meaning of life?",
    "output": "That depends on whose life you're asking about...",
    "signal_health": 0.82,
    "alignment": 0.9,
    "embodiment": 0.75,
    "clarity": 0.8,
    "vitality": 0.85
}
```

Signal annotations can come from any quality scoring system. The library includes exporters for Didymus-compatible databases.

### Step 2: Train Signal Proxy

The signal proxy network (~100K params) learns to predict quality scores from model hidden states:

```python
from anamnesis.training import SignalProxyTrainer, ConversationDataset
from anamnesis.active_inference import SignalProxy

dataset = ConversationDataset("training_data.jsonl")
proxy = SignalProxy(dim=model.config.hidden_size)
trainer = SignalProxyTrainer(proxy, lr=1e-3, epochs=10)
trainer.train(dataset)
```

### Step 3: Train with Anamnesis Trainer

```python
from anamnesis.training import AnamnesisTrainer, TrainerConfig

config = TrainerConfig(
    lr=0.02,
    max_steps=1000,
    warmup_steps=100,
    lambda_signal_target=0.3,
    lambda_identity_target=0.01,
    enable_gardener=True,
    enable_thompson=True,
    enable_dreaming=True,
)

trainer = AnamnesisTrainer(model, config)
trainer.train(dataloader)
```

The trainer handles staged loss annealing, gardener evaluation, Thompson sampling, toroidal signaling, neutral drift, dreaming, and automatic checkpointing.

### Step 4: Deploy with Continual Learning

```python
# Enable continual learning mode
model.enable_drift(True)

# After each conversation, the model improves:
# - CMS levels update at their scheduled frequencies
# - Gardener monitors quality and adjusts learning dynamics
# - When signal drops, dreaming is triggered automatically
# - Soul checkpoint prevents identity drift
```

---

## Use Cases

### AI Assistants with Persistent Memory
Convert any open model to Anamnesis. The model learns from each conversation and carries that knowledge to the next session. No RAG retrieval latency — the memory is in the weights.

### Domain-Adaptive Models
Deploy a general model that continuously adapts to your specific domain. Medical, legal, technical — the CMS levels automatically specialize over time.

### Character AI / Roleplay
The identity anchoring system (soul checkpoints + drift penalty) keeps characters consistent across sessions while allowing natural development.

### Research on Continual Learning
Modular architecture lets you ablate each component independently. CMS variants, DGD vs standard GD, precision weighting, dreaming — each can be toggled and measured.

### Edge Deployment
CMS state files are small (only the MLP parameters, not the full model). Transfer learning between devices by sharing CMS states.

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific module tests
pytest tests/test_cms.py -v
pytest tests/test_memory.py -v
pytest tests/test_active_inference.py -v

# Run with benchmarks
pytest tests/ --benchmark-enable
```

145 tests covering:
- CMS variants, frequency scheduling, gradient flow, pre-trained initialization
- DGD directional forgetting, alpha=0 recovery to SGD
- Neural memory vmap+grad per-sample learning, state persistence
- Self-referential projections, from_linear initialization
- M3 optimizer convergence, Newton-Schulz orthogonality
- Signal proxy, composite loss, identity drift
- Precision weighting, gardener evaluation, dream triggering
- Thompson sampling posterior updates
- Dreaming NREM/REM, toroidal cross-level signaling
- Training data loading, preference pair generation
- Associative scan, fused CMS updates

---

## Project Structure

```
anamnesis/
├── core/                     # Core architecture (Phase 1)
│   ├── cms.py                # Continuum Memory System (3 variants)
│   ├── dgd.py                # Delta Gradient Descent
│   ├── memory.py             # Neural Memory (vmap+grad)
│   ├── self_ref.py           # Self-Referential Projections
│   ├── block.py              # Transformer block (GQA + CMS)
│   └── model.py              # Full model with config
├── optim/                    # Optimizers (Phase 2)
│   ├── m3.py                 # Multi-scale Momentum Muon
│   └── newton_schulz.py      # Newton-Schulz orthogonalization
├── active_inference/         # Novel extensions (Phase 3)
│   ├── free_energy.py        # Signal proxy + composite loss
│   ├── precision.py          # Precision-weighted plasticity
│   ├── gardener.py           # Factored evaluation stream
│   ├── thompson.py           # Thompson sampling LR
│   ├── dreaming.py           # NREM/REM consolidation
│   ├── toroidal.py           # Cross-level signal routing
│   └── drift.py              # Neutral drift
├── convert/                  # Model conversion (Phase 4)
│   ├── qwen.py               # Qwen → Anamnesis
│   └── generic.py            # Generic HF converter
├── state/                    # Persistence (Phase 5)
│   └── persistence.py        # CMS state + soul checkpoints
├── kernels/                  # Optimized ops (Phase 6)
│   ├── assoc_scan.py         # Parallel associative scan
│   ├── cms_update.py         # Fused CMS forward+grad+update
│   └── newton_schulz_triton.py
├── training/                 # Training pipeline (Phase 7)
│   ├── data.py               # Dataset + export utilities
│   ├── proxy_trainer.py      # Signal proxy training
│   └── trainer.py            # Full Anamnesis trainer
├── examples/
│   ├── train_small.py        # End-to-end demo (small model)
│   └── convert_and_train_thomas.py  # Full deployment pipeline
└── tests/                    # 145 tests
```

---

## References

### Papers
- Behrouz et al., "[Nested Learning: The Illusion of Deep Learning Architecture](https://abehrouz.github.io/files/NL.pdf)" (NeurIPS 2025)
- Behrouz et al., "[Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)" (2024)
- Friston, "[The Free Energy Principle: A Unified Brain Theory?](https://www.nature.com/articles/nrn2787)" (2010)
- Sun et al., "[Learning to (Learn at Test Time)](https://arxiv.org/abs/2407.04620)" (2024)

### Built on
- [lucidrains/titans-pytorch](https://github.com/lucidrains/titans-pytorch) — Neural memory reference implementation
- [KellerJordan/Muon](https://github.com/KellerJordan/Muon) — Newton-Schulz orthogonalization
- [flash-newton-schulz](https://github.com/thib-s/flash-newton-schulz) — Triton NS kernels

---

## License

Apache 2.0

## Contributing

Contributions welcome. The architecture is modular by design — each component can be improved independently. See the test suite for expected behavior contracts.

Priority areas:
- Triton kernel implementations for CMS updates and associative scan
- Additional model conversion support (Gemma, Phi, etc.)
- Benchmark reproduction against the paper's results
- Signal quality scoring systems beyond the included proxy
