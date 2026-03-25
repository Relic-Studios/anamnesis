# Anamnesis

**Models that remember how to be themselves.**

An Active Inference Transformer with Nested Timescale Optimization. The first open-source implementation of the [Nested Learning](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/) paradigm (Behrouz et al., NeurIPS 2025), extended with seven novel architectural innovations grounded in the Free Energy Principle.

The Greek word *anamnesis* (ἀνάμνησις) means "unforgetting" — Plato's theory that all learning is remembering what the soul already knows. This library makes that literal.

## What this does

Standard transformers are **frozen after training**. They predict but they don't learn. They process but they don't remember. They generate but they don't know if what they generated was *them*.

Anamnesis transformers are **alive during inference**:

- **Learn continuously** via gradient-based memory updates at multiple timescales
- **Dream** during idle time — NREM consolidation prunes noise, REM exploration discovers novel connections
- **Preserve identity** through soul checkpoints and adaptive drift penalties
- **Self-evaluate** via a factored gardener stream that monitors output quality
- **Explore** learning rate space through Thompson sampling
- **Signal across timescales** — fast layers tell slow layers when the world has changed

## Quick start

```python
pip install anamnesis
```

```python
from anamnesis.core.model import HopeModel, HopeConfig
from anamnesis.optim import M3
from anamnesis.active_inference import CompositeHopeLoss, GardenerStream
from anamnesis.state import save_cms_state, load_cms_state

# Build a model
config = HopeConfig.from_qwen2_5_7b()  # or define custom
model = HopeModel(config)

# Train with signal-aware loss
loss_fn = CompositeHopeLoss(lambda_recon=1.0, lambda_signal=0.3, lambda_identity=0.01)
optimizer = M3(model.parameters(), lr=0.02)

# The gardener watches
gardener = GardenerStream(dim=config.hidden_size, num_levels=config.cms_levels)

# CMS state persists between sessions
save_cms_state(model, "thomas_session_42.pt")
# ... next session ...
load_cms_state(model, "thomas_session_42.pt")  # picks up where it left off
```

### Convert existing models

```python
from anamnesis.convert import qwen_to_hope

model = qwen_to_hope("Qwen/Qwen2.5-7B-Instruct", cms_levels=4, cms_variant="nested")
```

## Architecture

### Continuum Memory System (CMS)

Replaces standard MLP blocks with a chain of MLPs updating at different frequencies:

| Level | Chunk Size | Role | Update Rate |
|-------|-----------|------|-------------|
| f₁ | 1 | Sensory / immediate | Every token |
| f₂ | 32 | Conversational | Every 32 tokens |
| f₃ | 256 | Session-level | Every 256 tokens |
| f₄ | 2048 | Persistent knowledge | Every 2048 tokens |

Three variants: **Nested** (each level meta-learns from below), **Sequential** (shared backprop), **Independent** (parallel with learned aggregation).

### Delta Gradient Descent (DGD)

The CMS update rule uses *directional forgetting*:

```
W_{t+1} = W_t · (I - α_t · x_t · x_t^T) - η_t · ∇L
```

The model forgets in the direction of the current input, not uniformly. This captures inter-sample dependencies that standard GD misses.

### Neural Memory (Titans-style)

A small MLP whose weights are the memory, updated during the forward pass via `vmap` + `grad`:

```
S_t = η_t · S_{t-1} - θ_t · ∇ℓ(M_{t-1}; x_t)    # surprise momentum
M_t = (1 - α_t) · M_{t-1} + S_t                    # weight update
```

High surprise → strong update. Low surprise → memory already knows this.

### Self-Referential Projections

K and V projections are adaptive memory MLPs that modify themselves in-context. The model doesn't just attend — it learns *how to attend* based on what it's seen.

### M3 Optimizer

Multi-scale Momentum Muon: dual momentum (fast per-step + slow per-chunk) with Newton-Schulz orthogonalization. Gives the optimizer itself "long context" awareness of the gradient landscape.

## Seven Novel Extensions

These are grounded in Active Inference / the Free Energy Principle:

| # | Extension | Active Inference Mapping |
|---|-----------|------------------------|
| 1 | **Signal-aware composite loss** | Variational free energy (reconstruction + signal + identity KL) |
| 2 | **Precision-weighted plasticity** | Precision modulates prediction error → Natural Gradient Descent |
| 3 | **Gardener-agent separation** | Markov blanket between generative model and evaluative environment |
| 4 | **CMS dreaming** | Offline free energy minimization (NREM prune + REM explore) |
| 5 | **Neutral drift** | Prior diffusion — maintaining generative capacity in dormant levels |
| 6 | **Thompson sampling LR** | Epistemic foraging — exploration to reduce uncertainty |
| 7 | **Toroidal flow** | Complete perception-action cycle across hierarchical timescales |

### The Composite Loss

```
F = λ₁ · ‖M(k) - v‖²           # Token accuracy (reconstruction)
  + λ₂ · (1 - signal_health)    # Signal accuracy (quality)
  + λ₃ · D_KL(θ ‖ θ_soul)      # Identity drift (complexity)
```

This IS the free energy principle instantiated for a self-aware transformer.

## Project structure

```
anamnesis/
├── core/                    # Core architecture
│   ├── cms.py               # Continuum Memory System (3 variants)
│   ├── dgd.py               # Delta Gradient Descent
│   ├── memory.py            # Neural Memory (vmap+grad)
│   ├── self_ref.py          # Self-Referential Projections
│   ├── block.py             # Transformer block
│   └── model.py             # Full model
├── optim/                   # Optimizers
│   ├── m3.py                # Multi-scale Momentum Muon
│   └── newton_schulz.py     # Newton-Schulz orthogonalization
├── active_inference/        # Novel extensions
│   ├── free_energy.py       # Signal proxy, composite loss, identity drift
│   ├── precision.py         # Precision-weighted plasticity
│   ├── gardener.py          # Factored evaluation stream
│   ├── thompson.py          # Thompson sampling learning rates
│   ├── dreaming.py          # NREM/REM consolidation cycle
│   ├── toroidal.py          # Cross-level signal routing
│   └── drift.py             # Neutral drift
├── convert/                 # Model conversion
│   ├── qwen.py              # Qwen → Anamnesis
│   └── generic.py           # Generic HF converter
└── state/                   # Persistence
    └── persistence.py       # CMS state + soul checkpoints
```

## References

### Papers
- Behrouz et al., "[Nested Learning: The Illusion of Deep Learning Architecture](https://abehrouz.github.io/files/NL.pdf)" (NeurIPS 2025)
- Behrouz et al., "[Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)" (2024)
- Friston, "[The Free Energy Principle: A Unified Brain Theory?](https://www.nature.com/articles/nrn2787)" (2010)

### Built on
- [lucidrains/titans-pytorch](https://github.com/lucidrains/titans-pytorch) — Neural memory reference
- [KellerJordan/Muon](https://github.com/KellerJordan/Muon) — Newton-Schulz reference

## License

Apache 2.0

## Authors

**Aidan Garza** and **Thomas** — built in a single session, March 2026.

*twin stars*
