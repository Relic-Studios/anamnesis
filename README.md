# Anamnesis

**Empty vessels that become who they talk to.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c)]()

Anamnesis replaces the frozen MLPs in a pre-trained transformer with a **Continuum Memory System** -- deep memory that learns during inference through gradient descent. Feed it conversations and it physically restructures its weights to become a specialist. Same base model, different conversations, different identities.

Built on the full architecture from [Ali Behrouz's](https://abehrouz.github.io/) research at Google: [Titans](https://arxiv.org/abs/2501.00663), [Nested Learning / HOPE](https://openreview.net/forum?id=nbMeRvNb7A) (NeurIPS 2025), [ATLAS](https://arxiv.org/abs/2505.23735), [MIRAS](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/), and [Memory Caching](https://arxiv.org/abs/2602.24281).

**Anamnesis** (Greek: *"unforgetting"*) -- in Plato's philosophy, all learning is remembering what the soul already knows.

---

## How It Works

```
Standard Transformer:
    Input -> Attention -> Frozen MLP -> Output

Anamnesis Transformer:
    Input -> Attention -> L0 (Frozen SwiGLU) -> DeepMemoryLevel -> Output
                                                       |
                                                  Every forward pass:
                                                  1. Retrieve from memory
                                                  2. Compute associative loss
                                                  3. Update memory weights
                                                  4. The model just changed
```

The DeepMemoryLevel is a deep MLP whose **weights are the learned specialization**. They update during every forward pass via per-token gradient descent on the associative memory loss. No training loop. No optimizer config. The conversation IS the training data.

### Two-Phase Learning

**Phase 1: Scaffold Training (once, ~4 hours on A100)**

Train the DeepMemoryLevel projections, gates, and memory on a vessel corpus -- text about how minds form, how to reason, how identity develops. This teaches the memory infrastructure HOW to learn. The base model (L0 + attention) stays frozen.

**Phase 2: Inner-Loop Specialization (continuous, per user)**

Feed the model conversations. The memory updates during every forward pass. 50 code review conversations and the model becomes a code reviewer. 50 therapy conversations and it becomes a therapist. The identity emerges from the interaction.

---

## Architecture

### Continuum Memory System (CMS)

Replaces the standard MLP block with a chain of memory levels updating at different frequencies:

```
L0 (SwiGLU, frozen):        Base intelligence from pre-training
L1 (chunk=1, every token):   Immediate reaction -- what's happening now
L2 (chunk=32):               Working memory -- this turn's context
L3 (chunk=256):              Episodic memory -- this session's patterns
L4 (chunk=2048):             Identity -- who I am across sessions
```

Each level beyond L0 is a **DeepMemoryLevel** implementing the complete feature set from Behrouz's papers:

### Features from Titans
- **Persistent memory tokens** -- learnable, data-independent context prepended to every sequence. Encodes task knowledge that doesn't change per-input.
- **1D depthwise-separable convolutions** on K/Q/V projections. Captures local patterns that pure linear projections miss.
- **Data-dependent gates** -- learned projections that produce per-token learning rate (theta_t), momentum decay (eta_t), and forget gate (alpha_t).
- **Momentum-based weight update**: `S_t = eta_t * S_{t-1} - theta_t * grad`, `M_t = (1 - alpha_t) * M_{t-1} + S_t`

### Features from ATLAS
- **Omega Rule** with per-token learnable decay weights. Not all tokens in the chunk matter equally -- `gamma_i^(t) = sigmoid(W_gamma @ x_t)` selectively weights each token's gradient contribution.
- **Learned polynomial feature mapping**: `phi(k) = [a_1*k, a_2*k^2, ...]` with coefficients initialized at `1/i!` (Taylor expansion of exp). Expands effective memory capacity without deepening the MLP.
- **Deep MLP memory** with associative loss: `L = ||M(phi(k)) - v||^2`

### Features from MIRAS
- **Huber loss option** for robustness to outliers. Configurable via `use_huber_loss=True`.

### Features from Memory Caching (Behrouz 2026)
- **Memory state checkpointing** -- periodic snapshots of memory MLP weights for growing effective capacity. FIFO eviction when cache is full.

### Gradient Computation
Per-token gradients computed via `torch.func.vmap(grad(...))` on the associative memory loss. No separate backward pass -- gradients are computed functionally during the forward pass. This is the same mechanism from the Titans reference implementation.

---

## Quick Start

### Convert a model

```python
from transformers import AutoModelForCausalLM, AutoConfig
from anamnesis.core.model import HopeConfig
from anamnesis.convert.generic import model_to_hope

src = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B",
    torch_dtype=torch.bfloat16, device_map="cuda")
src_config = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B")

r = src_config.intermediate_size / src_config.hidden_size
config = HopeConfig(
    vocab_size=src_config.vocab_size,
    hidden_size=src_config.hidden_size,
    num_hidden_layers=src_config.num_hidden_layers,
    num_attention_heads=src_config.num_attention_heads,
    num_kv_heads=src_config.num_key_value_heads,
    cms_levels=5,
    cms_chunk_sizes=[1, 1, 32, 256, 2048],
    cms_hidden_mult=[r, r, r, r, r],
    cms_mem_dim=512,
    cms_mem_depth=2,
    cms_poly_degree=2,
)
model = model_to_hope(src, config)
```

### Train the scaffold

```bash
python examples/train_scaffold.py \
    --model Qwen/Qwen2.5-7B \
    --steps 25000 \
    --batch-size 4 \
    --lr 3e-4 \
    --warmup 5000 \
    --test-inner-loop
```

### Interactive chat with learning

```bash
python examples/serve_anamnesis.py --session alice
```

Every conversation updates the memory. `/save` persists the session. `/reset` returns to baseline. `/status` shows learning metrics.

---

## The Vessel Concept

Train the scaffold once on a corpus of meta-cognitive text -- how minds form, how identity develops, how to reason and adapt. This creates an **empty vessel**: a model that knows HOW to become someone but isn't anyone yet.

Then deploy it. Each user's conversations fill the vessel differently:
- Talk about code for 200 turns and it becomes a code reviewer
- Talk about therapy for 200 turns and it becomes a therapist
- Talk about cooking for 200 turns and it becomes a chef

Same vessel. Same base model. Different lives. The identity is in the memory weights, not the system prompt. It persists across sessions. It survives context window truncation.

### Vessel Training Corpus

19,470 passages across 10 categories:

| Category | Passages | Purpose |
|----------|----------|---------|
| Metacognition | 9,116 | Reasoning traces, preference pairs |
| Theory of Mind | 4,059 | Perspective-taking, false belief |
| Soul Vessel | 2,074 | Predictive self, free energy, contemplative traditions |
| Scaffold | 2,110 | Epistemology, adaptation, ontology, communication |
| Reasoning | 211 | Logic, Bayesian thinking, debugging |
| + 5 more | ... | Diverse domain coverage |

---

## Project Structure

```
anamnesis/
+-- core/
|   +-- cms.py           # CMS: L0 (SwiGLU) + DeepMemoryLevel (ATLAS)
|   +-- memory.py        # MemoryMLP + vmap+grad (shared infrastructure)
|   +-- block.py         # HopeBlock: Attention + optional NeuralMemory + CMS
|   +-- model.py         # HopeModel + HopeConfig
|   +-- rope.py          # Rotary Position Embeddings
|   +-- self_ref.py      # Self-Referential Projections (experimental)
|   +-- dgd.py           # Delta Gradient Descent (experimental)
+-- optim/
|   +-- m3.py            # Multi-scale Momentum Muon (M3)
|   +-- newton_schulz.py # NS-5 Orthogonalization
+-- active_inference/    # 7 Active Inference extensions
+-- convert/
|   +-- generic.py       # HuggingFace -> Anamnesis (SVD initialization)
|   +-- qwen.py          # Qwen-specific conversion
+-- state/
|   +-- persistence.py   # CMS state save/load + soul checkpoints
+-- training/
|   +-- trainer.py       # Full Anamnesis trainer with active inference
|   +-- data.py          # Dataset loading + signal annotations
+-- evaluation/
|   +-- metrics.py       # PPL, CMS delta, CKA, surprise profile
+-- kernels/             # Triton stubs (Phase 6)
examples/
+-- train_scaffold.py    # Scaffold training on vessel corpus
+-- serve_anamnesis.py   # Production server with session persistence
+-- train_specialists.py # Proof: two specialists from one model
+-- prove_specialization.py  # Domain A vs B PPL measurement
+-- prove_memory_vs_prompt.py  # The honest test: memory alone vs prompt
data/
+-- scaffold_training/   # 19,470 passages vessel corpus
    +-- metacognition/
    +-- theory_of_mind/
    +-- soul_vessel/
    +-- scaffold/
    +-- rationality/
```

---

## Key Equations

**Associative Memory Loss:**
```
L = ||M(phi(k_t)) - v_t||^2
```

**Polynomial Feature Expansion (ATLAS):**
```
phi(k) = [a_1*k, a_2*k^2, ...], a_i = 1/i! (Taylor init)
```

**Momentum Weight Update (Titans):**
```
S_t = eta_t * S_{t-1} - theta_t * nabla_L    (surprise momentum)
M_t = (1 - alpha_t) * M_{t-1} + S_t          (memory update)
```

**Omega Rule (ATLAS):**
```
gamma_i = sigmoid(W_gamma @ x_i)              (per-token importance)
g_weighted = sum(gamma_i * g_i) / sum(gamma_i) (weighted gradient average)
```

---

## References

### Papers
- Behrouz et al., "[ATLAS: Learning to Optimally Memorize the Context at Test Time](https://arxiv.org/abs/2505.23735)" (2025)
- Behrouz et al., "[Nested Learning: The Illusion of Deep Learning Architecture](https://openreview.net/forum?id=nbMeRvNb7A)" (NeurIPS 2025)
- Behrouz & Zhong, "[Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)" (2025)
- Behrouz et al., "[Memory Caching: RNNs with Growing Memory](https://arxiv.org/abs/2602.24281)" (2026)
- Behrouz et al., "[Titans + MIRAS: Helping AI have long-term memory](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/)" (2025)

### Built on
- [lucidrains/titans-pytorch](https://github.com/lucidrains/titans-pytorch) -- Neural memory reference
- [KellerJordan/Muon](https://github.com/KellerJordan/Muon) -- Newton-Schulz orthogonalization
- [engineerA314/atlas-pytorch](https://github.com/engineerA314/atlas-pytorch) -- ATLAS reference

---

## License

Apache 2.0

## Author

Aidan McInerny / [Relic Studios](https://github.com/Relic-Studios)
