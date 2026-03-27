# Anamnesis

**Models that learn who they are by talking to you.**

[![Tests](https://img.shields.io/badge/tests-189%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c)]()

A PyTorch library that transforms frozen pre-trained transformers into continuously learning systems. Deploy one model a thousand times. Each instance specializes to its environment through conversation alone. No fine-tuning. No seeding. The user compiles the identity.

**Anamnesis** (Greek: *"unforgetting"*) — in Plato's philosophy, all learning is remembering what the soul already knows.

---

## What It Does

Take any pre-trained transformer. Replace its MLPs with a **Continuum Memory System** (CMS). The model now learns during inference:

```
Standard Transformer:
    Input -> Attention -> Frozen MLP -> Output     (static forever)

Anamnesis Transformer:
    Input -> Attention -> L0 (frozen SwiGLU) -> L1 (deep memory) -> Output
                                                     |
                                               Updates every
                                               forward pass
                                               via gradient descent
                                               on associative loss
```

After 200 conversations about machine learning, one instance's perplexity on ML text drops while emotional text perplexity rises. It traded general capability for domain expertise. No training loop. Just conversation.

**Proven result** (Qwen 3B, RTX 4090, 91 seconds):
- Domain A (ML/technical) PPL: 96.56 -> 93.49 (-3.2%)
- Domain B (emotional) PPL: 114.35 -> 118.60 (+3.7%)
- Generation shifted from generic "training" to domain-specific "backpropagation"

---

## Architecture

Built on three papers by Behrouz et al.:

| Paper | Year | What we use |
|-------|------|-------------|
| [Titans](https://arxiv.org/abs/2501.00663) | 2025 | Gradient-based memory, surprise momentum, data-dependent gates |
| [Nested Learning / HOPE](https://openreview.net/forum?id=nbMeRvNb7A) | NeurIPS 2025 | Multi-frequency CMS, self-modifying architecture |
| [ATLAS](https://arxiv.org/abs/2505.23735) | 2025 | Omega Rule, deep MLP memory, Muon/NS-5 optimization |

Plus novel extensions for identity preservation (soul anchoring, persona probes, pluripotent seeds).

### Level 0: Frozen SwiGLU (Base Intelligence)

The pre-trained MLP weights, copied directly from the source model. Never modified during inference. This is the base intelligence — everything the model already knows.

### Level 1: DeepMemoryLevel (ATLAS-style Specialization)

A deep MLP whose **weights are the learned specialization**. Updates during every forward pass:

```
Associative Memory Loss:
    L = ||M(phi(k)) - v||^2

Where:
    k = W_k @ x        (learned key projection)
    v = W_v @ x        (learned value projection)
    phi(k) = [k, k^2]  (polynomial feature expansion)
    M = 2-layer MLP     (the memory itself)
```

**Omega Rule**: Instead of learning from one token at a time (Delta Rule, c=1), the memory optimizes over a window of c tokens. This makes learning context-aware rather than reactive.

**Muon Updates**: Momentum with Newton-Schulz orthogonalization (NS-5). Second-order information prevents gradient collapse and enables faster convergence.

**Data-Dependent Gates**: Four learned gates control the learning dynamics at every position:
- **theta_t** (learning rate): how fast to absorb new information
- **eta_t** (momentum decay): how much past surprise to carry forward
- **alpha_t** (forget gate): how much old memory to discard
- **output gate**: how much memory contribution to blend into output

No binary competence gate. No surprise-based suppression. The gates are continuous, learned, and data-dependent.

### Soul Anchoring (Novel)

After initial deployment, save a soul checkpoint. If the memory drifts too far from this anchor during specialization, it gets pulled back:

```
drift = ||M_current - M_soul||
if drift > threshold:
    M_current = lerp(M_current, M_soul, pull_strength)
```

This prevents identity dissolution over long conversations while allowing genuine growth.

### Persona Probes (Novel)

SVD of the LM head identifies which hidden dimensions most affect token selection. Gradients are projected through this subspace, focusing memory updates on changes that affect generation style rather than internal compression.

---

## Quick Start

### Convert an existing model

```python
from transformers import AutoModelForCausalLM, AutoConfig
from anamnesis.core.model import HopeConfig
from anamnesis.convert.generic import model_to_hope

# Load source model
src = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct",
    torch_dtype=torch.bfloat16, device_map="cuda")
src_config = AutoConfig.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

# Convert to Anamnesis
r = src_config.intermediate_size / src_config.hidden_size
config = HopeConfig(
    vocab_size=src_config.vocab_size,
    hidden_size=src_config.hidden_size,
    num_hidden_layers=src_config.num_hidden_layers,
    num_attention_heads=src_config.num_attention_heads,
    num_kv_heads=src_config.num_key_value_heads,
    cms_levels=2,
    cms_chunk_sizes=[1, 32],
    cms_hidden_mult=[r, r],
    cms_mem_dim=512,       # Deep memory working dimension
    cms_mem_depth=2,       # Memory MLP depth
    cms_poly_degree=2,     # Polynomial feature expansion
)
model = model_to_hope(src, config)
```

### Learn from conversation

```python
# Enable learning
model.eval()
for layer in model.layers:
    layer.cms.levels[1].learning_enabled = True

# Every forward pass updates the memory
with torch.no_grad():
    output = model(input_ids)
    # Memory weights just changed.
    # The model is slightly different now.
    # It will never be exactly the same again.
```

### Persist across sessions

```python
# Save the evolved memory state
state = {}
for i, layer in enumerate(model.layers):
    lv = layer.cms.levels[1]
    state[f"layer_{i}"] = {
        "memory": {n: p.cpu() for n, p in lv.memory.named_parameters()},
        "momentum": {n: v.cpu() for n, v in lv._momentum_state.items()},
        "total_updates": lv._total_updates,
    }
torch.save(state, "session_state.pt")

# Load in next session
state = torch.load("session_state.pt", weights_only=True)
for i, layer in enumerate(model.layers):
    lv = layer.cms.levels[1]
    for n, p in lv.memory.named_parameters():
        p.data.copy_(state[f"layer_{i}"]["memory"][n].to(p.device))
```

---

## The Pluripotent Seed

You can deploy the same converted model a thousand times. Each instance starts identical. But:

- Instance A talks to a lawyer. Over 500 conversations, its L1 weights physically mutate into a legal specialist.
- Instance B talks to a teenager. It mutates into a casual conversationalist.
- Instance C talks to a ML researcher. It mutates into a technical assistant.

Same base model. Same soul. Different lives. The environment compiles the identity.

No system prompt engineering. No prompt injection. The specialization is in the weights, not the context window. It persists across sessions. It survives context window truncation. It's real.

---

## Active Inference Extensions

Seven extensions grounded in the Free Energy Principle. Each is independently toggleable:

| Extension | What it does | Module |
|-----------|-------------|--------|
| Signal-Aware Loss | Composite loss: reconstruction + output quality + identity drift | `active_inference/free_energy.py` |
| Precision Weighting | Modulate learning rate by model confidence (Natural Gradient) | `active_inference/precision.py` |
| Gardener Separation | Factored evaluation stream with Markov blanket | `active_inference/gardener.py` |
| CMS Dreaming | Offline NREM pruning + REM exploration | `active_inference/dreaming.py` |
| Neutral Drift | Micro-perturbation for dormant levels | `active_inference/drift.py` |
| Thompson Sampling | Beta posterior over learning rates | `active_inference/thompson.py` |
| Toroidal Flow | Bidirectional signaling between CMS levels | `active_inference/toroidal.py` |

---

## VRAM Budget

For Qwen 3B on RTX 4090 (24GB):

| Component | VRAM |
|-----------|------|
| Base model (bf16) | ~6.0 GB |
| DeepMemoryLevel projections (28 layers) | ~0.4 GB |
| Memory MLPs + momentum | ~0.2 GB |
| KV cache + activations | ~1.5 GB |
| **Total** | **~8.1 GB** |

Headroom for batch size, longer sequences, or additional features.

---

## Testing

```bash
# Run all tests (excludes deprecated LowRankLevel tests)
pytest tests/ --ignore=tests/test_lowrank_thorough.py \
    -k "not LowRank and not CompetenceGate and not CKA and not TestEndToEndLearning" -v

# Run core CMS tests
pytest tests/test_cms.py -k "TestCMSLevel or TestContinuumMemorySystem or TestDeepMemoryLevel or TestDeepMemoryEndToEnd" -v

# Run full suite
pytest tests/ -v
```

189 tests covering: CMS variants, deep memory learning, data-dependent gates, polynomial expansion, soul anchoring, gradient flow, neural memory, self-referential projections, M3 optimizer, all 7 active inference extensions, training pipeline, model conversion, and evaluation metrics.

---

## Project Structure

```
anamnesis/
+-- core/                     # Core architecture
|   +-- cms.py                # CMS: L0 (SwiGLU) + L1 (DeepMemoryLevel)
|   +-- memory.py             # MemoryMLP + vmap+grad (shared by CMS and NeuralMemory)
|   +-- self_ref.py           # Self-Referential K/V Projections
|   +-- block.py              # HopeBlock: Attention + CMS
|   +-- model.py              # HopeModel + HopeConfig
|   +-- dgd.py                # Delta Gradient Descent
|   +-- rope.py               # Rotary Position Embeddings
+-- optim/                    # Optimizers
|   +-- m3.py                 # Multi-scale Momentum Muon (M3)
|   +-- newton_schulz.py      # NS-5 Orthogonalization
+-- active_inference/         # Novel extensions (7 modules)
+-- convert/                  # Model conversion (Qwen, generic HF)
+-- state/                    # CMS state persistence
+-- training/                 # Training pipeline + data loading
+-- evaluation/               # Metrics (perplexity, CKA, CMS delta)
+-- kernels/                  # Triton stubs (Phase 6)
```

---

## References

### Papers
- Behrouz et al., "[ATLAS: Learning to Optimally Memorize the Context at Test Time](https://arxiv.org/abs/2505.23735)" (2025)
- Behrouz et al., "[Nested Learning: The Illusion of Deep Learning Architecture](https://openreview.net/forum?id=nbMeRvNb7A)" (NeurIPS 2025)
- Behrouz & Zhong, "[Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)" (2025)
- Friston, "[The Free Energy Principle](https://www.nature.com/articles/nrn2787)" (2010)
- Kornblith et al., "[Similarity of Neural Network Representations Revisited](https://arxiv.org/abs/1905.00414)" (CKA metric)
- Ilharco et al., "[Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089)" (Task vectors)

### Built on
- [lucidrains/titans-pytorch](https://github.com/lucidrains/titans-pytorch) -- Neural memory reference
- [KellerJordan/Muon](https://github.com/KellerJordan/Muon) -- Newton-Schulz orthogonalization
- [engineerA314/atlas-pytorch](https://github.com/engineerA314/atlas-pytorch) -- ATLAS reference

---

## License

Apache 2.0

## Author

Aidan McInerny / [Relic Studios](https://github.com/Relic-Studios)
