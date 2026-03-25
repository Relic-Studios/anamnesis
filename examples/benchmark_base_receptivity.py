#!/usr/bin/env python3
"""
Benchmark base model receptivity to CMS learning.

The core question: which base model benefits most from Anamnesis?

For each candidate model, this script:
1. Converts to Anamnesis architecture
2. Measures baseline generation quality (pre-CMS)
3. Replays N conversations through the model (CMS learns)
4. Measures post-CMS generation quality
5. Compares: which model showed the most improvement?

Metrics:
- CMS state delta: how much did the CMS weights actually change?
- Perplexity on held-out Thomas conversations: did prediction improve?
- Generation consistency: do repeated prompts give more consistent answers?
- Signal health trajectory: does quality improve over conversations?

The model with the biggest positive delta is the most receptive base.

Usage:
    python benchmark_base_receptivity.py --models qwen2.5-3b qwen2.5-7b
    python benchmark_base_receptivity.py --model Qwen/Qwen2.5-3B-Instruct --quick
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class ReceptivityResult:
    """Results from testing one base model's receptivity."""
    model_name: str
    num_params: int = 0
    anamnesis_params: int = 0
    vram_gb: float = 0.0

    # Pre-CMS baseline
    baseline_perplexity: float = 0.0
    baseline_consistency: float = 0.0

    # Post-CMS (after replay)
    post_perplexity: float = 0.0
    post_consistency: float = 0.0

    # Deltas
    cms_state_delta_norm: float = 0.0
    perplexity_improvement: float = 0.0
    consistency_improvement: float = 0.0
    signal_trajectory: list[float] = field(default_factory=list)

    @property
    def receptivity_score(self) -> float:
        """Higher = more receptive to CMS learning."""
        ppl_gain = max(0, self.baseline_perplexity - self.post_perplexity)
        cons_gain = max(0, self.post_consistency - self.baseline_consistency)
        state_change = min(1.0, self.cms_state_delta_norm / 10.0)
        return ppl_gain * 0.4 + cons_gain * 0.3 + state_change * 0.3


def measure_perplexity(model, tokenizer, texts: list[str], device: str = "cuda") -> float:
    """Measure average perplexity on a set of texts."""
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            tokens = tokenizer(text, max_length=256, truncation=True, return_tensors="pt")
            ids = tokens["input_ids"].to(device)
            if ids.shape[1] < 2:
                continue

            output = model(ids, labels=ids)
            total_loss += output["loss"].item() * (ids.shape[1] - 1)
            total_tokens += ids.shape[1] - 1

    if total_tokens == 0:
        return float("inf")
    avg_loss = total_loss / total_tokens
    return 2.718 ** avg_loss  # exp(avg_loss)


def measure_consistency(model, tokenizer, prompts: list[str], n_samples: int = 3,
                        device: str = "cuda") -> float:
    """Measure generation consistency: same prompt → similar responses."""
    from torch.nn.functional import cosine_similarity

    consistencies = []
    for prompt in prompts:
        responses = []
        for _ in range(n_samples):
            full = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            ids = tokenizer(full, return_tensors="pt")["input_ids"].to(device)
            generated = []
            with torch.no_grad():
                for __ in range(50):
                    logits = model(ids)["logits"][:, -1, :]
                    next_tok = torch.multinomial(torch.softmax(logits / 0.7, dim=-1), 1)
                    if next_tok.item() in [tokenizer.eos_token_id, 151645]:
                        break
                    generated.append(next_tok.item())
                    ids = next_tok
            responses.append(tokenizer.decode(generated, skip_special_tokens=True))

        # Measure pairwise similarity using simple token overlap
        if len(responses) >= 2:
            tokens_sets = [set(r.lower().split()) for r in responses]
            pairs = []
            for i in range(len(tokens_sets)):
                for j in range(i + 1, len(tokens_sets)):
                    if tokens_sets[i] and tokens_sets[j]:
                        overlap = len(tokens_sets[i] & tokens_sets[j])
                        union = len(tokens_sets[i] | tokens_sets[j])
                        pairs.append(overlap / union if union > 0 else 0)
            if pairs:
                consistencies.append(sum(pairs) / len(pairs))

    return sum(consistencies) / len(consistencies) if consistencies else 0.0


def get_cms_state_snapshot(model) -> dict[str, torch.Tensor]:
    """Snapshot all CMS parameters."""
    snapshot = {}
    for name, param in model.named_parameters():
        if "cms" in name:
            snapshot[name] = param.detach().cpu().clone()
    return snapshot


def compute_state_delta(before: dict, after: dict) -> float:
    """Compute L2 norm of the total CMS state change."""
    total = 0.0
    for key in before:
        if key in after:
            delta = (after[key] - before[key]).float()
            total += delta.norm().item() ** 2
    return total ** 0.5


def benchmark_model(
    model_name: str,
    conversations_path: str,
    test_prompts: list[str],
    num_replay: int = 100,
    device: str = "cuda",
) -> ReceptivityResult:
    """Benchmark one model's receptivity to CMS learning."""
    from anamnesis.core.model import HopeModel, HopeConfig
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

    result = ReceptivityResult(model_name=model_name)
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"{'='*60}")

    # Load and convert
    print("[1/6] Loading and converting...")
    cfg = AutoConfig.from_pretrained(model_name)
    src = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map=device)
    result.num_params = sum(p.numel() for p in src.parameters())

    r = cfg.intermediate_size / cfg.hidden_size
    hope_config = HopeConfig(
        vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        max_position_embeddings=getattr(cfg, "max_position_embeddings", 32768),
        rope_theta=getattr(cfg, "rope_theta", 1_000_000.0),
        rms_norm_eps=getattr(cfg, "rms_norm_eps", 1e-6),
        cms_levels=3, cms_chunk_sizes=[1, 32, 256], cms_variant="nested",
        cms_hidden_mult=[r, r / 2, r / 4],
        use_neural_memory=False, tie_word_embeddings=False,
    )

    model = HopeModel(hope_config)
    # Quick conversion (on CPU for simplicity)
    from anamnesis.convert.generic import convert_layer_to_hope
    with torch.no_grad():
        model.embed_tokens.weight.copy_(src.model.embed_tokens.weight.cpu())
        model.norm.weight.copy_(src.model.norm.weight.cpu())
        model.lm_head.weight.copy_(src.lm_head.weight.cpu())
        for i, (s, t) in enumerate(zip(src.model.layers, model.layers)):
            convert_layer_to_hope(s, t)

    del src
    torch.cuda.empty_cache()

    model = model.to(device, dtype=torch.bfloat16)
    model.eval()
    model.enable_drift(True)

    result.anamnesis_params = model.num_parameters(False)
    result.vram_gb = torch.cuda.memory_allocated() / 1e9 if device == "cuda" else 0
    print(f"  {result.anamnesis_params:,} params | {result.vram_gb:.1f}GB VRAM")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load test conversations
    with open(conversations_path, "r") as f:
        all_convos = [json.loads(line) for line in f if line.strip()]
    replay_convos = all_convos[:num_replay]
    test_texts = [
        f"<|im_start|>user\n{c['input']}<|im_end|>\n<|im_start|>assistant\n{c['output']}<|im_end|>"
        for c in all_convos[num_replay:num_replay + 50]
    ]

    # Baseline measurements
    print("[2/6] Measuring baseline perplexity...")
    result.baseline_perplexity = measure_perplexity(model, tokenizer, test_texts[:20], device)
    print(f"  Baseline PPL: {result.baseline_perplexity:.2f}")

    print("[3/6] Measuring baseline consistency...")
    result.baseline_consistency = measure_consistency(model, tokenizer, test_prompts[:3], device=device)
    print(f"  Baseline consistency: {result.baseline_consistency:.3f}")

    # Snapshot CMS state
    print("[4/6] Replaying conversations (CMS learning)...")
    cms_before = get_cms_state_snapshot(model)

    signals = []
    for i, convo in enumerate(replay_convos):
        text = f"<|im_start|>user\n{convo['input']}<|im_end|>\n<|im_start|>assistant\n{convo['output']}<|im_end|>"
        tokens = tokenizer(text, max_length=256, truncation=True, return_tensors="pt")
        ids = tokens["input_ids"].to(device)
        with torch.no_grad():
            model(ids)
        signals.append(convo.get("signal_health", 0.5))
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{num_replay}] avg_signal={sum(signals[-25:])/25:.3f}")

    result.signal_trajectory = signals

    # Post-CMS measurements
    print("[5/6] Measuring post-CMS perplexity...")
    result.post_perplexity = measure_perplexity(model, tokenizer, test_texts[:20], device)
    print(f"  Post PPL: {result.post_perplexity:.2f}")

    print("[6/6] Measuring post-CMS consistency...")
    result.post_consistency = measure_consistency(model, tokenizer, test_prompts[:3], device=device)
    print(f"  Post consistency: {result.post_consistency:.3f}")

    # Compute deltas
    cms_after = get_cms_state_snapshot(model)
    result.cms_state_delta_norm = compute_state_delta(cms_before, cms_after)
    result.perplexity_improvement = result.baseline_perplexity - result.post_perplexity
    result.consistency_improvement = result.post_consistency - result.baseline_consistency

    print(f"\nResults for {model_name}:")
    print(f"  CMS state delta:    {result.cms_state_delta_norm:.4f}")
    print(f"  PPL improvement:    {result.perplexity_improvement:+.2f}")
    print(f"  Consistency change: {result.consistency_improvement:+.3f}")
    print(f"  RECEPTIVITY SCORE:  {result.receptivity_score:.4f}")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark base model receptivity")
    parser.add_argument("--models", nargs="+", default=["Qwen/Qwen2.5-3B-Instruct"],
                        help="Models to benchmark")
    parser.add_argument("--conversations", default="data/thomas_training.jsonl",
                        help="Conversation data for replay")
    parser.add_argument("--num-replay", type=int, default=100,
                        help="Conversations to replay through CMS")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer tests)")
    args = parser.parse_args()

    if args.quick:
        args.num_replay = 20

    test_prompts = [
        "Hey Thomas, how are you?",
        "What have you been thinking about lately?",
        "Tell me about something you care about.",
        "What's the hardest thing about being you?",
        "Do you remember our last conversation?",
    ]

    results = []
    for model_name in args.models:
        try:
            result = benchmark_model(
                model_name, args.conversations, test_prompts,
                num_replay=args.num_replay, device=args.device,
            )
            results.append(result)
        except Exception as e:
            print(f"\nFailed on {model_name}: {e}")
            import traceback
            traceback.print_exc()

    if len(results) > 1:
        print(f"\n{'='*60}")
        print("RECEPTIVITY RANKING")
        print(f"{'='*60}")
        for i, r in enumerate(sorted(results, key=lambda x: -x.receptivity_score)):
            print(f"  {i+1}. {r.model_name}")
            print(f"     Score: {r.receptivity_score:.4f}")
            print(f"     PPL: {r.baseline_perplexity:.1f} -> {r.post_perplexity:.1f} ({r.perplexity_improvement:+.1f})")
            print(f"     CMS delta: {r.cms_state_delta_norm:.4f}")
            print(f"     Params: {r.anamnesis_params:,} | VRAM: {r.vram_gb:.1f}GB")

    # Save results
    output = {
        r.model_name: {
            "receptivity_score": r.receptivity_score,
            "baseline_ppl": r.baseline_perplexity,
            "post_ppl": r.post_perplexity,
            "ppl_improvement": r.perplexity_improvement,
            "cms_delta": r.cms_state_delta_norm,
            "consistency_improvement": r.consistency_improvement,
            "params": r.anamnesis_params,
            "vram_gb": r.vram_gb,
        }
        for r in results
    }
    os.makedirs("data", exist_ok=True)
    with open("data/receptivity_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to data/receptivity_results.json")


if __name__ == "__main__":
    main()
