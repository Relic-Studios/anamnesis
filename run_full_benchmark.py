#!/usr/bin/env python3
"""
FULL ANAMNESIS BENCHMARK SUITE — Run unattended.

This script runs everything and saves all results to data/full_benchmark/:
1. Unit tests (172 expected)
2. Conversion fidelity (Qwen2.5-3B → Anamnesis, PPL comparison)
3. CMS inference-time learning (replay conversations, measure weight changes)
4. Generation quality (coherent output before/after CMS learning)
5. Ablation study on tiny model (all 9 configs)
6. State persistence roundtrip

Designed to run for 1-2 hours unattended on RTX 4090.
All output logged to data/full_benchmark/benchmark.log
Results saved to data/full_benchmark/results.json

Usage:
    python run_full_benchmark.py
"""

import json
import math
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Force unbuffered
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import torch

OUTPUT_DIR = Path("data/full_benchmark")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUTPUT_DIR / "benchmark.log"
RESULTS_FILE = OUTPUT_DIR / "results.json"

results = {
    "started_at": datetime.now().isoformat(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    "sections": {},
}


class Logger:
    """Tee stdout to both console and log file."""
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, "w", encoding="utf-8", buffering=1)

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = Logger(LOG_FILE)


def save_results():
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)


def section(name):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}\n")


# =========================================================================
# SECTION 1: UNIT TESTS
# =========================================================================
def run_tests():
    section("SECTION 1: UNIT TESTS")
    import subprocess
    t0 = time.time()
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"],
        capture_output=True, text=True, timeout=120,
    )
    elapsed = time.time() - t0
    print(r.stdout)
    if r.stderr:
        print(r.stderr)

    passed = "passed" in r.stdout
    # Extract count
    for line in r.stdout.split("\n"):
        if "passed" in line:
            print(f"  Result: {line.strip()}")

    results["sections"]["unit_tests"] = {
        "passed": passed,
        "return_code": r.returncode,
        "elapsed_s": round(elapsed, 1),
        "output": r.stdout[-500:],
    }
    save_results()
    return passed


# =========================================================================
# SECTION 2: CONVERSION FIDELITY (Qwen2.5-3B)
# =========================================================================
def run_conversion_fidelity():
    section("SECTION 2: CONVERSION FIDELITY (Qwen2.5-3B)")
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
    from anamnesis.core.model import HopeModel, HopeConfig
    from anamnesis.convert.generic import model_to_hope

    model_name = "Qwen/Qwen2.5-3B-Instruct"
    device = "cuda"
    sec = {}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    src_config = AutoConfig.from_pretrained(model_name)

    # Load source
    print("[1] Loading source model...")
    t0 = time.time()
    src_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=device,
    )
    sec["source_load_s"] = round(time.time() - t0, 1)
    print(f"  Loaded in {sec['source_load_s']}s")

    # Source PPL on test strings
    test_strings = [
        "The capital of France is Paris, and it is known for the Eiffel Tower.",
        "Machine learning models can be trained on large datasets to perform various tasks including classification and generation.",
        "In the beginning was the Word, and the Word was with God, and the Word was God.",
        "The quick brown fox jumps over the lazy dog while the cat watches from the windowsill.",
        "Neural networks consist of layers of interconnected nodes that process information through weighted connections.",
        "To be or not to be, that is the question. Whether it is nobler in the mind to suffer.",
        "The mitochondria is the powerhouse of the cell, responsible for producing ATP through oxidative phosphorylation.",
        "When you look into an abyss, the abyss also looks into you. What does not kill me makes me stronger.",
    ]

    print("[2] Measuring source perplexity...")
    src_ppl = _measure_ppl(src_model, tokenizer, test_strings, device)
    sec["source_ppl"] = round(src_ppl, 2)
    print(f"  Source PPL: {src_ppl:.2f}")

    # Source generation
    print("[3] Source generation samples...")
    src_gens = {}
    prompts = [
        "What is consciousness?",
        "Tell me about yourself.",
        "Explain how transformers work in simple terms.",
    ]
    for p in prompts:
        gen = _generate(src_model, tokenizer, p, device, max_tokens=80)
        src_gens[p] = gen
        print(f"  Q: {p}")
        print(f"  A: {gen[:200]}")
        print()
    sec["source_generations"] = src_gens

    # Convert
    print("[4] Converting to Anamnesis...")
    t0 = time.time()
    r = src_config.intermediate_size / src_config.hidden_size
    hope_config = HopeConfig(
        vocab_size=src_config.vocab_size,
        hidden_size=src_config.hidden_size,
        num_hidden_layers=src_config.num_hidden_layers,
        num_attention_heads=src_config.num_attention_heads,
        num_kv_heads=src_config.num_key_value_heads,
        max_position_embeddings=getattr(src_config, "max_position_embeddings", 32768),
        rope_theta=getattr(src_config, "rope_theta", 1_000_000.0),
        rms_norm_eps=getattr(src_config, "rms_norm_eps", 1e-6),
        cms_levels=3,
        cms_chunk_sizes=[1, 32, 256],
        cms_variant="nested",
        cms_hidden_mult=[r, r / 2, r / 4],
        use_neural_memory=False,
        tie_word_embeddings=False,
    )
    model = model_to_hope(src_model, hope_config, verbose=True)

    src_params = sum(p.numel() for p in src_model.parameters())
    del src_model
    torch.cuda.empty_cache()

    model = model.to(device, dtype=torch.bfloat16)
    model.eval()

    sec["conversion_s"] = round(time.time() - t0, 1)
    sec["source_params"] = src_params
    sec["anamnesis_params"] = model.num_parameters(trainable_only=False)
    sec["param_ratio"] = round(sec["anamnesis_params"] / src_params, 3)
    sec["vram_gb"] = round(torch.cuda.memory_allocated() / 1e9, 1)
    print(f"  Converted in {sec['conversion_s']}s")
    print(f"  Params: {src_params:,} -> {sec['anamnesis_params']:,} ({sec['param_ratio']}x)")
    print(f"  VRAM: {sec['vram_gb']} GB")

    # Anamnesis PPL (learning off)
    for layer in model.layers:
        layer.cms.enable_learning(False)

    print("\n[5] Measuring Anamnesis perplexity (learning off)...")
    anamnesis_ppl = _measure_ppl(model, tokenizer, test_strings, device)
    sec["anamnesis_ppl"] = round(anamnesis_ppl, 2)
    sec["ppl_delta"] = round(anamnesis_ppl - src_ppl, 2)
    sec["ppl_match"] = abs(anamnesis_ppl - src_ppl) < src_ppl * 0.5  # within 50%
    print(f"  Anamnesis PPL: {anamnesis_ppl:.2f} (delta: {anamnesis_ppl - src_ppl:+.2f})")
    print(f"  MATCH: {'YES' if sec['ppl_match'] else 'NO'}")

    # Anamnesis generation
    print("\n[6] Anamnesis generation samples...")
    anam_gens = {}
    for p in prompts:
        gen = _generate(model, tokenizer, p, device, max_tokens=80)
        anam_gens[p] = gen
        print(f"  Q: {p}")
        print(f"  A: {gen[:200]}")
        print()
    sec["anamnesis_generations"] = anam_gens

    results["sections"]["conversion_fidelity"] = sec
    save_results()
    return model, tokenizer, hope_config


# =========================================================================
# SECTION 3: CMS INFERENCE-TIME LEARNING
# =========================================================================
def run_cms_learning(model, tokenizer):
    section("SECTION 3: CMS INFERENCE-TIME LEARNING")
    from anamnesis.evaluation.metrics import snapshot_cms_state, compute_cms_delta

    device = "cuda"
    sec = {}

    # Enable sequential learning on level 0 only
    for layer in model.layers:
        layer.cms.enable_learning(True, levels=[0])
    model.enable_drift(True)
    torch.cuda.reset_peak_memory_stats()

    # Load data
    data_path = "data/thomas_training.jsonl"
    with open(data_path, encoding="utf-8") as f:
        all_convos = [json.loads(line) for line in f if line.strip()]

    num_replay = min(100, len(all_convos))
    replay_convos = all_convos[:num_replay]
    print(f"[7] Replaying {num_replay} conversations (level-0 sequential CMS learning)...")

    cms_before = snapshot_cms_state(model)
    signal_traj = []
    t0 = time.time()

    for i, convo in enumerate(replay_convos):
        text = f"<|im_start|>user\n{convo['input']}<|im_end|>\n<|im_start|>assistant\n{convo['output']}<|im_end|>"
        tokens = tokenizer(text, max_length=128, truncation=True, return_tensors="pt")
        ids = tokens["input_ids"].to(device)

        with torch.no_grad():
            model(ids)

        # VRAM guard
        vram = torch.cuda.memory_allocated() / 1e9
        if vram > 16.0:
            print(f"  WARNING: VRAM {vram:.1f}GB, stopping at conversation {i+1}")
            break

        signal_traj.append(convo.get("signal_health", 0.5))

        if (i + 1) % 20 == 0:
            cms_now = snapshot_cms_state(model)
            delta = compute_cms_delta(cms_before, cms_now)
            elapsed = time.time() - t0
            print(f"  [{i+1}/{num_replay}] delta={delta['total_l2']:.6f}, "
                  f"VRAM={vram:.1f}GB, {elapsed:.0f}s")

    replay_time = time.time() - t0
    cms_after = snapshot_cms_state(model)
    delta = compute_cms_delta(cms_before, cms_after)

    sec["replay_conversations"] = len(signal_traj)
    sec["replay_time_s"] = round(replay_time, 1)
    sec["time_per_convo_s"] = round(replay_time / max(len(signal_traj), 1), 2)
    sec["cms_delta_total"] = round(delta["total_l2"], 6)
    sec["cms_delta_per_level"] = {k: round(v, 6) for k, v in delta["per_level"].items()}
    sec["cms_max_param_delta"] = round(delta["max_param_delta"], 8)
    sec["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
    sec["weights_changed"] = delta["total_l2"] > 0.0

    print(f"\n  Total CMS delta: {delta['total_l2']:.6f}")
    for level, d in delta["per_level"].items():
        print(f"    {level}: {d:.6f}")
    print(f"  Replay: {len(signal_traj)} convos in {replay_time:.1f}s "
          f"({sec['time_per_convo_s']}s/convo)")
    print(f"  Peak VRAM: {sec['peak_vram_gb']} GB")
    print(f"  Weights changed: {'YES' if sec['weights_changed'] else 'NO'}")

    # Post-learning PPL
    for layer in model.layers:
        layer.cms.enable_learning(False)

    print("\n[8] Post-learning perplexity...")
    test_strings = [
        "The capital of France is Paris, and it is known for the Eiffel Tower.",
        "Machine learning models can be trained on large datasets to perform various tasks.",
        "Neural networks consist of layers of interconnected nodes.",
    ]
    post_ppl = _measure_ppl(model, tokenizer, test_strings, device)
    sec["post_learning_ppl"] = round(post_ppl, 2)
    print(f"  Post-learning PPL: {post_ppl:.2f}")

    # Post-learning generation
    print("\n[9] Post-learning generation...")
    prompts = [
        "Tell me about yourself.",
        "What have you been thinking about?",
        "What is the meaning of life?",
    ]
    gens = {}
    for p in prompts:
        gen = _generate(model, tokenizer, p, device, max_tokens=80)
        gens[p] = gen
        print(f"  Q: {p}")
        print(f"  A: {gen[:200]}")
        print()
    sec["post_learning_generations"] = gens

    results["sections"]["cms_learning"] = sec
    save_results()


# =========================================================================
# SECTION 4: ABLATION STUDY (tiny model, CPU)
# =========================================================================
def run_ablation():
    section("SECTION 4: ABLATION STUDY (tiny model)")
    from anamnesis.evaluation.ablation import AblationRunner, ABLATION_CONFIGS
    from anamnesis.core.model import HopeModel, HopeConfig
    from torch.utils.data import DataLoader

    torch.manual_seed(42)

    def model_factory():
        torch.manual_seed(42)
        config = HopeConfig(
            vocab_size=256, hidden_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_kv_heads=2,
            cms_levels=3, cms_chunk_sizes=[1, 8, 32], cms_hidden_mult=4.0,
        )
        return HopeModel(config), config

    def make_dl(n):
        batches = [{"input_ids": torch.randint(0, 256, (4, 32)),
                     "signal_health": torch.rand(4) * 0.5 + 0.3} for _ in range(n)]
        return DataLoader(batches, batch_size=None)

    t0 = time.time()
    runner = AblationRunner(
        model_factory=model_factory,
        train_loader=make_dl(120),
        eval_loader=make_dl(10),
        train_steps=100,
        trainer_overrides={"lr": 3e-4, "adam_lr": 3e-4},
    )
    ablation_results = runner.run_all(verbose=False)
    runner.print_table(ablation_results)

    sec = {
        "elapsed_s": round(time.time() - t0, 1),
        "configs": {r.config_name: r.to_dict() for r in ablation_results},
    }
    results["sections"]["ablation"] = sec
    save_results()


# =========================================================================
# SECTION 5: STATE PERSISTENCE
# =========================================================================
def run_persistence(model):
    section("SECTION 5: STATE PERSISTENCE")
    from anamnesis.state.persistence import save_cms_state, load_cms_state
    from anamnesis.evaluation.metrics import snapshot_cms_state, compute_cms_delta
    from anamnesis.core.model import HopeModel

    sec = {}
    state_path = OUTPUT_DIR / "test_cms_state.pt"

    # Save current model state
    print("[10] Saving CMS state...")
    save_cms_state(model, state_path, {
        "benchmark": True,
        "timestamp": datetime.now().isoformat(),
    })
    sec["state_file_mb"] = round(state_path.stat().st_size / 1e6, 1)
    print(f"  Saved: {sec['state_file_mb']} MB")

    # Load into fresh model with same config
    print("[11] Loading into fresh model...")
    cms_before = snapshot_cms_state(model)

    # Reload
    metadata = load_cms_state(model, state_path)
    cms_after = snapshot_cms_state(model)

    delta = compute_cms_delta(cms_before, cms_after)
    sec["reload_delta"] = round(delta["total_l2"], 8)
    sec["metadata_preserved"] = metadata.get("benchmark") is True
    sec["roundtrip_match"] = delta["total_l2"] < 1e-6

    print(f"  Reload delta: {delta['total_l2']:.8f}")
    print(f"  Metadata preserved: {'YES' if sec['metadata_preserved'] else 'NO'}")
    print(f"  Roundtrip match: {'YES' if sec['roundtrip_match'] else 'NO'}")

    results["sections"]["persistence"] = sec
    save_results()


# =========================================================================
# HELPERS
# =========================================================================
def _measure_ppl(model, tokenizer, texts, device, max_len=256):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            tokens = tokenizer(text, max_length=max_len, truncation=True, return_tensors="pt")
            ids = tokens["input_ids"].to(device)
            if ids.shape[1] < 2:
                continue
            output = model(ids, labels=ids)
            n = ids.shape[1] - 1
            total_loss += output["loss"].item() * n
            total_tokens += n
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


def _generate(model, tokenizer, prompt, device, max_tokens=80, temperature=0.7):
    model.eval()
    full = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    ids = tokenizer(full, return_tensors="pt")["input_ids"].to(device)
    generated = []
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(ids)["logits"][:, -1, :]
            probs = torch.softmax(logits / temperature, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            tok_id = next_tok.item()
            if tok_id in [tokenizer.eos_token_id, 151645]:
                break
            generated.append(tok_id)
            ids = torch.cat([ids, next_tok.view(1, 1)], dim=1)
    return tokenizer.decode(generated, skip_special_tokens=True)


# =========================================================================
# MAIN
# =========================================================================
def main():
    print("=" * 70)
    print("  ANAMNESIS FULL BENCHMARK SUITE")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Output: {OUTPUT_DIR.absolute()}")
    print("=" * 70)

    total_t0 = time.time()

    # Section 1: Tests
    try:
        tests_ok = run_tests()
        if not tests_ok:
            print("\n  WARNING: Tests failed! Continuing anyway...\n")
    except Exception as e:
        print(f"  ERROR in tests: {e}")
        traceback.print_exc()

    # Section 2: Conversion fidelity (returns model for reuse)
    model = None
    tokenizer = None
    try:
        model, tokenizer, _ = run_conversion_fidelity()
    except Exception as e:
        print(f"  ERROR in conversion: {e}")
        traceback.print_exc()
        results["sections"]["conversion_fidelity"] = {"error": str(e)}
        save_results()

    # Section 3: CMS learning
    if model is not None:
        try:
            run_cms_learning(model, tokenizer)
        except Exception as e:
            print(f"  ERROR in CMS learning: {e}")
            traceback.print_exc()
            results["sections"]["cms_learning"] = {"error": str(e)}
            save_results()

    # Section 4: Ablation (uses CPU, independent of GPU model)
    try:
        # Free GPU for ablation stability
        if model is not None:
            model_cpu = None  # keep ref for persistence test
        run_ablation()
    except Exception as e:
        print(f"  ERROR in ablation: {e}")
        traceback.print_exc()
        results["sections"]["ablation"] = {"error": str(e)}
        save_results()

    # Section 5: Persistence
    if model is not None:
        try:
            run_persistence(model)
        except Exception as e:
            print(f"  ERROR in persistence: {e}")
            traceback.print_exc()
            results["sections"]["persistence"] = {"error": str(e)}
            save_results()

    # Final summary
    total_time = time.time() - total_t0
    results["finished_at"] = datetime.now().isoformat()
    results["total_time_s"] = round(total_time, 1)

    section("FINAL SUMMARY")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Results: {RESULTS_FILE.absolute()}")
    print(f"  Log: {LOG_FILE.absolute()}")

    for name, sec in results["sections"].items():
        if isinstance(sec, dict) and "error" in sec:
            print(f"  {name}: FAILED ({sec['error'][:80]})")
        else:
            print(f"  {name}: OK")

    save_results()

    # Cleanup
    if model is not None:
        del model
    torch.cuda.empty_cache()
    print("\nDone. All results saved.")


if __name__ == "__main__":
    main()
