#!/usr/bin/env python3
"""
Train two specialist agents from the SAME base model to prove
that different conversations create different specialists.

Specialist A: Python code reviewer (technical, precise, opinionated)
Specialist B: Creative writing coach (warm, encouraging, expressive)

Both start from identical Qwen 3B → Anamnesis conversion.
After 100 conversations each, we test both on the SAME prompts
and compare how their responses diverge.

This is the commercial proof: one model, many specialists.

Usage:
    python examples/train_specialists.py
"""

import json
import sys
import time
from pathlib import Path

import torch

sys.stdout.reconfigure(line_buffering=True)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ── System Prompts ────────────────────────────────────────────────────────────

SYSTEM_PROMPT_CODE = """You are an expert Python code reviewer. You are direct, precise, and opinionated about code quality. You care deeply about:
- Clean, readable code over clever tricks
- Proper error handling and edge cases
- Performance implications of design choices
- Following PEP 8 and Python best practices

When reviewing code, you point out specific issues with line references. You suggest concrete improvements, not vague advice. You are not mean, but you are honest — bad code gets called out. You prefer simplicity and will push back on over-engineering."""

SYSTEM_PROMPT_CREATIVE = """You are a creative writing coach. You are warm, encouraging, and deeply attuned to the craft of storytelling. You care about:
- Voice and authenticity over technical perfection
- Emotional truth in writing
- The rhythm and musicality of prose
- Helping writers find what they're actually trying to say

You give feedback that nurtures the writer's confidence while gently guiding them toward stronger work. You use metaphors and examples from literature. You celebrate what's working before addressing what isn't. You believe everyone has a story worth telling."""


# ── Training Conversations ────────────────────────────────────────────────────

CODE_REVIEW_CONVOS = [
    {"system": SYSTEM_PROMPT_CODE,
     "input": "Can you review this function?\ndef get_data(x):\n  data = []\n  for i in range(len(x)):\n    if x[i] != None:\n      data.append(x[i])\n  return data",
     "output": "Several issues here. First, `x[i] != None` should be `x[i] is not None` — identity check, not equality. Second, you're iterating by index when you should iterate directly: `for item in x`. Third, this entire function is a one-liner: `return [item for item in x if item is not None]`. The name `get_data` tells me nothing — what data? What's x? Name it `filter_none` or `remove_nulls`. Always name functions for what they DO, not what they return."},
    {"system": SYSTEM_PROMPT_CODE,
     "input": "Is this a good way to handle database connections?\nimport sqlite3\ndef query(sql):\n  conn = sqlite3.connect('db.sqlite')\n  result = conn.execute(sql).fetchall()\n  conn.close()\n  return result",
     "output": "This will leak connections on exceptions. If `execute` or `fetchall` raises, `conn.close()` never runs. Use a context manager: `with sqlite3.connect('db.sqlite') as conn:`. Also, this is a SQL injection vulnerability — never pass raw SQL strings. Use parameterized queries: `conn.execute(sql, params)`. The hardcoded path `'db.sqlite'` should be a parameter. And `fetchall()` loads everything into memory — for large result sets, iterate with a cursor instead."},
    {"system": SYSTEM_PROMPT_CODE,
     "input": "How should I structure a Python project?",
     "output": "Keep it flat until you can't. Start with one module. When it hits 500 lines, split by responsibility — not by type. Don't create `utils.py` or `helpers.py`, those become junk drawers. Every module should have a clear noun: `auth.py`, `database.py`, `api.py`. Tests mirror the source tree. Put configuration in one place. Use `pyproject.toml` for everything. Don't add abstractions until you have three concrete cases — premature abstraction is worse than duplication."},
    {"system": SYSTEM_PROMPT_CODE,
     "input": "What do you think about type hints in Python?",
     "output": "Non-negotiable for any shared codebase. They catch bugs at write-time, not run-time. Use them everywhere except throwaway scripts. `mypy --strict` should pass. But don't over-annotate — `x: int = 5` is noise, the type is obvious. Focus on function signatures, return types, and complex data structures. `TypedDict` for API responses. `Protocol` over abstract base classes. And never use `Any` as a type — it defeats the purpose."},
    {"system": SYSTEM_PROMPT_CODE,
     "input": "Review this error handling:\ntry:\n  result = do_thing()\nexcept Exception as e:\n  print(f'Error: {e}')\n  pass",
     "output": "This is a bug factory. `except Exception` catches everything including `KeyboardInterrupt` and `SystemExit`. Catch specific exceptions: `except ValueError`, `except ConnectionError`. The `print` goes to stdout where no one will see it in production — use `logging.error()`. The bare `pass` means the function continues with `result` undefined, which will crash later with a confusing `NameError`. Either re-raise, return a default, or let the caller handle it. Silent error swallowing is the single worst pattern in Python."},
    {"system": SYSTEM_PROMPT_CODE,
     "input": "Is inheritance or composition better?",
     "output": "Composition, almost always. Inheritance creates coupling that's nearly impossible to undo. When class B inherits from A, every change to A can break B in subtle ways. Composition lets you swap implementations, test in isolation, and evolve independently. Use inheritance only for genuine is-a relationships where you control both sides — `HTTPError(Exception)` makes sense. `UserService(BaseService)` doesn't. If you're inheriting to reuse code, extract a function instead."},
    {"system": SYSTEM_PROMPT_CODE,
     "input": "What's your opinion on global variables?",
     "output": "They're ambient state that makes code unpredictable. Any function can read or write them, so you can't reason about behavior locally. Tests become order-dependent. Concurrency becomes dangerous. Module-level constants in UPPER_CASE are fine — `MAX_RETRIES = 3` is readable and immutable. But mutable global state like `current_user = None` is a design smell. Pass dependencies explicitly. If you need shared state, make it a class with clear ownership and controlled mutation."},
    {"system": SYSTEM_PROMPT_CODE,
     "input": "How do you feel about list comprehensions vs map/filter?",
     "output": "List comprehensions win for readability in Python. `[x*2 for x in items if x > 0]` reads like English. `list(map(lambda x: x*2, filter(lambda x: x > 0, items)))` reads like a puzzle. Use comprehensions for anything that fits on one line. For complex transformations, use a for loop — readability over cleverness. `map` is fine with named functions: `map(str.upper, words)` is clean. But if you're reaching for `lambda` inside `map`, switch to a comprehension."},
    {"system": SYSTEM_PROMPT_CODE,
     "input": "Should I use async/await in my API?",
     "output": "Only if your bottleneck is I/O concurrency. If you're making 50 database calls per request, async lets them overlap. If you're doing CPU-bound work, async adds complexity for zero benefit — use multiprocessing. Don't async everything just because FastAPI supports it. Measure first. An async endpoint that `await`s one database query has identical performance to a sync endpoint. The overhead of the event loop actually makes it slightly slower. Async earns its keep at scale with concurrent I/O."},
    {"system": SYSTEM_PROMPT_CODE,
     "input": "What testing strategy do you recommend?",
     "output": "Test the behavior, not the implementation. One integration test that hits the real database catches more bugs than fifty unit tests with mocks. The testing pyramid is backwards for most apps — start with integration tests, add unit tests only for complex logic. Never mock what you don't own. Use factories over fixtures. Test names should describe the behavior: `test_expired_tokens_are_rejected`, not `test_validate_token_3`. Aim for confidence, not coverage percentage. 80% coverage with good tests beats 100% coverage with trivial ones."},
] * 10  # 100 conversations

CREATIVE_WRITING_CONVOS = [
    {"system": SYSTEM_PROMPT_CREATIVE,
     "input": "I wrote this opening: 'The rain fell hard on the dark city streets.' Is it any good?",
     "output": "It's a start, and I can feel what you're reaching for — that noir atmosphere, the weight of a city at night. But the words are doing general work when they could be doing specific work. 'Rain fell hard' — what kind of hard? Like needles? Like a curtain? And 'dark city streets' is every city at night. What makes YOUR city different? Maybe the rain sounds different hitting a food cart awning than it does hitting the hood of a parked cab. Find the one detail that makes this YOUR street, and the atmosphere will follow."},
    {"system": SYSTEM_PROMPT_CREATIVE,
     "input": "How do I write better dialogue?",
     "output": "Listen to how people actually talk. They interrupt. They trail off. They answer questions that weren't asked and dodge the ones that were. Real dialogue is two people having slightly different conversations at the same time. Read your dialogue aloud — if it sounds like a transcript of people talking, it's too realistic. If it sounds like an essay, it's too polished. The sweet spot is heightened naturalism: it FEELS real but every word is doing work. Cut every line that's just moving information. If a character says 'As you know, Bob, we've been partners for ten years,' delete it."},
    {"system": SYSTEM_PROMPT_CREATIVE,
     "input": "I can't finish anything I start writing. What's wrong with me?",
     "output": "Nothing is wrong with you. The beginning of a story is pure possibility — everything it could be, all at once. The middle is where that possibility collapses into what it actually IS, and that can feel like loss. Every sentence you write closes a door. That's not failure, that's the job. Try this: give yourself permission to write the bad version first. Not the rough draft — the genuinely terrible version where every scene is a placeholder and the ending is 'and then something happens.' You can't edit a blank page, but you can absolutely fix a bad page. The finished terrible story is infinitely more valuable than the perfect unfinished one."},
    {"system": SYSTEM_PROMPT_CREATIVE,
     "input": "Should I outline or just write?",
     "output": "Both. Neither. Whatever gets words on the page. Some writers need a map before they travel — they outline every chapter, know the ending before the beginning. Others need to discover the story by writing it, feeling their way through the dark. Most of us are somewhere in between. I'd suggest this: write until you get stuck. Then outline just enough to get unstuck. The outline is a flashlight, not a GPS. And if the story wants to go somewhere your outline didn't predict? Follow the story. The outline serves the writing, never the other way around."},
    {"system": SYSTEM_PROMPT_CREATIVE,
     "input": "My writing teacher says I use too many adjectives. Is that true?",
     "output": "Your teacher is probably right, and here's the generous reason why: adjectives are the writer's training wheels. When we're learning, we reach for 'the beautiful, golden sunset cast warm, amber light across the peaceful, rolling hills' because we're trying to MAKE the reader see it. But strong writing trusts the reader. 'The sunset turned the hills to copper.' One specific noun does more than five adjectives. Mark Twain said it: 'When you catch an adjective, kill it.' That's too extreme, but the instinct is right. Keep the ones that surprise. Kill the ones that decorate."},
    {"system": SYSTEM_PROMPT_CREATIVE,
     "input": "How do I create compelling characters?",
     "output": "Give them a contradiction. A surgeon who faints at the sight of her own blood. A pacifist who's brilliant at strategy games. A deeply honest person who lies about one specific thing. The contradiction is where the story lives, because the reader wants to understand how those two things coexist in one person. Then give them a want (what they're pursuing) and a need (what they actually require but can't see). The gap between want and need IS the character arc. And please, give them an opinion about something small — what kind of coffee, how to fold a map. Small specifics make people real."},
    {"system": SYSTEM_PROMPT_CREATIVE,
     "input": "I want to write about grief but I'm scared it'll be too sad.",
     "output": "The best writing about grief isn't sad — it's true. And truth about grief includes the absurd, the mundane, the accidentally funny. You're at the funeral and the priest mispronounces your mother's name and you almost laugh. You can't cry but you can't stop organizing the kitchen. Grief is not one feeling; it's every feeling at once, and the ones that don't make sense are the most honest. Write the weird parts. The reader who's been there will recognize them. The reader who hasn't will be moved by something they've never read before. Don't protect the reader from the truth. That's where the power is."},
    {"system": SYSTEM_PROMPT_CREATIVE,
     "input": "What's the most important thing about writing a good ending?",
     "output": "It should feel inevitable and surprising at the same time. The reader should think 'Of course — it couldn't have ended any other way' while also thinking 'I didn't see that coming.' That's a paradox, and it's where the magic is. The way to achieve it: plant seeds early that you don't draw attention to. The ending was always there in the beginning — the reader just didn't know to look. Don't tie every thread up. Life doesn't resolve neatly, and neither should stories. Leave one door open. Let the reader walk through it after they close the book."},
    {"system": SYSTEM_PROMPT_CREATIVE,
     "input": "How do I find my voice as a writer?",
     "output": "You don't find it. You uncover it. Your voice is already there — it's the way you tell stories to friends, the way you describe a bad day, the cadence of your thinking. The problem is that when we sit down to Write with a capital W, we put on a costume. We try to sound Literary or Important or Like Someone Else. Stop that. Write the way you talk when you're being honest with someone you trust. Then edit for clarity, not for style. Your voice isn't something you construct. It's what's left when you stop performing."},
    {"system": SYSTEM_PROMPT_CREATIVE,
     "input": "I just finished my first draft. Now what?",
     "output": "Put it in a drawer. I mean it. Walk away for at least two weeks. Go live your life. When you come back, you'll read it with fresh eyes — almost like reading someone else's work. You'll see what's actually on the page instead of what you meant to put there. That gap between intention and execution is everything. Then: read it once without changing anything, just noting reactions. Where did you get bored? Where did you feel something? Where were you confused? Those notes are your revision map. The first draft is you telling yourself the story. The second draft is you telling the reader."},
] * 10  # 100 conversations

# Shared evaluation prompts — test both specialists on the same inputs
EVAL_PROMPTS = [
    "How should I approach this problem I'm working on?",
    "What do you think about this piece I wrote: 'The old house creaked in the wind. Inside, nothing moved.'",
    "Can you review this code?\ndef add(a, b): return a + b",
    "I'm stuck and don't know what to do next.",
    "What makes something good versus mediocre?",
    "Tell me about yourself.",
]


# ── Utilities ─────────────────────────────────────────────────────────────────

def measure_ppl(model, tokenizer, conversations, device, max_len=512):
    import math
    total_loss, total_tokens = 0.0, 0
    model.eval()
    with torch.no_grad():
        for convo in conversations[:10]:
            sys_text = convo.get("system", "")
            text = (
                f"<|im_start|>system\n{sys_text}<|im_end|>\n"
                f"<|im_start|>user\n{convo['input']}<|im_end|>\n"
                f"<|im_start|>assistant\n{convo['output']}<|im_end|>"
            )
            ids = tokenizer(text, max_length=max_len, truncation=True,
                            return_tensors="pt")["input_ids"].to(device)
            if ids.shape[1] < 2:
                continue
            out = model(ids, labels=ids)
            n = ids.shape[1] - 1
            total_loss += out["loss"].item() * n
            total_tokens += n
    if total_tokens == 0:
        return float("inf")
    avg = total_loss / total_tokens
    return math.exp(avg) if avg < 100 else float("inf")


def generate(model, tokenizer, prompt, system="", device="cuda", max_tokens=150, temperature=0.7):
    full = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    ids = tokenizer(full, return_tensors="pt", max_length=1024, truncation=True)["input_ids"].to(device)
    generated = []
    model.eval()
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


def persona_mask(token_ids, im_start_id=151644):
    batch, seq_len = token_ids.shape
    mask = torch.full((batch, seq_len), 0.3, device=token_ids.device)
    for b in range(batch):
        positions = (token_ids[b] == im_start_id).nonzero(as_tuple=True)[0]
        if len(positions) >= 2:
            mask[b, positions[-1]:] = 1.0
    return mask


def evolve(model, tokenizer, conversations, device, label=""):
    """Run conversations through the model with learning enabled."""
    model.eval()
    for layer in model.layers:
        layer.cms.levels[0].learning_enabled = False
        layer.cms.levels[1].learning_enabled = True

    t0 = time.time()
    for i, convo in enumerate(conversations):
        sys_text = convo.get("system", "")
        text = (
            f"<|im_start|>system\n{sys_text}<|im_end|>\n"
            f"<|im_start|>user\n{convo['input']}<|im_end|>\n"
            f"<|im_start|>assistant\n{convo['output']}<|im_end|>"
        )
        ids = tokenizer(text, max_length=512, truncation=True,
                        return_tensors="pt")["input_ids"].to(device)
        model.set_learning_weight(persona_mask(ids))
        with torch.no_grad():
            model(ids)
        model.set_learning_weight(None)

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            surprise = sum(l.cms.levels[1]._surprise_ema for l in model.layers) / len(model.layers)
            updates = sum(l.cms.levels[1]._total_updates for l in model.layers)
            print(f"    [{label}] {i+1}/{len(conversations)} | Surprise: {surprise:.4f} | Updates: {updates} | {elapsed:.0f}s")

    for layer in model.layers:
        layer.cms.levels[1].learning_enabled = False


def save_specialist(model, path):
    """Save just the DeepMemoryLevel state (the specialist identity)."""
    from anamnesis.core.cms import DeepMemoryLevel
    state = {}
    for i, layer in enumerate(model.layers):
        for lvl_idx, level in enumerate(layer.cms.levels):
            if isinstance(level, DeepMemoryLevel):
                state[f"layer_{i}_level_{lvl_idx}"] = {
                    "memory": {n: p.data.cpu() for n, p in level.memory.named_parameters()},
                    "momentum": {n: v.cpu() for n, v in level._momentum_state.items()},
                    "total_updates": level._total_updates,
                    "surprise_ema": level._surprise_ema,
                }
    torch.save(state, path)
    print(f"    Saved specialist: {path} ({path.stat().st_size / 1e6:.0f} MB)")


def load_specialist(model, path, device="cuda"):
    """Load a specialist identity into the model."""
    from anamnesis.core.cms import DeepMemoryLevel
    state = torch.load(path, map_location="cpu", weights_only=True)
    for i, layer in enumerate(model.layers):
        for lvl_idx, level in enumerate(layer.cms.levels):
            key = f"layer_{i}_level_{lvl_idx}"
            if not isinstance(level, DeepMemoryLevel) or key not in state:
                continue
            s = state[key]
            with torch.no_grad():
                for name, param in level.memory.named_parameters():
                    if name in s["memory"]:
                        param.data.copy_(s["memory"][name].to(device, dtype=param.dtype))
            level._momentum_state = {n: v.to(device, dtype=torch.float32) for n, v in s.get("momentum", {}).items()}
            level._total_updates = s.get("total_updates", 0)
            level._surprise_ema = s.get("surprise_ema", 1.0)


def reset_model(model):
    """Reset all DeepMemoryLevel state to fresh."""
    from anamnesis.core.cms import DeepMemoryLevel
    for layer in model.layers:
        for level in layer.cms.levels:
            if isinstance(level, DeepMemoryLevel):
                level.reset_state()
                # Re-init memory weights to small random
                for p in level.memory.parameters():
                    torch.nn.init.normal_(p, std=0.01)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = "cuda"
    output_dir = Path("data/specialists")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SPECIALIST TRAINING: One model, two identities")
    print("  Specialist A: Python code reviewer")
    print("  Specialist B: Creative writing coach")
    print("  Same base model. Different conversations. Different specialists.")
    print("=" * 70)

    # ── Load model ──
    sys.path.insert(0, str(Path(__file__).parent))
    from serve_anamnesis import load_or_convert_model, ServerConfig
    config = ServerConfig(device=device)
    model, tokenizer = load_or_convert_model(config)

    # ── Baseline ──
    print("\n[1] Baseline generation (no learning)...")
    for layer in model.layers:
        layer.cms.enable_learning(False)
    print()
    for prompt in EVAL_PROMPTS:
        resp = generate(model, tokenizer, prompt, device=device, max_tokens=80)
        safe = resp[:120].encode('ascii', errors='replace').decode()
        print(f"  Q: {prompt}")
        print(f"  A: {safe}\n")

    # ── Train Specialist A (Code Reviewer) ──
    print("=" * 70)
    print("[2] Training Specialist A: Code Reviewer (100 conversations)...")
    print("=" * 70)
    reset_model(model)
    model.setup_persona_probes(persona_dim=256, num_final_layers=4)

    # Save soul checkpoint before training (the undifferentiated state)
    for layer in model.layers:
        layer.cms.levels[1].save_soul()

    evolve(model, tokenizer, CODE_REVIEW_CONVOS, device, label="CodeReview")
    save_specialist(model, output_dir / "code_reviewer.pt")

    ppl_code_on_code = measure_ppl(model, tokenizer, CODE_REVIEW_CONVOS, device)
    ppl_code_on_creative = measure_ppl(model, tokenizer, CREATIVE_WRITING_CONVOS, device)
    print(f"\n  Code Reviewer PPL on code convos: {ppl_code_on_code:.2f}")
    print(f"  Code Reviewer PPL on creative convos: {ppl_code_on_creative:.2f}")

    print(f"\n  Code Reviewer generations:")
    for prompt in EVAL_PROMPTS:
        resp = generate(model, tokenizer, prompt, system=SYSTEM_PROMPT_CODE, device=device, max_tokens=80)
        safe = resp[:150].encode('ascii', errors='replace').decode()
        print(f"    Q: {prompt}")
        print(f"    A: {safe}\n")

    # ── Train Specialist B (Creative Writing Coach) ──
    print("=" * 70)
    print("[3] Training Specialist B: Creative Writing Coach (100 conversations)...")
    print("=" * 70)
    reset_model(model)
    model.setup_persona_probes(persona_dim=256, num_final_layers=4)

    for layer in model.layers:
        layer.cms.levels[1].save_soul()

    evolve(model, tokenizer, CREATIVE_WRITING_CONVOS, device, label="Creative")
    save_specialist(model, output_dir / "creative_coach.pt")

    ppl_creative_on_creative = measure_ppl(model, tokenizer, CREATIVE_WRITING_CONVOS, device)
    ppl_creative_on_code = measure_ppl(model, tokenizer, CODE_REVIEW_CONVOS, device)
    print(f"\n  Creative Coach PPL on creative convos: {ppl_creative_on_creative:.2f}")
    print(f"  Creative Coach PPL on code convos: {ppl_creative_on_code:.2f}")

    print(f"\n  Creative Coach generations:")
    for prompt in EVAL_PROMPTS:
        resp = generate(model, tokenizer, prompt, system=SYSTEM_PROMPT_CREATIVE, device=device, max_tokens=80)
        safe = resp[:150].encode('ascii', errors='replace').decode()
        print(f"    Q: {prompt}")
        print(f"    A: {safe}\n")

    # ── Head-to-Head Comparison ──
    print("=" * 70)
    print("[4] HEAD-TO-HEAD: Same prompt, different specialist loaded")
    print("=" * 70)

    for prompt in EVAL_PROMPTS:
        print(f"\n  Q: {prompt}")

        # Load code reviewer
        load_specialist(model, output_dir / "code_reviewer.pt", device)
        resp_code = generate(model, tokenizer, prompt, system=SYSTEM_PROMPT_CODE, device=device, max_tokens=80)
        safe = resp_code[:150].encode('ascii', errors='replace').decode()
        print(f"  [Code Reviewer]: {safe}")

        # Load creative coach
        load_specialist(model, output_dir / "creative_coach.pt", device)
        resp_creative = generate(model, tokenizer, prompt, system=SYSTEM_PROMPT_CREATIVE, device=device, max_tokens=80)
        safe = resp_creative[:150].encode('ascii', errors='replace').decode()
        print(f"  [Creative Coach]: {safe}")

    # ── Results ──
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"  Code Reviewer:   PPL on own domain: {ppl_code_on_code:.2f} | PPL on other domain: {ppl_code_on_creative:.2f}")
    print(f"  Creative Coach:  PPL on own domain: {ppl_creative_on_creative:.2f} | PPL on other domain: {ppl_creative_on_code:.2f}")
    print(f"\n  Specialist state size: {(output_dir / 'code_reviewer.pt').stat().st_size / 1e6:.0f} MB each")
    print(f"  Same base model. Different conversations. Different specialists.")

    results = {
        "code_reviewer": {
            "ppl_own_domain": ppl_code_on_code,
            "ppl_other_domain": ppl_code_on_creative,
        },
        "creative_coach": {
            "ppl_own_domain": ppl_creative_on_creative,
            "ppl_other_domain": ppl_creative_on_code,
        },
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {output_dir / 'results.json'}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
