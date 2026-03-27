#!/usr/bin/env python3
"""
Anamnesis Production Server — models that specialize through conversation.

Converts Qwen 3B to Anamnesis, manages per-user memory sessions, and
provides interactive chat where the model learns from every conversation.

Usage:
    python serve_anamnesis.py                    # Interactive, default session
    python serve_anamnesis.py --session alice     # Named session
    python serve_anamnesis.py --no-cache          # Force re-conversion

Commands during chat:
    /reset    — Reset memory to soul checkpoint (or blank)
    /save     — Force save current session
    /status   — Show learning stats (surprise, updates, drift)
    /soul     — Save current state as soul checkpoint
    /learn    — Toggle learning on/off
    /quit     — Save and exit
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

sys.stdout.reconfigure(line_buffering=True)


@dataclass
class ServerConfig:
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    cache_dir: Path = field(default_factory=lambda: Path("data/anamnesis_cache"))
    sessions_dir: Path = field(default_factory=lambda: Path("data/sessions"))
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cms_levels: int = 2
    cms_chunk_sizes: tuple = (1, 32)
    cms_mem_dim: int = 512
    cms_mem_depth: int = 2
    cms_poly_degree: int = 2
    auto_save_every: int = 5
    max_gen_tokens: int = 512
    temperature: float = 0.7
    persona_dim: int = 256
    persona_layers: int = 4


# ── Model Loading ─────────────────────────────────────────────────────────────

def load_or_convert_model(config: ServerConfig):
    """Load from cache or convert from HuggingFace. Returns (model, tokenizer)."""
    from transformers import AutoTokenizer

    cache_state = config.cache_dir / "model_state.pt"
    cache_config = config.cache_dir / "hope_config.json"

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    if cache_state.exists() and cache_config.exists():
        print(f"Loading cached model from {config.cache_dir}...")
        from anamnesis.core.model import HopeModel, HopeConfig
        with open(cache_config) as f:
            cfg_dict = json.load(f)
        hope_config = HopeConfig(**cfg_dict)
        model = HopeModel(hope_config)
        state = torch.load(cache_state, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
        del state
        model = model.to(config.device, dtype=torch.bfloat16)
    else:
        print(f"Converting {config.model_name} to Anamnesis (first run)...")
        from transformers import AutoModelForCausalLM, AutoConfig
        from anamnesis.core.model import HopeConfig
        from anamnesis.convert.generic import model_to_hope

        src_config = AutoConfig.from_pretrained(config.model_name)
        src_model = AutoModelForCausalLM.from_pretrained(
            config.model_name, torch_dtype=torch.bfloat16, device_map=config.device,
        )
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
            cms_levels=config.cms_levels,
            cms_chunk_sizes=list(config.cms_chunk_sizes),
            cms_variant="nested",
            cms_hidden_mult=[r, r],
            cms_mem_dim=config.cms_mem_dim,
            cms_mem_depth=config.cms_mem_depth,
            cms_poly_degree=config.cms_poly_degree,
            use_neural_memory=False,
            tie_word_embeddings=False,
        )
        model = model_to_hope(src_model, hope_config, verbose=True)
        del src_model
        torch.cuda.empty_cache()
        model = model.to(config.device, dtype=torch.bfloat16)

        # Cache for next time
        print(f"Caching converted model to {config.cache_dir}...")
        config.cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), cache_state)
        cfg_dict = {
            k: (list(v) if isinstance(v, (list, tuple)) else v)
            for k, v in hope_config.__dict__.items()
        }
        with open(cache_config, "w") as f:
            json.dump(cfg_dict, f, indent=2)
        print(f"  Cached ({cache_state.stat().st_size / 1e9:.1f} GB)")

    model.eval()
    vram = torch.cuda.memory_allocated() / 1e9 if config.device == "cuda" else 0
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params | {vram:.1f} GB VRAM")
    return model, tokenizer


# ── Session State Persistence ─────────────────────────────────────────────────

def save_session_state(model, path: Path, metadata: dict | None = None):
    """Save DeepMemoryLevel state for session persistence."""
    from anamnesis.core.cms import DeepMemoryLevel
    state = {"version": 2, "layers": {}, "metadata": metadata or {}}

    for i, layer in enumerate(model.layers):
        layer_state = {}
        for lvl_idx, level in enumerate(layer.cms.levels):
            if isinstance(level, DeepMemoryLevel):
                lvl_state = {
                    "memory": {n: p.data.cpu() for n, p in level.memory.named_parameters()},
                    "momentum": {n: v.cpu() for n, v in level._momentum_state.items()},
                    "total_updates": level._total_updates,
                    "surprise_ema": level._surprise_ema,
                }
                if level._soul_weights:
                    lvl_state["soul"] = {n: v.cpu() for n, v in level._soul_weights.items()}
                layer_state[f"level_{lvl_idx}"] = lvl_state
        state["layers"][str(i)] = layer_state

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return path


def load_session_state(model, path: Path, device: str = "cuda") -> dict:
    """Load DeepMemoryLevel state from a saved session."""
    from anamnesis.core.cms import DeepMemoryLevel
    state = torch.load(path, map_location="cpu", weights_only=True)

    for i, layer in enumerate(model.layers):
        layer_key = str(i)
        if layer_key not in state["layers"]:
            continue
        for lvl_idx, level in enumerate(layer.cms.levels):
            lvl_key = f"level_{lvl_idx}"
            if not isinstance(level, DeepMemoryLevel):
                continue
            if lvl_key not in state["layers"][layer_key]:
                continue
            s = state["layers"][layer_key][lvl_key]

            # Restore memory MLP weights
            with torch.no_grad():
                for name, param in level.memory.named_parameters():
                    if name in s["memory"]:
                        param.data.copy_(s["memory"][name].to(device, dtype=param.dtype))

            # Restore momentum
            level._momentum_state = {
                n: v.to(device, dtype=torch.float32) for n, v in s.get("momentum", {}).items()
            }
            level._total_updates = s.get("total_updates", 0)
            level._surprise_ema = s.get("surprise_ema", 1.0)

            # Restore soul checkpoint if present
            if "soul" in s:
                level._soul_weights = {
                    n: v.to(device, dtype=torch.float32) for n, v in s["soul"].items()
                }

    return state.get("metadata", {})


# ── Session Manager ───────────────────────────────────────────────────────────

class SessionManager:
    """Manages per-user memory state and conversation."""

    def __init__(self, model, tokenizer, config: ServerConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.user_id: str | None = None
        self.turn_count: int = 0
        self.learning_enabled: bool = True

        # Enable learning on L1, disable on L0
        for layer in model.layers:
            layer.cms.levels[0].learning_enabled = False
            layer.cms.levels[1].learning_enabled = True

    def _session_dir(self) -> Path:
        return self.config.sessions_dir / self.user_id

    def start_session(self, user_id: str) -> str:
        """Start or resume a session for the given user."""
        self.user_id = user_id
        self.turn_count = 0
        session_dir = self._session_dir()
        state_path = session_dir / "memory_state.pt"

        if state_path.exists():
            meta = load_session_state(self.model, state_path, self.config.device)
            self.turn_count = meta.get("turn_count", 0)
            return f"Resumed session '{user_id}' (turn {self.turn_count}, {meta.get('total_updates', '?')} updates)"
        else:
            # New user — set up persona probes
            self.model.setup_persona_probes(
                persona_dim=self.config.persona_dim,
                num_final_layers=self.config.persona_layers,
            )
            return f"New session '{user_id}'"

    def end_session(self) -> str:
        """Save and close current session."""
        if self.user_id is None:
            return "No active session"
        path = self.save_session()
        msg = f"Session '{self.user_id}' saved ({self.turn_count} turns)"
        self.user_id = None
        return msg

    def save_session(self) -> Path:
        """Save current memory state to disk."""
        total_updates = sum(
            layer.cms.levels[1]._total_updates for layer in self.model.layers
        )
        meta = {
            "user_id": self.user_id,
            "turn_count": self.turn_count,
            "total_updates": total_updates,
            "saved_at": datetime.now().isoformat(),
        }
        path = self._session_dir() / "memory_state.pt"
        save_session_state(self.model, path, meta)

        # Also save metadata as readable JSON
        meta_path = self._session_dir() / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return path

    def save_soul(self) -> str:
        """Save current state as soul checkpoint."""
        for layer in self.model.layers:
            layer.cms.levels[1].save_soul()
        path = self._session_dir() / "soul_checkpoint.pt"
        save_session_state(self.model, path, {"type": "soul", "turn": self.turn_count})
        return f"Soul checkpoint saved at turn {self.turn_count}"

    def reset_memory(self) -> str:
        """Reset to soul checkpoint if available, otherwise full reset."""
        restored = False
        for layer in self.model.layers:
            lv = layer.cms.levels[1]
            if lv.reset_to_soul():
                restored = True
            else:
                lv.reset_state()
        if restored:
            return "Memory reset to soul checkpoint"
        return "Memory fully reset (no soul checkpoint found)"

    def get_status(self) -> dict:
        """Get current learning statistics."""
        surprises = []
        updates = []
        drifts = []
        for layer in self.model.layers:
            lv = layer.cms.levels[1]
            surprises.append(lv._surprise_ema)
            updates.append(lv._total_updates)
            if lv._soul_weights:
                drift = sum(
                    (p.data.float() - lv._soul_weights.get(n, p.data.float())).norm().item()
                    for n, p in lv.memory.named_parameters()
                    if n in lv._soul_weights
                )
                drifts.append(drift)

        return {
            "user": self.user_id,
            "turns": self.turn_count,
            "learning": self.learning_enabled,
            "avg_surprise": sum(surprises) / len(surprises) if surprises else 0,
            "total_updates": sum(updates),
            "avg_drift": sum(drifts) / len(drifts) if drifts else None,
        }

    def toggle_learning(self) -> str:
        """Toggle learning on/off."""
        self.learning_enabled = not self.learning_enabled
        for layer in self.model.layers:
            layer.cms.levels[1].learning_enabled = self.learning_enabled
        return f"Learning {'enabled' if self.learning_enabled else 'disabled'}"

    def chat(self, user_message: str):
        """Process one conversation turn. Yields tokens for streaming.

        Flow:
        1. Ingest prompt with learning ON (model reads user's input)
        2. Generate response token-by-token (learning auto-skips for seq_len=1)
        3. Replay full conversation with persona mask (model learns from exchange)
        """
        device = self.config.device

        # 1. INGEST: Feed prompt through model with learning ON
        prompt = f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
        prompt_ids = self.tokenizer(
            prompt, return_tensors="pt", max_length=1024, truncation=True,
        )["input_ids"].to(device)

        with torch.no_grad():
            self.model(prompt_ids)

        # 2. GENERATE: Token-by-token, streaming
        generated_ids = []
        ids = prompt_ids
        with torch.no_grad():
            for _ in range(self.config.max_gen_tokens):
                logits = self.model(ids)["logits"][:, -1, :]
                if self.config.temperature > 0:
                    probs = torch.softmax(logits / self.config.temperature, dim=-1)
                    next_tok = torch.multinomial(probs, 1)
                else:
                    next_tok = logits.argmax(dim=-1, keepdim=True)

                tok_id = next_tok.item()
                if tok_id in (self.tokenizer.eos_token_id, 151645, 151643):
                    break

                generated_ids.append(tok_id)
                yield self.tokenizer.decode([tok_id])
                ids = torch.cat([ids, next_tok.view(1, 1)], dim=1)

        # 3. REPLAY: Full conversation with persona mask
        if self.learning_enabled and generated_ids:
            assistant_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            full_turn = (
                f"<|im_start|>user\n{user_message}<|im_end|>\n"
                f"<|im_start|>assistant\n{assistant_text}<|im_end|>"
            )
            full_ids = self.tokenizer(
                full_turn, return_tensors="pt", max_length=1024, truncation=True,
            )["input_ids"].to(device)

            # Persona mask: 1.0 for assistant tokens, 0.3 for user tokens
            mask = self._build_persona_mask(full_ids)
            self.model.set_learning_weight(mask)
            with torch.no_grad():
                self.model(full_ids)
            self.model.set_learning_weight(None)

        # 4. AUTO-SAVE
        self.turn_count += 1
        if self.turn_count % self.config.auto_save_every == 0:
            self.save_session()

    def _build_persona_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Build persona mask: 1.0 for assistant tokens, 0.3 for user tokens."""
        batch, seq_len = token_ids.shape
        mask = torch.full((batch, seq_len), 0.3, device=token_ids.device)
        im_start_id = 151644  # Qwen <|im_start|> token
        for b in range(batch):
            positions = (token_ids[b] == im_start_id).nonzero(as_tuple=True)[0]
            if len(positions) >= 2:
                mask[b, positions[-1]:] = 1.0
            elif len(positions) == 1:
                mask[b] = 1.0
        return mask


# ── CLI Interface ─────────────────────────────────────────────────────────────

def cli_loop(mgr: SessionManager):
    """Interactive chat loop with slash commands."""
    print()
    user_id = input("Session name (or Enter for 'default'): ").strip() or "default"
    status = mgr.start_session(user_id)
    print(f"  {status}")
    print()
    print("  Commands: /reset /save /status /soul /learn /quit")
    print("  Type a message to chat. The model learns from every conversation.")
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        # Slash commands
        if user_input.startswith("/"):
            cmd = user_input.lower().split()[0]
            if cmd == "/quit":
                break
            elif cmd == "/save":
                path = mgr.save_session()
                print(f"  Saved to {path}")
            elif cmd == "/reset":
                print(f"  {mgr.reset_memory()}")
            elif cmd == "/soul":
                print(f"  {mgr.save_soul()}")
            elif cmd == "/learn":
                print(f"  {mgr.toggle_learning()}")
            elif cmd == "/status":
                s = mgr.get_status()
                print(f"  Session: {s['user']} | Turns: {s['turns']} | Learning: {s['learning']}")
                print(f"  Updates: {s['total_updates']} | Surprise: {s['avg_surprise']:.4f}")
                if s['avg_drift'] is not None:
                    print(f"  Drift from soul: {s['avg_drift']:.4f}")
            else:
                print(f"  Unknown command: {cmd}")
            continue

        # Chat with streaming
        print("Assistant: ", end="", flush=True)
        for token in mgr.chat(user_input):
            print(token, end="", flush=True)
        print("\n")

    # Save on exit
    msg = mgr.end_session()
    print(f"\n  {msg}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Anamnesis Production Server")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--session", default="", help="Auto-start with this session name")
    parser.add_argument("--no-cache", action="store_true", help="Force re-conversion")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--auto-save", type=int, default=5, help="Auto-save every N turns")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    config = ServerConfig(
        model_name=args.model,
        device=args.device,
        temperature=args.temperature,
        auto_save_every=args.auto_save,
        max_gen_tokens=args.max_tokens,
    )

    if args.no_cache and config.cache_dir.exists():
        import shutil
        shutil.rmtree(config.cache_dir)
        print("Cache cleared.")

    print("=" * 60)
    print("ANAMNESIS — Models that specialize through conversation")
    print("=" * 60)

    model, tokenizer = load_or_convert_model(config)
    mgr = SessionManager(model, tokenizer, config)

    if args.session:
        status = mgr.start_session(args.session)
        print(f"  {status}")

    cli_loop(mgr)

    if config.device == "cuda":
        print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
