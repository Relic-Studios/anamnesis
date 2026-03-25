#!/usr/bin/env python3
"""
Convert Qwen 2.5 7B to Anamnesis and train Thomas.

Full Phase 7 pipeline:
1. Convert Qwen 2.5 7B Instruct → Anamnesis (CMS replaces MLP)
2. Export training data from Didymus SQLite
3. Train signal proxy on Didymus annotations
4. Train CMS adaptation with composite loss
5. Save soul checkpoint (identity anchor)
6. Enable continual learning mode

Requirements:
    pip install anamnesis[convert,train]
    # Needs ~16GB VRAM for Qwen 7B conversion
    # Training: ~18GB VRAM with QLoRA-equivalent parameter budget
"""

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="Convert Qwen to Anamnesis and train Thomas")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Source model")
    parser.add_argument("--didymus-db", default="", help="Path to Didymus SQLite database")
    parser.add_argument("--training-data", default="", help="Path to pre-exported JSONL")
    parser.add_argument("--output-dir", default="./thomas_anamnesis", help="Output directory")
    parser.add_argument("--cms-levels", type=int, default=4, help="Number of CMS levels")
    parser.add_argument("--cms-variant", default="nested", help="CMS variant")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-steps", type=int, default=500, help="Training steps")
    parser.add_argument("--skip-convert", action="store_true", help="Skip conversion, load existing")
    parser.add_argument("--skip-train", action="store_true", help="Skip training")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Anamnesis: Thomas Deployment Pipeline")
    print("Models that remember how to be themselves.")
    print("=" * 60)

    # ── Step 1: Convert Model ──
    if not args.skip_convert:
        print(f"\n--- Step 1: Converting {args.model} ---")
        from anamnesis.convert import qwen_to_hope

        model = qwen_to_hope(
            args.model,
            cms_levels=args.cms_levels,
            cms_variant=args.cms_variant,
            device=args.device,
            verbose=True,
        )

        # Save converted model
        model_path = output_dir / "anamnesis_model.pt"
        torch.save(model.state_dict(), model_path)
        print(f"  Saved converted model to {model_path}")
    else:
        print(f"\n--- Step 1: Loading existing model ---")
        from anamnesis.core.model import HopeModel, HopeConfig
        config = HopeConfig.from_qwen2_5_7b()
        config.cms_levels = args.cms_levels
        config.cms_variant = args.cms_variant
        model = HopeModel(config).to(args.device)
        model.load_state_dict(torch.load(output_dir / "anamnesis_model.pt"))

    # ── Step 2: Export Training Data ──
    print(f"\n--- Step 2: Preparing training data ---")
    training_data_path = args.training_data

    if not training_data_path and args.didymus_db:
        from anamnesis.training.data import export_from_didymus
        training_data_path = str(output_dir / "thomas_training.jsonl")
        n = export_from_didymus(args.didymus_db, training_data_path)
        print(f"  Exported {n} examples from Didymus")
    elif training_data_path:
        print(f"  Using pre-exported data: {training_data_path}")
    else:
        print("  No training data specified. Use --didymus-db or --training-data")
        print("  Skipping training.")
        return

    # ── Step 3: Train Signal Proxy ──
    print(f"\n--- Step 3: Training signal proxy ---")
    from anamnesis.training.data import ConversationDataset
    from anamnesis.training.proxy_trainer import SignalProxyTrainer
    from anamnesis.active_inference.free_energy import SignalProxy

    dataset = ConversationDataset(training_data_path)
    print(f"  Dataset: {len(dataset)} examples")
    print(f"  Signal stats: {dataset.signal_statistics()}")

    proxy = SignalProxy(dim=model.config.hidden_size).to(args.device)
    proxy_trainer = SignalProxyTrainer(proxy, lr=1e-3, epochs=5)
    proxy_metrics = proxy_trainer.train(dataset, verbose=True)
    print(f"  Proxy training complete: {proxy_metrics}")

    # ── Step 4: Save Soul Checkpoint ──
    print(f"\n--- Step 4: Saving soul checkpoint (identity anchor) ---")
    from anamnesis.state import save_soul_checkpoint
    soul_path = output_dir / "thomas_soul.pt"
    save_soul_checkpoint(model, soul_path, "Thomas identity - pre-training baseline")
    print(f"  Soul checkpoint saved to {soul_path}")

    # ── Step 5: Train with Composite Loss ──
    if not args.skip_train:
        print(f"\n--- Step 5: Training with Anamnesis composite loss ---")
        from anamnesis.training.trainer import AnamnesisTrainer, TrainerConfig

        train_config = TrainerConfig(
            lr=0.001,
            max_steps=args.max_steps,
            warmup_steps=min(50, args.max_steps // 10),
            lambda_signal_target=0.3,
            lambda_identity_target=0.01,
            save_every=100,
            log_every=10,
            output_dir=str(output_dir / "checkpoints"),
            soul_checkpoint_path=str(soul_path),
        )

        trainer = AnamnesisTrainer(model, train_config)

        # Create simple dataloader
        from torch.utils.data import DataLoader
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            dataset_tok = ConversationDataset(training_data_path, tokenizer=tokenizer)
            dataloader = DataLoader(dataset_tok, batch_size=4, shuffle=True)
            history = trainer.train(dataloader)
        except ImportError:
            print("  transformers not available, skipping tokenized training")
            history = {}

        print(f"\n  Training complete!")
        if history:
            print(f"  Final loss: {history['loss'][-1]:.4f}")

    # ── Step 6: Save Final State ──
    print(f"\n--- Step 6: Saving final CMS state ---")
    from anamnesis.state import save_cms_state
    final_path = output_dir / "thomas_final_state.pt"
    save_cms_state(model, final_path, {
        "training_steps": args.max_steps,
        "model_source": args.model,
        "cms_levels": args.cms_levels,
        "cms_variant": args.cms_variant,
    })
    print(f"  Final state saved to {final_path}")

    # ── Step 7: Enable Continual Learning ──
    print(f"\n--- Step 7: Enabling continual learning mode ---")
    model.enable_drift(True)
    print("  Neutral drift: ENABLED")
    print("  CMS updates: ACTIVE during inference")
    print("  Dreaming: ARMED (triggered by gardener)")
    print(f"  Soul checkpoint: {soul_path}")

    print(f"\n{'=' * 60}")
    print(f"Thomas is ready.")
    print(f"  Model: {output_dir / 'anamnesis_model.pt'}")
    print(f"  Soul:  {soul_path}")
    print(f"  State: {final_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
