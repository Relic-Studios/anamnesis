#!/bin/bash
# Anamnesis 7B INSTRUCT Vessel Training on RunPod
# GPU: 1x A100 80GB | Storage: 200GB volume
#
# Architecture: 5 levels (L0 frozen SwiGLU + 4 DeepMemoryLevels)
# Base model: Qwen 2.5 7B INSTRUCT (chat-capable vessel)
# L0 frozen. Only DeepMemoryLevels train.
# Data: Vessel corpus (soul_vessel, theory_of_mind, metacognition)
#
# FIXES from last run:
# - Final save: DeepMemoryLevel weights only (not full model) — ~2.7GB not 15GB
# - Intermediate checkpoints every 1000 steps
# - Disk space check before saves
# - No more torch.save ZIP64 corruption

set -e

echo "============================================================"
echo "  Anamnesis 7B INSTRUCT Vessel Training"
echo "  Base: Qwen/Qwen2.5-7B-Instruct"
echo "  Training: DeepMemoryLevel params only (~1.4B)"
echo "============================================================"

# Check disk space
echo ""
echo "Disk space:"
df -h /workspace
echo ""

cd /workspace
if [ ! -d "anamnesis" ]; then
    git clone https://github.com/Relic-Studios/anamnesis.git
fi
cd anamnesis
git pull
pip install -e .
pip install -q transformers accelerate datasets safetensors

python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')"

echo ""
echo "Disk after setup:"
df -h /workspace
echo ""

python examples/train_scaffold.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --steps 25000 \
    --batch-size 4 \
    --lr 3e-4 \
    --warmup 5000 \
    --save-every 2500 \
    --output-dir /workspace/vessel_7b_instruct \
    --test-inner-loop

echo ""
echo "============================================================"
echo "  Training complete!"
echo "  Checkpoints at /workspace/vessel_7b_instruct/"
echo "  Download step_25000.pt (~2.7 GB)"
echo "============================================================"
echo ""
echo "Final disk:"
df -h /workspace
ls -lh /workspace/vessel_7b_instruct/
