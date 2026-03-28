#!/bin/bash
# Anamnesis 7B Vessel Training on RunPod
# GPU: 1x A100 80GB
#
# Architecture: 5 levels (L0 frozen SwiGLU + 4 DeepMemoryLevels)
# Base model: Qwen 2.5 7B (NOT Instruct — empty vessel)
# L0 trains at 1/10th LR to become part of the vessel
# Data: Wikipedia + vessel corpus (soul_vessel, theory_of_mind, metacognition)

set -e

echo "============================================================"
echo "  Anamnesis 7B Vessel Training (Base Model + 4 Memory Levels)"
echo "============================================================"

cd /workspace
if [ ! -d "anamnesis" ]; then
    git clone https://github.com/Relic-Studios/anamnesis.git
fi
cd anamnesis
git pull
pip install -e .
pip install -q transformers accelerate datasets

# Copy vessel corpus if available locally, otherwise it loads from repo
if [ -d "/workspace/vessel_data" ]; then
    echo "Using local vessel corpus"
    VESSEL_DIR="/workspace/vessel_data"
else
    echo "Using repo vessel corpus"
    VESSEL_DIR="data/scaffold_training"
fi

python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')"

python examples/train_scaffold.py \
    --model Qwen/Qwen2.5-7B \
    --steps 25000 \
    --batch-size 4 \
    --lr 3e-4 \
    --warmup 500 \
    --output-dir /workspace/vessel_7b \
    --test-inner-loop

echo ""
echo "============================================================"
echo "  Training complete. Download /workspace/vessel_7b/final.pt"
echo "============================================================"
