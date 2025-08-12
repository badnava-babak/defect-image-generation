#!/bin/bash

# Check if arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <defect_type> <num_samples>"
    echo "Example: $0 color 10"
    exit 1
fi

DEFECT_TYPE="$1"
NUM_SAMPLES="$2"

# Model path
export MODEL_NAME="/scratch/b502b586/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b/"
export CACHE_DIR="/scratch/b502b586"

# Set data and output directories based on defect type and number of samples
export DATA_DIR="/home/b502b586/scratch/SiemensEnergy/dataset/ti/${DEFECT_TYPE}"
export OUT_DIR="/home/b502b586/scratch/SiemensEnergy/lora-weights/ti/${DEFECT_TYPE}-defect-pill-${NUM_SAMPLES}-samples"

# Set placeholder and initializer tokens
export TOKEN="<sks_${DEFECT_TYPE}_defect>"
export INIT="$DEFECT_TYPE"

# Go to training script directory
#git clone https://github.com/huggingface/diffusers
cd diffusers/examples/textual_inversion

# Run training
accelerate launch textual_inversion_sdxl.py \
  --pretrained_model_name_or_path "$MODEL_NAME" \
  --train_data_dir "$DATA_DIR" \
  --placeholder_token "$TOKEN" \
  --initializer_token "$INIT" \
  --resolution 256 \
  --mixed_precision bf16 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_train_steps 2000 \
  --learning_rate 1e-4 \
  --lr_scheduler cosine \
  --lr_warmup_steps 100 \
  --output_dir "$OUT_DIR"
