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

# Set data and output directories based on defect type and number of samples
export DATA_DIR="/home/b502b586/scratch/SiemensEnergy/dataset/ft/${DEFECT_TYPE}"
export OUT_DIR="/home/b502b586/scratch/SiemensEnergy/lora-weights/ft/${DEFECT_TYPE}-defect-pill-${NUM_SAMPLES}-samples"

export CACHE_DIR="/scratch/b502b586"

# Go to training script directory
#git clone https://github.com/huggingface/diffusers
cd diffusers/examples/text_to_image

# Run training
accelerate launch train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path "$MODEL_NAME" \
  --train_data_dir "$DATA_DIR" \
  --image_column image \
  --caption_column caption \
  --cache_dir "$CACHE_DIR" \
  --resolution 256 \
  --mixed_precision fp16 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --max_train_steps 1200 \
  --checkpointing_steps 200 \
  --lr_scheduler cosine \
  --lr_warmup_steps 100 \
  --train_text_encoder \
  --rank 16 \
  --output_dir "$OUT_DIR"
  # --validation_prompt "a macro photo of a pill with a ${DEFECT_TYPE} defect"
