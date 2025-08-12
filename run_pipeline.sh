#!/usr/bin/env bash

# ----------------------------
# Usage
# ----------------------------
# bash run_pipeline.sh <defect_type> <num_samples>
# e.g.:
# bash run_pipeline.sh color 20

# ----------------------------
# Inputs
# ----------------------------
DEFECT_TYPE="${1:-color}"
NUM_SAMPLES="${2:-20}"

# ----------------------------
# Project root (edit this once)
# ----------------------------
ROOT_DIR="/scratch/b502b586/SiemensEnergy"


# ----------------------------
# Derived paths
# ----------------------------
DATASET_DIR="${ROOT_DIR}/dataset"
SYNTH_DIR="${DATASET_DIR}/synthetic"
LORA_ROOT="${ROOT_DIR}/lora-weights"
EVAL_DIR="${ROOT_DIR}/evaluations"

# Few-shot sampler output (TI and FT splits)
TI_DATA_DIR="${DATASET_DIR}/ti/${DEFECT_TYPE}-${NUM_SAMPLES}-samples"
FT_DATA_DIR="${DATASET_DIR}/ft/${DEFECT_TYPE}-${NUM_SAMPLES}-samples"

# Inference output per defect
INFER_OUT_DIR="${SYNTH_DIR}/${DEFECT_TYPE}"

# ----------------------------
# Summary
# ----------------------------
echo "== PIPELINE START =="
echo "Defect type     : ${DEFECT_TYPE}"
echo "Num samples     : ${NUM_SAMPLES}"
echo "Project ROOT_DIR: ${ROOT_DIR}"
echo "TI data dir     : ${TI_DATA_DIR}"
echo "FT data dir     : ${FT_DATA_DIR}"
echo "LoRA root       : ${LORA_ROOT}"
echo "Inference out   : ${INFER_OUT_DIR}"
echo "Evaluations dir : ${EVAL_DIR}"
echo "===================="

# ----------------------------
# 1) Few-shot sampling (creates both TI and/or FT dirs, as you need)
#    Adjust flags to your sampler as appropriate.
# ----------------------------
python few_shot_sampler.py \
  --num_sample "${NUM_SAMPLES}" \
  --out_dir   "${DATASET_DIR}" \
  --ti_dataset True

# ----------------------------
# 2) Fine-tune (LoRA) for this defect & sample count
# ----------------------------
sh ./scripts/fine-tune.sh "${DEFECT_TYPE}" "${NUM_SAMPLES}"

# ----------------------------
# 3) Textual Inversion for this defect & sample count
# ----------------------------
sh ./scripts/fine-tune-textual-inversion.sh "${DEFECT_TYPE}" "${NUM_SAMPLES}"

# ----------------------------
# 4) Inference with the fine-tuned models
#    Adjust flags to match your inference.py signature.
# ----------------------------
python inference.py \
  --num_samples "${NUM_SAMPLES}" \
  --out_dir     "${INFER_OUT_DIR}" \
  --lora_dir    "${LORA_ROOT}" \
  --lora_samples "${NUM_SAMPLES}" \
  --enable_ti   True \
  --defect_type "${DEFECT_TYPE}"

# ----------------------------
# 5) Evaluation
#    Set orig_data_dir to your *reference* real data for this defect.
#    Set synthetic_data_dir to where inference saved images.
#    (If your inference saves directly under ${INFER_OUT_DIR}, adjust path below.)
# ----------------------------
python evaluate.py \
  --out_dir            "${EVAL_DIR}" \
  --orig_data_dir      "${TI_DATA_DIR%*-samples}"   \
  --synthetic_data_dir "${INFER_OUT_DIR}/img" \
  --file_name          "results_${DEFECT_TYPE}_${NUM_SAMPLES}s.csv"

echo "== PIPELINE DONE =="
