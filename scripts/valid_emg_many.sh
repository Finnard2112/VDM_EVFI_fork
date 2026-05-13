#!/bin/bash
#SBATCH --job-name=VDM_EVFI_EMG_VALID
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --qos=medium
#SBATCH --account=nexus
#SBATCH --partition=tron
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/fs/vulcan-projects/Force_Learning/phan2003/%j_valid.out
#SBATCH --error=/fs/vulcan-projects/Force_Learning/phan2003/%j_valid.err
#SBATCH --mail-user=finnard2112@gmail.com
#SBATCH --mail-type=END,FAIL

set -euo pipefail

source /nfshomes/phan2003/miniconda3/etc/profile.d/conda.sh
conda activate VDM_EVFI
cd /nfshomes/phan2003/VDM_EVFI/scripts

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

RUN_DIR="${RUN_DIR:-/fs/vulcan-projects/Force_Learning/phan2003/output_EMG_controlnet_run1}"
VIDEO_ROOT="${VIDEO_ROOT:-/fs/vulcan-projects/Force_Learning/phan2003/videos}"
EMG_DATA_ROOT="${EMG_DATA_ROOT:-/fs/vulcan-projects/Force_Learning/EMG}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-stabilityai/stable-video-diffusion-img2vid}"

if [[ -n "${CHECKPOINTS:-}" ]]; then
    read -r -a CHECKPOINT_LIST <<< "${CHECKPOINTS}"
elif [[ -n "${CHECKPOINT_DIR:-}" ]]; then
    CHECKPOINT_LIST=("${CHECKPOINT_DIR}")
else
    CHECKPOINT_LIST=("$(find "${RUN_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)")
fi

if [[ "${#CHECKPOINT_LIST[@]}" -eq 0 || -z "${CHECKPOINT_LIST[0]}" ]]; then
    echo "Could not find any checkpoint under ${RUN_DIR}" >&2
    exit 1
fi

OUT_ROOT="${OUT_ROOT:-${RUN_DIR}/extra_validation_notiled}"

# Override from the command line with:
#   VIDEOS="Sirguta2_video.mp4 Merawi1_video.mp4" sbatch scripts/valid_emg_many.sh
if [[ -n "${VIDEOS:-}" ]]; then
    read -r -a VALIDATION_VIDEOS <<< "${VIDEOS}"
else
    VALIDATION_VIDEOS=(
        "Eadom2_video.mp4"
        "Eadom3_video.mp4"
        "Eadom5_video.mp4"
        "Merawi1_video.mp4"
        "Merawi2_video.mp4"
        "Merawi3_video.mp4"
        "Sirguta1_video.mp4"
        "Sirguta2_video.mp4"
        "Sirguta3_video.mp4"
    )
fi

START_IDX="${START_IDX:-0}"
MAX_EVAL_CLIPS="${MAX_EVAL_CLIPS:-7}"
CLIP_STRIDE="${CLIP_STRIDE:-14}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-25}"
DECODE_CHUNK_SIZE="${DECODE_CHUNK_SIZE:-1}"
RESCALE_FACTOR="${RESCALE_FACTOR:-1.0}"
OVERLAPPING_RATIO="${OVERLAPPING_RATIO:-0.5}"
S_CHURN="${S_CHURN:-0.0}"
ZERO_EMG="${ZERO_EMG:-0}"
EMG_MODE="${EMG_MODE:-real}"
CONTROLNET_COND_SCALE="${CONTROLNET_COND_SCALE:-1.0}"
SHUFFLE_OFFSET="${SHUFFLE_OFFSET:-97}"

if [[ "${ZERO_EMG}" == "1" || "${ZERO_EMG}" == "true" || "${ZERO_EMG}" == "TRUE" ]]; then
    EMG_MODE="zero"
fi

SCALE_TAG="${CONTROLNET_COND_SCALE//./p}"

echo "Checkpoints: ${CHECKPOINT_LIST[*]}"
echo "Writing validation outputs to: ${OUT_ROOT}"
echo "Videos: ${VALIDATION_VIDEOS[*]}"
echo "Sampling ${MAX_EVAL_CLIPS} clips per video from start_idx=${START_IDX} with stride=${CLIP_STRIDE}"
echo "Inference geometry: rescale_factor=${RESCALE_FACTOR}, overlap=${OVERLAPPING_RATIO}, s_churn=${S_CHURN}"
echo "EMG mode: ${EMG_MODE}"
echo "ControlNet conditioning scale: ${CONTROLNET_COND_SCALE}"

for checkpoint in "${CHECKPOINT_LIST[@]}"; do
    if [[ "${checkpoint}" != /* ]]; then
        checkpoint="${RUN_DIR}/${checkpoint}"
    fi

    if [[ ! -d "${checkpoint}/controlnet" ]]; then
        echo "Skipping ${checkpoint}: missing controlnet directory" >&2
        continue
    fi

    checkpoint_name="$(basename "${checkpoint}")"
    echo "Using checkpoint: ${checkpoint}"

    for video in "${VALIDATION_VIDEOS[@]}"; do
        if [[ ! -f "${VIDEO_ROOT}/${video}" ]]; then
            echo "Skipping ${video}: missing MP4 under ${VIDEO_ROOT}" >&2
            continue
        fi

        video_stem="${video%.mp4}"
        output_dir="${OUT_ROOT}/${checkpoint_name}/emg_${EMG_MODE}/scale_${SCALE_TAG}/${video_stem}"
        echo "Validating ${checkpoint_name} ${video}"

        python valid_emg.py \
            --pretrained_model_name_or_path="${PRETRAINED_MODEL}" \
            --controlnet_model_name_or_path="${checkpoint}/controlnet" \
            --output_dir="${output_dir}" \
            --video_root="${VIDEO_ROOT}" \
            --emg_data_root="${EMG_DATA_ROOT}" \
            --validation_video="${video}" \
            --num_frames=14 \
            --width=512 \
            --height=320 \
            --emg_samples_per_interval=64 \
            --emg_fs=500 \
            --start_idx="${START_IDX}" \
            --max_eval_clips="${MAX_EVAL_CLIPS}" \
            --clip_stride="${CLIP_STRIDE}" \
            --rescale_factor="${RESCALE_FACTOR}" \
            --overlapping_ratio="${OVERLAPPING_RATIO}" \
            --s_churn="${S_CHURN}" \
            --emg_mode="${EMG_MODE}" \
            --shuffle_offset="${SHUFFLE_OFFSET}" \
            --controlnet_cond_scale="${CONTROLNET_COND_SCALE}" \
            --num_inference_steps="${NUM_INFERENCE_STEPS}" \
            --decode_chunk_size="${DECODE_CHUNK_SIZE}" \
            --mixed_precision=bf16 \
            --enable_xformers_memory_efficient_attention \
            --save_validation_gifs
    done
done
