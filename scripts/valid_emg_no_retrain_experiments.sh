#!/bin/bash
#SBATCH --job-name=VDM_EVFI_EMG_ABLATE_A5000
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --qos=medium
#SBATCH --account=nexus
#SBATCH --partition=tron
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/fs/vulcan-projects/Force_Learning/phan2003/%j_ablate.out
#SBATCH --error=/fs/vulcan-projects/Force_Learning/phan2003/%j_ablate.err
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

if [[ -n "${CHECKPOINT_DIR:-}" ]]; then
    CHECKPOINT="${CHECKPOINT_DIR}"
else
    CHECKPOINT="$(find "${RUN_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
fi

if [[ -z "${CHECKPOINT}" || ! -d "${CHECKPOINT}/controlnet" ]]; then
    echo "Could not find a ControlNet checkpoint under ${RUN_DIR}" >&2
    exit 1
fi

CHECKPOINT_NAME="$(basename "${CHECKPOINT}")"
OUT_ROOT="${OUT_ROOT:-${RUN_DIR}/no_retrain_experiments/${CHECKPOINT_NAME}}"

# Keep this default compact enough for a first pass. Override VIDEOS/MAX_EVAL_CLIPS
# from sbatch if you want a broader or smaller run.
if [[ -n "${VIDEOS:-}" ]]; then
    read -r -a VALIDATION_VIDEOS <<< "${VIDEOS}"
else
    VALIDATION_VIDEOS=(
        "Sirguta2_video.mp4"
        "Sirguta1_video.mp4"
        "Merawi1_video.mp4"
        "Eadom2_video.mp4"
    )
fi

START_IDX="${START_IDX:-0}"
MAX_EVAL_CLIPS="${MAX_EVAL_CLIPS:-2}"
CLIP_STRIDE="${CLIP_STRIDE:-28}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-20}"
DECODE_CHUNK_SIZE="${DECODE_CHUNK_SIZE:-1}"
RESCALE_FACTOR="${RESCALE_FACTOR:-1.0}"
OVERLAPPING_RATIO="${OVERLAPPING_RATIO:-0.5}"
S_CHURN="${S_CHURN:-0.0}"
SHUFFLE_OFFSET="${SHUFFLE_OFFSET:-97}"

if [[ -n "${SCALES:-}" ]]; then
    read -r -a SCALE_LIST <<< "${SCALES}"
else
    SCALE_LIST=(0.0 0.5 1.0)
fi

echo "Checkpoint: ${CHECKPOINT}"
echo "Output root: ${OUT_ROOT}"
echo "Videos: ${VALIDATION_VIDEOS[*]}"
echo "Scale sweep: ${SCALE_LIST[*]}"
echo "Clips per video: ${MAX_EVAL_CLIPS}, start=${START_IDX}, stride=${CLIP_STRIDE}"

run_validation() {
    local emg_mode="$1"
    local scale="$2"
    local scale_tag="${scale//./p}"
    local mode_root="${OUT_ROOT}/emg_${emg_mode}/scale_${scale_tag}"

    echo "Running emg_mode=${emg_mode}, controlnet_cond_scale=${scale}"

    for video in "${VALIDATION_VIDEOS[@]}"; do
        if [[ ! -f "${VIDEO_ROOT}/${video}" ]]; then
            echo "Skipping ${video}: missing MP4 under ${VIDEO_ROOT}" >&2
            continue
        fi

        local video_stem="${video%.mp4}"
        python valid_emg.py \
            --pretrained_model_name_or_path="${PRETRAINED_MODEL}" \
            --controlnet_model_name_or_path="${CHECKPOINT}/controlnet" \
            --output_dir="${mode_root}/${video_stem}" \
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
            --emg_mode="${emg_mode}" \
            --shuffle_offset="${SHUFFLE_OFFSET}" \
            --controlnet_cond_scale="${scale}" \
            --num_inference_steps="${NUM_INFERENCE_STEPS}" \
            --decode_chunk_size="${DECODE_CHUNK_SIZE}" \
            --mixed_precision=bf16 \
            --enable_xformers_memory_efficient_attention \
            --save_validation_gifs
    done
}

# Experiment 1: does a weaker/stronger ControlNet branch improve appearance
# or reduce wobble when using the real synchronized EMG?
for scale in "${SCALE_LIST[@]}"; do
    run_validation "real" "${scale}"
done

# Experiment 2: at the nominal scale, compare real EMG against no EMG and
# mismatched EMG. This tests whether the checkpoint is actually using the
# synchronized signal.
run_validation "zero" "1.0"
run_validation "shuffle" "1.0"

echo "Done. Inspect GIFs under ${OUT_ROOT}"
