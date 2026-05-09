#!/bin/bash
#SBATCH --job-name=EMG_SVD_SMOKE
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --qos=medium
#SBATCH --account=nexus
#SBATCH --partition=tron
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/fs/vulcan-projects/Force_Learning/phan2003/emg_smoke_%j.out
#SBATCH --error=/fs/vulcan-projects/Force_Learning/phan2003/emg_smoke_%j.err
#SBATCH --mail-user=finnard2112@gmail.com
#SBATCH --mail-type=END,FAIL

source /nfshomes/phan2003/miniconda3/etc/profile.d/conda.sh
conda activate VDM_EVFI
cd /nfshomes/phan2003/VDM_EVFI/scripts

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=0
export NCCL_TIMEOUT=3600

accelerate launch --num_processes=1 --mixed_precision=bf16 train_emg.py \
    --pretrained_model_name_or_path="stabilityai/stable-video-diffusion-img2vid" \
    --output_dir="/fs/vulcan-projects/Force_Learning/phan2003/output_EMG_smoke" \
    --video_root="/fs/vulcan-projects/Force_Learning/phan2003/videos" \
    --emg_data_root="/fs/vulcan-projects/Force_Learning/EMG" \
    --validation_video="Sirguta2_video.mp4" \
    --num_frames=14 \
    --width=512 \
    --height=320 \
    --per_gpu_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=2 \
    --samples_per_video=1 \
    --emg_samples_per_interval=64 \
    --emg_fs=500 \
    --learning_rate=1e-5 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --checkpointing_steps=1 \
    --checkpoints_total_limit=2 \
    --validation_steps=1 \
    --num_validation_images=1 \
    --decode_chunk_size=4 \
    --save_validation_gifs \
    --num_workers=0 \
    --seed=42
