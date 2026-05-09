#!/bin/bash
#SBATCH --job-name=VDM_EVFI100
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --qos=medium
#SBATCH --account=nexus
#SBATCH --partition=tron 
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/fs/vulcan-projects/Force_Learning/phan2003/%j.out
#SBATCH --error=/fs/vulcan-projects/Force_Learning/phan2003/%j.err

source /nfshomes/phan2003/miniconda3/etc/profile.d/conda.sh
conda activate VDM_EVFI
cd /nfshomes/phan2003/VDM_EVFI/scripts
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=0
export NCCL_TIMEOUT=3600


accelerate launch --multi_gpu --num_processes=2 --mixed_precision=bf16 train_emg.py \
    --pretrained_model_name_or_path="stabilityai/stable-video-diffusion-img2vid" \
    --output_dir="/fs/vulcan-projects/Force_Learning/phan2003/output_EMG_controlnet_run1" \
    --video_root="/fs/vulcan-projects/Force_Learning/phan2003/videos" \
    --emg_data_root="/fs/vulcan-projects/Force_Learning/EMG" \
    --validation_video="Sirguta2_video.mp4" \
    --num_frames=14 \
    --width=512 \
    --height=320 \
    --per_gpu_batch_size=1 \
    --gradient_accumulation_steps=8 \
    --gradient_checkpointing \
    \
    --num_train_epochs=100 \
    --samples_per_video=100 \
    --emg_samples_per_interval=64 \
    --emg_fs=500 \
    \
    --learning_rate=1e-5 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=200 \
    --adam_weight_decay=1e-4 \
    --max_grad_norm=1.0 \
    --enable_xformers_memory_efficient_attention \
    --allow_tf32 \
    \
    --checkpointing_steps=500 \
    --checkpoints_total_limit=3 \
    --validation_steps=500 \
    --num_validation_images=1 \
    --decode_chunk_size=8 \
    --save_validation_gifs \
    --num_workers=6 \
    --seed=42
