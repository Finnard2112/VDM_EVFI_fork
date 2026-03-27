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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=1800  # 1 hour instead of default 10 minutes
export TORCH_NCCL_BLOCKING_WAIT=1

accelerate launch --multi_gpu --num_processes=2 --dynamo_backend=inductor --mixed_precision=bf16 train.py \
    --pretrained_model_name_or_path="stabilityai/stable-video-diffusion-img2vid" \
    --output_dir="/fs/vulcan-projects/Force_Learning/phan2003/output_EMG_run3" \
    --train_data_path="/fs/vulcan-projects/Force_Learning/EMG" \
    --num_frames=14 \
    --width=512 \
    --height=320 \
    --per_gpu_batch_size=1 \
    --gradient_accumulation_steps=8 \
    --gradient_checkpointing \
    \
    --num_train_epochs=100 \
    --samples_per_folder=100 \
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
    \
    --valid_path1="/fs/vulcan-projects/Force_Learning/EMG/Eadom_1/frames" \
    --valid_path1_idx=0 \
    --num_workers=6 \
    --seed=42