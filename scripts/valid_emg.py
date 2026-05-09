import argparse
import os
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import AutoencoderKLTemporalDecoder
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from packaging import version
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

import diffusers
import transformers

from src.dataset_emg_mp4 import DEFAULT_EMG_SAMPLES_PER_INTERVAL, load_validation_clip, make_datasets
from src.models.fullControlnet_sdv_emg import ControlNetSDVModel
from src.models.unet_spatio_temporal_condition_fullControlnet import UNetSpatioTemporalConditionControlNetModel
from src.pipelines.pipeline_stable_video_diffusion_fullControlnet_emg_timereversal import (
    StableVideoDiffusionPipelineControlNet,
)


check_min_version("0.24.0.dev0")
logger = get_logger(__name__, log_level="INFO")


def export_to_gif(frames, output_gif_path, fps):
    pil_frames = [Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame for frame in frames]
    os.makedirs(os.path.dirname(output_gif_path), exist_ok=True)
    duration_ms = int(round(1000.0 / max(float(fps), 1e-6)))
    pil_frames[0].save(
        output_gif_path,
        format="GIF",
        append_images=pil_frames[1:],
        save_all=True,
        duration=duration_ms,
        loop=0,
    )


def export_to_images(frames, output_folder_path, start_idx):
    os.makedirs(output_folder_path, exist_ok=True)
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame
        img.save(os.path.join(output_folder_path, f"{start_idx + i:06d}.png"))


def make_side_by_side_frames(left_frames, right_frames):
    out = []
    for left, right in zip(left_frames, right_frames):
        left_img = Image.fromarray(left) if isinstance(left, np.ndarray) else left
        right_img = Image.fromarray(right) if isinstance(right, np.ndarray) else right
        canvas = Image.new("RGB", (left_img.width + right_img.width, max(left_img.height, right_img.height)))
        canvas.paste(left_img, (0, 0))
        canvas.paste(right_img, (left_img.width, 0))
        out.append(canvas)
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="EMG-conditioned SVD validation with two-side fusion.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--pretrain_unet", type=str, default=None)
    parser.add_argument("--controlnet_model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--video_root", type=str, required=True)
    parser.add_argument("--emg_data_root", type=str, required=True)
    parser.add_argument("--validation_video", type=str, default="Sirguta2_video.mp4")
    parser.add_argument("--num_frames", type=int, default=14)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--emg_samples_per_interval", type=int, default=DEFAULT_EMG_SAMPLES_PER_INTERVAL)
    parser.add_argument("--emg_fs", type=float, default=500.0)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--max_eval_clips", type=int, default=1)
    parser.add_argument("--clip_stride", type=int, default=None)
    parser.add_argument("--rescale_factor", type=float, default=2.0)
    parser.add_argument("--overlapping_ratio", type=float, default=0.1)
    parser.add_argument("--t0", type=int, default=0)
    parser.add_argument("--M", type=int, default=2)
    parser.add_argument("--s_churn", type=float, default=0.5)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--decode_chunk_size", type=int, default=2)
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_validation_gifs", action="store_true", default=True)
    parser.add_argument("--no_save_validation_gifs", dest="save_validation_gifs", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()
    args.width = args.width - args.width % 8
    args.height = args.height - args.height % 8
    os.makedirs(args.output_dir, exist_ok=True)

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    feature_extractor = CLIPImageProcessor.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="feature_extractor", revision=args.revision
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant="fp16"
    )
    unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(
        args.pretrained_model_name_or_path if args.pretrain_unet is None else args.pretrain_unet,
        subfolder="unet",
        low_cpu_mem_usage=True,
        variant="fp16",
    )
    controlnet = ControlNetSDVModel.from_pretrained(args.controlnet_model_name_or_path)

    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    image_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            if version.parse(xformers.__version__) == version.parse("0.0.16"):
                logger.warning("xFormers 0.0.16 may be unstable for this workload.")
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available.")

    _, valid_dataset = make_datasets(
        args.video_root,
        args.emg_data_root,
        args.validation_video,
        samples_per_video=1,
        width=args.width,
        height=args.height,
        sample_frames=args.num_frames,
        emg_samples_per_interval=args.emg_samples_per_interval,
        emg_fs=args.emg_fs,
    )
    record = valid_dataset.records[0]
    stride = args.clip_stride if args.clip_stride is not None else max(1, args.num_frames - 1)
    max_start = max(0, record.num_video_frames - args.num_frames)
    starts = list(range(min(args.start_idx, max_start), max_start + 1, stride))[: args.max_eval_clips]

    pipeline = StableVideoDiffusionPipelineControlNet.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet,
        image_encoder=image_encoder,
        controlnet=controlnet,
        vae=vae,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.enable_model_cpu_offload()
    pipeline.set_progress_bar_config(disable=True)

    pred_root = os.path.join(args.output_dir, "test_images", record.sequence_name)
    gt_root = os.path.join(args.output_dir, "gt_test_images", record.sequence_name)
    gif_root = os.path.join(args.output_dir, "validation_gifs", record.sequence_name)

    controlnet.eval()
    for clip_i, start_idx in enumerate(tqdm(starts, disable=not accelerator.is_local_main_process)):
        image_1, emg_1, image_2, emg_2, gt_frames, _ = load_validation_clip(valid_dataset, 0, start_idx)
        emg_1 = emg_1.to(accelerator.device, dtype=weight_dtype)
        emg_2 = emg_2.to(accelerator.device, dtype=weight_dtype)

        with torch.autocast(str(accelerator.device).replace(":0", ""), enabled=accelerator.mixed_precision == "fp16"):
            video_frames, org_frames = pipeline(
                image_1,
                emg_1,
                image_2,
                emg_2,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                decode_chunk_size=args.decode_chunk_size,
                motion_bucket_id=127,
                fps=max(1, int(round(record.fps))),
                noise_aug_strength=0.02,
                rescale_factor=args.rescale_factor,
                num_inference_steps=args.num_inference_steps,
                overlap_ratio=args.overlapping_ratio,
                t0=args.t0,
                M=args.M,
                s_churn=args.s_churn,
                org_frames=gt_frames,
            )

        pred_frames = [np.array(frame.resize((args.width, args.height), Image.Resampling.LANCZOS)) for frame in video_frames.frames[0]]
        gt_np_frames = [np.array(frame.resize((args.width, args.height), Image.Resampling.LANCZOS)) for frame in gt_frames]

        export_to_images(pred_frames, pred_root, start_idx)
        export_to_images(gt_np_frames, gt_root, start_idx)

        if args.save_validation_gifs:
            prefix = os.path.join(gif_root, f"clip_{clip_i:04d}_start_{start_idx:06d}")
            export_to_gif(pred_frames, f"{prefix}_pred.gif", record.fps)
            export_to_gif(gt_np_frames, f"{prefix}_gt.gif", record.fps)
            export_to_gif(make_side_by_side_frames(pred_frames, gt_np_frames), f"{prefix}_side_by_side.gif", record.fps)

    del pipeline
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
