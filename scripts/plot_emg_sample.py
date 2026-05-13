import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from src.dataset_emg_mp4 import make_datasets  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Plot a synchronized processed EMG sample from an MP4 validation clip.")
    parser.add_argument("--video_root", default="/fs/vulcan-projects/Force_Learning/phan2003/videos")
    parser.add_argument("--emg_data_root", default="/fs/vulcan-projects/Force_Learning/EMG")
    parser.add_argument("--validation_video", default="Sirguta2_video.mp4")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=14)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--emg_samples_per_interval", type=int, default=64)
    parser.add_argument("--emg_fs", type=float, default=500.0)
    parser.add_argument(
        "--output",
        default="/nfshomes/phan2003/Report/figures/sample_emg_sirguta2_start0.png",
    )
    return parser.parse_args()


def main():
    args = parse_args()
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
    start_idx = min(max(0, args.start_idx), record.num_video_frames - args.num_frames)
    _, _, emg_values = valid_dataset.load_clip(record, start_idx)
    frame_times, _, _ = valid_dataset._cached_arrays(record)
    clip_times = frame_times[start_idx : start_idx + args.num_frames]
    rel_frame_times = clip_times - clip_times[0]

    emg_np = emg_values.numpy()
    # emg_values[0] is zero because there is no interval before the first clip frame.
    intervals = np.abs(emg_np[1:])
    interval_times = []
    for i in range(1, args.num_frames):
        interval_times.append(
            np.linspace(rel_frame_times[i - 1], rel_frame_times[i], args.emg_samples_per_interval)
        )
    t = np.concatenate(interval_times)
    channel_values = intervals.transpose(1, 0, 2).reshape(intervals.shape[1], -1)

    left_mean = channel_values[:8].mean(axis=0)
    right_mean = channel_values[8:16].mean(axis=0)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.3]},
        constrained_layout=True,
    )

    axes[0].plot(t, left_mean, label="Left hand mean", color="#1f77b4", linewidth=2.0)
    axes[0].plot(t, right_mean, label="Right hand mean", color="#d62728", linewidth=2.0)
    axes[0].set_ylabel("Processed EMG\nmean |activation|")
    axes[0].set_title(f"{record.sequence_name}: processed EMG over a {args.num_frames}-frame clip")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", frameon=False)

    vmax = np.quantile(channel_values, 0.99)
    im = axes[1].imshow(
        channel_values,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[t[0], t[-1], 0.5, 16.5],
        vmin=0.0,
        vmax=max(float(vmax), 1e-6),
        cmap="magma",
    )
    axes[1].set_yticks([1, 4, 8, 9, 12, 16])
    axes[1].set_yticklabels(["L1", "L4", "L8", "R1", "R4", "R8"])
    axes[1].set_ylabel("EMG channel")
    axes[1].set_xlabel("Time from clip start (s)")
    cbar = fig.colorbar(im, ax=axes[1], pad=0.01)
    cbar.set_label("Processed |activation|")

    for ax in axes:
        for boundary in rel_frame_times:
            ax.axvline(boundary, color="white" if ax is axes[1] else "black", alpha=0.18, linewidth=0.8)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
