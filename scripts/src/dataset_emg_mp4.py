import argparse
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.signal import butter, filtfilt, iirnotch
from torch.utils.data import Dataset


EMG_CHANNELS_PER_HAND = 8
EMG_NUM_HANDS = 2
EMG_CHANNELS = EMG_CHANNELS_PER_HAND * EMG_NUM_HANDS
DEFAULT_EMG_SAMPLES_PER_INTERVAL = 64

DEFAULT_PANEL_WIDTH = 1100
DEFAULT_PANEL_HEIGHT = 720


@dataclass(frozen=True)
class VideoEMGRecord:
    sequence_name: str
    video_path: str
    frame_timestamps_csv: str
    left_emg_csv: Optional[str]
    right_emg_csv: Optional[str]
    num_video_frames: int
    fps: float
    width: int
    height: int


def video_stem_to_sequence_name(video_path: str) -> str:
    stem = Path(video_path).stem
    stem = re.sub(r"_?video$", "", stem, flags=re.IGNORECASE)
    match = re.match(r"^([A-Za-z]+)_?(\d+)$", stem)
    if not match:
        raise ValueError(f"Could not infer sequence name from video file: {video_path}")
    return f"{match.group(1)}_{match.group(2)}"


def _video_metadata(video_path: str) -> Tuple[int, float, int, int]:
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if frames <= 0:
        raise RuntimeError(f"Video has no readable frames: {video_path}")
    if not np.isfinite(fps) or fps <= 0:
        fps = 15.0
    return frames, fps, width, height


def discover_video_emg_records(video_root: str, emg_data_root: str) -> List[VideoEMGRecord]:
    video_paths = sorted(Path(video_root).glob("*.mp4"))
    records: List[VideoEMGRecord] = []
    for video_path in video_paths:
        sequence_name = video_stem_to_sequence_name(str(video_path))
        sequence_dir = Path(emg_data_root) / sequence_name
        timestamp_csv = sequence_dir / f"{sequence_name}_frame_timestamps.csv"
        if not timestamp_csv.exists():
            continue

        left_csv = sequence_dir / "left.csv"
        right_csv = sequence_dir / "right.csv"
        left_alt_csv = sequence_dir / "left.CSV"
        right_alt_csv = sequence_dir / "right.CSV"
        if not left_csv.exists() and left_alt_csv.exists():
            left_csv = left_alt_csv
        if not right_csv.exists() and right_alt_csv.exists():
            right_csv = right_alt_csv

        nframes, fps, width, height = _video_metadata(str(video_path))
        records.append(
            VideoEMGRecord(
                sequence_name=sequence_name,
                video_path=str(video_path),
                frame_timestamps_csv=str(timestamp_csv),
                left_emg_csv=str(left_csv) if left_csv.exists() else None,
                right_emg_csv=str(right_csv) if right_csv.exists() else None,
                num_video_frames=nframes,
                fps=fps,
                width=width,
                height=height,
            )
        )
    if not records:
        raise ValueError(f"No MP4/EMG records found in {video_root} with metadata under {emg_data_root}")
    return records


def split_records(records: Sequence[VideoEMGRecord], validation_video: str) -> Tuple[List[VideoEMGRecord], List[VideoEMGRecord]]:
    validation_name = Path(validation_video).name
    valid = [r for r in records if Path(r.video_path).name == validation_name]
    train = [r for r in records if Path(r.video_path).name != validation_name]
    if not valid:
        raise ValueError(f"Validation video {validation_video} was not found in discovered records")
    if not train:
        raise ValueError("No training videos remain after validation split")
    return train, valid


def _estimate_fs(t_sec: np.ndarray) -> float:
    dt = np.diff(t_sec)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return 500.0
    return float(1.0 / np.median(dt))


def _moving_rms(x: np.ndarray, window_size: int) -> np.ndarray:
    window_size = max(1, int(window_size))
    kernel = np.ones(window_size, dtype=np.float64) / float(window_size)
    out = np.empty_like(x, dtype=np.float32)
    for ch in range(x.shape[1]):
        out[:, ch] = np.sqrt(np.convolve(x[:, ch] ** 2, kernel, mode="same")).astype(np.float32)
    return out


def _raw_emg_columns(df: pd.DataFrame) -> List[str]:
    expected = [f"EMG{i}_Raw" for i in range(1, EMG_CHANNELS_PER_HAND + 1)]
    cols = [c for c in expected if c in df.columns]
    if len(cols) == EMG_CHANNELS_PER_HAND:
        return cols
    cols = [c for c in df.columns if "emg" in c.lower() and "raw" in c.lower()]
    cols = cols[:EMG_CHANNELS_PER_HAND]
    if not cols:
        raise ValueError("No raw EMG columns found")
    return cols


def _load_one_hand_emg(
    csv_path: Optional[str],
    emg_fs_override: Optional[float],
    hp_cutoff: float,
    hp_order: int,
    notch_freq: float,
    notch_q: float,
    rms_window_ms: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if csv_path is None:
        return None, None
    df = pd.read_csv(csv_path)
    if "unix_time_s" not in df.columns:
        raise ValueError(f"{csv_path} must contain unix_time_s")
    df = df.sort_values("unix_time_s").reset_index(drop=True)
    cols = _raw_emg_columns(df)
    for col in cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    t = df["unix_time_s"].to_numpy(dtype=np.float64)
    x = df[cols].to_numpy(dtype=np.float64)
    fs = float(emg_fs_override) if emg_fs_override is not None else _estimate_fs(t)

    if len(x) > max(32, hp_order * 6):
        nyq = 0.5 * fs
        if 0 < hp_cutoff < nyq:
            b_high, a_high = butter(hp_order, hp_cutoff / nyq, btype="high")
            x = filtfilt(b_high, a_high, x, axis=0)
        if 0 < notch_freq < nyq:
            b_notch, a_notch = iirnotch(notch_freq, notch_q, fs)
            x = filtfilt(b_notch, a_notch, x, axis=0)

    x = np.abs(x)
    rms_window = int(round(fs * rms_window_ms / 1000.0))
    x = _moving_rms(x, rms_window)

    med = np.median(x, axis=0, keepdims=True)
    x = x - med
    scale = np.quantile(np.abs(x), 0.995, axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    x = np.clip(x / scale, -5.0, 5.0).astype(np.float32)
    keep = np.concatenate([[True], np.diff(t) > 0])
    if not np.all(keep):
        t = t[keep]
        x = x[keep]
    return t, x


def load_emg_pair(
    record: VideoEMGRecord,
    emg_fs_override: Optional[float] = 500.0,
    hp_cutoff: float = 10.0,
    hp_order: int = 4,
    notch_freq: float = 60.0,
    notch_q: float = 30.0,
    rms_window_ms: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray]:
    left_t, left_x = _load_one_hand_emg(
        record.left_emg_csv, emg_fs_override, hp_cutoff, hp_order, notch_freq, notch_q, rms_window_ms
    )
    right_t, right_x = _load_one_hand_emg(
        record.right_emg_csv, emg_fs_override, hp_cutoff, hp_order, notch_freq, notch_q, rms_window_ms
    )

    if left_t is None and right_t is None:
        raise ValueError(f"{record.sequence_name} has no readable left or right EMG CSV")

    base_t = left_t if left_t is not None else right_t
    out = np.zeros((len(base_t), EMG_CHANNELS), dtype=np.float32)
    if left_t is not None and left_x is not None:
        out[:, : left_x.shape[1]] = left_x[: len(base_t)]
    if right_t is not None and right_x is not None:
        if right_t is base_t:
            right_aligned = right_x[: len(base_t)]
        else:
            right_aligned = np.empty((len(base_t), right_x.shape[1]), dtype=np.float32)
            for ch in range(right_x.shape[1]):
                right_aligned[:, ch] = np.interp(base_t, right_t, right_x[:, ch], left=right_x[0, ch], right=right_x[-1, ch])
        out[:, EMG_CHANNELS_PER_HAND : EMG_CHANNELS_PER_HAND + right_aligned.shape[1]] = right_aligned
    return base_t.astype(np.float64), out


def load_frame_times(record: VideoEMGRecord) -> np.ndarray:
    ts = pd.read_csv(record.frame_timestamps_csv)
    if "utc_sec" not in ts.columns:
        raise ValueError(f"{record.frame_timestamps_csv} must contain utc_sec")
    times = ts.sort_values("frame_idx")["utc_sec"].to_numpy(dtype=np.float64)
    if len(times) < record.num_video_frames:
        raise ValueError(
            f"{record.sequence_name} has {record.num_video_frames} MP4 frames but only {len(times)} timestamp rows"
        )
    # The MP4s were generated with make_single_videos.py's default frame_fraction=0.05,
    # so frame k corresponds to timestamp row k from the original sequence.
    return times[: record.num_video_frames]


def infer_rgb_crop(width: int, height: int) -> Tuple[int, int, int, int]:
    scaled_panel_w = int(round(DEFAULT_PANEL_WIDTH * height / float(DEFAULT_PANEL_HEIGHT)))
    rgb_w = width - scaled_panel_w
    if rgb_w <= 0 or rgb_w > width:
        rgb_w = min(width, height)
    if abs(rgb_w - height) <= 2:
        rgb_w = height
    rgb_w = max(1, min(rgb_w, width))
    return 0, 0, rgb_w, height


def _center_crop_to_aspect(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    src_w, src_h = img.size
    target_aspect = target_w / float(target_h)
    src_aspect = src_w / float(src_h)
    if src_aspect > target_aspect:
        new_w = int(round(src_h * target_aspect))
        left = max(0, (src_w - new_w) // 2)
        return img.crop((left, 0, left + new_w, src_h))
    new_h = int(round(src_w / target_aspect))
    top = max(0, (src_h - new_h) // 2)
    return img.crop((0, top, src_w, top + new_h))


def preprocess_rgb_frame(frame_bgr: np.ndarray, crop: Tuple[int, int, int, int], width: int, height: int) -> Image.Image:
    x, y, w, h = crop
    rgb = cv2.cvtColor(frame_bgr[y : y + h, x : x + w], cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    img = _center_crop_to_aspect(img, width, height)
    return img.resize((width, height), Image.Resampling.LANCZOS)


def read_mp4_clip(record: VideoEMGRecord, start_idx: int, num_frames: int, width: int, height: int) -> List[Image.Image]:
    if start_idx < 0 or start_idx + num_frames > record.num_video_frames:
        raise IndexError(f"Clip {start_idx}:{start_idx + num_frames} is out of bounds for {record.video_path}")
    cap = cv2.VideoCapture(record.video_path)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {record.video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        crop = infer_rgb_crop(record.width, record.height)
        frames: List[Image.Image] = []
        for offset in range(num_frames):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Failed to read frame {start_idx + offset} from {record.video_path}")
            frames.append(preprocess_rgb_frame(frame, crop, width, height))
    finally:
        cap.release()
    return frames


def pil_frames_to_tensor(frames: Sequence[Image.Image]) -> torch.Tensor:
    out = torch.empty((len(frames), 3, frames[0].height, frames[0].width), dtype=torch.float32)
    for i, img in enumerate(frames):
        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
        out[i] = torch.from_numpy(arr).permute(2, 0, 1)
    return out


def _nearest_index(sorted_times: np.ndarray, t: float) -> int:
    idx = int(np.searchsorted(sorted_times, t))
    if idx <= 0:
        return 0
    if idx >= len(sorted_times):
        return len(sorted_times) - 1
    left = idx - 1
    right = idx
    return right if abs(sorted_times[right] - t) < abs(sorted_times[left] - t) else left


def _resample_interval(
    emg_t: np.ndarray,
    emg_x: np.ndarray,
    t0: float,
    t1: float,
    samples_per_interval: int,
) -> np.ndarray:
    if not np.isfinite(t0) or not np.isfinite(t1):
        return np.zeros((emg_x.shape[1], samples_per_interval), dtype=np.float32)
    if t1 < t0:
        t0, t1 = t1, t0
    lo = int(np.searchsorted(emg_t, t0, side="left"))
    hi = int(np.searchsorted(emg_t, t1, side="right"))
    if hi - lo <= 1:
        idx = _nearest_index(emg_t, 0.5 * (t0 + t1))
        return np.repeat(emg_x[idx : idx + 1].T, samples_per_interval, axis=1).astype(np.float32)

    interval_t = emg_t[lo:hi]
    interval_x = emg_x[lo:hi]
    target_t = np.linspace(t0, t1, samples_per_interval, dtype=np.float64)
    out = np.empty((emg_x.shape[1], samples_per_interval), dtype=np.float32)
    for ch in range(emg_x.shape[1]):
        out[ch] = np.interp(target_t, interval_t, interval_x[:, ch]).astype(np.float32)
    return out


def build_emg_condition(
    frame_times: np.ndarray,
    start_idx: int,
    num_frames: int,
    emg_t: np.ndarray,
    emg_x: np.ndarray,
    samples_per_interval: int,
) -> torch.Tensor:
    clip_times = frame_times[start_idx : start_idx + num_frames]
    cond = np.zeros((num_frames, EMG_CHANNELS, samples_per_interval), dtype=np.float32)
    for i in range(1, num_frames):
        cond[i] = _resample_interval(emg_t, emg_x, clip_times[i - 1], clip_times[i], samples_per_interval)
    return torch.from_numpy(cond)


def reverse_emg_condition(emg_values: torch.Tensor) -> torch.Tensor:
    reversed_values = torch.zeros_like(emg_values)
    if emg_values.shape[0] > 1:
        reversed_values[1:] = torch.flip(emg_values[1:], dims=(0, 2))
    return reversed_values


class EMGVideoDataset(Dataset):
    def __init__(
        self,
        records: Sequence[VideoEMGRecord],
        samples_per_video: int = 100,
        width: int = 512,
        height: int = 320,
        sample_frames: int = 14,
        emg_samples_per_interval: int = DEFAULT_EMG_SAMPLES_PER_INTERVAL,
        emg_fs: Optional[float] = 500.0,
    ):
        self.records = list(records)
        if not self.records:
            raise ValueError("EMGVideoDataset requires at least one record")
        self.samples_per_video = int(samples_per_video)
        self.width = int(width)
        self.height = int(height)
        self.sample_frames = int(sample_frames)
        self.emg_samples_per_interval = int(emg_samples_per_interval)
        self.emg_fs = emg_fs
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        for record in self.records:
            if record.num_video_frames < self.sample_frames:
                raise ValueError(
                    f"{record.video_path} has {record.num_video_frames} frames, need {self.sample_frames}"
                )

    def __len__(self):
        return len(self.records) * self.samples_per_video

    def _cached_arrays(self, record: VideoEMGRecord) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cached = self._cache.get(record.sequence_name)
        if cached is None:
            frame_times = load_frame_times(record)
            emg_t, emg_x = load_emg_pair(record, emg_fs_override=self.emg_fs)
            cached = (frame_times, emg_t, emg_x)
            self._cache[record.sequence_name] = cached
        return cached

    def load_clip(self, record: VideoEMGRecord, start_idx: int):
        frame_times, emg_t, emg_x = self._cached_arrays(record)
        frames = read_mp4_clip(record, start_idx, self.sample_frames, self.width, self.height)
        pixel_values = pil_frames_to_tensor(frames)
        emg_values = build_emg_condition(
            frame_times,
            start_idx,
            self.sample_frames,
            emg_t,
            emg_x,
            self.emg_samples_per_interval,
        )
        return frames, pixel_values, emg_values

    def __getitem__(self, idx):
        record = self.records[idx % len(self.records)]
        max_start = record.num_video_frames - self.sample_frames
        start_idx = random.randint(0, max_start)
        _, pixel_values, emg_values = self.load_clip(record, start_idx)
        return {
            "pixel_values": pixel_values,
            "emg_values": emg_values,
            "video_path": record.video_path,
            "sequence_name": record.sequence_name,
            "start_idx": start_idx,
        }


def make_datasets(
    video_root: str,
    emg_data_root: str,
    validation_video: str,
    samples_per_video: int,
    width: int,
    height: int,
    sample_frames: int,
    emg_samples_per_interval: int,
    emg_fs: Optional[float] = 500.0,
) -> Tuple[EMGVideoDataset, EMGVideoDataset]:
    records = discover_video_emg_records(video_root, emg_data_root)
    train_records, valid_records = split_records(records, validation_video)
    train_dataset = EMGVideoDataset(
        train_records,
        samples_per_video=samples_per_video,
        width=width,
        height=height,
        sample_frames=sample_frames,
        emg_samples_per_interval=emg_samples_per_interval,
        emg_fs=emg_fs,
    )
    valid_dataset = EMGVideoDataset(
        valid_records,
        samples_per_video=1,
        width=width,
        height=height,
        sample_frames=sample_frames,
        emg_samples_per_interval=emg_samples_per_interval,
        emg_fs=emg_fs,
    )
    return train_dataset, valid_dataset


def load_validation_clip(
    dataset: EMGVideoDataset,
    record_idx: int = 0,
    start_idx: int = 0,
) -> Tuple[Image.Image, torch.Tensor, Image.Image, torch.Tensor, List[Image.Image], VideoEMGRecord]:
    record = dataset.records[record_idx]
    start_idx = min(max(0, start_idx), record.num_video_frames - dataset.sample_frames)
    frames, _, emg_values = dataset.load_clip(record, start_idx)
    return frames[0], emg_values, frames[-1], reverse_emg_condition(emg_values), frames, record


def _smoke():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", default="/fs/vulcan-projects/Force_Learning/phan2003/videos")
    parser.add_argument("--emg_data_root", default="/fs/vulcan-projects/Force_Learning/EMG")
    parser.add_argument("--validation_video", default="Sirguta2_video.mp4")
    parser.add_argument("--num_frames", type=int, default=14)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--emg_samples_per_interval", type=int, default=64)
    args = parser.parse_args()

    _, valid_dataset = make_datasets(
        args.video_root,
        args.emg_data_root,
        args.validation_video,
        samples_per_video=1,
        width=args.width,
        height=args.height,
        sample_frames=args.num_frames,
        emg_samples_per_interval=args.emg_samples_per_interval,
    )
    _, pixel_values, emg_values = valid_dataset.load_clip(valid_dataset.records[0], 0)
    print("pixel_values", tuple(pixel_values.shape), "finite", bool(torch.isfinite(pixel_values).all()))
    print("emg_values", tuple(emg_values.shape), "finite", bool(torch.isfinite(emg_values).all()))


if __name__ == "__main__":
    _smoke()
