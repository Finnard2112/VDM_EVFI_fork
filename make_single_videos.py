#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.signal import butter, filtfilt, iirnotch

def moving_window_rms(signal, window_size):
 signal = np.asarray(signal, dtype=np.float64)
 kernel = np.ones(window_size, dtype=np.float64) / float(window_size)
 return np.sqrt(np.convolve(signal ** 2, kernel, mode="same"))


def get_emg_data(
 data,
 window_size,
 fs=500,
 cutoff=8.0,
 notch_freq=60.0,
 notch_q=30.0,
 highpass_order=4,
):
 data = np.asarray(data, dtype=np.float64)

 if data.ndim != 2:
 raise ValueError(f"get_emg_data expects shape (n_samples, n_channels), got {data.shape}")

 nyq = 0.5 * fs
 if cutoff >= nyq:
 raise ValueError(f"cutoff={cutoff} must be < Nyquist={nyq}.")

 high = cutoff / nyq

 b_high, a_high = butter(highpass_order, high, btype="high")
 b_notch, a_notch = iirnotch(notch_freq, notch_q, fs)

 out = np.zeros_like(data, dtype=np.float32)

 for ch in range(data.shape[1]):
 x = data[:, ch]
 x = filtfilt(b_high, a_high, x)
 x = filtfilt(b_notch, a_notch, x)
 # x = np.abs(x)
 # x = moving_window_rms(x, window_size)
 out[:, ch] = x.astype(np.float32)

 return out

def _find_numeric_columns(df: pd.DataFrame):
 return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _pick_signal_columns(df: pd.DataFrame, emg_only: bool = True):
 cols = list(df.columns)
 numeric_cols = _find_numeric_columns(df)

 time_like = {"unix_time_s", "unix_time", "timestamp", "time", "time_s"}
 numeric_signal_cols = [c for c in numeric_cols if c not in time_like]

 emg_cols = [c for c in numeric_signal_cols if "emg" in c.lower()]
 accel_cols = [c for c in numeric_signal_cols if any(k in c.lower() for k in ["accel", "acc_", "accx", "accy", "accz"])]
 gyro_cols = [c for c in numeric_signal_cols if any(k in c.lower() for k in ["gyro", "gyr_", "gyrox", "gyroy", "gyroz"])]

 if len(emg_cols) == 0:
 if emg_only:
 emg_cols = numeric_signal_cols
 else:
 emg_cols = numeric_signal_cols
 accel_cols = []
 gyro_cols = []

 emg_cols = [c for c in cols if c in emg_cols]
 accel_cols = [c for c in cols if c in accel_cols and c not in emg_cols]
 gyro_cols = [c for c in cols if c in gyro_cols and c not in emg_cols and c not in accel_cols]

 if emg_only:
 return emg_cols, [], []
 return emg_cols, accel_cols, gyro_cols


def _reorder_left_hand_emg_if_needed(csv_path: str, emg_cols, emg_data: np.ndarray):
 _ = csv_path
 _ = emg_cols
 return emg_data


def estimate_fs_from_time(t_sec: np.ndarray):
 dt = np.diff(t_sec)
 dt = dt[np.isfinite(dt) & (dt > 0)]
 if len(dt) == 0:
 raise ValueError("Could not estimate sampling rate from timestamps.")
 return 1.0 / np.median(dt)


def extract_frame_index(path: Path):
 stem = path.stem
 nums = re.findall(r"\d+", stem)
 if len(nums) == 0:
 return None
 return int(nums[-1])


def list_frame_files(frames_dir: str):
 exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
 files = []
 for p in sorted(Path(frames_dir).iterdir()):
 if p.is_file() and p.suffix.lower() in exts:
 idx = extract_frame_index(p)
 if idx is not None:
 files.append((idx, str(p)))
 if len(files) == 0:
 raise ValueError(f"No image files with numeric frame indices found in {frames_dir}")
 files.sort(key=lambda x: x[0])
 return files


def resize_to_height(img, target_h):
 h, w = img.shape[:2]
 if h == target_h:
 return img
 scale = target_h / h
 new_w = max(1, int(round(w * scale)))
 return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def nearest_index(sorted_times: np.ndarray, t: float) -> int:
 idx = np.searchsorted(sorted_times, t)
 if idx <= 0:
 return 0
 if idx >= len(sorted_times):
 return len(sorted_times) - 1
 left = idx - 1
 right = idx
 if abs(sorted_times[right] - t) < abs(sorted_times[left] - t):
 return right
 return left


class FastEMGPanelRenderer:
 def __init__(
 self,
 t_emg: np.ndarray,
 emg_hp: np.ndarray,
 channel_names,
 window_sec=4.0,
 panel_height=720,
 panel_width=1100,
 left_pad=80,
 right_pad=20,
 top_pad=20,
 bottom_pad=35,
 ):
 self.t_emg = t_emg
 self.emg_hp = emg_hp
 self.channel_names = list(channel_names)
 self.window_sec = float(window_sec)
 self.panel_height = int(panel_height)
 self.panel_width = int(panel_width)
 self.left_pad = int(left_pad)
 self.right_pad = int(right_pad)
 self.top_pad = int(top_pad)
 self.bottom_pad = int(bottom_pad)

 self.n_ch = emg_hp.shape[1]
 self.inner_w = self.panel_width - self.left_pad - self.right_pad
 self.inner_h = self.panel_height - self.top_pad - self.bottom_pad
 self.ch_h = max(20, self.inner_h // self.n_ch)

 # Precompute robust vertical scaling per channel.
 # This avoids recomputing min/max on every frame.
 q = np.quantile(np.abs(emg_hp), 0.995, axis=0)
 q[q < 1e-6] = 1.0
 self.scales = q.astype(np.float32)

 def render(self, t_center: float):
 half = self.window_sec / 2.0
 t0 = t_center - half
 t1 = t_center + half

 lo = int(np.searchsorted(self.t_emg, t0, side="left"))
 hi = int(np.searchsorted(self.t_emg, t1, side="right"))

 if hi <= lo:
 idx = int(np.argmin(np.abs(self.t_emg - t_center)))
 lo = max(0, idx - 1)
 hi = min(len(self.t_emg), idx + 2)

 tt = self.t_emg[lo:hi]
 xx = self.emg_hp[lo:hi]

 panel = np.zeros((self.panel_height, self.panel_width, 3), dtype=np.uint8)
 panel[:] = 255

 # vertical cursor position for t_center
 x_cursor = self.left_pad + int(round((t_center - t0) / self.window_sec * (self.inner_w - 1)))
 x_cursor = max(self.left_pad, min(self.panel_width - self.right_pad - 1, x_cursor))

 for ch in range(self.n_ch):
 y_top = self.top_pad + ch * self.ch_h
 y_bot = min(self.top_pad + (ch + 1) * self.ch_h - 1, self.panel_height - self.bottom_pad)
 y_mid = (y_top + y_bot) // 2

 # background guide line
 cv2.line(panel, (self.left_pad, y_mid), (self.panel_width - self.right_pad, y_mid), (220, 220, 220), 1)

 # label
 cv2.putText(
 panel,
 str(self.channel_names[ch]),
 (8, y_mid + 5),
 cv2.FONT_HERSHEY_SIMPLEX,
 0.45,
 (0, 0, 0),
 1,
 cv2.LINE_AA,
 )

 if len(tt) >= 2:
 x = self.left_pad + ((tt - t0) / self.window_sec * (self.inner_w - 1)).astype(np.int32)
 y_amp = xx[:, ch] / self.scales[ch]
 y_amp = np.clip(y_amp, -1.0, 1.0)

 amp_px = max(4, int(0.42 * (y_bot - y_top)))
 y = (y_mid - y_amp * amp_px).astype(np.int32)

 pts = np.stack([x, y], axis=1).reshape(-1, 1, 2)
 cv2.polylines(panel, [pts], isClosed=False, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

 # cursor line
 cv2.line(
 panel,
 (x_cursor, self.top_pad),
 (x_cursor, self.panel_height - self.bottom_pad),
 (0, 0, 255),
 1,
 cv2.LINE_AA,
 )

 # x-axis label
 cv2.putText(
 panel,
 "UTC time (s)",
 (self.panel_width // 2 - 40, self.panel_height - 8),
 cv2.FONT_HERSHEY_SIMPLEX,
 0.45,
 (0, 0, 0),
 1,
 cv2.LINE_AA,
 )

 return panel


def render_side_by_side(frame_bgr, panel_bgr):
 target_h = max(frame_bgr.shape[0], panel_bgr.shape[0])
 frame_bgr = resize_to_height(frame_bgr, target_h)
 panel_bgr = resize_to_height(panel_bgr, target_h)
 return np.concatenate([frame_bgr, panel_bgr], axis=1)


def parse_args():
 p = argparse.ArgumentParser(
 description="Create one full-length MP4 over the entire frame folder, side-by-side with high-pass EMG."
 )
 p.add_argument("--frames_dir", type=str, required=True)
 p.add_argument("--frame_timestamps_csv", type=str, required=True,
 help="CSV with at least columns: frame_idx, utc_sec")
 p.add_argument("--emg_csv", type=str, required=True,
 help="EMG CSV with unix_time_s and EMG channels")
 p.add_argument("--output_mp4", type=str, required=True)
 p.add_argument("--fps", type=float, default=None,
 help="Output fps. If omitted, estimated from frame timestamps.")
 p.add_argument("--emg_fs", type=float, default=None,
 help="EMG sampling rate. If omitted, estimated from unix_time_s.")
 p.add_argument("--hp_cutoff", type=float, default=10.0)
 p.add_argument("--hp_order", type=int, default=4)
 p.add_argument("--min_filter_len", type=int, default=128)
 p.add_argument("--max_channels", type=int, default=8)
 p.add_argument("--plot_window_sec", type=float, default=4.0)
 p.add_argument("--plot_height", type=int, default=720)
 p.add_argument("--plot_width", type=int, default=1100)
 p.add_argument("--codec", type=str, default="mp4v")
 p.add_argument("--emg_window_size", type=int, default=100)
 p.add_argument("--notch_freq", type=float, default=60.0)
 p.add_argument("--notch_q", type=float, default=30.0)
 p.add_argument("--rotate_clockwise_90", action="store_true")
 p.add_argument("--line_utc_offset_sec", type=float, default=0.0,
 help="Optional manual time offset added to frame UTC before drawing the EMG cursor.")
 p.add_argument("--frame_fraction", type=float, default=0.05,
 help="Fraction of matched frames to render, from the start. 0.1 means first 10%%.")
 return p.parse_args()


def main():
 args = parse_args()
 out_dir = os.path.dirname(args.output_mp4)
 if out_dir:
 os.makedirs(out_dir, exist_ok=True)

 frame_files = list_frame_files(args.frames_dir)
 frame_indices = [idx for idx, _ in frame_files]
 frame_path_by_idx = {idx: path for idx, path in frame_files}

 ts = pd.read_csv(args.frame_timestamps_csv)
 required_ts = {"frame_idx", "utc_sec"}
 if not required_ts.issubset(ts.columns):
 raise ValueError(f"frame_timestamps_csv must contain columns {sorted(required_ts)}")
 ts = ts.sort_values("frame_idx").reset_index(drop=True)
 frame_to_utc = dict(zip(ts["frame_idx"].astype(int), ts["utc_sec"].astype(float)))

 valid_frame_indices_all = [idx for idx in frame_indices if idx in frame_to_utc]
 if len(valid_frame_indices_all) == 0:
 raise ValueError("None of the frame files have matching entries in frame_timestamps_csv.")

 if not (0 < args.frame_fraction <= 1.0):
 raise ValueError("--frame_fraction must be in (0, 1].")

 n_keep = max(1, int(round(len(valid_frame_indices_all) * args.frame_fraction)))
 valid_frame_indices = valid_frame_indices_all[:n_keep]

 start_idx = min(valid_frame_indices)
 end_idx = max(valid_frame_indices)
 print(
 f"Using first {n_keep}/{len(valid_frame_indices_all)} matched frames "
 f"({args.frame_fraction:.3f} of total): {start_idx} to {end_idx}"
 )

 emg_df = pd.read_csv(args.emg_csv)
 if "unix_time_s" not in emg_df.columns:
 raise ValueError("emg_csv must contain unix_time_s")
 emg_df = emg_df.sort_values("unix_time_s").reset_index(drop=True)

 emg_cols, _, _ = _pick_signal_columns(emg_df, emg_only=True)
 if len(emg_cols) == 0:
 raise ValueError("Could not find EMG columns in EMG CSV.")
 emg_cols = emg_cols[:args.max_channels]

 for c in emg_cols:
 if emg_df[c].isna().any():
 emg_df[c] = emg_df[c].fillna(emg_df[c].mean())

 t_emg = emg_df["unix_time_s"].to_numpy(dtype=np.float64)
 emg = emg_df[emg_cols].to_numpy(dtype=np.float32)
 emg = _reorder_left_hand_emg_if_needed(args.emg_csv, emg_cols, emg)

 emg_fs = args.emg_fs if args.emg_fs is not None else estimate_fs_from_time(t_emg)
 if len(emg) >= args.min_filter_len:
 emg_proc = get_emg_data(
 emg,
 window_size=args.emg_window_size,
 fs=emg_fs,
 cutoff=args.hp_cutoff,
 notch_freq=args.notch_freq,
 notch_q=args.notch_q,
 highpass_order=args.hp_order,
 ).astype(np.float32)
 else:
 emg_proc = emg.astype(np.float32)

 if args.fps is None:
 ts_valid = ts[ts["frame_idx"].isin(valid_frame_indices)].copy()
 frame_dt = np.diff(ts_valid["utc_sec"].to_numpy(dtype=np.float64))
 frame_dt = frame_dt[np.isfinite(frame_dt) & (frame_dt > 0)]
 fps = 30.0 if len(frame_dt) == 0 else 1.0 / np.median(frame_dt)
 else:
 fps = float(args.fps)

 first_idx = valid_frame_indices[0]
 first_frame = cv2.imread(frame_path_by_idx[first_idx], cv2.IMREAD_COLOR)
 if first_frame is None:
 raise RuntimeError(f"Failed to read frame {frame_path_by_idx[first_idx]}")
 if args.rotate_clockwise_90:
 first_frame = cv2.rotate(first_frame, cv2.ROTATE_90_CLOCKWISE)

 renderer = FastEMGPanelRenderer(
 t_emg=t_emg,
 emg_hp=emg_proc,
 channel_names=emg_cols,
 window_sec=args.plot_window_sec,
 panel_height=args.plot_height,
 panel_width=args.plot_width,
 )

 first_t = frame_to_utc[first_idx] + args.line_utc_offset_sec
 first_panel = renderer.render(first_t)
 first_canvas = render_side_by_side(first_frame, first_panel)
 h_out, w_out = first_canvas.shape[:2]

 fourcc = cv2.VideoWriter_fourcc(*args.codec)
 writer = cv2.VideoWriter(args.output_mp4, fourcc, fps, (w_out, h_out))
 if not writer.isOpened():
 raise RuntimeError(f"Could not open video writer for {args.output_mp4}")

 print(f"Writing {args.output_mp4}")
 print(f"EMG columns: {emg_cols}")
 print(f"Estimated EMG fs: {emg_fs:.3f} Hz")
 print(f"Output fps: {fps:.3f}")

 alignment_errors_ms = []

 for k, fidx in enumerate(valid_frame_indices):
 frame = cv2.imread(frame_path_by_idx[fidx], cv2.IMREAD_COLOR)
 if frame is None:
 print(f"warning: failed to read frame {frame_path_by_idx[fidx]}")
 continue

 if args.rotate_clockwise_90:
 frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

 t_frame = frame_to_utc[fidx] + args.line_utc_offset_sec

 emg_idx = nearest_index(t_emg, t_frame)
 nearest_emg_t = float(t_emg[emg_idx])
 alignment_error_sec = nearest_emg_t - t_frame
 alignment_error_ms = 1000.0 * alignment_error_sec
 alignment_errors_ms.append(alignment_error_ms)

 if (k + 1) % 100 == 0 or k < 10:
 print(
 f"frame {fidx}: frame_utc={t_frame:.6f}, "
 f"nearest_emg_utc={nearest_emg_t:.6f}, "
 f"alignment_error_ms={alignment_error_ms:+.2f}"
 )

 panel = renderer.render(t_frame)
 canvas = render_side_by_side(frame, panel)

 if canvas.shape[0] != h_out or canvas.shape[1] != w_out:
 canvas = cv2.resize(canvas, (w_out, h_out), interpolation=cv2.INTER_AREA)

 text = f"frame {fidx} | utc {t_frame:.6f}"
 cv2.putText(
 canvas,
 text,
 (20, 30),
 cv2.FONT_HERSHEY_SIMPLEX,
 0.8,
 (255, 255, 255),
 2,
 cv2.LINE_AA,
 )

 writer.write(canvas)

 if (k + 1) % 100 == 0 or (k + 1) == len(valid_frame_indices):
 print(f" rendered {k + 1}/{len(valid_frame_indices)} frames")

 writer.release()

 if len(alignment_errors_ms) > 0:
 errs = np.asarray(alignment_errors_ms, dtype=np.float64)
 print("Alignment error summary (nearest EMG timestamp - frame timestamp):")
 print(f" mean = {errs.mean():+.3f} ms")
 print(f" median = {np.median(errs):+.3f} ms")
 print(f" std = {errs.std():.3f} ms")
 print(f" min = {errs.min():+.3f} ms")
 print(f" max = {errs.max():+.3f} ms")
 print(f" mean |err| = {np.mean(np.abs(errs)):.3f} ms")

 print("Done.")


if __name__ == "__main__":
 main()
 
#  Code for running script:

# python make_single_videos.py --frames_dir ../../EMG/Sirguta_2/frames/ --frame_timestamps_csv ../../EMG/Sirguta_2/Sirguta_2_frame_timestamps.csv --emg_csv ../../EMG/Sirguta_2/right.csv --output_mp4 new3.mp4 --emg_fs 500 --rotate_clockwise_90