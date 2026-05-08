#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
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


def load_and_process_emg(csv_path: str, hand_label: str, max_channels: int,
                          min_filter_len: int, emg_window_size: int,
                          emg_fs_override, hp_cutoff: float, hp_order: int,
                          notch_freq: float, notch_q: float):
    """Load one EMG CSV, filter it, and return (t_emg, emg_proc, channel_names, emg_fs)."""
    df = pd.read_csv(csv_path)
    if "unix_time_s" not in df.columns:
        raise ValueError(f"{csv_path} must contain unix_time_s")
    df = df.sort_values("unix_time_s").reset_index(drop=True)

    emg_cols, _, _ = _pick_signal_columns(df, emg_only=True)
    if len(emg_cols) == 0:
        raise ValueError(f"Could not find EMG columns in {csv_path}")
    emg_cols = emg_cols[:max_channels]

    for c in emg_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mean())

    t_emg = df["unix_time_s"].to_numpy(dtype=np.float64)
    emg = df[emg_cols].to_numpy(dtype=np.float32)
    emg = _reorder_left_hand_emg_if_needed(csv_path, emg_cols, emg)

    emg_fs = emg_fs_override if emg_fs_override is not None else estimate_fs_from_time(t_emg)

    if len(emg) >= min_filter_len:
        emg_proc = get_emg_data(
            emg,
            window_size=emg_window_size,
            fs=emg_fs,
            cutoff=hp_cutoff,
            notch_freq=notch_freq,
            notch_q=notch_q,
            highpass_order=hp_order,
        ).astype(np.float32)
    else:
        emg_proc = emg.astype(np.float32)

    # Prefix channel names with hand label so they are distinguishable in the panel
    prefixed_names = [f"{hand_label}_{c}" for c in emg_cols]
    return t_emg, emg_proc, prefixed_names, emg_fs


class FastEMGPanelRenderer:
    """Renders a scrolling EMG waveform panel.

    Supports one or two hands. When two hands are provided the panel is split
    vertically: Left hand on top, Right hand on bottom, separated by a thin
    divider line.
    """

    # Colours used for the two-hand divider header bars (BGR)
    _HAND_COLOURS = {
        "left":  (200, 120,  50),   # blue-ish
        "right": ( 50, 120, 200),   # orange-ish
    }

    def __init__(
        self,
        t_emg_left: np.ndarray,
        emg_hp_left: np.ndarray,
        channel_names_left,
        t_emg_right: np.ndarray = None,
        emg_hp_right: np.ndarray = None,
        channel_names_right=None,
        window_sec=4.0,
        panel_height=720,
        panel_width=1100,
        left_pad=80,
        right_pad=20,
        top_pad=20,
        bottom_pad=35,
    ):
        self.t_left = t_emg_left
        self.emg_left = emg_hp_left
        self.names_left = list(channel_names_left)

        self.dual = (t_emg_right is not None and emg_hp_right is not None)
        self.t_right = t_emg_right
        self.emg_right = emg_hp_right
        self.names_right = list(channel_names_right) if channel_names_right else []

        self.window_sec = float(window_sec)
        self.panel_height = int(panel_height)
        self.panel_width = int(panel_width)
        self.left_pad = int(left_pad)
        self.right_pad = int(right_pad)
        self.top_pad = int(top_pad)
        self.bottom_pad = int(bottom_pad)

        self.inner_w = self.panel_width - self.left_pad - self.right_pad

        # Precompute robust vertical scaling per channel
        q_left = np.quantile(np.abs(emg_hp_left), 0.995, axis=0)
        q_left[q_left < 1e-6] = 1.0
        self.scales_left = q_left.astype(np.float32)

        if self.dual:
            q_right = np.quantile(np.abs(emg_hp_right), 0.995, axis=0)
            q_right[q_right < 1e-6] = 1.0
            self.scales_right = q_right.astype(np.float32)

        # Divide the inner height between hands
        self._divider_h = 18 if self.dual else 0
        usable_h = self.panel_height - self.top_pad - self.bottom_pad - self._divider_h
        if self.dual:
            n_left = len(self.names_left)
            n_right = len(self.names_right)
            n_total = n_left + n_right
            # Allocate rows proportionally
            self._left_h = max(60, int(round(usable_h * n_left / n_total)))
            self._right_h = usable_h - self._left_h
            self._ch_h_left = max(20, self._left_h // n_left)
            self._ch_h_right = max(20, self._right_h // n_right)
        else:
            n_left = len(self.names_left)
            self._left_h = usable_h
            self._ch_h_left = max(20, self._left_h // n_left)

    # ------------------------------------------------------------------
    def _draw_hand_section(self, panel, t_emg, emg_hp, scales, names,
                           ch_h, section_top, section_bot, t_center, t0, colour_label):
        half = self.window_sec / 2.0
        t1 = t_center + half

        lo = int(np.searchsorted(t_emg, t0, side="left"))
        hi = int(np.searchsorted(t_emg, t1, side="right"))
        if hi <= lo:
            idx = int(np.argmin(np.abs(t_emg - t_center)))
            lo = max(0, idx - 1)
            hi = min(len(t_emg), idx + 2)

        tt = t_emg[lo:hi]
        xx = emg_hp[lo:hi]
        n_ch = len(names)

        for ch in range(n_ch):
            y_top = section_top + ch * ch_h
            y_bot = min(section_top + (ch + 1) * ch_h - 1, section_bot)
            y_mid = (y_top + y_bot) // 2

            cv2.line(panel, (self.left_pad, y_mid),
                     (self.panel_width - self.right_pad, y_mid), (220, 220, 220), 1)

            cv2.putText(panel, str(names[ch]),
                        (4, y_mid + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)

            if len(tt) >= 2:
                x = self.left_pad + ((tt - t0) / self.window_sec * (self.inner_w - 1)).astype(np.int32)
                y_amp = xx[:, ch] / scales[ch]
                y_amp = np.clip(y_amp, -1.0, 1.0)
                amp_px = max(4, int(0.42 * (y_bot - y_top)))
                y = (y_mid - y_amp * amp_px).astype(np.int32)
                pts = np.stack([x, y], axis=1).reshape(-1, 1, 2)
                cv2.polylines(panel, [pts], isClosed=False,
                              color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    # ------------------------------------------------------------------
    def render(self, t_center: float):
        half = self.window_sec / 2.0
        t0 = t_center - half

        panel = np.full((self.panel_height, self.panel_width, 3), 255, dtype=np.uint8)

        x_cursor = self.left_pad + int(round((t_center - t0) / self.window_sec * (self.inner_w - 1)))
        x_cursor = max(self.left_pad, min(self.panel_width - self.right_pad - 1, x_cursor))

        left_top = self.top_pad
        left_bot = self.top_pad + self._left_h

        self._draw_hand_section(
            panel, self.t_left, self.emg_left, self.scales_left, self.names_left,
            self._ch_h_left, left_top, left_bot, t_center, t0, "left"
        )

        if self.dual:
            # Divider bar between left and right sections
            divider_y = left_bot
            colour = self._HAND_COLOURS["right"]
            cv2.rectangle(panel,
                          (0, divider_y),
                          (self.panel_width, divider_y + self._divider_h),
                          colour, -1)
            cv2.putText(panel, "RIGHT HAND",
                        (self.left_pad, divider_y + self._divider_h - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            right_top = divider_y + self._divider_h
            right_bot = right_top + self._right_h

            self._draw_hand_section(
                panel, self.t_right, self.emg_right, self.scales_right, self.names_right,
                self._ch_h_right, right_top, right_bot, t_center, t0, "right"
            )

            # Label the left section header retroactively
            colour_left = self._HAND_COLOURS["left"]
            cv2.rectangle(panel,
                          (0, self.top_pad - self._divider_h),
                          (self.panel_width, self.top_pad),
                          colour_left, -1)
            cv2.putText(panel, "LEFT HAND",
                        (self.left_pad, self.top_pad - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Cursor line spanning full inner height
        cv2.line(panel,
                 (x_cursor, self.top_pad),
                 (x_cursor, self.panel_height - self.bottom_pad),
                 (0, 0, 255), 1, cv2.LINE_AA)

        cv2.putText(panel, "UTC time (s)",
                    (self.panel_width // 2 - 40, self.panel_height - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        return panel


def render_side_by_side(frame_bgr, panel_bgr):
    target_h = max(frame_bgr.shape[0], panel_bgr.shape[0])
    frame_bgr = resize_to_height(frame_bgr, target_h)
    panel_bgr = resize_to_height(panel_bgr, target_h)
    return np.concatenate([frame_bgr, panel_bgr], axis=1)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Create one full-length MP4 over the entire frame folder, "
            "side-by-side with high-pass EMG for one or both hands.\n"
            "Provide --left_emg_csv and/or --right_emg_csv (at least one required).\n"
            "When both are given the EMG panel is split: Left on top, Right on bottom."
        )
    )
    p.add_argument("--frames_dir", type=str, required=True)
    p.add_argument("--frame_timestamps_csv", type=str, required=True,
                   help="CSV with at least columns: frame_idx, utc_sec")

    # Dual-hand inputs (replaces the old --emg_csv)
    p.add_argument("--left_emg_csv", type=str, default=None,
                   help="Left-hand EMG CSV with unix_time_s and EMG channels")
    p.add_argument("--right_emg_csv", type=str, default=None,
                   help="Right-hand EMG CSV with unix_time_s and EMG channels")

    p.add_argument("--output_mp4", type=str, required=True)
    p.add_argument("--fps", type=float, default=None,
                   help="Output fps. If omitted, estimated from frame timestamps.")
    p.add_argument("--emg_fs", type=float, default=None,
                   help="EMG sampling rate (applied to both hands). If omitted, estimated from unix_time_s.")
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

    if args.left_emg_csv is None and args.right_emg_csv is None:
        raise ValueError("At least one of --left_emg_csv or --right_emg_csv must be provided.")

    out_dir = os.path.dirname(args.output_mp4)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # ── Frames ────────────────────────────────────────────────────────
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
    print(
        f"Using first {n_keep}/{len(valid_frame_indices_all)} matched frames "
        f"({args.frame_fraction:.3f} of total): "
        f"{min(valid_frame_indices)} to {max(valid_frame_indices)}"
    )

    # ── EMG loading ───────────────────────────────────────────────────
    common_kw = dict(
        max_channels=args.max_channels,
        min_filter_len=args.min_filter_len,
        emg_window_size=args.emg_window_size,
        emg_fs_override=args.emg_fs,
        hp_cutoff=args.hp_cutoff,
        hp_order=args.hp_order,
        notch_freq=args.notch_freq,
        notch_q=args.notch_q,
    )

    t_left = emg_left = names_left = emg_fs_left = None
    t_right = emg_right = names_right = emg_fs_right = None

    if args.left_emg_csv:
        t_left, emg_left, names_left, emg_fs_left = load_and_process_emg(
            args.left_emg_csv, "L", **common_kw)
        print(f"Left  EMG columns : {names_left}")
        print(f"Left  EMG fs      : {emg_fs_left:.3f} Hz")

    if args.right_emg_csv:
        t_right, emg_right, names_right, emg_fs_right = load_and_process_emg(
            args.right_emg_csv, "R", **common_kw)
        print(f"Right EMG columns : {names_right}")
        print(f"Right EMG fs      : {emg_fs_right:.3f} Hz")

    # Use whichever hand is available for alignment-error reporting
    t_emg_primary = t_left if t_left is not None else t_right

    # ── FPS ───────────────────────────────────────────────────────────
    if args.fps is None:
        ts_valid = ts[ts["frame_idx"].isin(valid_frame_indices)].copy()
        frame_dt = np.diff(ts_valid["utc_sec"].to_numpy(dtype=np.float64))
        frame_dt = frame_dt[np.isfinite(frame_dt) & (frame_dt > 0)]
        fps = 30.0 if len(frame_dt) == 0 else 1.0 / np.median(frame_dt)
    else:
        fps = float(args.fps)

    # ── First frame → canvas size ─────────────────────────────────────
    first_idx = valid_frame_indices[0]
    first_frame = cv2.imread(frame_path_by_idx[first_idx], cv2.IMREAD_COLOR)
    if first_frame is None:
        raise RuntimeError(f"Failed to read frame {frame_path_by_idx[first_idx]}")
    if args.rotate_clockwise_90:
        first_frame = cv2.rotate(first_frame, cv2.ROTATE_90_CLOCKWISE)

    renderer = FastEMGPanelRenderer(
        t_emg_left=t_left if t_left is not None else np.array([0.0]),
        emg_hp_left=emg_left if emg_left is not None else np.zeros((1, 1), dtype=np.float32),
        channel_names_left=names_left if names_left is not None else ["(none)"],
        t_emg_right=t_right,
        emg_hp_right=emg_right,
        channel_names_right=names_right,
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

    print(f"Writing {args.output_mp4}  |  fps={fps:.3f}")

    alignment_errors_left_ms = []
    alignment_errors_right_ms = []

    for k, fidx in enumerate(valid_frame_indices):
        frame = cv2.imread(frame_path_by_idx[fidx], cv2.IMREAD_COLOR)
        if frame is None:
            print(f"warning: failed to read frame {frame_path_by_idx[fidx]}")
            continue

        if args.rotate_clockwise_90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        t_frame = frame_to_utc[fidx] + args.line_utc_offset_sec

        # Per-hand alignment diagnostics
        if t_left is not None:
            li = nearest_index(t_left, t_frame)
            err_l = 1000.0 * (float(t_left[li]) - t_frame)
            alignment_errors_left_ms.append(err_l)

        if t_right is not None:
            ri = nearest_index(t_right, t_frame)
            err_r = 1000.0 * (float(t_right[ri]) - t_frame)
            alignment_errors_right_ms.append(err_r)

        if (k + 1) % 100 == 0 or k < 10:
            msg = f"frame {fidx}: frame_utc={t_frame:.6f}"
            if t_left is not None:
                msg += f"  L_err={err_l:+.2f}ms"
            if t_right is not None:
                msg += f"  R_err={err_r:+.2f}ms"
            print(msg)

        panel = renderer.render(t_frame)
        canvas = render_side_by_side(frame, panel)

        if canvas.shape[0] != h_out or canvas.shape[1] != w_out:
            canvas = cv2.resize(canvas, (w_out, h_out), interpolation=cv2.INTER_AREA)

        cv2.putText(canvas,
                    f"frame {fidx} | utc {t_frame:.6f}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(canvas)

        if (k + 1) % 100 == 0 or (k + 1) == len(valid_frame_indices):
            print(f"  rendered {k + 1}/{len(valid_frame_indices)} frames")

    writer.release()

    def _print_err_summary(label, errs_ms):
        if len(errs_ms) == 0:
            return
        e = np.asarray(errs_ms, dtype=np.float64)
        print(f"Alignment error summary — {label} (nearest EMG t - frame t):")
        print(f"  mean   = {e.mean():+.3f} ms")
        print(f"  median = {np.median(e):+.3f} ms")
        print(f"  std    = {e.std():.3f} ms")
        print(f"  min    = {e.min():+.3f} ms")
        print(f"  max    = {e.max():+.3f} ms")
        print(f"  mean|e|= {np.mean(np.abs(e)):.3f} ms")

    _print_err_summary("LEFT hand", alignment_errors_left_ms)
    _print_err_summary("RIGHT hand", alignment_errors_right_ms)
    print("Done.")


if __name__ == "__main__":
    main()

# Example usage (both hands):
# python make_single_videos.py \
#   --frames_dir       ../../EMG/Sirguta_2/frames/ \
#   --frame_timestamps_csv ../../EMG/Sirguta_2/Sirguta_2_frame_timestamps.csv \
#   --left_emg_csv     ../../EMG/Sirguta_2/left.csv \
#   --right_emg_csv    ../../EMG/Sirguta_2/right.csv \
#   --output_mp4       new3.mp4 \
#   --emg_fs           500 \
#   --rotate_clockwise_90
#
# Single-hand (right only):
# python make_single_videos.py \
#   --frames_dir       ../../EMG/Sirguta_2/frames/ \
#   --frame_timestamps_csv ../../EMG/Sirguta_2/Sirguta_2_frame_timestamps.csv \
#   --right_emg_csv    ../../EMG/Sirguta_2/right.csv \
#   --output_mp4       new3.mp4 \
#   --emg_fs           500 \
#   --rotate_clockwise_90
