import numpy as np
import os
from PIL import Image

# ── CONFIG ──────────────────────────────────────────────────────────────
NPZ_DIR   = "event/"       # folder with 000000.npz, 000001.npz ...
IMG_DIR   = "image/"        # to read image size (h, w)
SAVE_DIR  = "event/"        # output: will create 000001_00.png etc.
NUM_BINS  = 6
EVENT_SCALE = 1                          # use 1 for non-BS-ERGB data
# ────────────────────────────────────────────────────────────────────────

os.makedirs(SAVE_DIR, exist_ok=True)

# get image size from first frame
sample_img = Image.open(os.path.join(IMG_DIR, sorted(os.listdir(IMG_DIR))[0]))
W, H = sample_img.size  # PIL gives (width, height)


def events_to_voxel_grid(events, num_bins, width, height):
    """Bilinear interpolation voxel grid. events: (N,4) as [ts, x, y, pol]"""
    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    if len(events) < 5:
        return np.reshape(voxel_grid, (num_bins, height, width))

    last_stamp  = events[-1, 0]
    first_stamp = events[0,  0]
    deltaT = last_stamp - first_stamp if last_stamp != first_stamp else 1.0

    ts   = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    xs   = events[:, 1].astype(np.int64)
    ys   = events[:, 2].astype(np.int64)
    pols = events[:, 3].copy()
    pols[pols == 0] = -1

    tis        = ts.astype(np.int64)
    dts        = ts - tis
    vals_left  = pols * (1.0 - dts)
    vals_right = pols * dts

    valid = tis < num_bins
    np.add.at(voxel_grid,
              xs[valid] + ys[valid] * width + tis[valid] * width * height,
              vals_left[valid])

    valid = (tis + 1) < num_bins
    np.add.at(voxel_grid,
              xs[valid] + ys[valid] * width + (tis[valid] + 1) * width * height,
              vals_right[valid])

    return np.reshape(voxel_grid, (num_bins, height, width))


def voxel_norm(voxel):
    """Percentile-clip and normalize each bin to [-1, 1]."""
    voxel = voxel.copy()
    pos = voxel[voxel > 0]
    neg = voxel[voxel < 0]
    if len(pos) == 0 or len(neg) == 0:
        return voxel

    p2,  p98 = np.percentile(pos, 2),  np.percentile(pos, 98)
    n2,  n98 = np.percentile(neg, 2),  np.percentile(neg, 98)

    voxel[voxel > 0] = np.clip(voxel[voxel > 0], p2, p98)
    voxel[voxel < 0] = np.clip(voxel[voxel < 0], n2, n98)

    denom_p = (p98 - p2) if p98 != p2 else 1.0
    denom_n = (n98 - n2) if n98 != n2 else 1.0

    voxel[voxel > 0] = (voxel[voxel > 0] - p2) / denom_p
    voxel[voxel < 0] = -1.0 * (voxel[voxel < 0] - n2) / denom_n

    return voxel


def npz_to_mstack_pngs(npz_path, frame_idx, save_dir, h, w, num_bins, event_scale):
    """Convert one NPZ to NUM_BINS grayscale PNGs."""
    data = np.load(npz_path)
    x  = data['x'].astype(np.float32)
    y  = data['y'].astype(np.float32)
    ts = data['timestamp'].astype(np.float64)
    p  = data['polarity'].astype(np.int32)

    # scale coordinates (use event_scale=1 unless BS-ERGB)
    x = np.round(x / event_scale).astype(np.int32)
    y = np.round(y / event_scale).astype(np.int32)

    # polarity: {0,1} → {-1,+1}
    p = np.where(p == 0, -1, p)

    # clip to image bounds
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)

    # assemble (ts, x, y, p)
    events = np.stack((ts, x, y, p), axis=1)

    # build voxel grid
    voxel = events_to_voxel_grid(events, num_bins, w, h)
    voxel = voxel_norm(voxel)

    # save each bin as a grayscale PNG  [-1,1] → [0,255]
    voxel_vis = np.clip(voxel, -1.0, 1.0)
    voxel_vis = ((voxel_vis + 1.0) / 2.0 * 255).astype(np.uint8)

    stem = f"{frame_idx:06d}"
    for b in range(num_bins):
        img = Image.fromarray(voxel_vis[b])
        img.save(os.path.join(save_dir, f"{stem}_{str(b).zfill(2)}.png"))


# ── MAIN LOOP ────────────────────────────────────────────────────────────
# NPZ 000000.npz holds events BETWEEN frame 0 and frame 1,
# so it gets saved as event files for frame index 1.
npz_files = sorted([f for f in os.listdir(NPZ_DIR) if f.endswith('.npz')])

for npz_file in npz_files:
    frame_idx = int(npz_file.split('.')[0])   # 000000 → 0
    save_frame_idx = frame_idx + 1            # events go to frame 1's slot

    npz_path = os.path.join(NPZ_DIR, npz_file)
    npz_to_mstack_pngs(npz_path, save_frame_idx, SAVE_DIR, H, W, NUM_BINS, EVENT_SCALE)
    print(f"Converted {npz_file} → event/{save_frame_idx:06d}_00..05.png")

print("Done.")
