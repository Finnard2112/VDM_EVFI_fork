from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


OUT_PATH = Path("/nfshomes/phan2003/Report/figures/evaluation_conditions_table.png")


ROWS = [
    (
        "EMG training run",
        "Frozen SVD + trainable ControlNet + 1D EMG encoder",
        "Main model; tests whether EMG-conditioned residual branch can improve interpolation.",
    ),
    (
        "Checkpoint validation",
        "GIFs every 500 steps: 500 to 3500",
        "Tracks training progress, color stability, wobble, and temporal consistency over time.",
    ),
    (
        "ControlNet scale 0.0",
        "Disable ControlNet residual at inference",
        "Tests frozen SVD / frame prior without the learned control branch.",
    ),
    (
        "ControlNet scale 0.5",
        "Use half-strength ControlNet residual",
        "Tests whether weaker learned control gives smoother or less distorted outputs.",
    ),
    (
        "ControlNet scale 1.0",
        "Use full learned ControlNet residual",
        "Nominal setting; checks the best quality from the trained control branch.",
    ),
    (
        "Zero-EMG validation",
        "Replace EMG with all zeros at inference",
        "Tests whether the trained model needs EMG magnitude information.",
    ),
    (
        "Shuffled-EMG validation",
        "Use EMG from the wrong temporal segment",
        "Tests whether the model depends on synchronized EMG timing.",
    ),
    (
        "Frame-only baseline",
        "Train ControlNet with EMG zeroed from the beginning",
        "Tests whether gains come from EMG or from a visual adaptation branch.",
    ),
]


def wrap(text, width):
    words = text.split()
    lines = []
    current = []
    for word in words:
        trial = " ".join(current + [word])
        if len(trial) > width and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 9), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    bg = "#f7f8fb"
    header = "#1d3557"
    band = "#e8eef7"
    line = "#c4cedd"
    accent = "#2a9d8f"
    text = "#17202a"
    muted = "#4f5d6b"

    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    ax.text(
        0.035,
        0.94,
        "Evaluation and Baseline Conditions",
        fontsize=27,
        fontweight="bold",
        color=text,
        va="center",
    )
    ax.text(
        0.035,
        0.895,
        "Ablations used to test whether the ControlNet branch is using synchronized EMG or mostly acting as a visual adapter",
        fontsize=13.5,
        color=muted,
        va="center",
    )

    left = 0.035
    right = 0.965
    top = 0.83
    bottom = 0.065
    table_w = right - left
    table_h = top - bottom

    col_w = [0.255, 0.355, 0.39]
    x = [left, left + col_w[0] * table_w, left + (col_w[0] + col_w[1]) * table_w, right]

    n_rows = len(ROWS) + 1
    row_h = table_h / n_rows

    ax.add_patch(Rectangle((left, top - row_h), table_w, row_h, facecolor=header, edgecolor=header))
    headers = ["Condition", "What Changed", "What It Tests"]
    for i, label in enumerate(headers):
        ax.text(
            x[i] + 0.015,
            top - row_h / 2,
            label,
            fontsize=14.5,
            fontweight="bold",
            color="white",
            va="center",
        )

    for r, row in enumerate(ROWS):
        y_top = top - (r + 2) * row_h
        fill = "white" if r % 2 == 0 else band
        ax.add_patch(Rectangle((left, y_top), table_w, row_h, facecolor=fill, edgecolor=line, linewidth=0.7))
        ax.add_patch(Rectangle((left, y_top), 0.0075, row_h, facecolor=accent, edgecolor=accent, linewidth=0))

        condition, changed, purpose = row
        cells = [
            wrap(condition, 23),
            wrap(changed, 39),
            wrap(purpose, 50),
        ]
        sizes = [13.5, 12.2, 12.2]
        weights = ["bold", "normal", "normal"]
        colors = [text, text, muted]

        for c, cell in enumerate(cells):
            ax.text(
                x[c] + 0.015,
                y_top + row_h / 2,
                cell,
                fontsize=sizes[c],
                fontweight=weights[c],
                color=colors[c],
                va="center",
                linespacing=1.25,
            )

    for xi in x:
        ax.plot([xi, xi], [bottom, top], color=line, linewidth=0.8)
    ax.plot([left, right], [top, top], color=line, linewidth=0.8)
    ax.plot([left, right], [bottom, bottom], color=line, linewidth=0.8)

    footer = "Dataset: 9 MP4 videos, 15 fps, 14-frame clips; validation GIFs saved every 500 steps on held-out Sirguta2_video.mp4."
    ax.text(0.035, 0.03, footer, fontsize=11.5, color=muted, va="center")

    fig.savefig(OUT_PATH, bbox_inches="tight", facecolor=bg)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
