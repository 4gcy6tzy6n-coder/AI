#!/usr/bin/env python3
"""Figure 2: TRACE reduces evidence-to-answer black-box failures.

Nature-style 3-panel figure:
  a: Error rate by failure type (grouped bar chart)
  b: Relative reduction (horizontal forest plot)
  c: Pooled risk failures (before-after slope plot)

Saves PDF, SVG, and 300-dpi PNG.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Global style ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})

# Colours (colourblind-friendly)
RAW_GREY = "#555555"
TRACE_BLUE = "#2166AC"
CI_BLACK = "#000000"
LIGHT_GREY = "#BBBBBB"

# ── Data ──────────────────────────────────────────────────────────
failure_types = ["Conflict", "Evidence\ngap", "Misleading\nhint", "Direct\nevidence"]
n = [40, 41, 40, 40]
raw_errors = [39, 40, 32, 2]
trace_errors = [1, 2, 30, 2]
raw_rates = [e / n_i * 100 for e, n_i in zip(raw_errors, n)]
trace_rates = [e / n_i * 100 for e, n_i in zip(trace_errors, n)]

reduction_labels = ["Conflict", "Evidence gap", "Misleading hint", "Pooled"]
reductions = [97.4, 95.0, 6.3, 70.3]
ci_low = [92.1, 87.5, -14.3, 61.1]
ci_high = [100.0, 100.0, 24.2, 78.8]

# ── Create figure ─────────────────────────────────────────────────
fig = plt.figure(figsize=(180 / 25.4, 108 / 25.4))  # 180 × 108 mm

# Panel layout: a (left, full height), b+c (right column, stacked)
gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], height_ratios=[1.0, 0.85],
                      hspace=0.42, wspace=0.38,
                      left=0.08, right=0.98, top=0.94, bottom=0.10)

ax_a = fig.add_subplot(gs[:, 0])   # Panel a: left column, full height
ax_b = fig.add_subplot(gs[0, 1])   # Panel b: top right
ax_c = fig.add_subplot(gs[1, 1])   # Panel c: bottom right

# ═══════════════════════════════════════════════════════════════════
# PANEL A — Grouped bar chart: error rate by failure type
# ═══════════════════════════════════════════════════════════════════
x = np.arange(len(failure_types))
bar_w = 0.32

bars_raw = ax_a.bar(x - bar_w / 2, raw_rates, bar_w,
                     color=RAW_GREY, edgecolor="white", linewidth=0.3,
                     label="Raw")
bars_trace = ax_a.bar(x + bar_w / 2, trace_rates, bar_w,
                       color=TRACE_BLUE, edgecolor="white", linewidth=0.3,
                       label="TRACE")

# Annotate fractions on top of bars
for i, (bx, rate) in enumerate(zip(bars_raw, raw_rates)):
    ax_a.text(bx.get_x() + bx.get_width() / 2, rate + 1.2,
              f"{raw_errors[i]}/{n[i]}", ha="center", va="bottom",
              fontsize=5.5, color=RAW_GREY, fontweight="bold")
for i, (bx, rate) in enumerate(zip(bars_trace, trace_rates)):
    y_pos = rate + 1.2
    ax_a.text(bx.get_x() + bx.get_width() / 2, y_pos,
              f"{trace_errors[i]}/{n[i]}", ha="center", va="bottom",
              fontsize=5.5, color=TRACE_BLUE, fontweight="bold")

ax_a.set_xticks(x)
ax_a.set_xticklabels(failure_types, fontsize=6.5)
ax_a.set_ylabel("Error rate (%)")
ax_a.set_ylim(0, 108)
ax_a.yaxis.set_major_locator(mticker.MultipleLocator(25))
ax_a.legend(frameon=False, loc="upper left", fontsize=6,
            handlelength=0.8, handleheight=0.6)
ax_a.text(-0.28, 1.02, "a", transform=ax_a.transAxes,
          fontsize=9, fontweight="bold", va="bottom", ha="left")
ax_a.set_title("Error rate by failure type", fontsize=8, pad=4)

# Dashed horizontal at 50% for reference
ax_a.axhline(y=50, color=LIGHT_GREY, linestyle="--", linewidth=0.5, zorder=0)

# ═══════════════════════════════════════════════════════════════════
# PANEL B — Horizontal forest plot: relative error reduction
# ═══════════════════════════════════════════════════════════════════
n_items = len(reduction_labels)
y_positions = np.arange(n_items)[::-1]  # top to bottom

for i, (yp, red, lo, hi) in enumerate(zip(y_positions, reductions, ci_low, ci_high)):
    marker_style = "D" if reduction_labels[n_items - 1 - i] == "Pooled" else "o"
    ms = 5.5 if reduction_labels[n_items - 1 - i] == "Pooled" else 4.5
    colour = TRACE_BLUE

    ax_b.plot(red, yp, marker_style, color=colour, markersize=ms, zorder=5)
    ax_b.plot([lo, hi], [yp, yp], "-", color=CI_BLACK, linewidth=1.2, zorder=4)
    ax_b.plot([lo, lo], [yp - 0.15, yp + 0.15], "-", color=CI_BLACK, linewidth=0.8)
    ax_b.plot([hi, hi], [yp - 0.15, yp + 0.15], "-", color=CI_BLACK, linewidth=0.8)

    # Annotation
    lbl = reduction_labels[n_items - 1 - i]
    ann = f"{red:.1f}%"
    if lo is not None and hi is not None:
        ann += f"  [{lo:.1f}, {hi:.1f}]"
    ax_b.text(red + 1.5, yp + 0.22, ann, fontsize=5.2, va="bottom", color="black")

ax_b.set_yticks(y_positions)
ax_b.set_yticklabels(reduction_labels[::-1])
ax_b.axvline(x=0, color=LIGHT_GREY, linestyle="--", linewidth=0.7, zorder=0)
ax_b.set_xlabel("Relative error reduction (%)")
ax_b.set_xlim(-28, 112)
ax_b.xaxis.set_major_locator(mticker.MultipleLocator(25))
ax_b.text(-0.25, 1.02, "b", transform=ax_b.transAxes,
          fontsize=9, fontweight="bold", va="bottom", ha="left")
ax_b.set_title("Relative reduction", fontsize=8, pad=4)

# ═══════════════════════════════════════════════════════════════════
# PANEL C — Before-after slope plot: pooled risk failures
# ═══════════════════════════════════════════════════════════════════
pooled_raw_rate = 111 / 121 * 100   # 91.7%
pooled_trace_rate = 33 / 121 * 100  # 27.3%
pooled_red = 70.3

ax_c.plot([0, 1], [pooled_raw_rate, pooled_trace_rate],
          "-", color=TRACE_BLUE, linewidth=2.8, solid_capstyle="round", zorder=5)
ax_c.plot(0, pooled_raw_rate, "o", color=RAW_GREY, markersize=9, zorder=6)
ax_c.plot(1, pooled_trace_rate, "o", color=TRACE_BLUE, markersize=9, zorder=6)

# Annotations
ax_c.text(0, pooled_raw_rate + 3.5, f"111/121\n({pooled_raw_rate:.1f}%)",
          ha="center", fontsize=6.5, color=RAW_GREY, fontweight="bold")
ax_c.text(1, pooled_trace_rate + 3.5, f"33/121\n({pooled_trace_rate:.1f}%)",
          ha="center", fontsize=6.5, color=TRACE_BLUE, fontweight="bold")
ax_c.text(0.5, (pooled_raw_rate + pooled_trace_rate) / 2 - 5,
          f"{pooled_red:.1f}% reduction\np < 10⁻⁶",
          ha="center", fontsize=6.5, fontweight="bold", color="black")

ax_c.set_xticks([0, 1])
ax_c.set_xticklabels(["Raw", "TRACE"])
ax_c.set_ylabel("Pooled error rate (%)")
ax_c.set_ylim(0, 105)
ax_c.yaxis.set_major_locator(mticker.MultipleLocator(25))
ax_c.text(-0.25, 1.02, "c", transform=ax_c.transAxes,
          fontsize=9, fontweight="bold", va="bottom", ha="left")
ax_c.set_title("Pooled risk failures", fontsize=8, pad=4)

# ── Save ──────────────────────────────────────────────────────────
for fmt in ("pdf", "svg", "png"):
    path = OUTPUT_DIR / f"fig2_trace_primary_results.{fmt}"
    kwargs = dict(facecolor="white", edgecolor="none")
    if fmt == "png":
        kwargs["dpi"] = 300
        kwargs["pil_kwargs"] = {"compress_level": 1}
    fig.savefig(str(path), **kwargs)
    print(f"  Saved {path}")

plt.close(fig)
print("Figure 2 done.")
