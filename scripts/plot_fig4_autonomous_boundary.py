#!/usr/bin/env python3
"""Figure 4: Autonomous TRACE exposes a safety-utility frontier and a
misleading-cue boundary.

Nature-style 2-panel figure:
  a: Autonomous trigger frontier (scatter plot)
  b: Misleading cue hijacking (dumbbell plot)

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

TRACE_BLUE = "#2166AC"
ORANGE = "#E07020"
RAW_GREY = "#555555"
LIGHT_GREY = "#BBBBBB"
CI_BLACK = "#000000"
RED_CUE = "#C02020"

# ── Create figure ─────────────────────────────────────────────────
fig = plt.figure(figsize=(180 / 25.4, 88 / 25.4))

gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.95], wspace=0.35,
                      left=0.08, right=0.97, top=0.92, bottom=0.12)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])

# ═══════════════════════════════════════════════════════════════════
# PANEL A — Autonomous trigger frontier: scatter plot
# ═══════════════════════════════════════════════════════════════════
versions = ["Raw", "Oracle", "Trace-only", "TRACE\nV3.1", "V3.2\nfrontier"]
fire_rate = [0, 75, 92, 71, 61]
error_rate = [76.2, 30.0, 6.2, 8.8, 18.8]
ci_low = [65.9, 21.1, 2.7, 4.3, 11.7]
ci_high = [84.2, 40.8, 13.8, 17.0, 28.7]
gold_label_free = [False, False, True, True, True]

# Point styles
markers = ["o", "o", "s", "D", "o"]
colours = [RAW_GREY, "#999999", ORANGE, TRACE_BLUE, TRACE_BLUE]
sizes = [50, 50, 55, 80, 55]
face_colours = [RAW_GREY, "#999999", ORANGE, TRACE_BLUE, "white"]
edge_colours = [RAW_GREY, "#999999", ORANGE, TRACE_BLUE, TRACE_BLUE]

for i, (xp, yp, lo, hi, m, c, fc, ec, sz) in enumerate(
        zip(fire_rate, error_rate, ci_low, ci_high,
            markers, colours, face_colours, edge_colours, sizes)):
    ax_a.plot(xp, yp, m, color=c, markerfacecolor=fc,
              markersize=np.sqrt(sz), markeredgewidth=1.0 if fc == "white" else 0,
              zorder=5)
    ax_a.plot([xp, xp], [lo, hi], "-", color=CI_BLACK if i > 0 else RAW_GREY,
              linewidth=1.0, zorder=3)

# Connection lines: Trace-only → V3.1 → V3.2 (calibration frontier)
frontier_idx = [2, 3, 4]
ax_a.plot([fire_rate[i] for i in frontier_idx],
          [error_rate[i] for i in frontier_idx],
          "--", color=LIGHT_GREY, linewidth=0.8, zorder=2)

# Annotations
ax_a.annotate("V3.1 primary", xy=(71, 8.8),
              xytext=(60, 1.5), fontsize=6, fontweight="bold",
              color=TRACE_BLUE, ha="center",
              arrowprops=dict(arrowstyle="->", color=LIGHT_GREY, lw=0.6))
ax_a.annotate("lower fire,\nhigher error", xy=(61, 18.8),
              xytext=(52, 13), fontsize=5.5, color=RAW_GREY,
              ha="center")
ax_a.annotate("Raw", xy=(0, 76.2),
              xytext=(6, 72), fontsize=5.8, color=RAW_GREY, ha="left")
ax_a.annotate("Oracle", xy=(75, 30.0),
              xytext=(78, 28), fontsize=5.8, color="#999999", ha="left")
ax_a.annotate("Trace-only", xy=(92, 6.2),
              xytext=(88, 0), fontsize=5.8, color=ORANGE, ha="right")

ax_a.set_xlabel("Fire rate (%)")
ax_a.set_ylabel("Error rate (%)")
ax_a.set_xlim(-5, 102)
ax_a.set_ylim(-2, 88)
ax_a.xaxis.set_major_locator(mticker.MultipleLocator(25))
ax_a.yaxis.set_major_locator(mticker.MultipleLocator(20))
ax_a.text(-0.22, 1.02, "a", transform=ax_a.transAxes,
          fontsize=9, fontweight="bold", va="bottom", ha="left")
ax_a.set_title("Gold-label-free trigger frontier", fontsize=8, pad=4)

# ═══════════════════════════════════════════════════════════════════
# PANEL B — Misleading cue hijacking: dumbbell plot
# ═══════════════════════════════════════════════════════════════════
groups = ["Misleading", "Conflict"]
y_pos = [0, 1]

# Misleading: gold vs cue
misleading_a = 0.010   # gold_span_RQK mean
misleading_b = 0.480   # cue_RQK mean
misleading_err_a = 0.005
misleading_err_b = 0.030

# Conflict: span1 vs span2
conflict_a = 0.0125
conflict_b = 0.100
conflict_err_a = 0.0115
conflict_err_b = 0.030

# --- Misleading row ---
ax_b.plot(misleading_a, y_pos[0], "o", color=TRACE_BLUE, markersize=7, zorder=5)
ax_b.plot(misleading_b, y_pos[0], "o", color=RED_CUE, markersize=9, zorder=5)
ax_b.plot([misleading_a, misleading_b], [y_pos[0], y_pos[0]],
          "-", color=RAW_GREY, linewidth=1.5, zorder=3)
ax_b.plot([misleading_a - misleading_err_a, misleading_a + misleading_err_a],
          [y_pos[0], y_pos[0]], "-", color=TRACE_BLUE, linewidth=2.5, zorder=4)
ax_b.plot([misleading_b - misleading_err_b, misleading_b + misleading_err_b],
          [y_pos[0], y_pos[0]], "-", color=RED_CUE, linewidth=2.5, zorder=4)

ax_b.text(misleading_a, y_pos[0] - 0.33, "Gold\nevidence",
          ha="center", fontsize=5.5, color=TRACE_BLUE, fontweight="bold")
ax_b.text(misleading_b, y_pos[0] - 0.33, "Misleading\ncue",
          ha="center", fontsize=5.5, color=RED_CUE, fontweight="bold")
ax_b.annotate("cue hijacking:\n13–108×",
              xy=(0.25, y_pos[0] + 0.25), fontsize=6,
              color=RED_CUE, fontweight="bold", ha="center")

# --- Conflict row ---
ax_b.plot(conflict_a, y_pos[1], "o", color=TRACE_BLUE, markersize=7, zorder=5)
ax_b.plot(conflict_b, y_pos[1], "o", color="#888888", markersize=7, zorder=5)
ax_b.plot([conflict_a, conflict_b], [y_pos[1], y_pos[1]],
          "-", color=RAW_GREY, linewidth=1.5, zorder=3)
ax_b.plot([conflict_a - conflict_err_a, conflict_a + conflict_err_a],
          [y_pos[1], y_pos[1]], "-", color=TRACE_BLUE, linewidth=2.5, zorder=4)
ax_b.plot([conflict_b - conflict_err_b, conflict_b + conflict_err_b],
          [y_pos[1], y_pos[1]], "-", color="#888888", linewidth=2.5, zorder=4)

ax_b.text(conflict_a, y_pos[1] - 0.33, "Evidence\nspan 1",
          ha="center", fontsize=5.5, color=TRACE_BLUE, fontweight="bold")
ax_b.text(conflict_b, y_pos[1] - 0.33, "Alternative\nspan",
          ha="center", fontsize=5.5, color=RAW_GREY, fontweight="bold")
ax_b.annotate("distributed evidence\nrouting",
              xy=(0.06, y_pos[1] + 0.25), fontsize=6,
              color="black", ha="center")

ax_b.set_yticks(y_pos)
ax_b.set_yticklabels(groups)
ax_b.set_xlabel("R_QK")
ax_b.set_xlim(-0.02, 0.58)
ax_b.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
ax_b.text(-0.22, 1.02, "b", transform=ax_b.transAxes,
          fontsize=9, fontweight="bold", va="bottom", ha="left")
ax_b.set_title("Misleading cues hijack QK routing", fontsize=8, pad=4)

# ── Save ──────────────────────────────────────────────────────────
for fmt in ("pdf", "svg", "png"):
    path = OUTPUT_DIR / f"fig4_autonomous_boundary.{fmt}"
    kwargs = dict(facecolor="white", edgecolor="none")
    if fmt == "png":
        kwargs["dpi"] = 300
    fig.savefig(str(path), **kwargs)
    print(f"  Saved {path}")

plt.close(fig)
print("Figure 4 done.")
