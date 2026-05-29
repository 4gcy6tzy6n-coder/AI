#!/usr/bin/env python3
"""Figure 3: TRACE depends on mechanism-matched intervention and
transfers across architectures.

Nature-style 2-panel figure:
  a: Intervention ablation (point plot + vertical CI error bars)
  b: Cross-architecture transfer (horizontal dot/forest plot)

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
RAW_GREY = "#555555"
LIGHT_GREY = "#BBBBBB"
CI_BLACK = "#000000"

# ── Create figure ─────────────────────────────────────────────────
fig = plt.figure(figsize=(180 / 25.4, 88 / 25.4))

gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.35,
                      left=0.08, right=0.97, top=0.92, bottom=0.12)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])

# ═══════════════════════════════════════════════════════════════════
# PANEL A — Intervention ablation: point plot with CI error bars
# ═══════════════════════════════════════════════════════════════════
variants = ["TRACE\nfull", "Random", "Mismatched", "No\nintervention"]
errors = [42, 61, 78, 118]
n_total = 121
rates = [e / n_total * 100 for e in errors]
ci_low_pct = [26.8, 41.6, 55.6, 93.0]
ci_high_pct = [43.5, 59.2, 72.4, 99.2]

colours = [TRACE_BLUE, RAW_GREY, RAW_GREY, RAW_GREY]
x_pos = np.arange(len(variants))

for i, (xp, rate, lo, hi, colour) in enumerate(
        zip(x_pos, rates, ci_low_pct, ci_high_pct, colours)):
    ms = 7 if i == 0 else 5.5
    ax_a.plot(xp, rate, "o", color=colour, markersize=ms, zorder=5)
    ax_a.plot([xp, xp], [lo, hi], "-", color=CI_BLACK, linewidth=1.2, zorder=4)
    ax_a.plot([xp - 0.12, xp + 0.12], [lo, lo], "-", color=CI_BLACK, linewidth=0.8)
    ax_a.plot([xp - 0.12, xp + 0.12], [hi, hi], "-", color=CI_BLACK, linewidth=0.8)
    # Fraction label
    ax_a.text(xp, hi + 1.5, f"{errors[i]}/{n_total}", ha="center",
              fontsize=5.8, fontweight="bold", color=colour)

# Significance annotations
# mismatched vs full
ax_a.annotate("mismatched worse\np = 0.0013",
              xy=(2, rates[2]), xytext=(2.4, rates[2] + 8),
              fontsize=5.2, color="black",
              arrowprops=dict(arrowstyle="->", color=LIGHT_GREY, lw=0.6),
              ha="left", va="bottom")
# random vs full
ax_a.annotate("random worse\np = 0.0212",
              xy=(1, rates[1]), xytext=(1.4, rates[1] + 5),
              fontsize=5.2, color="black",
              arrowprops=dict(arrowstyle="->", color=LIGHT_GREY, lw=0.6),
              ha="left", va="bottom")

ax_a.set_xticks(x_pos)
ax_a.set_xticklabels(variants, fontsize=6.5)
ax_a.set_ylabel("Error rate (%)")
ax_a.set_ylim(0, 110)
ax_a.yaxis.set_major_locator(mticker.MultipleLocator(25))
ax_a.text(-0.22, 1.02, "a", transform=ax_a.transAxes,
          fontsize=9, fontweight="bold", va="bottom", ha="left")
ax_a.set_title("Mechanism-matched intervention is required", fontsize=8, pad=4)

# ═══════════════════════════════════════════════════════════════════
# PANEL B — Cross-architecture transfer: horizontal dot/forest plot
# ═══════════════════════════════════════════════════════════════════
models = [
    "Qwen2.5-1.5B",
    "Qwen2.5-3B",
    "LLaMA-3.2-1B\nInstruct",
    "LLaMA-3.2-1B\nbase",
]
reductions = [70.3, 57.2, 22.1, 0.0]
ci_lo = [61.1, None, 12.8, None]
ci_hi = [78.8, None, 32.1, None]

n_models = len(models)
y_pos = np.arange(n_models)[::-1]

for i, (yp, red, lo, hi, model) in enumerate(
        zip(y_pos, reductions, ci_lo, ci_hi, models)):
    # Base model: hollow marker (execution boundary)
    is_base = "base" in models[n_models - 1 - i]
    marker = "o" if not is_base else "s"
    face = "white" if is_base else TRACE_BLUE
    edge = TRACE_BLUE if is_base else TRACE_BLUE
    ms = 5.5

    ax_b.plot(red, yp, marker, color=edge, markerfacecolor=face,
              markersize=ms, markeredgewidth=1.0 if is_base else 0, zorder=5)

    if lo is not None and hi is not None:
        ax_b.plot([lo, hi], [yp, yp], "-", color=CI_BLACK, linewidth=1.2, zorder=4)
        ax_b.plot([lo, lo], [yp - 0.15, yp + 0.15], "-", color=CI_BLACK, linewidth=0.8)
        ax_b.plot([hi, hi], [yp - 0.15, yp + 0.15], "-", color=CI_BLACK, linewidth=0.8)

    # Annotation
    if is_base:
        ax_b.text(red + 1.5, yp - 0.22, "Base: execution\nboundary",
                  fontsize=5.2, va="top", color=RAW_GREY, fontstyle="italic")
    else:
        ann = f"{red:.1f}%"
        if lo is not None and hi is not None:
            ann += f"  [{lo:.1f}, {hi:.1f}]"
        ax_b.text(red + 1.5, yp + 0.2, ann, fontsize=5, va="bottom", color="black")

ax_b.set_yticks(y_pos)
ax_b.set_yticklabels(models[::-1])
ax_b.axvline(x=0, color=LIGHT_GREY, linestyle="--", linewidth=0.7, zorder=0)
ax_b.set_xlabel("Pooled error reduction (%)")
ax_b.set_xlim(-8, 85)
ax_b.xaxis.set_major_locator(mticker.MultipleLocator(20))
ax_b.text(-0.22, 1.02, "b", transform=ax_b.transAxes,
          fontsize=9, fontweight="bold", va="bottom", ha="left")
ax_b.set_title("Cross-architecture transfer", fontsize=8, pad=4)

# ── Save ──────────────────────────────────────────────────────────
for fmt in ("pdf", "svg", "png"):
    path = OUTPUT_DIR / f"fig3_ablation_cross_architecture.{fmt}"
    kwargs = dict(facecolor="white", edgecolor="none")
    if fmt == "png":
        kwargs["dpi"] = 300
    fig.savefig(str(path), **kwargs)
    print(f"  Saved {path}")

plt.close(fig)
print("Figure 3 done.")
