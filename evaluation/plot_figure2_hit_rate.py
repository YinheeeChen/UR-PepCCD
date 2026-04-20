import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


# Replace this dictionary with your real per-target data.
# Each method needs paired target-level values:
#   - template: template ipTM for each target
#   - generated: generated peptide ipTM(avg) for each target
hitrate_data = {
    "RFdiffusion": {"template": [], "generated": []},
    "PepPrCLIP": {"template": [], "generated": []},
    "PepCCD": {"template": [], "generated": []},
    "PepGUIDE": {"template": [], "generated": []},
}


METHOD_ORDER = ["RFdiffusion", "PepPrCLIP", "PepCCD", "PepGUIDE"]
COLORS = {
    "RFdiffusion": "#5F6B7A",
    "PepPrCLIP": "#8B7D6B",
    "PepCCD": "#4F7C82",
    "PepGUIDE": "#A65E58",
}


def set_publication_style():
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_figure(fig, output_stem):
    fig.savefig(f"{output_stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_stem}.png", dpi=300, bbox_inches="tight")


def compute_hit_rate(template_values, generated_values):
    template = np.asarray(template_values, dtype=float)
    generated = np.asarray(generated_values, dtype=float)
    if template.size == 0 or generated.size == 0:
        return 0.0
    return float(np.mean(generated >= template))


def global_axis_limits(hitrate_dict):
    values = []
    for method in METHOD_ORDER:
        if method not in hitrate_dict:
            continue
        values.extend(hitrate_dict[method].get("template", []))
        values.extend(hitrate_dict[method].get("generated", []))
    if not values:
        return 0.0, 1.0
    low = max(0.0, float(min(values)) - 0.02)
    high = min(1.0, float(max(values)) + 0.02)
    if high <= low:
        high = low + 0.1
    return low, high


def plot_hit_rate_figure(hitrate_dict, output_stem):
    set_publication_style()
    fig = plt.figure(figsize=(7.2, 6.4))
    outer = GridSpec(2, 2, figure=fig, wspace=0.30, hspace=0.32)
    panel_labels = ["(A)", "(B)", "(C)", "(D)"]
    x_min, x_max = global_axis_limits(hitrate_dict)

    for idx, method in enumerate(METHOD_ORDER):
        if method not in hitrate_dict:
            continue

        panel = GridSpecFromSubplotSpec(
            2, 2, subplot_spec=outer[idx],
            width_ratios=[4.0, 1.1],
            height_ratios=[1.1, 4.0],
            wspace=0.05,
            hspace=0.05,
        )
        ax_top = fig.add_subplot(panel[0, 0])
        ax_main = fig.add_subplot(panel[1, 0])
        ax_right = fig.add_subplot(panel[1, 1])

        template = np.asarray(hitrate_dict[method]["template"], dtype=float)
        generated = np.asarray(hitrate_dict[method]["generated"], dtype=float)
        color = COLORS.get(method, "#4C4C4C")
        hit_rate = compute_hit_rate(template, generated)

        ax_main.scatter(
            template,
            generated,
            s=18,
            alpha=0.78,
            color=color,
            edgecolors="none",
        )
        ax_main.plot(
            [x_min, x_max],
            [x_min, x_max],
            linestyle="--",
            linewidth=1.0,
            color="#B22222",
        )
        ax_main.set_xlim(x_min, x_max)
        ax_main.set_ylim(x_min, x_max)
        ax_main.set_xlabel("Template ipTM")
        ax_main.set_ylabel("Generated peptide ipTM(avg)")
        ax_main.set_title(f"{panel_labels[idx]} {method}", loc="left", pad=2)
        ax_main.text(
            0.97,
            0.05,
            f"Hit rate = {hit_rate:.3f}",
            transform=ax_main.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
        )

        bins = np.linspace(x_min, x_max, 16)
        ax_top.hist(
            template,
            bins=bins,
            color=color,
            alpha=0.55,
            edgecolor="white",
            linewidth=0.4,
        )
        ax_right.hist(
            generated,
            bins=bins,
            orientation="horizontal",
            color=color,
            alpha=0.55,
            edgecolor="white",
            linewidth=0.4,
        )

        ax_top.set_xlim(x_min, x_max)
        ax_right.set_ylim(x_min, x_max)
        ax_top.tick_params(axis="x", labelbottom=False)
        ax_right.tick_params(axis="y", labelleft=False)
        ax_top.tick_params(axis="y", left=False, labelleft=False)
        ax_right.tick_params(axis="x", bottom=False, labelbottom=False)
        for ax in (ax_top, ax_right):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    save_figure(fig, output_stem)
    plt.close(fig)


def main():
    out_dir = Path("/workspace/guest/cyh/workspace/PepCCD/evaluation/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_hit_rate_figure(hitrate_data, str(out_dir / "figure_hit_rate"))


if __name__ == "__main__":
    main()
