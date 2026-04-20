import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# ---------------------------------------------------------------------
# Replace these dictionaries with your final results, or point them to
# JSON files in `main()`.
# ---------------------------------------------------------------------
hitrate_data = {
    "RFdiffusion": {"template": [], "generated": []},
    "PepPrCLIP": {"template": [], "generated": []},
    "PepCCD": {"template": [], "generated": []},
    "UR-PepCCD": {"template": [], "generated": []},
}

aa_freq_data = {
    "Template": {},
    "PepCCD": {},
    "UR-PepCCD": {},
    # "RFdiffusion": {},
    # "PepPrCLIP": {},
}

gacd_scores = {
    # "PepCCD": 0.1754,
    # "UR-PepCCD": 0.1600,
}

energy_data = {
    "MMGBSA": {
        # "RFdiffusion": [],
        # "PepPrCLIP": [],
        # "PepCCD": [],
        # "UR-PepCCD": [],
    },
    "MMPBSA": {
        # "RFdiffusion": [],
        # "PepPrCLIP": [],
        # "PepCCD": [],
        # "UR-PepCCD": [],
    },
}


AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
METHOD_ORDER = ["RFdiffusion", "PepPrCLIP", "PepCCD", "UR-PepCCD"]
COLORS = {
    "RFdiffusion": "#5F6B7A",
    "PepPrCLIP": "#8B7D6B",
    "PepCCD": "#4F7C82",
    "UR-PepCCD": "#A65E58",
    "Template": "#444444",
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
        "legend.fontsize": 8,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_figure(fig, output_stem):
    fig.savefig(f"{output_stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_stem}.png", dpi=300, bbox_inches="tight")


def load_json_if_exists(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def compute_hit_rate(template_values, generated_values):
    template = np.asarray(template_values, dtype=float)
    generated = np.asarray(generated_values, dtype=float)
    if template.size == 0 or generated.size == 0:
        return 0.0
    return float(np.mean(generated >= template))


def _global_axis_limits(hitrate_dict):
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


def plot_hit_rate_figure(hitrate_dict, output_stem="figure_hit_rate"):
    set_publication_style()
    fig = plt.figure(figsize=(7.2, 6.4))
    outer = GridSpec(2, 2, figure=fig, wspace=0.30, hspace=0.32)
    x_min, x_max = _global_axis_limits(hitrate_dict)
    panel_labels = ["(A)", "(B)", "(C)", "(D)"]

    for idx, method in enumerate(METHOD_ORDER):
        if method not in hitrate_dict:
            continue
        panel = GridSpecFromSubplotSpec(
            2, 2, subplot_spec=outer[idx],
            width_ratios=[4.0, 1.1], height_ratios=[1.1, 4.0],
            wspace=0.05, hspace=0.05
        )
        ax_top = fig.add_subplot(panel[0, 0])
        ax_main = fig.add_subplot(panel[1, 0])
        ax_right = fig.add_subplot(panel[1, 1])

        template = np.asarray(hitrate_dict[method]["template"], dtype=float)
        generated = np.asarray(hitrate_dict[method]["generated"], dtype=float)
        color = COLORS.get(method, "#4C4C4C")
        hit_rate = compute_hit_rate(template, generated)

        ax_main.scatter(
            template, generated, s=18, alpha=0.75,
            color=color, edgecolors="none"
        )
        ax_main.plot([x_min, x_max], [x_min, x_max],
                     linestyle="--", linewidth=1.0, color="#B22222")
        ax_main.set_xlim(x_min, x_max)
        ax_main.set_ylim(x_min, x_max)
        ax_main.set_xlabel("Template ipTM")
        ax_main.set_ylabel("Generated peptide ipTM(avg)")
        ax_main.set_title(f"{panel_labels[idx]} {method}", loc="left", pad=2)
        ax_main.text(
            0.97, 0.05, f"Hit rate = {hit_rate:.3f}",
            transform=ax_main.transAxes, ha="right", va="bottom", fontsize=8
        )

        bins = np.linspace(x_min, x_max, 16)
        ax_top.hist(template, bins=bins, color=color, alpha=0.55, edgecolor="white", linewidth=0.4)
        ax_right.hist(generated, bins=bins, orientation="horizontal",
                      color=color, alpha=0.55, edgecolor="white", linewidth=0.4)

        ax_top.set_xlim(x_min, x_max)
        ax_right.set_ylim(x_min, x_max)
        ax_top.tick_params(axis="x", labelbottom=False)
        ax_right.tick_params(axis="y", labelleft=False)
        ax_top.tick_params(axis="y", left=False, labelleft=False)
        ax_right.tick_params(axis="x", bottom=False, labelbottom=False)
        for ax in (ax_top, ax_right):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    # Example caption:
    # Figure X: Hit-rate comparison across peptide design methods. Each point
    # represents one target, comparing template ipTM and the average ipTM of
    # generated peptides. Points above the diagonal are considered hits.
    save_figure(fig, output_stem)
    plt.close(fig)


def plot_aa_composition_figure(aa_dict, gacd_dict=None, output_stem="figure_aacomposition"):
    set_publication_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))

    methods = [m for m in ["Template", "RFdiffusion", "PepPrCLIP", "PepCCD", "UR-PepCCD"] if m in aa_dict]
    x = np.arange(len(AA_ORDER))
    width = 0.82 / max(1, len(methods))

    for idx, method in enumerate(methods):
        freq_map = aa_dict.get(method, {})
        values = [freq_map.get(aa, 0.0) for aa in AA_ORDER]
        label = method
        if gacd_dict and method in gacd_dict:
            label = f"{method} (GACD={gacd_dict[method]:.4f})"
        ax.bar(
            x + (idx - (len(methods) - 1) / 2.0) * width,
            values, width=width,
            color=COLORS.get(method, "#666666"),
            edgecolor="black", linewidth=0.3, label=label
        )

    ax.set_xticks(x)
    ax.set_xticklabels(AA_ORDER)
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Amino acid")
    ax.set_title("Global Amino-acid Composition Discrepancy")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.set_xlim(-0.6, len(AA_ORDER) - 0.4)

    # Example caption:
    # Figure Y: Amino-acid frequency distributions of template peptides and
    # generated peptides from different methods. Lower GACD indicates closer
    # agreement with the natural template distribution.
    save_figure(fig, output_stem)
    plt.close(fig)


def plot_binding_energy_figure(energy_dict, output_stem="figure_binding_energy"):
    mmgbsa = energy_dict.get("MMGBSA", {})
    mmpbsa = energy_dict.get("MMPBSA", {})
    if not mmgbsa and not mmpbsa:
        return

    set_publication_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.8))

    methods = [m for m in METHOD_ORDER if m in mmgbsa or m in mmpbsa]
    positions = []
    datasets = []
    box_colors = []
    labels = []

    left_positions = np.arange(len(methods)) + 1
    right_positions = np.arange(len(methods)) + len(methods) + 2

    for pos, method in zip(left_positions, methods):
        if method in mmgbsa and mmgbsa[method]:
            positions.append(pos)
            datasets.append(mmgbsa[method])
            box_colors.append(COLORS.get(method, "#666666"))
            labels.append(method)

    for pos, method in zip(right_positions, methods):
        if method in mmpbsa and mmpbsa[method]:
            positions.append(pos)
            datasets.append(mmpbsa[method])
            box_colors.append(COLORS.get(method, "#666666"))
            labels.append(method)

    box = ax.boxplot(
        datasets, positions=positions, widths=0.65,
        patch_artist=True, showfliers=False,
        medianprops={"color": "black", "linewidth": 1.0},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
        boxprops={"linewidth": 0.8},
    )
    for patch, color in zip(box["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    separator = len(methods) + 1
    ax.axvline(separator, color="black", linestyle="--", linewidth=0.8)
    ax.text(np.mean(left_positions), 1.02, "MM/GBSA", ha="center", va="bottom",
            transform=ax.get_xaxis_transform(), fontsize=9)
    ax.text(np.mean(right_positions), 1.02, "MM/PBSA", ha="center", va="bottom",
            transform=ax.get_xaxis_transform(), fontsize=9)

    tick_positions = list(left_positions) + list(right_positions)
    tick_labels = methods + methods
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=25, ha="right")
    ax.set_ylabel("Binding free energy (kcal/mol)")

    legend_handles = [Patch(facecolor=COLORS[m], edgecolor="black", linewidth=0.3, label=m, alpha=0.75)
                      for m in methods]
    ax.legend(handles=legend_handles, frameon=False, loc="upper right")

    # Example caption:
    # Figure Z: Comparison of binding free energy distributions across methods
    # using MM/GBSA and MM/PBSA. Lower values indicate stronger predicted binding.
    save_figure(fig, output_stem)
    plt.close(fig)


def main():
    out_dir = Path("/workspace/guest/cyh/workspace/PepCCD/evaluation/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional: replace these with your own JSON files later.
    # Example:
    # hitrate_from_file = load_json_if_exists("path/to/hitrate_data.json")
    # if hitrate_from_file is not None:
    #     global hitrate_data
    #     hitrate_data = hitrate_from_file

    plot_hit_rate_figure(hitrate_data, output_stem=str(out_dir / "figure_hit_rate"))
    plot_aa_composition_figure(
        aa_freq_data,
        gacd_dict=gacd_scores,
        output_stem=str(out_dir / "figure_aacomposition"),
    )
    plot_binding_energy_figure(energy_data, output_stem=str(out_dir / "figure_binding_energy"))


if __name__ == "__main__":
    main()
