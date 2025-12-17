#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pandas as pd
import seaborn as sns

TOOLKIT_ORDER = ["geomstats", "pymanopt", "geoopt", "tf_riemopt", "mctorch"]

TOOLKIT_LABELS = {
    "geomstats": "Geomstats",
    "pymanopt": "Pymanopt",
    "geoopt": "Geoopt",
    "tf_riemopt": "RiemOpt",
    "mctorch": "McTorch",
}

MANIFOLD_LABELS = {
    "euclidean": r"$\mathbb{R}^n$",
    "sphere": r"$\mathbb{S}^{n-1}$",
    "poincare": r"$\mathbb{H}^n$",
    "hyperbolic": r"$\mathbb{H}^n$",
    "so_n": r"$\mathrm{SO}(n)$",
    "spd": r"$\mathcal{S}_{++}^n$",
}

OPERATION_LABELS = {
    "dist": r"$d(x, y)$",
    "exp": r"$\mathrm{Exp}_x(v)$",
    "log": r"$\mathrm{Log}_x(y)$",
}

COLORBLIND_PALETTE = sns.color_palette("colorblind", n_colors=len(TOOLKIT_ORDER))
TOOLKIT_COLORS = {tk: COLORBLIND_PALETTE[i] for i, tk in enumerate(TOOLKIT_ORDER)}

MARKERS = ["o", "s", "^", "D", "v"]
TOOLKIT_MARKERS = {tk: MARKERS[i] for i, tk in enumerate(TOOLKIT_ORDER)}

LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
TOOLKIT_LINESTYLES = {tk: LINESTYLES[i] for i, tk in enumerate(TOOLKIT_ORDER)}


def load_and_combine(input_dir, pattern="*_results.csv"):
    input_path = Path(input_dir)
    csv_files = sorted(input_path.glob(pattern))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files matching '{pattern}' in {input_dir}")

    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  - {f.name}")

    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)
        print(f"    {len(df)} rows from {csv_file.name}")

    combined = pd.concat(dfs, ignore_index=True)
    combined["manifold"] = combined["manifold"].replace({"hyperbolic": "poincare"})

    print(f"\nCombined: {len(combined)} total rows")
    return combined


def create_plot(df_subset, manifold, operation, output_dir):
    fig, ax = plt.subplots(figsize=(6, 4))

    toolkits_present = df_subset["toolkit"].unique()
    toolkits_sorted = [tk for tk in TOOLKIT_ORDER if tk in toolkits_present]

    for toolkit in toolkits_sorted:
        tk_data = df_subset[df_subset["toolkit"] == toolkit].sort_values("dimension")

        ax.errorbar(
            tk_data["dimension"],
            tk_data["mean_time_sec"],
            yerr=tk_data["std_time_sec"],
            label=TOOLKIT_LABELS[toolkit],
            color=TOOLKIT_COLORS[toolkit],
            marker=TOOLKIT_MARKERS[toolkit],
            linestyle=TOOLKIT_LINESTYLES[toolkit],
            linewidth=1.5,
            markersize=6,
            capsize=3,
            capthick=1,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")

    dims = sorted(df_subset["dimension"].unique())
    if max(dims) <= 100:
        ax.set_xticks(dims)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.get_major_formatter().set_scientific(False)

    ax.set_xlabel("Dimension $n$", fontsize=11)
    ax.set_ylabel("Time (seconds)", fontsize=11)

    manifold_label = MANIFOLD_LABELS.get(manifold, manifold)
    operation_label = OPERATION_LABELS.get(operation, operation)
    ax.set_title(f"{operation_label} on {manifold_label}", fontsize=12)

    ax.legend(frameon=True, fancybox=False, edgecolor="gray", fontsize=9)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.tick_params(labelsize=10)

    plt.tight_layout()

    base_name = f"{manifold}_{operation}"
    fig.savefig(output_dir / f"{base_name}.pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)

    return base_name


def create_legend_figure(output_dir):
    fig, ax = plt.subplots(figsize=(6, 0.8))
    ax.axis("off")

    handles = []
    labels = []
    for toolkit in TOOLKIT_ORDER:
        line = plt.Line2D(
            [0], [0],
            color=TOOLKIT_COLORS[toolkit],
            marker=TOOLKIT_MARKERS[toolkit],
            linestyle=TOOLKIT_LINESTYLES[toolkit],
            linewidth=1.5,
            markersize=6,
        )
        handles.append(line)
        labels.append(TOOLKIT_LABELS[toolkit])

    ax.legend(
        handles, labels,
        loc="center",
        ncol=len(TOOLKIT_ORDER),
        frameon=True,
        fancybox=False,
        edgecolor="gray",
        fontsize=10,
    )

    plt.tight_layout()
    fig.savefig(output_dir / "legend.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(output_dir / "legend.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Combine benchmark CSVs and generate plots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="results",
        help="Directory containing *_results.csv files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Output directory for plots",
    )
    args = parser.parse_args()

    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "cm"

    df = load_and_combine(args.input_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifolds = df["manifold"].unique()
    operations = df["operation"].unique()

    print(f"\nManifolds: {list(manifolds)}")
    print(f"Operations: {list(operations)}")
    print(f"Toolkits: {list(df['toolkit'].unique())}")
    print()

    plots_created = []
    for manifold in manifolds:
        for operation in operations:
            df_subset = df[(df["manifold"] == manifold) & (df["operation"] == operation)]

            if df_subset.empty:
                continue

            plot_name = create_plot(df_subset, manifold, operation, output_dir)
            plots_created.append(plot_name)
            print(f"Created: {plot_name}.pdf, {plot_name}.png")

    print(f"\nGenerated {len(plots_created)} plots in {output_dir}/")


if __name__ == "__main__":
    main()
