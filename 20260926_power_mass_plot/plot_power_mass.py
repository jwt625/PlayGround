"""Render the supplied power/mass dataset without modifying its values or labels."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


HERE = Path(__file__).resolve().parent


def render(data):
    """Return the figure and axes; all explicit visual choices live in JSON."""
    cfg = data["plot_config"]
    points = data["points"]
    fig, ax = plt.subplots(figsize=cfg["figsize_in"])
    ax.set_xscale(cfg["xscale"])
    ax.set_yscale(cfg["yscale"])

    colors = matplotlib.rcParamsDefault["axes.prop_cycle"].by_key()["color"]
    category_colors = {
        category: colors[index]
        for category, index in cfg["category_color_indices"].items()
    }
    marker_style = cfg["marker_style"]
    for point in points:
        category = point["category"]
        color = category_colors[category]
        marker = cfg["category_markers"][category]
        linestyle = (
            cfg["estimated_connector_linestyle"]
            if point["estimated"] else cfg["sourced_connector_linestyle"]
        )
        ax.plot(
            [point["mass_kg"], point["mass_kg"]],
            [point["idle_W"], point["peak_W"]],
            linewidth=cfg["connector_linewidth"], linestyle=linestyle, color=color,
        )
        for endpoint in ("idle", "peak"):
            ax.scatter(
                point["mass_kg"], point[f"{endpoint}_W"],
                s=marker_style["sizes"][category][endpoint], marker=marker,
                facecolors="none" if cfg[f"{endpoint}_marker"] == "open" else color,
                edgecolors=color, linewidths=marker_style["linewidths"][endpoint],
                zorder=marker_style["zorder"],
            )

    dense = set(cfg["dense_labels"])
    for point in points:
        name = point["name"]
        x, y, alignment = cfg["label_positions"][name]
        font_size = cfg["font_sizes"]["dense_labels" if name in dense else "labels"]
        if name not in dense:
            font_size = min(
                font_size, cfg["category_label_font_caps"].get(point["category"], font_size)
            )
        ax.text(x, y, name, fontsize=font_size, ha=alignment, va=cfg["label_va"],
                zorder=cfg["label_zorder"])

    xmin, xmax = cfg["xlim"]
    curve = np.logspace(np.log10(xmin), np.log10(xmax), cfg["guide_samples"])
    guide_label = cfg["guide_label_style"]
    for specific_power in cfg["specific_power_guides_W_per_kg"]:
        ax.plot(
            curve, specific_power * curve,
            linestyle=cfg["specific_power_guide_linestyle"],
            linewidth=cfg["specific_power_guide_linewidth"],
            color=cfg["specific_power_guide_color"], zorder=cfg["guide_zorder"],
        )
        x = guide_label["x"]
        ax.text(
            x, specific_power * x * guide_label["y_factor"],
            guide_label["format"].format(specific_power=specific_power),
            fontsize=cfg["font_sizes"]["guide_labels"], color=guide_label["color"],
            ha=guide_label["ha"], va=guide_label["va"], zorder=cfg["label_zorder"],
        )

    ax.set_xlabel(cfg["xlabel"], fontsize=cfg["font_sizes"]["axis"])
    ax.set_ylabel(cfg["ylabel"], fontsize=cfg["font_sizes"]["axis"])
    ax.tick_params(axis="both", which="major", labelsize=cfg["font_sizes"]["ticks"])
    for level in ("major", "minor"):
        ax.grid(
            True, which=level, linewidth=cfg[f"grid_{level}_linewidth"],
            alpha=cfg[f"grid_{level}_alpha"],
        )
    ax.set_xlim(*cfg["xlim"])
    ax.set_ylim(*cfg["ylim"])

    legend = []
    legend_style = cfg["legend_style"]
    for category, pretty in cfg["category_legend_labels"].items():
        color = category_colors[category]
        for endpoint in ("idle", "peak"):
            legend.append(Line2D(
                [0], [0], marker=cfg["category_markers"][category], color="none",
                markeredgecolor=color,
                markerfacecolor="none" if cfg[f"{endpoint}_marker"] == "open" else color,
                markersize=legend_style["markersize"],
                markeredgewidth=legend_style["markeredgewidths"][endpoint],
                label=f"{pretty} {endpoint}",
            ))
    legend.append(Line2D(
        [0, 1], [0, 0], color=category_colors[legend_style["estimated_color_category"]],
        linewidth=cfg["connector_linewidth"], linestyle=cfg["estimated_connector_linestyle"],
        label=legend_style["estimated_label"],
    ))
    ax.legend(
        handles=legend, fontsize=cfg["font_sizes"]["legend"],
        loc=legend_style["loc"], frameon=legend_style["frameon"],
    )
    fig.tight_layout()
    return fig, ax


def main():
    with (HERE / "power_mass_data.json").open(encoding="utf-8") as source:
        data = json.load(source)
    # Use a consistent base style even when a machine has a custom matplotlibrc.
    with plt.rc_context(matplotlib.rcParamsDefault):
        fig, _ = render(data)
        output = HERE / "power_mass_plot.png"
        fig.savefig(output, dpi=data["plot_config"]["dpi"],
                    bbox_inches=data["plot_config"]["save_bbox_inches"])
        plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
