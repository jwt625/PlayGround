"""Render the supplied power/mass dataset without modifying its values or labels."""

import argparse
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
    hidden_endpoints = cfg.get("hidden_endpoints", {})
    marker_style = cfg["marker_style"]
    for point in points:
        category = point["category"]
        color = category_colors[category]
        marker = cfg["category_markers"][category]
        visible = [
            endpoint for endpoint in ("idle", "peak")
            if endpoint not in hidden_endpoints.get(category, [])
        ]
        linestyle = (
            cfg["estimated_connector_linestyle"]
            if point["estimated"] else cfg["sourced_connector_linestyle"]
        )
        if len(visible) == 2:
            ax.plot(
                [point["mass_kg"], point["mass_kg"]],
                [point["idle_W"], point["peak_W"]],
                linewidth=cfg["connector_linewidth"], linestyle=linestyle, color=color,
            )
        for endpoint in visible:
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

    for guide in data.get("reference_guides", []):
        specific_power = guide["specific_power_W_per_kg"]
        ax.plot(curve, specific_power * curve, **guide["line_style"])
        label = guide["label_style"]
        ax.text(label["x"], specific_power * label["x"] * label["y_factor"],
                guide["label"], **label["text_kwargs"])

    arrow_style = cfg.get("offscale_arrow_style")
    for arrow in data.get("offscale_arrows", []):
        ax.annotate(
            "", xy=(arrow["mass_kg"], arrow_style["y_head"]),
            xytext=(arrow["mass_kg"], arrow_style["y_tail"]),
            arrowprops={
                "arrowstyle": arrow_style["arrowstyle"],
                "color": arrow_style["color"],
                "linewidth": arrow_style["linewidth"],
                "mutation_scale": arrow_style["mutation_scale"],
            },
            zorder=arrow_style["label_zorder"],
        )
        label = arrow["label_style"]
        ax.text(
            label["x"], label["y"], arrow["label"],
            fontsize=arrow_style["fontsize"], color=arrow_style["color"],
            ha=label["ha"], va=label["va"], zorder=arrow_style["label_zorder"],
            bbox=arrow_style["bbox"],
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
            if endpoint in hidden_endpoints.get(category, []):
                continue
            endpoint_label = cfg.get("category_endpoint_labels", {}).get(category, {}).get(endpoint, endpoint)
            legend.append(Line2D(
                [0], [0], marker=cfg["category_markers"][category], color="none",
                markeredgecolor=color,
                markerfacecolor="none" if cfg[f"{endpoint}_marker"] == "open" else color,
                markersize=legend_style["markersize"],
                markeredgewidth=legend_style["markeredgewidths"][endpoint],
                label=f"{pretty} {endpoint_label}",
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
    for note in cfg.get("figure_notes", []):
        fig.text(**note)
    fig.tight_layout(rect=cfg.get("layout_rect", [0, 0, 1, 1]))
    return fig, ax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "power_mass_data.json")
    parser.add_argument("--output", type=Path, help="PNG path; default follows the dataset's version")
    args = parser.parse_args()
    with args.data.open(encoding="utf-8") as source:
        data = json.load(source)
    # Use a consistent base style even when a machine has a custom matplotlibrc.
    with plt.rc_context(matplotlib.rcParamsDefault):
        fig, _ = render(data)
        output = args.output or args.data.with_name(args.data.stem.replace("_data", "_plot") + ".png")
        fig.savefig(output, dpi=data["plot_config"]["dpi"],
                    bbox_inches=data["plot_config"]["save_bbox_inches"])
        plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
