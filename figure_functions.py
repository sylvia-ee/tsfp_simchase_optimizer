import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


action_label_map = {
    "very_small": "chocolate [2-9]",
    "small": "small [1-20]",
    "large": "large [20-45]"
}

def get_palette(n, mode="default"):

    if mode == "colorblind":
        cmap = plt.get_cmap("tab10")

    else:
        cmap = plt.get_cmap("viridis")

    colors = [mpl.colors.to_hex(cmap(i / max(n-1, 1))) for i in range(n)]

    return colors


def _draw_thresholds(ax, df, trials, n_vs, lower_only=True):

    for i, t in enumerate(trials):

        rule_row = df[df["trial"] == t].iloc[0]

        win_low = rule_row.get("win_low")
        win_high = rule_row.get("win_high")
        conv_low = rule_row.get("conv_low")
        conv_high = rule_row.get("conv_high")

        start = i * n_vs
        end = start + n_vs

        # draw win line 
        if pd.notna(win_low):
            ax.plot([start, end], [win_low, win_low],
                    linestyle="--", color="black",
                    linewidth=4, zorder=6)

        if not lower_only and pd.notna(win_high):
            ax.plot([start, end], [win_high, win_high],
                    linestyle="--", color="black",
                    linewidth=4, zorder=6)

        # draw convince line
        if pd.notna(conv_low):
            ax.plot([start, end], [conv_low, conv_low],
                    linestyle=":", color="black",
                    linewidth=4, zorder=6)

        if not lower_only and pd.notna(conv_high):
            ax.plot([start, end], [conv_high, conv_high],
                    linestyle=":", color="black",
                    linewidth=4, zorder=6)


def plot_policy_heatmaps(
    optimal_Q,
    lower_only=True,
    highlight_state=None,
    figsize_scale=1.0,
    font_scale=1.0,
    color_mode="default"
):

    figs = {}

    stages = sorted(optimal_Q["round"].unique())

    for r in stages:

        df = optimal_Q[optimal_Q["round"] == r].copy()

        actions = sorted(df["action"].unique())
        action_labels = [action_label_map.get(a, a) for a in actions]

        action_map = {a: i for i, a in enumerate(actions)}
        df["action_id"] = df["action"].map(action_map)

        df = df.sort_values(["trial", "vs_left", "score"])

        scores = sorted(df["score"].unique())
        trials = sorted(df["trial"].unique())
        vs_vals = sorted(df["vs_left"].unique())

        col_order = [(t, vs) for t in trials for vs in vs_vals]

        pivot = df.pivot(
            index="score",
            columns=["trial", "vs_left"],
            values="action_id"
        ).reindex(columns=pd.MultiIndex.from_tuples(col_order))

        Z = pivot.values

        # fill win regions to be black
        win_mask = np.zeros_like(Z, dtype=bool)

        for i, t in enumerate(trials):
            rule_row = df[df["trial"] == t].iloc[0]
            win_low = rule_row.get("win_low")

            if pd.notna(win_low):
                col_start = i * len(vs_vals)
                col_end = col_start + len(vs_vals)

                for j, score in enumerate(scores):
                    if score >= win_low:
                        win_mask[j, col_start:col_end] = True

        fig, ax = plt.subplots(
            figsize=(
                max(15, Z.shape[1]*0.35*figsize_scale),
                max(10, Z.shape[0]*0.12*figsize_scale)
            )
        )

        # colors
        base_colors = get_palette(len(actions), mode=color_mode)

        win_color = "#000000" if color_mode == "colorblind" else "#2b2b2b"

        colors = base_colors + [win_color]

        cmap = ListedColormap(colors)
        norm = BoundaryNorm(np.arange(len(colors)+1)-0.5, cmap.N)

        # generate grid for heatmap
        mesh = ax.pcolormesh(
            np.arange(Z.shape[1] + 1),
            np.arange(Z.shape[0] + 1),
            Z,
            cmap=cmap,
            norm=norm,
            shading="flat",
            edgecolors="black",
            linewidth=0.4
        )

        # slap the win mask on 
        win_overlay = np.full_like(Z, np.nan, dtype=float)
        win_overlay[win_mask] = len(actions)

        ax.pcolormesh(
            np.arange(Z.shape[1] + 1),
            np.arange(Z.shape[0] + 1),
            win_overlay,
            cmap=cmap,
            norm=norm,
            shading="flat",
            edgecolors="none"
        )

        ax.set_title(
            f"Optimal Decision (Stage {r})",
            fontsize=20 * font_scale,
            pad=30
        )
        ax.set_ylabel("Score", fontsize=14 * font_scale)
        ax.set_xlabel("Chocolate Left (nested within round)", fontsize=14 * font_scale)

        x_centers = np.arange(Z.shape[1]) + 0.5
        vs_labels = [vs for (_, vs) in col_order]
        ax.set_xticks(x_centers)
        ax.set_xticklabels(vs_labels, fontsize=10 * font_scale)

        y_centers = np.arange(Z.shape[0]) + 0.5
        ax.set_yticks(y_centers[::10])
        ax.set_yticklabels(scores[::10], fontsize=10 * font_scale)

        n_vs = len(vs_vals)

        for i, t in enumerate(trials):
            start = i * n_vs

            ax.axvline(start, color="black", linewidth=2)

            center = start + n_vs / 2
            ax.text(center, Z.shape[0] + 1,
                    f"R{t}",
                    ha="center",
                    fontsize=10 * font_scale)

        ax.axvline(Z.shape[1], color="black", linewidth=2)

        _draw_thresholds(ax, df, trials, n_vs, lower_only=lower_only)

        # box for highlighting where user is
        if highlight_state is not None:
            r_h, t_h, s_h, vs_h = highlight_state

            if r_h == r:
                try:
                    col_idx = (
                        trials.index(t_h) * len(vs_vals)
                        + vs_vals.index(vs_h)
                    )
                    row_idx = scores.index(s_h)

                    ax.add_patch(plt.Rectangle(
                        (col_idx, row_idx), 1, 1,
                        fill=False,
                        edgecolor="black",
                        linewidth=4,
                        zorder=10
                    ))

                    ax.add_patch(plt.Rectangle(
                        (col_idx + 0.05, row_idx + 0.05),
                        0.9, 0.9,
                        fill=False,
                        edgecolor="white",
                        linewidth=2,
                        zorder=11
                    ))

                except ValueError:
                    pass

        # colorbar
        cbar = plt.colorbar(mesh, ax=ax, pad=0.02)
        cbar.set_ticks(range(len(actions)))
        cbar.set_ticklabels(action_labels)

        # legend
        legend_elements = [
            Line2D([0], [0], color='black', linestyle='--', linewidth=4, label='Win Threshold'),
            Line2D([0], [0], color='black', linestyle=':', linewidth=4, label='Convince Threshold'),
            Patch(facecolor=win_color, label='Win Region')
        ]

        # adjust legend position
        ax.legend(
            handles=legend_elements,
            bbox_to_anchor=(1.15, 1),
            loc="upper left",
            frameon=False
        )

        plt.subplots_adjust(right=0.80)

        figs[r] = fig

    return figs