import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D

# config

BASE_PATH = "data"
st.title("Influencer Island Decision Helper")

action_label_map = {
    "very_small": "chocolate [2-9]",
    "small": "small [1-20]",
    "large": "large [20-45]"
}

dataset_options = [
    d for d in os.listdir(BASE_PATH)
    if os.path.isdir(os.path.join(BASE_PATH, d))
]

dataset = st.selectbox("Dataset", dataset_options)
dataset_path = os.path.join(BASE_PATH, dataset)

optimal_Q = pd.read_csv(
    os.path.join(dataset_path, "tables", "optimal_Q_table.csv")
)

full_Q = pd.read_csv(
    os.path.join(dataset_path, "tables", "full_Q_table.csv")
)

stages = sorted(optimal_Q["round"].unique())
stage = st.selectbox("Stage", stages)

rounds = sorted(
    optimal_Q[optimal_Q["round"] == stage]["trial"].unique()
)
round_num = st.selectbox("Round", rounds)

vs_vals = sorted(optimal_Q["vs_left"].unique())
default_vs = 3 if 3 in vs_vals else vs_vals[0]
vs_left = st.selectbox(
    "Chocolate Left",
    vs_vals,
    index=vs_vals.index(default_vs)
)

score_input = st.number_input("Score", step=1)

valid_scores = optimal_Q[
    (optimal_Q["round"] == stage) &
    (optimal_Q["trial"] == round_num) &
    (optimal_Q["vs_left"] == vs_left)
]["score"].unique()

if len(valid_scores) == 0:
    st.error("No valid states for this configuration")
    st.stop()

if score_input not in valid_scores:
    st.warning("Score is not reachable in this state")
    st.stop()

score = int(score_input)

mask = (
    (optimal_Q["round"] == stage) &
    (optimal_Q["trial"] == round_num) &
    (optimal_Q["score"] == score) &
    (optimal_Q["vs_left"] == vs_left)
)

if not mask.any():
    st.error("Invalid state")
    st.stop()

opt_row = optimal_Q[mask].iloc[0]

df = full_Q[
    (full_Q["round"] == stage) &
    (full_Q["trial"] == round_num) &
    (full_Q["score"] == score) &
    (full_Q["vs_left"] == vs_left)
].copy()

df = df.sort_values("win_probability", ascending=False)

best_action = opt_row["action"]
best_prob = opt_row["win_probability"]

df["regret"] = best_prob - df["win_probability"]
df["action_display"] = df["action"].map(action_label_map)

st.subheader("Recommendation")
st.success(f"Best action: {action_label_map.get(best_action, best_action)}")
st.write(f"Win probability: {best_prob:.4f}")

st.subheader("Action Comparison")
st.bar_chart(df.set_index("action_display")["win_probability"])

df_display = df.copy()
df_display["action"] = df_display["action_display"]
st.dataframe(df_display.drop(columns=["action_display"]))


def plot_policy_heatmap_specific_state(optimal_Q, highlight_state=None):

    TITLE_SIZE = 34
    LABEL_SIZE = 28
    TICK_SIZE = 22
    LEGEND_SIZE = 22
    CBAR_SIZE = 22

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
        rounds = sorted(df["trial"].unique())
        vs_vals = sorted(df["vs_left"].unique())

        col_order = [(t, vs) for t in rounds for vs in vs_vals]

        pivot = df.pivot(
            index="score",
            columns=["trial", "vs_left"],
            values="action_id"
        ).reindex(columns=pd.MultiIndex.from_tuples(col_order))

        Z = pivot.values

        fig, ax = plt.subplots(
            figsize=(max(20, Z.shape[1]*0.45),
                     max(14, Z.shape[0]*0.16))
        )

        cmap = ListedColormap([
            "#0072B2",
            "#E69F00",
            "#009E73",
            "#D55E00",
            "#CC79A7"
        ][:len(actions)])

        norm = BoundaryNorm(np.arange(len(actions)+1)-0.5, cmap.N)

        mesh = ax.pcolormesh(
            np.arange(Z.shape[1] + 1),
            np.arange(Z.shape[0] + 1),
            Z,
            cmap=cmap,
            norm=norm,
            shading="flat",
            edgecolors="black",
            linewidth=0.3
        )

        ax.set_title(f"Optimal Decision (Stage {r})", fontsize=TITLE_SIZE, pad=60)
        ax.set_ylabel("Score", fontsize=LABEL_SIZE, labelpad=15)
        ax.set_xlabel("Chocolate Left (nested within round)", fontsize=LABEL_SIZE, labelpad=15)

        x_centers = np.arange(Z.shape[1]) + 0.5
        vs_labels = [vs for (_, vs) in col_order]
        ax.set_xticks(x_centers)
        ax.set_xticklabels(vs_labels, fontsize=TICK_SIZE)

        y_centers = np.arange(Z.shape[0]) + 0.5
        ax.set_yticks(y_centers[::10])
        ax.set_yticklabels(scores[::10], fontsize=TICK_SIZE)

        ax.tick_params(axis='both', labelsize=TICK_SIZE, pad=10)

        n_vs = len(vs_vals)

        for i, t in enumerate(rounds):
            start = i * n_vs
            ax.axvline(start, linewidth=10, color="white", zorder=5)

            center = start + n_vs / 2
            ax.text(center, Z.shape[0] + 2,
                    f"R{t}",
                    ha="center",
                    va="bottom",
                    fontsize=24)

        ax.axvline(Z.shape[1], linewidth=1.5)

        for i, t in enumerate(rounds):

            rule_row = df[df["trial"] == t].iloc[0]

            win_low = rule_row["win_low"]
            conv_low = rule_row["conv_low"]

            start = i * n_vs
            end = start + n_vs
                
            if pd.notna(win_low):
                ax.plot([start, end], [win_low, win_low],
                        linestyle="--", color="black", linewidth=2)

            if pd.notna(conv_low):
                ax.plot([start, end], [conv_low, conv_low],
                        linestyle=":", color="black", linewidth=2.5)

        if highlight_state is not None:
            r_h, t_h, s_h, vs_h = highlight_state

            if r_h == r:
                try:
                    col_idx = (
                        rounds.index(t_h) * len(vs_vals)
                        + vs_vals.index(vs_h)
                    )
                    row_idx = scores.index(s_h)

                    ax.add_patch(plt.Rectangle(
                        (col_idx, row_idx), 1, 1,
                        fill=False, linewidth=4
                    ))
                except ValueError:
                    pass

        cbar = plt.colorbar(mesh, ax=ax, pad=0.03, fraction=0.05)
        cbar.set_ticks(range(len(actions)))
        cbar.set_ticklabels(action_labels)
        cbar.ax.tick_params(labelsize=CBAR_SIZE)

        legend_elements = [
            Line2D([0], [0], linestyle='--', label='Win Threshold'),
            Line2D([0], [0], linestyle=':', label='Convince Threshold')
        ]

        ax.legend(
            handles=legend_elements,
            bbox_to_anchor=(1.15, 1),
            loc="upper left",
            frameon=False,
            fontsize=LEGEND_SIZE
        )

        plt.subplots_adjust(right=0.75, top=0.88)

        figs[r] = fig

    return figs


figs = plot_policy_heatmap_specific_state(
    optimal_Q,
    highlight_state=(stage, round_num, score, vs_left)
)

st.subheader("Decision Map")
st.pyplot(figs[stage])