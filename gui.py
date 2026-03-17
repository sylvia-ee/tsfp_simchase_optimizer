import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D


# config

BASE_PATH = "data"
st.title("Decision Helper")

# select precomputed data 

dataset_options = [
    d for d in os.listdir(BASE_PATH)
    if os.path.isdir(os.path.join(BASE_PATH, d))
]

dataset = st.selectbox("Dataset", dataset_options)
dataset_path = os.path.join(BASE_PATH, dataset)

# load tables from precomputed data # TODO: make pathing flexible

optimal_Q = pd.read_csv(
    os.path.join(dataset_path, "tables", "optimal_Q_table.csv")
)

full_Q = pd.read_csv(
    os.path.join(dataset_path, "tables", "full_Q_table.csv")
)

# input

rounds = sorted(optimal_Q["round"].unique())
round_num = st.selectbox("Round", rounds)

trials = sorted(
    optimal_Q[optimal_Q["round"] == round_num]["trial"].unique()
)
trial = st.selectbox("Trial", trials)

vs_vals = sorted(optimal_Q["vs_left"].unique())
vs_left = st.selectbox("Very Small Left", vs_vals)

# score inputs
score_input = st.number_input("Score", step=1)

valid_scores = optimal_Q[
    (optimal_Q["round"] == round_num) &
    (optimal_Q["trial"] == trial) &
    (optimal_Q["vs_left"] == vs_left)
]["score"].unique()

if len(valid_scores) == 0:
    st.error("No valid states for this configuration")
    st.stop()

if score_input not in valid_scores:
    st.warning("Score is not reachable in this state")
    st.stop()

score = int(score_input)

# lookup info

mask = (
    (optimal_Q["round"] == round_num) &
    (optimal_Q["trial"] == trial) &
    (optimal_Q["score"] == score) &
    (optimal_Q["vs_left"] == vs_left)
)

if not mask.any():
    st.error("Invalid state")
    st.stop()

opt_row = optimal_Q[mask].iloc[0]

df = full_Q[
    (full_Q["round"] == round_num) &
    (full_Q["trial"] == trial) &
    (full_Q["score"] == score) &
    (full_Q["vs_left"] == vs_left)
].copy()

df = df.sort_values("win_probability", ascending=False)

best_action = opt_row["action"]
best_prob = opt_row["win_probability"]

df["regret"] = best_prob - df["win_probability"]

# output

st.subheader("Recommendation")

st.success(f"Best action: {best_action}")
st.write(f"Win probability: {best_prob:.4f}")

st.subheader("Action Comparison")

st.bar_chart(df.set_index("action")["win_probability"])
st.dataframe(df)

# dynamic plot...literally the same but with 5 lines added
# need to make this less dumb 

def plot_policy_heatmap_specific_state(optimal_Q, highlight_state=None):

    figs = {}

    rounds = sorted(optimal_Q["round"].unique())

    for r in rounds:

        df = optimal_Q[optimal_Q["round"] == r].copy()

        actions = sorted(optimal_Q["action"].unique())
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

        fig, ax = plt.subplots(
            figsize=(max(12, Z.shape[1]*0.35),
                     max(6, Z.shape[0]*0.12))
        )

        cmap = ListedColormap(plt.cm.tab10.colors[:len(actions)])
        norm = BoundaryNorm(np.arange(len(actions)+1)-0.5, cmap.N)

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

        # ax
        ax.set_title(f"Optimal Policy (Round {r})", pad=40)
        ax.set_ylabel("Score")
        ax.set_xlabel("vs_left (nested within trial)")

        x_centers = np.arange(Z.shape[1]) + 0.5
        vs_labels = [vs for (_, vs) in col_order]
        ax.set_xticks(x_centers)
        ax.set_xticklabels(vs_labels)

        y_centers = np.arange(Z.shape[0]) + 0.5
        ax.set_yticks(y_centers[::10])
        ax.set_yticklabels(scores[::10])

        # t groups
        n_vs = len(vs_vals)

        for i, t in enumerate(trials):
            start = i * n_vs
            ax.axvline(start, color="black", linewidth=1.5)

            center = start + n_vs / 2
            ax.text(center, Z.shape[0] + 1.5, f"T{t}",
                    ha="center", va="bottom", fontsize=10)

        ax.axvline(Z.shape[1], color="black", linewidth=1.5)

        # range
        for i, t in enumerate(trials):

            rule_row = df[df["trial"] == t].iloc[0]

            win_low = rule_row["win_low"]
            win_high = rule_row["win_high"]
            conv_low = rule_row["conv_low"]
            conv_high = rule_row["conv_high"]

            start = i * n_vs
            end = start + n_vs

            if pd.notna(win_low):
                ax.plot([start, end], [win_low, win_low],
                        linestyle="--", color="black", linewidth=1.2)
                ax.plot([start, end], [win_high, win_high],
                        linestyle="--", color="black", linewidth=1.2)

            if pd.notna(conv_low):
                ax.plot([start, end], [conv_low, conv_low],
                        linestyle=":", color="black", linewidth=1.2)
                ax.plot([start, end], [conv_high, conv_high],
                        linestyle=":", color="black", linewidth=1.2)

        # highlight new block
        if highlight_state is not None:
            r_h, t_h, s_h, vs_h = highlight_state

            if r_h == r:
                try:
                    col_idx = (
                        trials.index(t_h) * len(vs_vals)
                        + vs_vals.index(vs_h)
                    )
                    row_idx = scores.index(s_h)

                    ax.add_patch(
                        plt.Rectangle(
                            (col_idx, row_idx),
                            1, 1,
                            fill=False,
                            edgecolor="red",
                            linewidth=3
                        )
                    )
                except ValueError:
                    pass

        # cb
        cbar = plt.colorbar(mesh, ax=ax, pad=0.02)
        cbar.set_ticks(range(len(actions)))
        cbar.set_ticklabels(actions)
        cbar.set_label("Action")

        # legend
        legend_elements = [
            Line2D([0], [0], color='black', linestyle='--', label='Win Range'),
            Line2D([0], [0], color='black', linestyle=':', label='Convince Range')
        ]

        ax.legend(
            handles=legend_elements,
            bbox_to_anchor=(1.18, 1),
            loc="upper left",
            frameon=False
        )

        plt.subplots_adjust(right=0.78, top=0.85)

        figs[r] = fig

    return figs

# display range

figs = plot_policy_heatmap_specific_state(
    optimal_Q,
    highlight_state=(round_num, trial, score, vs_left)
)

st.subheader("Decision Map")
st.pyplot(figs[round_num])
