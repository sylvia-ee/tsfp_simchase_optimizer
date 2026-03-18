import streamlit as st
import pandas as pd
import os

from figure_functions import plot_policy_heatmaps


st.set_page_config(layout="wide")

st.title("Influencer Island Decision Helper")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

BASE_PATH = "data"


# interface
left_col, right_col = st.columns([3.5, 4.5])


# left panel
with left_col:

    col1, col2 = st.columns(2)

    dataset_options = [
        d for d in os.listdir(BASE_PATH)
        if os.path.isdir(os.path.join(BASE_PATH, d))
    ]

    with col1:
        dataset = st.selectbox("Dataset", dataset_options)

    with col2:
        color_mode = st.selectbox(
            "Color Mode",
            ["default", "colorblind"]
        )

    theme = st.get_option("theme.base")

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

    # FIX: default chocolate = 3 if available
    default_vs = 3 if 3 in vs_vals else vs_vals[0]

    vs_left = st.selectbox(
        "Chocolate Left",
        vs_vals,
        index=vs_vals.index(default_vs)
    )

    score = int(st.number_input("Score", step=1))

    mask = (
        (optimal_Q["round"] == stage) &
        (optimal_Q["trial"] == round_num) &
        (optimal_Q["score"] == score) &
        (optimal_Q["vs_left"] == vs_left)
    )

    if not mask.any():
        st.error("Invalid state")
        st.stop()

    action_label_map = {
        "very_small": "chocolate [2-9]",
        "small": "small [1-20]",
        "large": "large [20-45]"
    }

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

    st.subheader("Recommendation")
    st.success(f"{action_label_map.get(best_action, best_action)}")
    st.write(f"Win probability: {best_prob:.4f}")


# right panel
with right_col:

    figs = plot_policy_heatmaps(
        optimal_Q,
        lower_only=True,
        highlight_state=(stage, round_num, score, vs_left),
        figsize_scale=1.5,
        font_scale=1.2,
        color_mode=color_mode,
        theme=theme
    )

    st.pyplot(figs[stage], use_container_width=True)