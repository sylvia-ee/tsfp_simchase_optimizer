import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from compute_functions import solve_round


actions = {
    "small": np.arange(1,21),
    "large": np.arange(20,46),
    "very_small": np.arange(2,10)
}

action_probs = {
    "small": [1/20]*20,
    "large": [1/26]*26,
    "very_small": [1/8]*8
}

p_convince = 0.5


round1 = [
    {"conv":None,"win":(70,100)},
    {"conv":(60,70),"win":(70,100)},
    {"conv":(60,70),"win":(70,100)},
    {"conv":(65,75),"win":(75,100)},
    {"conv":(65,75),"win":(75,100)},
    {"conv":(70,80),"win":(80,100)}
]

round2 = [
    {"conv":(60,70),"win":(70,100)},
    {"conv":(60,70),"win":(70,100)},
    {"conv":(65,75),"win":(75,100)},
    {"conv":(65,75),"win":(75,100)},
    {"conv":(70,80),"win":(80,100)},
    {"conv":(70,85),"win":(85,100)},
    {"conv":(75,85),"win":(85,100)},
    {"conv":(75,85),"win":(85,100)}
]

round3 = [
    {"conv":(65,75),"win":(75,100)},
    {"conv":(65,75),"win":(75,100)},
    {"conv":(70,80),"win":(80,100)},
    {"conv":(70,80),"win":(80,100)},
    {"conv":(75,85),"win":(85,100)},
    {"conv":(75,85),"win":(85,100)},
    {"conv":(80,90),"win":(90,100)},
    {"conv":(80,98),"win":(98,100)},
    {"conv":(90,95),"win":(95,100)},
    {"conv":(90,95),"win":(95,100)}
]

round_map = {1: round1, 2: round2, 3: round3}


st.title("Influence Island Decision Helper")


round_num = st.selectbox("Round", [1,2,3])
trial = st.number_input("Trial", min_value=1, step=1)
score = st.number_input("Score", min_value=0, max_value=101)
vs_left = st.number_input("Very Small Left", min_value=0, max_value=3)


if st.button("Recommend Action"):

    rules = round_map[round_num]

    V, Q = solve_round(rules, actions, action_probs, p_convince)

    action_values = {}

    for action in actions:

        if action == "very_small" and vs_left == 0:
            continue

        val = Q(trial-1, score, vs_left, action)
        action_values[action] = val


    df = pd.DataFrame({
        "action": list(action_values.keys()),
        "win_probability": list(action_values.values())
    })


    best_action = df.loc[df["win_probability"].idxmax(), "action"]
    best_prob = df["win_probability"].max()


    st.success(f"Best action: {best_action}")
    st.write(f"Win probability: {best_prob:.6f}")


    st.subheader("Action Comparison")


    fig, ax = plt.subplots()

    bars = ax.bar(df["action"], df["win_probability"])

    for i, action in enumerate(df["action"]):
        if action == best_action:
            bars[i].set_linewidth(3)

    ax.set_ylabel("Win Probability")
    ax.set_title("Expected Success Probability by Action")

    st.pyplot(fig)


    st.subheader("Numerical Values")
    st.dataframe(df)