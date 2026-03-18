from functools import lru_cache
import pandas as pd

from preprocess_functions import load_game_config


def solve_game(round_map, actions, action_probs, p_convince):

    """ 
    description: solves one round of game 

    inputs: 
    - round_rules (list of dicts): bounds of "convince" and "win" states for each trial in ORDER of rounds
    - actions (dict): increments for each possible action 
    - action_probs (dict): probability for each possible increment for each action 
    - p_convince (float): probability of success if in convince range and "convince" action chosen 

    outputs: 
    - V (dict): value table
    - Q (function): action-value lookup
    """

    rounds = sorted(round_map.keys())
    n_rounds = len(rounds)

    V = {}

    def get_V(r_idx, t, score, vs_left):
        return V.get((r_idx, t, score, vs_left), 0.0)

    action_list = list(actions.keys()) + ["convince"]

    for r_idx in reversed(range(n_rounds)):

        rules = round_map[rounds[r_idx]]
        n_trials = len(rules)

        for t in reversed(range(n_trials + 1)):

            for score in reversed(range(0, 101)):
                for vs_left in range(4):

                    if t == n_trials:
                        if r_idx == n_rounds - 1:
                            V[(r_idx, t, score, vs_left)] = 1.0
                        else:
                            V[(r_idx, t, score, vs_left)] = get_V(r_idx + 1, 0, 0, 3)
                        continue

                    best = 0

                    for action in action_list:

                        if action == "very_small" and vs_left == 0:
                            continue

                        rule = rules[t]
                        win_low, win_high = rule["win"]
                        conv_range = rule["conv"]

                        if action == "convince":
                            if conv_range and conv_range[0] <= score <= conv_range[1]:
                                conv_low, conv_high = conv_range

                                if t < n_trials - 1:
                                    success_val = get_V(r_idx, t + 1, 0, vs_left)
                                else:
                                    if r_idx == n_rounds - 1:
                                        success_val = 1.0
                                    else:
                                        success_val = get_V(r_idx + 1, 0, 0, 3)

                                failure_val = get_V(r_idx, 0, 0, 3)

                                p_success = (score - conv_low + 1) / (conv_high - conv_low + 1)
                                p_success = max(0, min(1, p_success))

                                val = (
                                    p_success * success_val
                                    + (1 - p_success) * failure_val
                                )
                            else:
                                val = 0

                            best = max(best, val)
                            continue

                        increments = actions[action]
                        probs = action_probs[action]

                        vs_next = vs_left - 1 if action == "very_small" else vs_left

                        expected = 0

                        for inc, p in zip(increments, probs):

                            new_score = score + inc

                            if action == "very_small":
                                new_score = min(new_score, win_high)

                            if new_score > win_high:
                                val = get_V(r_idx, 0, 0, 3)

                            elif win_low <= new_score <= win_high:
                                if t < n_trials - 1:
                                    val = get_V(r_idx, t + 1, 0, vs_next)
                                else:
                                    if r_idx == n_rounds - 1:
                                        val = 1.0
                                    else:
                                        val = get_V(r_idx + 1, 0, 0, 3)

                            else:
                                val = get_V(r_idx, t, new_score, vs_next)

                            expected += p * val

                        best = max(best, expected)

                    V[(r_idx, t, score, vs_left)] = best

    def Q(r_idx, t, score, vs_left, action):

        rules = round_map[rounds[r_idx]]
        rule = rules[t]

        win_low, win_high = rule["win"]
        conv_range = rule["conv"]

        if action == "convince":
            if conv_range and conv_range[0] <= score <= conv_range[1]:
                conv_low, conv_high = conv_range

                if t < len(rules) - 1:
                    success_val = V[(r_idx, t + 1, 0, vs_left)]
                else:
                    if r_idx == n_rounds - 1:
                        success_val = 1.0
                    else:
                        success_val = V[(r_idx + 1, 0, 0, 3)]

                failure_val = V[(r_idx, 0, 0, 3)]

                p_success = (score - conv_low + 1) / (conv_high - conv_low + 1)
                p_success = max(0, min(1, p_success))

                return (
                    p_success * success_val
                    + (1 - p_success) * failure_val
                )
            return 0

        increments = actions[action]
        probs = action_probs[action]

        vs_next = vs_left - 1 if action == "very_small" else vs_left

        expected = 0

        for inc, p in zip(increments, probs):

            new_score = score + inc

            if action == "very_small":
                new_score = min(new_score, win_high)

            if new_score > win_high:
                val = V[(r_idx, 0, 0, 3)]

            elif win_low <= new_score <= win_high:
                if t < len(rules) - 1:
                    val = V[(r_idx, t + 1, 0, vs_next)]
                else:
                    if r_idx == n_rounds - 1:
                        val = 1.0
                    else:
                        val = V[(r_idx + 1, 0, 0, 3)]

            else:
                val = V[(r_idx, t, new_score, vs_next)]

            expected += p * val

        return expected

    return V, Q


def compute_policy(round_map, actions, action_probs, p_convince):
    """
    description: computes optimal user decision for a trial 
    inputs: see solve_round function
    outputs: df of every state and every action with win probabilities 
    """

    V, Q = solve_game(round_map, actions, action_probs, p_convince)

    rows = []

    rounds = sorted(round_map.keys())
    action_list = list(actions.keys()) + ["convince"]

    for r_idx, r in enumerate(rounds):

        rules = round_map[r]
        n_trials = len(rules)

        for t in range(n_trials):
            for score in range(0, 101):
                for vs_left in range(4):

                    for action in action_list:

                        if action == "very_small" and vs_left == 0:
                            continue

                        val = Q(r_idx, t, score, vs_left, action)

                        rows.append({
                            "round": r,
                            "trial": t + 1,
                            "score": score,
                            "vs_left": vs_left,
                            "action": action,
                            "win_probability": val
                        })

    full_Q_table = pd.DataFrame(rows)

    idx = full_Q_table.groupby(
        ["round", "trial", "score", "vs_left"]
    )["win_probability"].idxmax()

    optimal_Q_table = full_Q_table.loc[idx].reset_index(drop=True)

    return full_Q_table, optimal_Q_table


def build_decision_tbl(folder_path):

    actions, action_probs, round_map, vs_limit, p_convince = load_game_config(folder_path)

    full_Q_all, optimal_Q_all = compute_policy(
        round_map,
        actions,
        action_probs,
        p_convince
    )

    for r, rules in round_map.items():

        for t, rule in enumerate(rules, start=1):

            win_low, win_high = rule["win"]

            full_Q_all.loc[
                (full_Q_all["round"] == r) & (full_Q_all["trial"] == t),
                "win_low"
            ] = win_low

            full_Q_all.loc[
                (full_Q_all["round"] == r) & (full_Q_all["trial"] == t),
                "win_high"
            ] = win_high

            optimal_Q_all.loc[
                (optimal_Q_all["round"] == r) & (optimal_Q_all["trial"] == t),
                "win_low"
            ] = win_low

            optimal_Q_all.loc[
                (optimal_Q_all["round"] == r) & (optimal_Q_all["trial"] == t),
                "win_high"
            ] = win_high

            if rule["conv"] is not None:
                conv_low, conv_high = rule["conv"]
            else:
                conv_low, conv_high = None, None

            full_Q_all.loc[
                (full_Q_all["round"] == r) & (full_Q_all["trial"] == t),
                "conv_low"
            ] = conv_low

            full_Q_all.loc[
                (full_Q_all["round"] == r) & (full_Q_all["trial"] == t),
                "conv_high"
            ] = conv_high

            optimal_Q_all.loc[
                (optimal_Q_all["round"] == r) & (optimal_Q_all["trial"] == t),
                "conv_low"
            ] = conv_low

            optimal_Q_all.loc[
                (optimal_Q_all["round"] == r) & (optimal_Q_all["trial"] == t),
                "conv_high"
            ] = conv_high

    return full_Q_all, optimal_Q_all