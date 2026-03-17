from functools import lru_cache
import pandas as pd

def solve_round(round_rules, actions, action_probs, p_convince):
    n_trials = len(round_rules)

    @lru_cache(None)
    def V(t, score, vs_left):
        if t == n_trials:
            return 1.0

        best = 0
        for action in actions:
            if action == "very_small" and vs_left == 0:
                continue
            best = max(best, Q(t, score, vs_left, action))

        return min(1.0, max(0.0, best))


    def Q(t, score, vs_left, action):

        rule = round_rules[t]
        win_low, win_high = rule["win"]
        conv_range = rule["conv"]

        if action == "convince":
            if conv_range and conv_range[0] <= score <= conv_range[1]:
                return p_convince
            return 0

        increments = actions[action]
        probs = action_probs[action]

        vs_next = vs_left - 1 if action == "very_small" else vs_left

        expected = 0

        for inc, p in zip(increments, probs):

            new_score = score + inc

            if new_score > 101:
                val = 0

            elif win_low <= new_score <= win_high:
                val = V(t + 1, new_score, vs_next)

            else:
                val = V(t, new_score, vs_next)

            expected += p * val

        return expected

    return V, Q

def compute_policy(round_rules, actions, action_probs, p_convince):

    V, Q = solve_round(round_rules, actions, action_probs, p_convince)
    n_trials = len(round_rules)

    policy = []

    for t in range(n_trials):
        for score in range(0, 101):
            for vs_left in range(4):

                best_action = None
                best_val = -1

                for action in actions:

                    if action == "very_small" and vs_left == 0:
                        continue

                    val = Q(t, score, vs_left, action)

                    if val > best_val:
                        best_val = val
                        best_action = action

                policy.append({
                    "trial": t + 1,
                    "score": score,
                    "vs_left": vs_left,
                    "best_action": best_action,
                    "win_probability": best_val
                })

    return pd.DataFrame(policy)