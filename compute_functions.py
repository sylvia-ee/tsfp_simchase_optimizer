from functools import lru_cache
import pandas as pd

def solve_round(round_rules, actions, action_probs, p_convince):

    """ 
    description: solves one round (score carries over between trials) of game 

    inputs: 
    - round_rules (list of dicts): bounds of "convince" and "win" states for each trial in ORDER of rounds
        e.g. {"conv": None, "win": (50, 100), 
               "conv": (60, 70), "win": (70, 100)} 
    - actions (dict): increments for each possible action 
        e.g. {"small": [1, 2, 3...20], "large": [20, 21..., 44, 45]} 
    - action_probs (dict): probability for each possible increment for each action 
        e.g. {"small": [1/20,...,1/20], "large": [1/25,...,1/25]} 
    - p_convince (float): probability of success if in convince range and "convince" action chosen 
        e.g. 0.5 # 50% chance that convince succeeds

    outputs: 
    - V (function): value function caching max. win probability for a given state (trial, score, vs_left) assuming optimal action
    - Q (function): action-value function caching best possible outcome for a given state over all possible actions
    """

    n_trials = len(round_rules) 

    @lru_cache(None)
    def V(t, score, vs_left):
        
        """
        description: caches the maximum win probability for a given state (trial, score, vs_left) assuming optimal action
        inputs:
        - t (int): current trial
        - score (int): current score
        - vs_left (int): "very small" actions left
        outputs:
        - max_prob (float): maximum win probability for the given state
    
        """
        if t == n_trials:
            return 1.0

        best = 0
        for action in actions:
            if action == "very_small" and vs_left == 0:
                continue
            best = max(best, Q(t, score, vs_left, action))
        
        max_prob = min(1.0, max(0.0, best))

        return max_prob


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