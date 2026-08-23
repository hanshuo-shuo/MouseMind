import numpy as np


# ── BotEvade reward ──────────────────────────────────────────────────────────

def custom_reward(obs):
    def _get_from_nonstack(field):
        if hasattr(obs, field):
            return getattr(obs, field)
        if isinstance(obs, np.ndarray):
            stack_fields = [
                "prey_x",
                "prey_y",
                "prey_direction",
                "predator_visible",
                "predator_x",
                "predator_y",
                "predator_direction",
                "time_prey_seen_predator",
            ]
            obs_fields = [
                "prey_x",
                "prey_y",
                "prey_direction",
                "predator_visible",
                "predator_x",
                "predator_y",
                "predator_direction",
                "time_prey_seen_predator",
                "puffed",
                "puff_cooled_down",
                "finished",
                "peeking",
                "prey_goal_distance"
            ]
            nonstack_fields = [f for f in obs_fields if f not in stack_fields]
            nonstack = obs[-len(nonstack_fields):]
            return nonstack[nonstack_fields.index(field)]

    reward = 0.0
    if _get_from_nonstack("puffed") > 0:
        reward = -1.0
    if _get_from_nonstack("prey_goal_distance") < 0.1:
        reward = 1.0
    return reward


# ── Oasis reward ─────────────────────────────────────────────────────────────
#
# OasisObservation non-stacked fields (last 7 elements of the stacked obs):
#   obs[-7]  puffed
#   obs[-6]  puff_cooled_down
#   obs[-5]  finished
#   obs[-4]  prey_goal_distance
#   obs[-3]  goal_x
#   obs[-2]  goal_y
#   obs[-1]  goals_remaining
#
# Reward design:
#   +1.0   each time a goal in the sequence is completed (goals_remaining drops)
#   -1.0   each puff
#   -0.05 * prey_goal_distance   dense shaping toward current goal
#   +3.0   episode completion bonus (all goals visited + returned to start)

def oasis_reward():
    """
    Factory that returns a stateful reward function for OasisEnv.
    Call once per environment: reward_fn = oasis_reward()
    """
    _prev_goals_remaining = [None]

    def _reward(obs):
        puffed          = obs[-7]
        finished        = obs[-5]
        prey_goal_dist  = obs[-4]
        goals_remaining = obs[-1]

        reward = 0.0

        # Per-goal bonus: fires when goals_remaining decreases
        if _prev_goals_remaining[0] is not None:
            if goals_remaining < _prev_goals_remaining[0]:
                reward += 1.0
        _prev_goals_remaining[0] = goals_remaining

        # Puff penalty
        if puffed > 0:
            reward -= 1.0

        # Dense distance shaping (encourages moving toward active goal)
        reward -= 0.05 * float(prey_goal_dist)

        # Completion bonus: prey returned to start after visiting all goals
        if finished > 0:
            reward += 3.0

        return reward

    return _reward
