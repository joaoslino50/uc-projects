import torch
import torch.nn as nn
import numpy as np
import marioai



class HunterTask(marioai.Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "Hunter"

    def compute_reward(self, current_obs, last_obs):
        # `mario_pos` is the correct attribute on the Observation object.
        # (The Agent stores it as self.mario_floats, but Observation uses mario_pos.)\
        if current_obs is None or last_obs is None:
            return 0
        if current_obs.mario_pos is None or last_obs.mario_pos is None:
            return 0

        reward = 0

        delta_x = current_obs.mario_pos[0] - last_obs.mario_pos[0]
        reward += delta_x

        # ---- Stagnation penalty ----
        # If Mario barely moved this frame he is stuck against a wall or
        # standing still.  Penalise every idle frame heavily so the EA is
        # forced to find a way past obstacles (i.e. jumping) rather than
        # accepting a score of 0 per frame indefinitely.
        STAGNATION_THRESHOLD = 0.5   # pixels; tweak if too sensitive
        STAGNATION_PENALTY   = 1.0   # per idle frame (~24 frames/sec in game)
        if abs(delta_x) < STAGNATION_THRESHOLD:
            reward -= STAGNATION_PENALTY

        # ---- Kill bonus ----
        kills_now    = getattr(current_obs, 'kills_total', 0)
        kills_before = getattr(last_obs,    'kills_total', 0)
        if kills_now > kills_before:
            reward += (kills_now - kills_before) * 1000

        return reward