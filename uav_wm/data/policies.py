"""Collection policies for swm/UAVTurret-v0.

``ExplorationPolicy`` is a *reactive*, stateless, vectorized policy that
generates danger-rich training data. A pure random walk under-produces danger
frames and kills (the smoke dataset was ~0.3% danger / 2% kill), but the
world model + danger head need many "barrel tracks toward the drone" and
"sustained-LOS kill" sequences. So this policy deliberately approaches the
turret when safe, then -- once in danger -- either flees (escape sequences,
the most informative "predicted future danger that was averted") or commits
(sustained-LOS kills).

Everything is derived from the 16-dim ``state`` vector only, which is always
stacked by the vectorized env and encodes drone / goal / turret / danger. This
avoids depending on custom info keys surviving the swm wrapper stack.
"""
import numpy as np
from stable_worldmodel.policy import BasePolicy

from uav_wm.envs.uav_turret import ARENA, DRONE_MAX_SPEED


class ExplorationPolicy(BasePolicy):
    """Reactive turret-interaction explorer for UAVTurretEnv.

    Per step, per env:
      * in danger  -> flee (away from turret) with prob (1-r), else commit.
      * safe       -> approach turret with prob q, else head to the goal.

    Parameters:
      q:    P(approach turret | safe). Higher -> more danger exposure.
      r:    P(commit | in danger). 1-r is the flee probability. Higher -> more
            kills; lower -> more escapes.
      noise: Gaussian action noise std.

    Stateless across resets; danger is read from ``state[:, 13]``.
    """

    def __init__(self, seed=0, q=0.3, r=0.3, noise=0.1, **kwargs):
        super().__init__(**kwargs)
        self.type = "exploration"
        self.seed = seed
        self.q = q
        self.r = r
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def get_action(self, obs, **kwargs):
        # World passes self.infos as the first positional arg. The wrapper
        # stack adds a length-1 time axis, so state is (n, 1, 16) -> (n, 16).
        state = np.asarray(obs["state"])
        if state.ndim == 3 and state.shape[1] == 1:
            state = state[:, 0, :]
        n = state.shape[0]
        # Decode geometry from state (see UAVTurretEnv._state).
        rel_goal = state[:, 4:6] * ARENA             # goal - drone
        rel_tur = state[:, 6:8] * ARENA              # turret - drone
        danger = state[:, 13] > 0.0                  # in FOV + range + LOS
        near = (state[:, 11] > 0.0) & (state[:, 12] > 0.0) & ~danger  # in range + LOS, not yet aimed

        # Default: head to the goal.
        act = np.clip(rel_goal / DRONE_MAX_SPEED, -1.0, 1.0).astype(np.float32)

        # Safe envs (not near, not danger): with prob q, approach the turret.
        safe = ~near & ~danger
        approach = safe & (self.rng.random(n) < self.q)
        if approach.any():
            act[approach] = np.clip(rel_tur[approach] / DRONE_MAX_SPEED,
                                    -1.0, 1.0)

        # Perpendicular-to-bearing direction toward the goal side: used both to
        # evade when "near" (precautionary -- averts danger, yields long
        # goal-reaching detours + near-miss "future danger" frames) and to flee
        # when in danger. Retreating radially stays on the same bearing and
        # never leaves the FOV cone before the shot lands.
        perp = np.stack([-rel_tur[:, 1], rel_tur[:, 0]], axis=1)  # 90 deg
        flip = (perp * rel_goal).sum(axis=1) < 0
        perp = np.where(flip[:, None], -perp, perp)
        perp = np.clip(perp / DRONE_MAX_SPEED, -1.0, 1.0)

        if near.any():
            act[near] = perp[near]

        # In-danger envs: flee with prob (1-r), else commit (suicidal approach).
        flee = danger & (self.rng.random(n) > self.r)
        commit = danger & ~flee
        if flee.any():
            act[flee] = perp[flee]
        if commit.any():
            act[commit] = np.clip(rel_tur[commit] / DRONE_MAX_SPEED, -1.0, 1.0)

        act += self.rng.normal(0.0, self.noise, act.shape).astype(np.float32)
        return np.clip(act, -1.0, 1.0).astype(np.float32)
