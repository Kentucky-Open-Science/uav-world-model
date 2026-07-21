"""3D collection policy for ``Isaac-UAVTurret3D-v0``.

Pure-``torch`` (no swm / pygame / 2D-env deps) so it imports inside the Isaac
Lab container. The collector calls it each step with the env's GPU state
tensor; it returns a ``(N, 4)`` **body-frame** command in ``[-1, 1]``::

    [ vx_fwd, vy_strafe, vz_climb, yaw_rate ]

Body frame (not world) because the WM sees ONLY the first-person POV and has no
notion of "North": a world-frame ``vx=+1`` produces different pixel motion at
every heading (many-to-one), so the latent never converges. Grounding the action
to the camera -- ``vx_fwd=+1`` always means "scene center grows" -- makes the
action->pixel map consistent across headings. Yaw is an EXPLICIT action (the env
no longer auto-yaws to velocity heading), so the policy can strafe while holding
heading or spin in place -- both real ISR maneuvers the WM must see. The env's
``VelocityToThrustAction`` rotates body->world before calling the controller.

**Drunken explorer / max-entropy** (replaces the old A->B path-follower). The
policy does NOT fly to a goal. Instead it holds a *maneuver* for 10-30 steps
(re-rolled on a timer, NOT every step) and mixes:

  * ~70% wander: cruise / strafe-hold-heading / spin-in-place / banked-turn /
    hover-and-look. Teaches general flight dynamics + scene geometry.
  * ~30% provoke: STAND-OFF approach the turret (loiter at ~7 m, the range edge)
    when safe, EVADE radially outward when near/in-danger (exit range -> fire
    resets -> survive), or COMMIT to a suicide dive (danger, with prob r) -> the
    kill+fall tail. Teaches threat geometry + manufactures SURVIVABLE danger
    cycles (approach->standoff->danger->evade->survive->re-provoke): danger
    triggers SHALLOW (7 m) so a full-strength radial-out EVADE exits the 8 m
    range in ~5 steps, inside the env's 7-step FIRE_INTERVAL -> the drone lives
    to re-provoke. The 50 deg/s turret + 0.7 s fire are tuned for this margin.

**Anti-jitter (Evan's "still very jittery" feedback):** the previous policy
re-sampled its approach-vs-goal and flee-vs-commit decision EVERY step (10 Hz);
the two headings can be ~180 deg apart, so the velocity heading flipped
step-to-step and the (then-auto-yawing) drone snapped left-right. The fix is
structural: (1) yaw is explicit, so heading only changes when the policy
commands it; (2) each maneuver is HELD 10-30 steps in a per-env buffer, so the
command is constant within a maneuver and only changes ~every 1-3 s (a
deliberate direction change, not a 10 Hz oscillation). APPROACH/EVADE commands
are computed LIVE from geometry each step, but as a *continuous* function of
continuous state (no discrete re-sampling) -> smooth tracking, no flips.

State layout (21-dim, ``danger`` at index 13, ``drone_yaw`` at index 20):
    [0:3] drone pos (env-local)   [3:6] drone vel   [6:9] goal_rel
    [9:12] turret_rel             [12] dist_t       [13] danger
    [14] in_range   [15] los   [16] aimed   [17] fire_prog   [18] tyaw   [19] tpit
    [20] drone_yaw (world)
``goal_rel`` is kept in state for the (unused-here) goal terminations; the
drunken explorer ignores it.
"""
from __future__ import annotations

import random

import torch

# Maneuver ids (per-env, held in self._mid).
CRUISE, STRAFE, SPIN, BANKED, HOVER, APPROACH, EVADE, COMMIT = 0, 1, 2, 3, 4, 5, 6, 7
_N_WANDER_MODES = 5  # CRUISE..HOVER
# Turret keep-out radius (m): radial-OUT repulsion below this prevents deep entry
# (see moat in get_action). Just inside the 8 m TURRET_RANGE so the drone still
# loiters/provokes in-range (7.5-7.7 m) but can't coast past to ~6 m.
MOAT_R = 7.5
# APPROACH persists through danger while fire_prog < LOITER_FIRE, so the drone
# loiters at the stand-off ACCUMULATING danger frames before evading -- this
# lengthens each encounter (~5 -> ~15 frames) to raise threat density while still
# surviving (EVADE exits in ~5 steps once fire is high, << the 20-step window).
# 0.5 = 10 steps loiter + ~5 escape = 15, fire ~0.75 at exit -> survives.
LOITER_FIRE = 0.5


class ExplorationPolicy3D:
    """Drunken-explorer collector policy -> body-frame 4D action.

    Per-env maneuver state is held in GPU buffers and re-rolled on a timer
    (10-30 steps) or when danger forces an EVADE/COMMIT. ``reset_envs`` clears
    state for envs that just started a new episode (call from the collector on
    each episode end) so a fresh spawn doesn't inherit a stale evade-spin.
    """

    def __init__(self, q: float = 0.3, r: float = 0.3, noise: float = 0.0,
                 seed: int = 0, device: str | torch.device = "cuda") -> None:
        # q = provoke (vs wander) fraction; r = commit (vs evade) fraction in danger.
        self.q = q
        self.r = r
        self.noise = noise
        self.device = torch.device(device)
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)
        self._gen = g
        self._n = 0  # buffers allocated lazily on first get_action

    # -- buffer management ---------------------------------------------------
    def _ensure_buffers(self, n: int) -> None:
        if self._n == n:
            return
        dev = self.device
        self._n = n
        z = torch.zeros(n, device=dev)
        self._mid = torch.zeros(n, dtype=torch.long, device=dev)      # maneuver id
        self._mtimer = torch.zeros(n, dtype=torch.long, device=dev)   # steps left
        # frozen wander params (used by CRUISE..HOVER); APPROACH/EVADE compute live
        self._mfwd = z.clone()
        self._mstrafe = z.clone()
        self._mvz = z.clone()
        self._myaw = z.clone()
        # (EVADE/APPROACH compute their command live each step from geometry --
        #  no frozen per-env params needed.)

    def reset_envs(self, env_ids) -> None:
        """Clear maneuver state for envs starting a new episode (force re-roll)."""
        if self._n == 0:
            return
        if env_ids is None:
            self._mtimer.zero_()
            return
        idx = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if idx.numel() == 0:
            return
        self._mtimer[idx] = 0  # -> re-roll on the next get_action

    # -- main entry point ----------------------------------------------------
    def get_action(self, state: torch.Tensor,
                   building_centers: torch.Tensor | None = None,
                   building_half: torch.Tensor | None = None) -> torch.Tensor:
        """``state`` (N, 21) on the env device -> body-frame action (N, 4) in [-1,1].

        Buildings ((N, K, 3) env-local AABBs) add a potential-field repulsion
        rotated into body frame so the drone threads the streets. They are NOT in
        the 21-dim state (the WM state dim is what the env reports) -- passed
        separately as a per-env channel.
        """
        state = state.to(self.device)
        n = state.shape[0]
        self._ensure_buffers(n)

        yaw = state[:, 20]
        cy = torch.cos(yaw)
        sy = torch.sin(yaw)
        danger = state[:, 13] > 0.0
        los = state[:, 15] > 0.0
        tur_rel = state[:, 9:12]  # world-frame turret - drone
        drone_z = state[:, 2]
        dist_t = state[:, 12]     # drone->turret distance (m)

        # Re-roll maneuvers whose timer expired, OR danger envs not already in a
        # "safe-to-hold" maneuver. EVADE/COMMIT always persist (they ARE the danger
        # response). APPROACH -- a stand-off loiter at the range edge -- PERSISTS
        # through danger while fire_prog < LOITER_FIRE, so the drone loiters at the
        # edge accumulating danger frames before evading (lengthens encounters for
        # threat density). Once fire is high (>= LOITER_FIRE) APPROACH rerolls to
        # EVADE/COMMIT. WANDER drones reroll immediately (they don't hold the edge).
        fire_prog = state[:, 17]
        self._mtimer -= 1
        in_danger_maneuver = ((self._mid == EVADE) | (self._mid == COMMIT)
                              | ((self._mid == APPROACH) & (fire_prog < LOITER_FIRE)))
        reroll = (self._mtimer <= 0) | (danger & ~in_danger_maneuver)
        if reroll.any():
            self._roll_maneuvers(reroll, danger)

        # Body-frame turret direction (used by APPROACH + EVADE + the moat).
        tx, ty = tur_rel[:, 0], tur_rel[:, 1]
        tur_fwd = tx * cy + ty * sy          # body-forward component
        tur_right = -tx * sy + ty * cy       # body-right component
        bearing = torch.atan2(tur_right, tur_fwd)  # body-frame angle to turret
        # (cb, sb) = toward-turret BODY unit vector (forward, right). Radial-OUT
        # (away from turret) is (-cb, -sb). Hoisted: EVADE + the moat both use it.
        cb = torch.cos(bearing)
        sb = torch.sin(bearing)

        # Turret keep-out "moat": a radial-OUT repulsion active below MOAT_R that
        # prevents DEEP entry. Without it, WANDER maneuvers (CRUISE/BANKED) fly
        # straight into the turret at full speed -- the stand-off regulator only
        # acts in APPROACH -- so the drone coasts to ~6 m before danger triggers,
        # and EVADE (which retreats fine at ~1 m/s) can't open 1.8 m inside the
        # 20-step FIRE_INTERVAL -> dies. The moat pushes every non-COMMIT drone
        # back out to >= MOAT_R, so danger triggers SHALLOW (7.5-7.7 m, 0.3-0.5 m
        # exit) and EVADE exits in ~3-5 steps -> survives. COMMIT is exempt (it
        # must dive in past the moat to die). Also arrests APPROACH overshoot.
        moat_strength = ((MOAT_R - dist_t) / 1.0).clamp(0.0, 1.0)  # 0 >= MOAT_R

        # Building repulsion (computed once: APPROACH peels along its lateral
        # sign to regain LOS around an occluding building; the avoidance add-on
        # below applies it to all envs). rep_world points AWAY from buildings.
        if building_centers is not None and building_half is not None:
            rep_world = self._building_repulsion(state, building_centers, building_half)
            rep_fwd = rep_world[:, 0] * cy + rep_world[:, 1] * sy
            rep_right = -rep_world[:, 0] * sy + rep_world[:, 1] * cy
        else:
            rep_fwd = torch.zeros(n, device=self.device)
            rep_right = torch.zeros(n, device=self.device)

        # Start from frozen wander params; overwrite APPROACH/EVADE envs below.
        act = torch.stack([self._mfwd, self._mstrafe, self._mvz, self._myaw], dim=-1).clone()

        is_approach = self._mid == APPROACH
        if is_approach.any():
            # Stand-off provoke (survivable-danger core): loiter at ~7.7 m -- just
            # inside the 8 m range edge -- facing the turret. tanh((d-7.7)/2) is a
            # stand-off regulator: d=10 -> +0.80 (close in), d=7.7 -> 0 (hold), d=5
            # -> -0.83 (back off). Loitering at the EDGE (0.3 m inside range) means
            # danger triggers SHALLOW (7.7 m), so the radial-out EVADE has only
            # ~0.3 m to open to break in_range -- at the controller's sluggish
            # ~0.5 m/s^2 that is ~1.5 s < the 2.0 s (20-step) FIRE_INTERVAL -> the
            # drone survives to re-provoke. (7 m gave a 1 m exit -> ~2.8 s -> died
            # mid-reversal; 7.7 m is the minimum-stable edge loiter.)
            act[is_approach, 0] = torch.tanh((dist_t[is_approach] - 7.7) / 2.0)
            act[is_approach, 1] = 0.0
            act[is_approach, 2] = 0.0
            act[is_approach, 3] = 0.8 * torch.tanh(bearing[is_approach])
            # LOS peel: a stand-off loiter often lands in a building's shadow
            # (in_range + aimed but LOS=0 -> danger never fires). Slide laterally
            # AWAY from the occluding building (repulsion sign -- rep_right points
            # away from buildings; +right fallback when no lateral building near)
            # to peel into a clear-LOS street angle while holding the standoff.
            # Once LOS clears the mask empties -> pure loiter -> aimed -> danger.
            occluded = is_approach & ~los
            if occluded.any():
                peel = torch.where(rep_right.abs() > 0.1,
                                   torch.sign(rep_right),
                                   torch.ones_like(rep_right))
                act[occluded, 1] = 0.7 * peel[occluded]

        is_commit = self._mid == COMMIT
        if is_commit.any():
            # Suicide dive (the kill+fall tail): fly straight at the turret with
            # no stand-off -> deep entry -> sustained danger -> death. Provides
            # the lethal encounters the WM must see (the env's fall recording
            # runs after ground impact).
            act[is_commit, 0] = torch.tanh(tur_fwd[is_commit] / 1.5)
            act[is_commit, 1] = torch.tanh(tur_right[is_commit] / 1.5)
            act[is_commit, 2] = 0.0
            act[is_commit, 3] = 0.9 * torch.tanh(bearing[is_commit])

        is_evade = self._mid == EVADE
        if is_evade.any():
            # Survivable danger (Modification 1): full-strength RADIAL-OUT (fly
            # directly away from the turret -> break in_range -> fire_timer resets
            # -> survive) blended with a close-range LATERAL strafe (at <3 m a
            # 2 m/s lateral move outpaces the 50 deg/s turret -> breaks `aimed`).
            # Both are unit body-frame vectors built from `bearing` (angle to
            # turret): radial-out = bearing+pi, lateral = bearing+pi/2. Full
            # strength (no /3 damping) -> MAX_SPEED. Hold heading (yaw=0,
            # anti-jitter): the turret recedes steadily in POV (no snap). With the
            # moat ensuring a SHALLOW entry (7.5-7.7 m), the 0.3-0.5 m exit is
            # cleared in ~3-5 steps << the 2.0 s FIRE_INTERVAL -> survives.
            # (cb, sb) hoisted above; radial-out body = (-cb, -sb).
            rad_f, rad_r = -cb, -sb                 # body (fwd, right) = away from turret
            lat_f, lat_r = -sb, cb                  # body (fwd, right) = perpendicular
            close = (dist_t < 3.0).float().clamp(0.0, 1.0)
            w_lat = 0.6 * close
            w_rad = 1.0 - w_lat
            ef = w_rad * rad_f + w_lat * lat_f
            er = w_rad * rad_r + w_lat * lat_r
            act[is_evade, 0] = ef[is_evade]
            act[is_evade, 1] = er[is_evade]
            act[is_evade, 2] = 0.0   # altitude handled by the safety band below
            act[is_evade, 3] = 0.0   # hold heading (anti-jitter)

        # Building avoidance add-on (repulsion computed above, before maneuvers).
        act[:, 0] = act[:, 0] + 0.7 * torch.tanh(rep_fwd)
        act[:, 1] = act[:, 1] + 0.7 * torch.tanh(rep_right)

        # Turret keep-out moat add-on (computed above). Full-strength radial-OUT
        # below MOAT_R, EXEMPTING COMMIT (the suicide dive must cross the moat to
        # die). Prevents WANDER deep entry + arrests APPROACH overshoot so danger
        # triggers shallow and EVADE survives. EVADE already commands radial-out,
        # so this just reinforces it inside the moat (clamped, no double-count).
        not_commit = self._mid != COMMIT
        moat = not_commit & (dist_t < MOAT_R)
        if moat.any():
            act[moat, 0] = act[moat, 0] - 1.0 * cb[moat] * moat_strength[moat]
            act[moat, 1] = act[moat, 1] - 1.0 * sb[moat] * moat_strength[moat]

        # Altitude safety band (non-shot wander drones only; shot drones are
        # thrust-overridden by the env, and CRASH_Z is excluded for them). Keeps
        # the drone off the ground during wander so it doesn't crash-terminate,
        # and caps aimless climbing. low/high are disjoint so order is safe.
        low = drone_z < 0.8
        high = drone_z > 2.5
        act[low, 2] = 0.6
        act[high, 2] = -0.3

        if self.noise > 0.0:
            act = act + torch.randn(n, 4, device=self.device, generator=self._gen) * self.noise
        return act.clamp(-1.0, 1.0).to(torch.float32)

    # -- maneuver re-roll ----------------------------------------------------
    def _roll_maneuvers(self, mask: torch.Tensor, danger: torch.Tensor) -> None:
        idx = mask.nonzero(as_tuple=False).squeeze(-1)
        m = idx.numel()
        if m == 0:
            return
        dev = self.device
        g = self._gen
        rnd = torch.rand(m, device=dev, generator=g)
        spd = 0.4 + 0.6 * torch.rand(m, device=dev, generator=g)        # [0.4, 1.0]
        sgn = torch.where(torch.rand(m, device=dev, generator=g) < 0.5,
                          torch.ones(m, device=dev), -torch.ones(m, device=dev))
        sgn2 = torch.where(torch.rand(m, device=dev, generator=g) < 0.5,
                           torch.ones(m, device=dev), -torch.ones(m, device=dev))
        yawmag = 0.5 + 0.5 * torch.rand(m, device=dev, generator=g)     # [0.5, 1.0]

        dur_wander = torch.randint(12, 31, (m,), device=dev, generator=g)
        # Stand-off loiter must outlast the turret's aim slew (50 deg/s; a ~90 deg
        # average slew ~ 18 steps) so danger actually triggers before the timer
        # rerolls the drone out of range. Danger cuts it short anyway.
        dur_prov = torch.randint(20, 35, (m,), device=dev, generator=g)
        dur_danger = torch.randint(6, 11, (m,), device=dev, generator=g)

        mid = torch.zeros(m, dtype=torch.long, device=dev)
        dur = torch.zeros(m, dtype=torch.long, device=dev)
        fwd = torch.zeros(m, device=dev)
        strafe = torch.zeros(m, device=dev)
        vz = torch.zeros(m, device=dev)
        yrt = torch.zeros(m, device=dev)

        is_danger = danger[idx]
        commit = is_danger & (rnd < self.r)          # danger -> suicide dive (COMMIT)
        evade = is_danger & ~commit                  # danger -> radial-out (EVADE)
        safe = ~is_danger
        wander = safe & (rnd < self.q)               # q = wander fraction
        provoke = safe & ~wander                     # safe -> stand-off loiter (APPROACH)

        # Wander: uniform among CRUISE..HOVER.
        wchoice = torch.randint(0, _N_WANDER_MODES, (m,), device=dev, generator=g)
        dur[wander] = dur_wander[wander]
        sel = wander & (wchoice == CRUISE)
        fwd[sel] = sgn[sel] * spd[sel]
        sel = wander & (wchoice == STRAFE)
        strafe[sel] = sgn[sel] * spd[sel]
        sel = wander & (wchoice == SPIN)
        yrt[sel] = sgn[sel] * yawmag[sel]
        sel = wander & (wchoice == BANKED)
        fwd[sel] = spd[sel]
        yrt[sel] = sgn2[sel] * yawmag[sel]
        sel = wander & (wchoice == HOVER)
        fwd[sel] = 0.1 * sgn[sel]
        yrt[sel] = sgn2[sel] * 0.3 * yawmag[sel]

        # COMMIT (suicide dive): danger-commit. Flies straight at the turret ->
        # deep entry -> death -> fall tail. Live-computed in get_action.
        sel = commit
        mid[sel] = COMMIT
        dur[sel] = dur_danger[sel]

        # APPROACH (stand-off provoke): safe (not in danger) -> loiter at the ~7 m
        # range edge facing the turret until danger triggers. Live-computed in
        # get_action (stand-off regulator). Persists through the turret slew
        # (dur_prov > slew time) so danger actually fires.
        sel = provoke
        mid[sel] = APPROACH
        dur[sel] = dur_prov[sel]

        # EVADE (radial-out escape): danger-evade. Command computed live each step
        # in get_action (full-strength radial-out + close-range lateral).
        sel = evade
        mid[sel] = EVADE
        dur[sel] = dur_danger[sel]

        self._mid[idx] = mid
        self._mtimer[idx] = dur
        self._mfwd[idx] = fwd
        self._mstrafe[idx] = strafe
        self._mvz[idx] = vz
        self._myaw[idx] = yrt

    # -- helpers -------------------------------------------------------------
    def _building_repulsion(self, state, building_centers, building_half) -> torch.Tensor:
        drone_xy = state[:, 0:2]
        bc = building_centers.to(self.device)[:, :, :2]   # (N, K, 2)
        bh = building_half.to(self.device)[:, :, :2]      # (N, K, 2)
        rel = bc - drone_xy.unsqueeze(1)                  # (N, K, 2)
        dist = torch.linalg.vector_norm(rel, dim=-1)      # (N, K)
        radius = bh.max(dim=-1).values + 1.5              # (N, K) half-extent + margin
        strength = ((radius - dist) / 1.5).clamp(0.0, 1.0)
        rep_dir = -rel / (dist.unsqueeze(-1) + 1e-6)
        return (rep_dir * strength.unsqueeze(-1)).sum(dim=1)  # (N, 2) world-frame


class WaypointPolicy:
    """Fixed-route collector policy -> body-frame 4D action. Does NOT react to the
    turret (the "fixed flight + WM as a safety monitor" use case): the drone
    follows a per-episode MANHATTAN route weaving through the street grid and may
    fly into the turret's FOV -> danger/shot/fall, exactly the transition the WM
    must learn to flag. Paired with EP_NO_TURRET it is a pure normal-flight baseline.

    Routes are axis-aligned segments between street-grid intersections (the clear
    corridors between buildings), generated per episode in reset_envs as a goal-less
    random walk from the env's actual spawn (Evan: the drone "follows fixed paths
    throughout the map", not an A->B flight). The controller commands body-forward
    velocity toward the current waypoint + yaw to face it, advancing when within
    WP_ADVANCE. A building-repulsion safety net + an altitude band guard the turns
    (the route itself stays in streets, so these rarely engage). Per-episode cruise
    speed is randomized in [0.5, 1.0] of MAX_SPEED. No moat, no evade -- the drone
    never retreats, by design (it is the oblivious fixed flight).
    """
    # Street grid + spawn height DEFAULTS -- must match uav_turret_3d.py. Kept as
    # local defaults (not imported) so policies_3d stays pure-torch / dependency-free.
    STREET_GRID = (-12.0, -6.0, 0.0, 6.0, 12.0)
    DRONE_SPAWN_Z = 1.5
    MAX_WP = 12        # route buffer width (a 3-turn route is ~10 waypoints)
    WP_ADVANCE = 1.5   # m; switch to the next waypoint when within this

    def __init__(self, seed: int = 0, device: str = "cuda",
                 street_grid=STREET_GRID, drone_spawn_z: float = DRONE_SPAWN_Z):
        self.device = torch.device(device)
        self._street = list(street_grid)
        self._spawn_z = float(drone_spawn_z)
        self._rng = random.Random(seed)
        self._n = 0  # buffers allocated lazily (grow-preserving) on first use

    # -- buffer management (grow-preserving so routes set in reset_envs survive a
    # later get_action that widens the buffer to the full env count) -------------
    def _ensure_buffers(self, n: int) -> None:
        if self._n >= n:
            return
        dev = self.device
        new_active = torch.zeros(n, dtype=torch.bool, device=dev)
        new_wp = torch.zeros(n, self.MAX_WP, 3, device=dev)
        new_len = torch.zeros(n, dtype=torch.long, device=dev)
        new_idx = torch.zeros(n, dtype=torch.long, device=dev)
        new_speed = torch.ones(n, device=dev)   # per-episode cruise-speed scale (1.0 = max)
        if self._n:
            new_active[: self._n] = self._active
            new_wp[: self._n] = self._waypoints
            new_len[: self._n] = self._wp_len
            new_idx[: self._n] = self._wp_idx
            new_speed[: self._n] = self._speed_scale
        self._active, self._waypoints, self._wp_len, self._wp_idx, self._speed_scale = (
            new_active, new_wp, new_len, new_idx, new_speed)
        self._n = n

    def reset_envs(self, env_ids, drone_pos) -> None:
        """Generate a fresh goal-less Manhattan route for each env_id from its spawn,
        and roll a per-episode cruise-speed scale (Evan: "randomize the speed of the
        drone for episodes with a path").

        drone_pos: (m, 3) env-local, aligned with env_ids (length m). No goal -- the
        drone follows a fixed wandering path through the map, not an A->B flight.
        """
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        self._ensure_buffers(int(env_ids.max().item()) + 1)
        for j in range(int(env_ids.numel())):
            route = self._gen_route(drone_pos[j][:2].tolist())
            ei = int(env_ids[j].item())
            L = len(route)
            self._waypoints[ei, :L] = torch.tensor(route, device=self.device, dtype=torch.float32)
            self._wp_len[ei] = L
            self._wp_idx[ei] = 0
            self._active[ei] = True
            # per-episode cruise speed in [0.5, 1.0] of MAX_SPEED -> 1.0..2.0 m/s
            self._speed_scale[ei] = self._rng.uniform(0.5, 1.0)

    def _gen_route(self, spawn):
        """Goal-less Manhattan random walk through street corridors (building-free).

        The drone follows a FIXED wandering path through the map (Evan: "remove the
        target location; the drone follows fixed paths throughout the map"). Start at
        the spawn (a street intersection), then alternate x/y moves to random street
        values -> a bounded staircase covering the city. Every segment lies on a
        street x or street y (no building), so the path is always clear.
        """
        rng = self._rng
        street = self._street
        x, y = float(spawn[0]), float(spawn[1])
        z = self._spawn_z
        route = [(x, y, z)]
        axis = 0  # 0 = move along x (change x, hold y); 1 = move along y
        for _ in range(self.MAX_WP - 1):
            axis = 1 - axis                      # alternate x/y turns -> staircase
            cur = x if axis == 0 else y
            choices = [v for v in street if abs(v - cur) > 1e-3]
            if not choices:
                break
            nv = rng.choice(choices)
            if axis == 0:
                x = nv
            else:
                y = nv
            route.append((x, y, z))
        return route

    # -- main entry point -----------------------------------------------------
    def get_action(self, state: torch.Tensor,
                   building_centers: torch.Tensor | None = None,
                   building_half: torch.Tensor | None = None) -> torch.Tensor:
        state = state.to(self.device)
        n = int(state.shape[0])
        self._ensure_buffers(n)
        dp = state[:, 0:3]
        yaw = state[:, 20]
        cy = torch.cos(yaw)
        sy = torch.sin(yaw)
        ar = torch.arange(n, device=self.device)

        def _to_wp():
            idx = torch.clamp(torch.minimum(self._wp_idx, self._wp_len - 1), min=0)
            wp = self._waypoints[ar, idx]            # (n, 3)
            return wp[:, :2] - dp[:, :2], idx        # (n, 2) world->body later

        to_wp, _ = _to_wp()
        dist = torch.linalg.vector_norm(to_wp, dim=-1)
        advance = self._active & (dist < self.WP_ADVANCE) & (self._wp_idx < self._wp_len - 1)
        self._wp_idx = torch.where(advance, self._wp_idx + 1, self._wp_idx)
        to_wp, _ = _to_wp()
        dist = torch.linalg.vector_norm(to_wp, dim=-1)

        # body-frame direction to the current waypoint
        fwd = to_wp[:, 0] * cy + to_wp[:, 1] * sy
        right = -to_wp[:, 0] * sy + to_wp[:, 1] * cy
        bearing = torch.atan2(right, fwd)
        vx = torch.tanh(dist / 3.0)            # cruise: ease in/out by distance
        yrt = torch.tanh(bearing)              # yaw to face the waypoint
        act = torch.stack([vx, torch.zeros(n, device=self.device),
                           torch.zeros(n, device=self.device), yrt], dim=-1)
        # finished routes (at the last wp + close): hold still
        done = self._active & (self._wp_idx >= self._wp_len - 1) & (dist < self.WP_ADVANCE)
        act[done] = 0.0
        # building-avoidance safety net (route stays in streets; guards turns)
        if building_centers is not None and building_half is not None:
            rep = self._building_repulsion(state, building_centers, building_half)
            rep_fwd = rep[:, 0] * cy + rep[:, 1] * sy
            rep_right = -rep[:, 0] * sy + rep[:, 1] * cy
            act[:, 0] = act[:, 0] + 0.7 * torch.tanh(rep_fwd)
            act[:, 1] = act[:, 1] + 0.7 * torch.tanh(rep_right)
        # per-episode cruise speed (Evan: "randomize the speed... within reasonable
        # speeds"): scale the HORIZONTAL channels only -- altitude band + yaw_rate
        # are safety/heading controls, not cruise, so they stay unscaled.
        act[:, 0] = act[:, 0] * self._speed_scale
        act[:, 1] = act[:, 1] * self._speed_scale
        # altitude band (same as the explorer: keep off the ground + below roofline)
        drone_z = state[:, 2]
        act[drone_z < 0.8, 2] = 0.6
        act[drone_z > 2.5, 2] = -0.3
        # inactive envs: zero LAST (repulsion/altitude above run over all envs, so a
        # trailing zero guarantees inactive envs return true zeros -- the collector
        # discards them via the per-env mode combine regardless).
        act[~self._active] = 0.0
        return act.clamp(-1.0, 1.0).to(torch.float32)

    def _building_repulsion(self, state, building_centers, building_half) -> torch.Tensor:
        drone_xy = state[:, 0:2]
        bc = building_centers.to(self.device)[:, :, :2]
        bh = building_half.to(self.device)[:, :, :2]
        rel = bc - drone_xy.unsqueeze(1)
        dist = torch.linalg.vector_norm(rel, dim=-1)
        radius = bh.max(dim=-1).values + 1.5
        strength = ((radius - dist) / 1.5).clamp(0.0, 1.0)
        rep_dir = -rel / (dist.unsqueeze(-1) + 1e-6)
        return (rep_dir * strength.unsqueeze(-1)).sum(dim=1)
