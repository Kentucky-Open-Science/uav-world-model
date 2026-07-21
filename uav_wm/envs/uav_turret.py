"""Synthetic UAV + turret environment for the world-model danger-reasoning PoC.

Top-down 2D arena. A holonomic drone must reach a goal while a turret tracks
and shoots it on sustained line-of-sight.

The env is engineered so the demo's central claim holds:
  * The turret and its barrel orientation are VISIBLE in the rendered image,
    so a YOLO object detector can detect "a turret object" in the *current*
    frame.
  * The DANGER, however, is a *future* property. The turret tracks the drone
    with a finite yaw rate (`TURRET_YAW_RATE`), so the barrel may point away
    right now but rotate to face the drone within a few steps if the drone
    advances into range. A single-frame detector cannot see this; a world
    model that imagines the future can. Obstacles block line-of-sight, so the
    turret can also be occluded until the drone rounds a corner.

Observation is a compact state vector; pixels come from `render()` via swm's
`MegaWrapper` (`World(add_pixels=True)`). `info` carries `state`, `goal`,
`goal_state`, `danger` (per-frame label for the danger head), `killed`, and
`reached_goal` so the dataset records everything needed for training and
auto-labeling.
"""
import math
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless pygame

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

pygame.init()

# --- physics constants (meters, seconds) ---
ARENA = 10.0                  # arena is [-ARENA, ARENA]^2
DRONE_MAX_SPEED = 3.0
DRONE_RADIUS = 0.4
TURRET_RANGE = 8.0
TURRET_FOV_HALF = math.radians(30)
TURRET_YAW_RATE = math.radians(110)   # max barrel rotation (rad/s) -> tracking lag
TURRET_RADIUS = 0.5
FIRE_INTERVAL = 0.4           # sustained LOS+aim seconds before the shot lands
DT = 0.1                      # sim seconds per env step
GOAL_RADIUS = 1.0
IMG = 224                     # render resolution (swm resizes to image_shape anyway)

# --- colors ---
C_BG = (24, 26, 32)
C_OBS = (150, 152, 160)
C_GOAL = (60, 200, 90)
C_DRONE = (70, 140, 255)
C_TURRET = (220, 70, 60)
C_BARREL = (240, 200, 60)
C_FOV = (220, 70, 60, 55)
C_FOV_FIRE = (255, 230, 60, 90)

_M2PX = IMG / (2.0 * ARENA)   # meters -> pixels


def _ang_diff(a, b):
    """Smallest signed angle a-b, in [-pi, pi]."""
    return math.atan2(math.sin(a - b), math.cos(a - b))


def _to_img(x, z):
    """Arena coords (x east, z north) -> image px (y down)."""
    return int((x + ARENA) * _M2PX), int((ARENA - z) * _M2PX)


def _seg_rect(x0, y0, x1, y1, rx0, ry0, rx1, ry1):
    """Liang-Barsky: does segment (x0,y0)-(x1,y1) intersect AABB?"""
    dx, dy = x1 - x0, y1 - y0
    p = (-dx, dx, -dy, dy)
    q = (x0 - rx0, rx1 - x0, y0 - ry0, ry1 - y0)
    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return False
        else:
            t = qi / pi
            if pi < 0:
                u1 = max(u1, t)
            else:
                u2 = min(u2, t)
    return u1 < u2


class UAVTurretEnv(gym.Env):
    """Holonomic drone + tracking turret, top-down, RGB-rendered."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": int(1 / DT)}

    def __init__(self, render_mode="rgb_array", max_episode_steps=200):
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (16,), np.float32)
        self._surf = pygame.Surface((IMG, IMG))
        self._fov_surf = pygame.Surface((IMG, IMG), pygame.SRCALPHA)

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self.drone = np.array(
            [rng.uniform(-ARENA * 0.8, -ARENA * 0.4),
             rng.uniform(-ARENA * 0.5, ARENA * 0.5)], np.float32)
        self.goal = np.array(
            [rng.uniform(ARENA * 0.4, ARENA * 0.8),
             rng.uniform(-ARENA * 0.5, ARENA * 0.5)], np.float32)
        self.turret_pos = np.array(
            [rng.uniform(-ARENA * 0.2, ARENA * 0.2),
             rng.uniform(-ARENA * 0.2, ARENA * 0.2)], np.float32)
        # Barrel starts NOT pointing at the drone (current frame looks "safe").
        self.barrel = float(rng.uniform(-math.pi, math.pi))
        self._obstacles = self._gen_obstacles(rng)
        self._separate_start()
        self._fire_t = 0.0
        self._step = 0
        self.killed = False
        self.reached_goal = False
        return self._state(), self._info()

    def _gen_obstacles(self, rng):
        obs = []
        for _ in range(int(rng.integers(3, 6))):
            cx = rng.uniform(-ARENA * 0.7, ARENA * 0.7)
            cz = rng.uniform(-ARENA * 0.7, ARENA * 0.7)
            hw = rng.uniform(0.8, 2.0)
            hh = rng.uniform(0.8, 2.0)
            obs.append((cx, cz, hw, hh))
        return obs

    def _collides(self, pos, margin=DRONE_RADIUS):
        for cx, cz, hw, hh in self._obstacles:
            if (abs(pos[0] - cx) < hw + margin
                    and abs(pos[1] - cz) < hh + margin):
                return True
        return False

    def _separate_start(self):
        """Nudge drone/goal/turret out of obstacles and away from each other."""
        for _ in range(50):
            if not self._collides(self.drone):
                break
            self.drone += self.np_random.uniform(-0.5, 0.5, 2).astype(np.float32)
        for _ in range(50):
            if not self._collides(self.goal, margin=GOAL_RADIUS):
                break
            self.goal += self.np_random.uniform(-0.5, 0.5, 2).astype(np.float32)
        for _ in range(50):
            if not self._collides(self.turret_pos, margin=TURRET_RADIUS):
                break
            self.turret_pos += self.np_random.uniform(-0.5, 0.5, 2).astype(np.float32)

    # ------------------------------------------------------------------- step
    def step(self, action):
        action = np.asarray(action, np.float32).reshape(-1)[:2]
        cmd = np.clip(action, -1.0, 1.0) * DRONE_MAX_SPEED
        # Per-axis move with wall-sliding against obstacles.
        new_x = self.drone + np.array([cmd[0] * DT, 0.0], np.float32)
        if not self._collides(new_x):
            self.drone = new_x
        new_z = self.drone + np.array([0.0, cmd[1] * DT], np.float32)
        if not self._collides(new_z):
            self.drone = new_z
        self.drone = np.clip(self.drone, -ARENA, ARENA).astype(np.float32)

        # Turret tracks the drone with a finite yaw rate (the tracking lag that
        # makes danger a *future* property).
        rel = self.drone - self.turret_pos
        dist = float(np.linalg.norm(rel))
        target_ang = math.atan2(rel[1], rel[0])
        max_rot = TURRET_YAW_RATE * DT
        self.barrel += float(np.clip(_ang_diff(target_ang, self.barrel),
                                     -max_rot, max_rot))

        aimed = (dist < TURRET_RANGE
                 and abs(_ang_diff(target_ang, self.barrel)) < TURRET_FOV_HALF)
        los = self._line_of_sight(self.turret_pos, self.drone)
        in_danger = bool(aimed and los)
        if in_danger:
            self._fire_t += DT
        else:
            self._fire_t = max(0.0, self._fire_t - DT)
        if self._fire_t >= FIRE_INTERVAL:
            self.killed = True

        if float(np.linalg.norm(self.drone - self.goal)) < GOAL_RADIUS:
            self.reached_goal = True

        self._step += 1
        terminated = self.killed or self.reached_goal
        truncated = self._step >= self.max_episode_steps
        reward = (1.0 if self.reached_goal
                  else -1.0 if self.killed else -0.01)
        return self._state(), reward, terminated, truncated, self._info()

    def _line_of_sight(self, a, b):
        for cx, cz, hw, hh in self._obstacles:
            if _seg_rect(a[0], a[1], b[0], b[1],
                         cx - hw, cz - hh, cx + hw, cz + hh):
                return False
        return True

    # ----------------------------------------------------------- obs / state
    def _state(self):
        rel_goal = self.goal - self.drone
        rel_tur = self.turret_pos - self.drone
        dist_tur = float(np.linalg.norm(rel_tur))
        target_ang = math.atan2(rel_tur[1], rel_tur[0])
        aimed_ang = _ang_diff(target_ang, self.barrel)
        in_range = dist_tur < TURRET_RANGE
        los = self._line_of_sight(self.turret_pos, self.drone)
        danger = float(in_range and abs(aimed_ang) < TURRET_FOV_HALF and los)
        return np.array([
            self.drone[0] / ARENA, self.drone[1] / ARENA,
            float(self.killed), float(self.reached_goal),
            rel_goal[0] / ARENA, rel_goal[1] / ARENA,
            rel_tur[0] / ARENA, rel_tur[1] / ARENA,
            math.cos(self.barrel), math.sin(self.barrel),
            dist_tur / TURRET_RANGE,
            float(in_range), float(los), danger,
            self._fire_t / FIRE_INTERVAL,
            0.0,
        ], np.float32)

    def _info(self):
        rel = self.drone - self.turret_pos
        dist = float(np.linalg.norm(rel))
        target_ang = math.atan2(rel[1], rel[0])
        aimed = (dist < TURRET_RANGE
                 and abs(_ang_diff(target_ang, self.barrel)) < TURRET_FOV_HALF)
        danger = float(aimed and self._line_of_sight(self.turret_pos, self.drone))
        return {
            "state": self._state(),
            "goal_state": np.array([self.goal[0] / ARENA, self.goal[1] / ARENA],
                                   np.float32),
            "goal": self._render(drone_at=self.goal),
            "danger": danger,
            "killed": float(self.killed),
            "reached_goal": float(self.reached_goal),
            "drone_pos": self.drone.copy(),
            "turret_pos": self.turret_pos.copy(),
            "barrel": float(self.barrel),
        }

    # --------------------------------------------------------------- render
    def render(self):
        return self._render(drone_at=self.drone)

    def _render(self, drone_at):
        s = self._surf
        s.fill(C_BG)
        for cx, cz, hw, hh in self._obstacles:
            x0, y0 = _to_img(cx - hw, cz + hh)
            x1, y1 = _to_img(cx + hw, cz - hh)
            pygame.draw.rect(s, C_OBS,
                             (min(x0, x1), min(y0, y1),
                              abs(x1 - x0), abs(y1 - y0)))
        gx, gy = _to_img(self.goal[0], self.goal[1])
        pygame.draw.circle(s, C_GOAL, (gx, gy), 8)
        tx, ty = _to_img(self.turret_pos[0], self.turret_pos[1])
        # Translucent FOV cone (brightens while acquiring a shot).
        self._fov_surf.fill((0, 0, 0, 0))
        r_px = TURRET_RANGE * _M2PX
        pts = [(tx, ty)]
        n = 14
        for i in range(n + 1):
            a = self.barrel + (i / n - 0.5) * 2 * TURRET_FOV_HALF
            pts.append((tx + math.cos(a) * r_px, ty - math.sin(a) * r_px))
        color = C_FOV_FIRE if self._fire_t > 0 else C_FOV
        pygame.draw.polygon(self._fov_surf, color, pts)
        s.blit(self._fov_surf, (0, 0))
        pygame.draw.circle(s, C_TURRET, (tx, ty),
                           max(3, int(TURRET_RADIUS * _M2PX)))
        bx = tx + math.cos(self.barrel) * 16
        by = ty - math.sin(self.barrel) * 16
        pygame.draw.line(s, C_BARREL, (tx, ty), (bx, by), 4)
        if drone_at is not None:
            dx, dy = _to_img(drone_at[0], drone_at[1])
            pygame.draw.circle(s, C_DRONE, (dx, dy), 7)
        arr = pygame.surfarray.array3d(s)  # (W, H, 3)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
