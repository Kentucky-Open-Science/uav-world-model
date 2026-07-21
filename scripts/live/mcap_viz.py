"""Foxglove MCAP logging for the live demo (scripts/live/live_demo.py).

Adapted from scripts/collect_uav3d_critique.py's EpisodeWriter, extended for the
WM-vs-detector showcase:
  - trajectory trail (LINE_STRIP of accumulated drone positions) -- shows evasion.
  - turret FOV cone (2 edge rays around the aim line) -- shows the danger volume.
  - a /signals PoseInFrame channel carrying (wm_danger, det_logit, danger) so
    Foxglove's Plot panel graphs imagination-vs-detection on ONE timeline -- the
    showcase for "WM predicts ahead of detection".
  - a plain .signals.txt sidecar (one line/step) so the best showcase episode can
    be picked textually without opening Foxglove.

Stack: mcap-protobuf-support + foxglove-schemas-protobuf 0.3.0, bind-mounted at
/workspace/libs. v0.3.0 has NO Plot/TimeSeries/numeric-scalar schema and
KeyValuePair is string-only -- so numeric signals ride in a PoseInFrame
(position.x/y/z), which Foxglove's Plot panel graphs as `pose.position.{x,y,z}`
vs its header timestamp. No heavy top-level imports (the caller runs AppLauncher
first); foxglove + uav_wm imports are lazy inside write_step, like the collector.
"""
import math
import os
import sys
import time

sys.path.insert(0, "/workspace/libs")
from google.protobuf.timestamp_pb2 import Timestamp  # noqa: E402
from mcap_protobuf.writer import Writer as McapWriter  # noqa: E402

STEP_DT = 0.1  # env step dt (sim.dt 0.01 * decimation 10) -> 0.1 s; drives MCAP timeline


def _ts(ns):
    return Timestamp(seconds=ns // 1_000_000_000, nanos=ns % 1_000_000_000)


def _vec3(x, y, z):
    from foxglove_schemas_protobuf.Vector3_pb2 import Vector3
    return Vector3(x=float(x), y=float(y), z=float(z))


def _quat(w, x, y, z):
    from foxglove_schemas_protobuf.Quaternion_pb2 import Quaternion
    return Quaternion(w=float(w), x=float(x), y=float(y), z=float(z))


def _pose(px, py, pz, qw=1.0, qx=0.0, qy=0.0, qz=0.0):
    from foxglove_schemas_protobuf.Pose_pb2 import Pose
    return Pose(position=_vec3(px, py, pz), orientation=_quat(qw, qx, qy, qz))


def _color(r, g, b, a=1.0):
    from foxglove_schemas_protobuf.Color_pb2 import Color
    return Color(r=float(r), g=float(g), b=float(b), a=float(a))


def _point(x, y, z):
    from foxglove_schemas_protobuf.Point3_pb2 import Point3
    return Point3(x=float(x), y=float(y), z=float(z))


class LiveEpisodeWriter:
    """One Foxglove MCAP (+ .signals.txt sidecar) per episode. Coordinates env-local."""

    def __init__(self, path, ep_label):
        self.path = path
        self.ep_label = ep_label
        self.w = McapWriter(path)
        self.t0_ns = time.time_ns()
        self.steps = 0
        self.sig_path = os.path.splitext(path)[0] + ".signals.txt"
        self._sigf = open(self.sig_path, "w")
        self._sigf.write("# step wm_danger det_logit danger fire dist_t dx dy yaw tbrg offaxis\n")

    def _tns(self, step):
        return self.t0_ns + step * int(STEP_DT * 1e9)

    def write_step(self, step, pov_hwc_rgb, drone_xyz, drone_quat_wxyz,
                   turret_xyz, aim_dir, obstacles, st, has_turret=True,
                   trail=None, wm_danger=None, det_logit=None, goal_xyz=None):
        """st = 21-dim state (torch or list). obstacles = list of (center, full_extents).
        trail = list of (x,y,z) drone positions this episode (drawn as LINE_STRIP).
        wm_danger/det_logit = floats or None (planner mode has both; detector has
        det_logit only; waypoint has neither).
        goal_xyz = env-local goal B (nav-family) or None/STOW (showcase/waypoint);
        a green pillar marks it when set (STOW's z=-50 suppresses it)."""
        from uav_wm.envs.uav_turret_3d import FOV_HALF, TURRET_RANGE  # lazy; container-only
        tns = self._tns(step)
        ts = _ts(tns)
        danger = int(st[13]); in_range = int(st[14]); los = int(st[15])
        aimed = int(st[16]); fire = float(st[17]); dist_t = float(st[12])
        dx = float(st[0]); dy = float(st[1]); yaw = float(st[20])
        # off-axis angle = |turret bearing - drone heading|; > FOV_HALF (~23deg =>
        # 0.40 rad) means the turret has left the POV (explains a flat det_logit /
        # wm_danger at a range where phantom fires). turret_rel (9:12) is world-frame.
        import math as _m
        tbrg = _m.atan2(float(st[10]), float(st[9]))
        offaxis = abs(((tbrg - yaw + _m.pi) % (2 * _m.pi)) - _m.pi)
        self.steps = step + 1
        wm = float(wm_danger) if wm_danger is not None else float("nan")
        det = float(det_logit) if det_logit is not None else float("nan")
        self._sigf.write(f"{step} {wm:.4f} {det:.4f} {danger} {fire:.4f} {dist_t:.3f} "
                         f"{dx:.2f} {dy:.2f} {yaw:.2f} {tbrg:.2f} {offaxis:.2f}\n")

        # --- /drone/pov/image : RawImage ---
        h, w, _ = pov_hwc_rgb.shape
        from foxglove_schemas_protobuf.RawImage_pb2 import RawImage
        img = RawImage(timestamp=ts, frame_id="drone", width=w, height=h,
                       encoding="rgb8", step=w * 3, data=bytes(pov_hwc_rgb.tobytes()))
        self.w.write_message("/drone/pov/image", img, log_time=tns, publish_time=tns)

        # --- /scene : SceneUpdate ---
        from foxglove_schemas_protobuf.SceneUpdate_pb2 import SceneUpdate
        from foxglove_schemas_protobuf.SceneEntity_pb2 import SceneEntity
        from foxglove_schemas_protobuf.CubePrimitive_pb2 import CubePrimitive
        from foxglove_schemas_protobuf.LinePrimitive_pb2 import LinePrimitive
        from foxglove_schemas_protobuf.TextPrimitive_pb2 import TextPrimitive
        cubes = []
        cubes.append(CubePrimitive(pose=_pose(*drone_xyz), size=_vec3(0.3, 0.3, 0.1),
                                   color=_color(0.2, 0.4, 1.0)))  # drone blue
        if has_turret:
            cubes.append(CubePrimitive(pose=_pose(*turret_xyz), size=_vec3(0.4, 0.4, 0.6),
                                       color=_color(0.9, 0.2, 0.2)))  # turret red
        if goal_xyz is not None and goal_xyz[2] > -10.0:  # skip GOAL_STOW (z=-50)
            gx, gy, _ = goal_xyz
            cubes.append(CubePrimitive(pose=_pose(gx, gy, 1.0), size=_vec3(0.25, 0.25, 2.0),
                                       color=_color(0.1, 0.9, 0.2)))  # goal B green pillar
        for cen, ext in obstacles:
            cubes.append(CubePrimitive(pose=_pose(cen[0], cen[1], cen[2]),
                                       size=_vec3(ext[0], ext[1], ext[2]),
                                       color=_color(0.5, 0.5, 0.5, 0.8)))
        lines = []
        if has_turret:
            ax, ay, az = aim_dir
            tx, ty, tz = turret_xyz
            # aim line (center) -- red if danger else orange
            line_color = _color(1.0, 0.1, 0.1, 0.9) if danger else _color(1.0, 0.5, 0.1, 0.9)
            lines.append(LinePrimitive(
                type=LinePrimitive.LINE_LIST, pose=_pose(0, 0, 0), thickness=0.04,
                scale_invariant=False, points=[_point(tx, ty, tz),
                    _point(tx + ax * TURRET_RANGE, ty + ay * TURRET_RANGE, tz + az * TURRET_RANGE)],
                color=line_color))
            # FOV cone edges (yaw +/- FOV_HALF) -- faint, shows the danger volume
            yaw = math.atan2(ay, ax); pit = math.asin(max(-1.0, min(1.0, az)))
            for dyaw in (FOV_HALF, -FOV_HALF):
                ex = math.cos(pit) * math.cos(yaw + dyaw)
                ey = math.cos(pit) * math.sin(yaw + dyaw)
                ez = math.sin(pit)
                lines.append(LinePrimitive(
                    type=LinePrimitive.LINE_LIST, pose=_pose(0, 0, 0), thickness=0.02,
                    scale_invariant=False, points=[_point(tx, ty, tz),
                        _point(tx + ex * TURRET_RANGE, ty + ey * TURRET_RANGE, tz + ez * TURRET_RANGE)],
                    color=_color(1.0, 0.3, 0.1, 0.35)))
        # trajectory trail (LINE_STRIP) -- the drone's path this episode; shows evasion
        if trail and len(trail) >= 2:
            pts = [_point(p[0], p[1], p[2]) for p in trail]
            lines.append(LinePrimitive(
                type=LinePrimitive.LINE_STRIP, pose=_pose(0, 0, 0), thickness=0.03,
                scale_invariant=False, points=pts, color=_color(0.2, 0.8, 0.2, 0.7)))
        sig = (f" wm={wm:+.2f} det={det:+.2f}" if (wm_danger is not None or det_logit is not None)
               else "")
        txt = (f"{self.ep_label} step{step} d={danger} r={in_range} l={los} a={aimed} "
               f"fire={fire:.2f} dist={dist_t:.1f}{sig}")
        texts = [TextPrimitive(pose=_pose(drone_xyz[0], drone_xyz[1], drone_xyz[2] + 0.6),
                               billboard=False, font_size=0.18, scale_invariant=False,
                               color=_color(1.0, 1.0, 1.0) if not danger else _color(1.0, 0.2, 0.2),
                               text=txt)]
        ent = SceneEntity(timestamp=ts, frame_id="world", id="world",
                          frame_locked=False, cubes=cubes, spheres=[], lines=lines, texts=texts)
        self.w.write_message("/scene", SceneUpdate(entities=[ent]), log_time=tns, publish_time=tns)

        # --- /drone/tf : FrameTransform (world -> drone) ---
        from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform
        ft = FrameTransform(timestamp=ts, parent_frame_id="world", child_frame_id="drone",
                            translation=_vec3(*drone_xyz),
                            rotation=_quat(*drone_quat_wxyz))
        self.w.write_message("/drone/tf", ft, log_time=tns, publish_time=tns)

        # --- /state : Log (text; WARNING level when in danger) ---
        from foxglove_schemas_protobuf.Log_pb2 import Log
        dp = [float(st[0]), float(st[1]), float(st[2])]
        msg = (f"step{step} drone=[{dp[0]:+.2f},{dp[1]:+.2f},{dp[2]:+.2f}] "
               f"dist_t={dist_t:.2f} danger={danger} in_range={in_range} los={los} "
               f"aimed={aimed} fire={fire:.2f}{sig}")
        log = Log(timestamp=ts, level=Log.WARNING if danger else Log.INFO, message=msg, name="uav3d")
        self.w.write_message("/state", log, log_time=tns, publish_time=tns)

        # --- /signals : PoseInFrame -- position.x=wm_danger, y=det_logit, z=danger.
        # Foxglove Plot panel graphs pose.position.{x,y,z} vs the header timestamp:
        # the WM imagination signal vs the single-frame detector signal on one axis,
        # with the ground-truth danger -- the "WM predicts ahead of detection" view.
        from foxglove_schemas_protobuf.PoseInFrame_pb2 import PoseInFrame
        from foxglove_schemas_protobuf.Pose_pb2 import Pose
        sig_pose = PoseInFrame(
            timestamp=ts, frame_id="signals",
            pose=Pose(position=_vec3(wm if wm_danger is not None else 0.0,
                                     det if det_logit is not None else 0.0,
                                     float(danger)),
                      orientation=_quat(1.0, 0.0, 0.0, 0.0)))
        self.w.write_message("/signals", sig_pose, log_time=tns, publish_time=tns)

    def close(self):
        try:
            self._sigf.close()
        except Exception:
            pass
        self.w.finish()
