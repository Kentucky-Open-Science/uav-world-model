"""Inspect a UAVTurret3D critique MCAP: print the /state Log timeline.

Text-only diagnostic (no image inspection). Parses the per-step /state Log
messages the collector writes and prints step / drone_z / danger / fire /
in_range / los / aimed / dist_t, every N steps + every danger/fire step, so we
can see WHY an episode ran a given length (e.g. did truncation fire? did the
shot drone fall+rest?).

Run in the isaac-lab container (mcap + foxglove-schemas-protobuf in /workspace/libs):
    /workspace/isaaclab/isaaclab.sh -p /workspace/scripts/inspect_mcap.py \
        --mcap /workspace/output/uav3d_critique/episode_016.mcap --every 25
"""
import argparse
import re
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--mcap", type=str, required=True)
parser.add_argument("--every", type=int, default=25)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, "/workspace/libs")
from mcap.reader import make_reader  # noqa: E402
from foxglove_schemas_protobuf.Log_pb2 import Log  # noqa: E402

pat = re.compile(
    r"step(\d+) drone=\[([+\-0-9.]+),([+\-0-9.]+),([+\-0-9.]+)\] "
    r"dist_t=([0-9.]+) danger=(\d) in_range=(\d) los=(\d) aimed=(\d) fire=([0-9.]+)"
)

rows = []
with open(args_cli.mcap, "rb") as f:
    r = make_reader(f)
    for schema, channel, msg in r.iter_messages(topics=["/state"]):
        log = Log()
        log.ParseFromString(msg.data)
        m = pat.search(log.message)
        if not m:
            continue
        rows.append({
            "step": int(m.group(1)),
            "z": float(m.group(4)),
            "dist_t": float(m.group(5)),
            "danger": int(m.group(6)),
            "in_range": int(m.group(7)),
            "los": int(m.group(8)),
            "aimed": int(m.group(9)),
            "fire": float(m.group(10)),
        })

print(f"=== {Path(args_cli.mcap).name}: {len(rows)} state frames ===")
if not rows:
    print("(no /state messages parsed)")
else:
    zs = [r["z"] for r in rows]
    print(f"z range: {min(zs):.2f} .. {max(zs):.2f}")
    print(f"first: {rows[0]}")
    print(f"last:  {rows[-1]}")
    print(f"\n--- every {args_cli.every}th step + all danger/fire>0 ---")
    print(f"{'step':>4} {'z':>6} {'dist':>5} {'dang':>4} {'rng':>3} {'los':>3} {'aim':>3} {'fire':>5}")
    for r in rows:
        if r["step"] % args_cli.every == 0 or r["danger"] or r["fire"] > 0 or r["step"] == rows[-1]["step"]:
            print(f"{r['step']:>4} {r['z']:>6.2f} {r['dist_t']:>5.1f} {r['danger']:>4} "
                  f"{r['in_range']:>3} {r['los']:>3} {r['aimed']:>3} {r['fire']:>5.2f}")

simulation_app.close()
