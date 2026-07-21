"""Stage 2: read Stage-1 episode pickles -> swm LanceWriter -> training lance.

Runs in the host swm training venv on the GPU box (has swm + lancedb + pyarrow + PIL; no
Isaac). Decodes each episode's JPEG pixels -> HWC uint8 and feeds LanceWriter, which
re-encodes to JPEG for the lance `pixels` column and assigns episode_idx/step_idx
(episode-contiguous by construction -- the writer assigns them, caller must not pass).
The density gate + load_dataset sanity run after this.

swm is editable-installed in the venv, so `import stable_worldmodel`
resolves directly -- no sys.path hack.

Usage (host swm training venv on the GPU box):
    python scripts/lance_from_episodes.py \
        --episodes_dir ~/docker/isaac-sim/output/uav3d_episodes \
        --out ~/.stable_worldmodel/datasets/uav_isaac_train.lance
"""
import argparse
import io
import pickle
from pathlib import Path

import numpy as np
from PIL import Image


def decode_jpeg(b: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype="uint8")


def load_episode(path):
    """Pickle -> LanceWriter episode dict (equal-length lists; pixels as raw HWC uint8)."""
    with open(path, "rb") as f:
        ep = pickle.load(f)
    return {
        "pixels": [decode_jpeg(b) for b in ep["pixels"]],
        "action": [np.asarray(a, dtype="float32") for a in ep["action"]],
        "state": [np.asarray(s, dtype="float32") for s in ep["state"]],
        "shot": [int(x) for x in ep["shot"]],
        "danger": [int(x) for x in ep["danger"]],
        "drone_pos": [np.asarray(p, dtype="float32") for p in ep["drone_pos"]],
        "turret_pos": [np.asarray(p, dtype="float32") for p in ep["turret_pos"]],
        "barrel": [np.asarray(p, dtype="float32") for p in ep["barrel"]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes_dir", required=True)
    ap.add_argument("--out", required=True, help="lance path ending in .lance")
    args = ap.parse_args()

    from stable_worldmodel.data.formats.lance import LanceWriter

    ep_files = sorted(Path(args.episodes_dir).glob("episode_*.pkl"))
    print(f"[LANCE] {len(ep_files)} episode pickles in {args.episodes_dir}")
    if not ep_files:
        print("[LANCE] no episodes -- nothing to write")
        return

    def episode_iter():
        tot_d = 0; tot = 0
        for k, p in enumerate(ep_files):
            ep = load_episode(p)
            tot += len(ep["pixels"]); tot_d += sum(ep["danger"])
            if (k + 1) % 50 == 0:
                print(f"[LANCE] decoded {k+1}/{len(ep_files)} eps  frames={tot} "
                      f"danger={tot_d} density={100.0*tot_d/max(tot,1):.2f}%", flush=True)
            yield ep
        print(f"[LANCE] decoded all: frames={tot} danger={tot_d} "
              f"density={100.0*tot_d/max(tot,1):.2f}%")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with LanceWriter(args.out, mode="overwrite") as w:
        w.write_episodes(episode_iter())
    print(f"[LANCE] wrote {args.out}")
    print("[LANCE] UAVTURRET3D LANCE WRITE OK")


if __name__ == "__main__":
    main()
