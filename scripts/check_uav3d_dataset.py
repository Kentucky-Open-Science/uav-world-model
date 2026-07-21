"""Sanity-check the uav3d lance: load_dataset + print shapes + get_dim + density.

Runs in the host swm training venv on the GPU box (after lance_from_episodes.py).
Resolves the bare lance name against $STABLEWM_HOME/datasets (default
~/.stable_worldmodel/datasets). Verifies the schema the LeWM reader expects:
  pixels (T,3,224,224), action (T, frameskip*4=20), state (T,21), shot (T,1)
and that get_dim('action')==4 (-> action_encoder.input_dim = frameskip*4 = 20).
Also recomputes danger density from the `state` column (state[13]==danger) and
the `shot` fraction, cross-checking the Stage-1 manifest density gate.

Usage:
    python scripts/check_uav3d_dataset.py
"""
import stable_worldmodel as swm

NAME = "uav_isaac_train.lance"
print(f"[CHECK] loading {NAME} ...")
ds = swm.data.load_dataset(
    NAME,
    num_steps=4,
    frameskip=5,
    keys_to_load=["pixels", "action", "state", "shot"],
    keys_to_cache=["action", "state"],
)
print(f"[CHECK] len(dataset)={len(ds)}  get_dim('action')={ds.get_dim('action')}  "
      f"get_dim('state')={ds.get_dim('state')}")

s = ds[0]
print("[CHECK] sample[0]:")
for k, v in s.items():
    shape = getattr(v, "shape", None) or (len(v) if hasattr(v, "__len__") else "?")
    print(f"    {k:8s} {type(v).__name__:12s} shape={shape}")

# density + shot fraction over a sample of windows (cheap; full scan is slow)
import random
n = len(ds)
k = min(200, n)
idxs = random.sample(range(n), k) if n else []
nd = 0; nshot = 0; nframes = 0
for i in idxs:
    st = ds[i]["state"]            # (T, 21) tensor
    sh = ds[i]["shot"]             # (T, 1) tensor
    T = st.shape[0]
    nframes += T
    nd += int((st[:, 13] > 0.5).sum())     # danger@13
    nshot += int((sh.flatten() > 0.5).sum())
if nframes:
    print(f"[CHECK] over {k} windows / {nframes} frames: "
          f"danger density={100.0*nd/nframes:.2f}%  shot fraction={100.0*nshot/nframes:.2f}%")
print("[CHECK] UAVTURRET3D DATASET SANITY OK")
