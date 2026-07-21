#!/usr/bin/env python
"""Model-half smoke for danger_probe.py: load the latest checkpoint, encode 8
real frames, assert emb[:,0,:].shape == (8,192) and non-collapse. Validates
load_pretrained + encode on a single-frame (T=1) batch before the full probe
run at epoch 25. ~0.1s GPU inference; safe alongside training.

  python scripts/danger_probe_smoke.py
"""
import torch
import stable_worldmodel as swm
from stable_worldmodel.wm.utils import load_pretrained
from stable_pretraining import data as spt_data

d = swm.data.utils.get_cache_dir(sub_folder="checkpoints") / "lewm"
ck = sorted(d.glob("weights_epoch_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))[-1]
print(f"[SMOKE] ckpt={ck.name}")
m = load_pretrained(f"lewm/{ck.name}").eval().to("cuda")
for p in m.parameters():
    p.requires_grad_(False)
print("[SMOKE] model loaded+frozen")

ds = swm.data.load_dataset(
    "uav_isaac_train.lance", num_steps=1, frameskip=1, keys_to_load=["pixels", "state"]
)
tr = spt_data.transforms
ds.transform = tr.Compose(
    tr.ToImage(**spt_data.dataset_stats.ImageNet, source="pixels", target="pixels"),
    tr.Resize(224, source="pixels", target="pixels"),
)
# 4 danger frames + 4 early frames
idxs, di = [], 0
while len(idxs) < 4:
    s = ds[di]
    di += 1
    if float(s["state"][0, 13]) > 0.5:
        idxs.append(di - 1)
idxs = idxs[:4] + list(range(0, 4))
px = torch.stack([ds[i]["pixels"] for i in idxs]).to("cuda")  # (8,1,C,H,W)
print(f"[SMOKE] pixels in={tuple(px.shape)} dtype={px.dtype}")
with torch.inference_mode():
    emb = m.encode({"pixels": px})["emb"][:, 0, :]
print(f"[SMOKE] emb={tuple(emb.shape)} (expect (8,192)) dtype={emb.dtype}")
e = emb.float()
print(f"[SMOKE] emb mean={e.mean():.4f} std={e.std():.4f} min={e.min():.4f} max={e.max():.4f}")
norm = e / (e.norm(dim=1, keepdim=True) + 1e-6)
print(f"[SMOKE] row0-1 cosine={(norm[0]*norm[1]).sum():.4f} (1.0=collapsed, <0.9=distinct)")
assert emb.shape == (8, 192), f"SHAPE FAIL {emb.shape}"
assert e.std() > 1e-4, "COLLAPSE (zero std)"
print("[SMOKE] danger_probe model-half OK: shape + non-collapse verified")
