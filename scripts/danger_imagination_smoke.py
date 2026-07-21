#!/usr/bin/env python
"""Model-half smoke for danger_imagination.py: load latest ckpt, encode + rollout
8 windows, assert emb=(8,7,192), predicted_emb=(8,1,8,192), imagined[t+k] at
index 2+k is non-collapsed, and imagination differs from the present latent.
Validates encode + rollout + the 2+k indexing before the full run at epoch 25.

  python scripts/danger_imagination_smoke.py
"""
import torch
import stable_worldmodel as swm
from stable_pretraining import data as spt_data
from stable_worldmodel.wm.utils import load_pretrained

HISTORY, K_MAX = 3, 4

d = swm.data.utils.get_cache_dir(sub_folder="checkpoints") / "lewm"
ck = sorted(d.glob("weights_epoch_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))[-1]
print(f"[SMOKE] ckpt={ck.name}")
m = load_pretrained(f"lewm/{ck.name}").eval().to("cuda")
for p in m.parameters():
    p.requires_grad_(False)
print("[SMOKE] model loaded+frozen")

ds = swm.data.load_dataset(
    "uav_isaac_train.lance",
    num_steps=HISTORY + K_MAX,
    frameskip=5,
    keys_to_load=["pixels", "action", "state", "shot"],
)
tr = spt_data.transforms
ds.transform = tr.Compose(
    tr.ToImage(**spt_data.dataset_stats.ImageNet, source="pixels", target="pixels"),
    tr.Resize(224, source="pixels", target="pixels"),
)
idxs = list(range(8))
px = torch.stack([ds[i]["pixels"] for i in idxs]).to("cuda")  # (8,7,C,H,W)
act = torch.stack([ds[i]["action"] for i in idxs]).to("cuda")  # (8,7,20) -- rollout's action_encoder is on cuda
print(f"[SMOKE] pixels={tuple(px.shape)} action={tuple(act.shape)} dtype={px.dtype}")

with torch.inference_mode():
    emb = m.encode({"pixels": px})["emb"]  # (8,7,192)
    px_ctx = px[:, :HISTORY].unsqueeze(1)  # (8,1,3,C,H,W)
    act_seq = act.unsqueeze(1)  # (8,1,7,20)
    pred = m.rollout({"pixels": px_ctx}, act_seq, history_size=HISTORY)["predicted_emb"]
print(f"[SMOKE] emb={tuple(emb.shape)} (expect (8,7,192))")
print(f"[SMOKE] pred={tuple(pred.shape)} (expect (8,1,8,192))")
assert emb.shape == (8, 7, 192), f"EMB SHAPE FAIL {emb.shape}"
assert pred.shape == (8, 1, 8, 192), f"PRED SHAPE FAIL {pred.shape}"

present = emb[:, HISTORY - 1, :].float()  # frame t (idx 2), (8,192)
imgs = [pred[:, 0, HISTORY - 1 + k, :].float() for k in range(1, K_MAX + 1)]  # idx 2+k
for k, ik in zip(range(1, K_MAX + 1), imgs):
    print(
        f"[SMOKE] imagined[t+{k}] std={ik.std():.4f} "
        f"range[{ik.min():.3f},{ik.max():.3f}] "
        f"cos-vs-present={torch.nn.functional.cosine_similarity(ik, present).mean():.4f}"
    )
    assert ik.std() > 1e-4, f"COLLAPSE at k={k}"
# imagination must differ from the present latent (else predictor is identity)
diff = (imgs[0] - present).abs().mean()
print(f"[SMOKE] |imagined[t+1] - present| mean={diff:.4f} (>0 => predictor moves the latent)")
assert diff > 1e-4, "predictor is identity (imagined == present)"
print("[SMOKE] danger_imagination model-half OK: encode + rollout + indexing verified")
