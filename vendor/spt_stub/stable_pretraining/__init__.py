"""Minimal stub of stable_pretraining for the live closed loop (Phase 7).

The real package's top-level __init__ unconditionally imports loguru, kornia,
lightning, timm, sklearn, ... which are NOT in the Isaac Lab container. The only
spt symbol the live loop actually needs is `vit_hf` (the LeWM encoder factory),
which `load_pretrained` instantiates via the hydra `_target_` in config.json
(`stable_pretraining.backbone.utils.vit_hf`). This stub provides exactly that,
with loguru replaced by stdlib logging, so it imports under torch+transformers
alone. See briefs/phase7-live-demo.md.
"""
