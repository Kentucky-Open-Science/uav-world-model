"""Stub of stable_pretraining.backbone.utils — only `vit_hf`.

Faithful copy of the real `vit_hf` (size_configs + ViT construction), with the
sole change `from loguru import logger as logging` -> stdlib logging. The real
module also defines EvalOnly/FeaturesConcat/etc.; those are not instantiated by
`load_pretrained` so they are omitted. `pretrained=False` is the path LeWM uses
(no network), matching the training config.
"""
import logging

import torch
from torch import nn
from transformers import ViTConfig, ViTModel


def vit_hf(
    size: str = "tiny",
    patch_size: int = 16,
    image_size: int = 224,
    pretrained: bool = False,
    use_mask_token: bool = True,
    **kwargs,
) -> nn.Module:
    size_configs = {
        "tiny": {"hidden_size": 192, "num_hidden_layers": 12, "num_attention_heads": 3},
        "small": {"hidden_size": 384, "num_hidden_layers": 12, "num_attention_heads": 6},
        "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12},
        "large": {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16},
        "huge": {"hidden_size": 1280, "num_hidden_layers": 32, "num_attention_heads": 16},
    }
    if size not in size_configs:
        raise ValueError(f"Invalid size '{size}'. Choose from {list(size_configs.keys())}")
    p = dict(size_configs[size])
    p["intermediate_size"] = p["hidden_size"] * 4
    p["image_size"] = image_size
    p["patch_size"] = patch_size
    p.update(kwargs)
    if pretrained:
        model_name = f"google/vit-{size}-patch{patch_size}-{image_size}"
        logging.info("Loading pretrained ViT from %s", model_name)
        model = ViTModel.from_pretrained(
            model_name, add_pooling_layer=False, use_mask_token=use_mask_token
        )
    else:
        config = ViTConfig(**p)
        model = ViTModel(config, add_pooling_layer=False, use_mask_token=use_mask_token)
        logging.info("Created ViT-%s from scratch", size)
    # match the real vit_hf: allow dynamic input sizes
    model.config.interpolate_pos_encoding = True
    return model
