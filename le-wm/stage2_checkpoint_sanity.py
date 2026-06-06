#!/usr/bin/env python3
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from diagnose_pusht_latent_contacts import clean_cfg, vit_hf_from_config
from jepa import JEPA
from module import ARPredictor, Embedder, MLP

root = Path(__file__).resolve().parent / ".cache" / "stage2_model"
cfg_path = Path(hf_hub_download("quentinll/lewm-pusht", "config.json", local_dir=root))
weights_path = Path(hf_hub_download("quentinll/lewm-pusht", "weights.pt", local_dir=root))
cfg = json.loads(cfg_path.read_text())
norm = torch.nn.BatchNorm1d
mlp = lambda k: MLP(norm_fn=norm, **clean_cfg({x: y for x, y in cfg[k].items() if x != "norm_fn"}))
model = JEPA(vit_hf_from_config(**clean_cfg(cfg["encoder"])), ARPredictor(**clean_cfg(cfg["predictor"])), Embedder(**clean_cfg(cfg["action_encoder"])), mlp("projector"), mlp("pred_proj"))
state = torch.load(weights_path, map_location="cpu")
print("state_dict_top_level", type(state).__name__, "sample_keys", list(state)[:5])
result = model.load_state_dict(state, strict=False)
print("missing_keys", result.missing_keys)
print("unexpected_keys", result.unexpected_keys)
print("parameter_count", sum(p.numel() for p in model.parameters()))
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.eval().requires_grad_(False).to(device)
print("moved_to", device)
with torch.inference_mode():
    latent = model.encode({"pixels": torch.rand(1, 1, 3, 224, 224, device=device)})["emb"]
print("latent_shape", tuple(latent.shape), "latent_mean", float(latent.mean().cpu()))
