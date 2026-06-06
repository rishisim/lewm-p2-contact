#!/usr/bin/env python3
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from diagnose_pusht_latent_contacts import vit_hf_from_config
from jepa import JEPA
from module import ARPredictor, Embedder, MLP

def strip(d): return {k: v for k, v in d.items() if k != "_target_"}
root = Path(__file__).resolve().parent
cfg_path = Path(hf_hub_download("quentinll/lewm-pusht", "config.json", local_dir=root / ".cache" / "sanity_model"))
w_path = Path(hf_hub_download("quentinll/lewm-pusht", "weights.pt", local_dir=root / ".cache" / "sanity_model"))
cfg = json.loads(cfg_path.read_text())
encoder = vit_hf_from_config(**strip(cfg["encoder"]))
norm = torch.nn.BatchNorm1d if cfg["projector"]["norm_fn"]["_target_"].endswith("BatchNorm1d") else torch.nn.LayerNorm
mlp = lambda k: MLP(norm_fn=norm, **strip({x: y for x, y in cfg[k].items() if x != "norm_fn"}))
model = JEPA(encoder, ARPredictor(**strip(cfg["predictor"])), Embedder(**strip(cfg["action_encoder"])), mlp("projector"), mlp("pred_proj"))
state = torch.load(w_path, map_location="cpu")
result = model.load_state_dict(state, strict=False)
print("missing_keys", result.missing_keys)
print("unexpected_keys", result.unexpected_keys)
print("parameter_count", sum(p.numel() for p in model.parameters()))
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.eval().requires_grad_(False).to(device)
print("moved_to", device)
with torch.inference_mode():
    latent = model.encode({"pixels": torch.rand(1, 1, 3, 224, 224, device=device)})["emb"]
print("latent_shape", tuple(latent.shape))
