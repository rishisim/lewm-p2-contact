"""Small pretrained model and planning gates."""

import shutil
import tempfile
import time
from pathlib import Path

import torch
from stable_worldmodel.wm.loss import SIGReg

from .device import resolve_device
from .lewm import encode, load_lewm
from .paths import checkpoint_dir
from .planning import config_dict, make_solver, pusht_config
from .runs import create_run, write_metrics


def synthetic_info(device: torch.device) -> dict[str, torch.Tensor]:
    """Make one observation and goal in normalized image space."""
    return {
        "pixels": torch.randn(1, 1, 3, 224, 224, device=device),
        "goal": torch.randn(1, 1, 3, 224, 224, device=device),
        "action": torch.zeros(1, 1, 10, device=device),
    }


def _timed(checks: dict, name: str, callback) -> None:
    started = time.perf_counter()
    value = callback()
    checks[name] = {"seconds": round(time.perf_counter() - started, 4), **value}


def gates(requested_device: str = "auto") -> dict:
    """Run five small model correctness and performance gates."""
    device = resolve_device(requested_device)
    cfg = pusht_config(requested_device=requested_device)
    run_dir = create_run("gates", config_dict(cfg))
    checks: dict = {}
    state: dict = {}

    def load() -> dict:
        state["model"] = load_lewm(checkpoint_dir("lewm-pusht"), device)
        return {"status": "passed", "device": str(device)}

    _timed(checks, "load_pretrained", load)
    model = state["model"]

    def embedding() -> dict:
        with torch.inference_mode():
            cls, projected = encode(model, torch.randn(2, 3, 224, 224, device=device))
        assert cls.shape == projected.shape == (2, 192)
        return {"status": "passed", "cls_shape": list(cls.shape), "projected_shape": list(projected.shape)}

    _timed(checks, "encode", embedding)

    def plan() -> dict:
        solver = make_solver(model, cfg)
        solver.num_samples = 8
        solver.topk = 2
        solver.n_steps = 2
        output = solver.solve(synthetic_info(device))
        assert output["actions"].shape == (1, 5, 10)
        return {"status": "passed", "actions_shape": list(output["actions"].shape), "samples": 8, "iterations": 2}

    _timed(checks, "cem_plan", plan)

    def backward() -> dict:
        model.train().requires_grad_(True)
        batch = {
            "pixels": torch.randn(8, 4, 3, 224, 224, device=device),
            "action": torch.randn(8, 3, 10, device=device),
        }
        output = model.encode(batch)
        emb = output["emb"]
        pred = model.predict(emb[:, :3], output["act_emb"][:, :3])
        pred_loss = (pred - emb[:, 1:]).square().mean()
        sigreg_loss = SIGReg(knots=17, num_proj=1024).to(device)(emb.transpose(0, 1))
        loss = pred_loss + 0.09 * sigreg_loss
        loss.backward()
        assert torch.isfinite(loss).item()
        assert any(p.grad is not None for p in model.parameters())
        model.eval().requires_grad_(False)
        return {"status": "passed", "batch_size": 8, "loss": round(float(loss.detach().cpu()), 6)}

    _timed(checks, "forward_backward", backward)

    def parity() -> dict:
        with tempfile.TemporaryDirectory(dir=run_dir) as temporary:
            temp = Path(temporary)
            shutil.copyfile(checkpoint_dir("lewm-pusht") / "config.json", temp / "config.json")
            torch.save(model.state_dict(), temp / "weights.pt")
            reloaded = load_lewm(temp, device)
            max_diff = max(
                (a - b).abs().max().item()
                for a, b in zip(model.state_dict().values(), reloaded.state_dict().values())
                if a.is_floating_point()
            )
            assert max_diff == 0.0
        return {"status": "passed", "max_abs_diff": max_diff}

    _timed(checks, "save_reload_parity", parity)
    result = {"device": str(device), "checks": checks, "run_dir": str(run_dir)}
    write_metrics(run_dir, result)
    return result
