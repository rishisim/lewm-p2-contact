#!/usr/bin/env python3
"""Diagnose LeWM PushT latent prediction error around contact events.

This script intentionally bypasses eval.py. It downloads the Hugging Face
weights/config, instantiates the local jepa.JEPA class, reads PushT trajectories,
computes one-step latent prediction MSEs, aligns them with contact labels, and
writes plots plus a short summary.

The current HF dataset repo exposes PushT as one large compressed HDF5 artifact
(`pusht_expert_train.h5.zst`). A true small range-readable subset is therefore
not available from the repo today. The script first tries HF streaming if the
optional datasets package can make it work; otherwise use --dataset-h5 for a
local HDF5 file/subset, or opt into the full artifact download with
--allow-full-dataset-download. When the full artifact is used, the script
materializes a small local HDF5 subset for repeated diagnostic runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODEL_REPO = "quentinll/lewm-pusht"
DATASET_REPO = "quentinll/lewm-pusht"
DATASET_ARTIFACT = "pusht_expert_train.h5.zst"
HF_BASE_URL = "https://huggingface.co"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class Trajectory:
    episode_id: int
    pixels: Any
    actions: Any
    states: Any | None
    contact_blocks: Any
    contact_source: str


@dataclass
class PredictionRecord:
    episode_id: int
    model_step: int
    raw_step: int
    transition_block: int
    mse: float
    n_contacts: float
    rel_contact: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a PushT latent prediction/contact diagnostic for the LeWM "
            "Hugging Face checkpoint."
        )
    )
    parser.add_argument("--model-repo", default=MODEL_REPO)
    parser.add_argument("--dataset-repo", default=DATASET_REPO)
    parser.add_argument("--dataset-artifact", default=DATASET_ARTIFACT)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("LEWM_DIAG_CACHE", "~/.cache/lewm-pusht-diagnostic")).expanduser(),
        help="Cache directory for HF files and extracted/subset datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("diagnostics/pusht_latent_contacts"),
        help="Directory for PNG plots, CSV records, and summary.md.",
    )
    parser.add_argument(
        "--dataset-h5",
        type=Path,
        default=None,
        help="Optional local PushT HDF5 file or subset. Skips HF dataset download.",
    )
    parser.add_argument(
        "--allow-full-dataset-download",
        action="store_true",
        help=(
            "Download and decompress the full official HF dataset artifact if "
            "streaming/local data are unavailable. The compressed artifact is "
            "about 13 GB and the extracted HDF5 may be substantially larger."
        ),
    )
    parser.add_argument(
        "--delete-full-after-subset",
        action="store_true",
        help="Delete the full compressed/extracted dataset after making the subset.",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=30,
        help="Number of trajectories to evaluate. The requested diagnostic range is 20-50.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--max-model-steps",
        type=int,
        default=None,
        help="Optional cap on strided model steps per trajectory to reduce runtime.",
    )
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=64,
        help="Number of frames to encode per forward pass.",
    )
    parser.add_argument(
        "--predict-batch-size",
        type=int,
        default=256,
        help="Number of one-step prediction windows per predictor pass.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cpu", "cuda"),
        default="auto",
        help="Device for model inference. auto prefers MPS, then CUDA, then CPU.",
    )
    parser.add_argument(
        "--contact-window",
        type=int,
        default=30,
        help="Symmetric plot window, in model timesteps, around contact onset.",
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Redownload model/dataset artifacts even if cached files exist.",
    )
    parser.add_argument(
        "--prepare-dataset-only",
        action="store_true",
        help="Download/decompress/materialize the dataset subset, then exit before model inference.",
    )
    return parser.parse_args()


def vit_hf_from_config(
    size: str = "tiny",
    patch_size: int = 16,
    image_size: int = 224,
    pretrained: bool = False,
    use_mask_token: bool = True,
    **kwargs: Any,
) -> Any:
    """Build the same Hugging Face ViT used by stable_pretraining.vit_hf."""
    from transformers import ViTConfig, ViTModel

    size_configs = {
        "tiny": {"hidden_size": 192, "num_hidden_layers": 12, "num_attention_heads": 3},
        "small": {"hidden_size": 384, "num_hidden_layers": 12, "num_attention_heads": 6},
        "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12},
        "large": {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16},
        "huge": {"hidden_size": 1280, "num_hidden_layers": 32, "num_attention_heads": 16},
    }
    if size not in size_configs:
        raise ValueError(f"Unknown ViT size {size!r}")

    config_params = dict(size_configs[size])
    config_params["intermediate_size"] = config_params["hidden_size"] * 4
    config_params["image_size"] = image_size
    config_params["patch_size"] = patch_size
    config_params.update(kwargs)

    if pretrained:
        model = ViTModel.from_pretrained(
            f"google/vit-{size}-patch{patch_size}-{image_size}",
            add_pooling_layer=False,
            use_mask_token=use_mask_token,
        )
    else:
        config = ViTConfig(**config_params)
        model = ViTModel(config, add_pooling_layer=False, use_mask_token=use_mask_token)
    model.config.interpolate_pos_encoding = True
    return model


def print_step(message: str) -> None:
    print(f"[lewm-diagnostic] {message}", flush=True)


def validate_args(args: argparse.Namespace) -> None:
    if not (20 <= args.num_trajectories <= 50):
        print_step(
            f"warning: --num-trajectories={args.num_trajectories}; "
            "the requested diagnostic range is 20-50."
        )
    if args.frameskip <= 0:
        raise ValueError("--frameskip must be positive")
    if args.history_size <= 0:
        raise ValueError("--history-size must be positive")
    if args.encode_batch_size <= 0 or args.predict_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if args.max_model_steps is not None and args.max_model_steps <= args.history_size:
        raise ValueError("--max-model-steps must exceed --history-size")


def hf_download(
    repo_id: str,
    filename: str,
    repo_type: str,
    cache_dir: Path,
    *,
    force: bool = False,
) -> Path:
    local_dir = cache_dir / repo_type / repo_id.replace("/", "--")
    local_dir.mkdir(parents=True, exist_ok=True)
    dest = local_dir / filename
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return dest

    if dest.exists() and force:
        dest.unlink()

    try:
        from huggingface_hub import hf_hub_download

        if repo_type == "model":
            path = hf_hub_download(
                repo_id=repo_id,
                repo_type=None,
                filename=filename,
                local_dir=str(local_dir),
                force_download=force,
            )
            return Path(path)
    except Exception as exc:
        print_step(f"huggingface_hub download unavailable/failed ({exc}); falling back to urllib")

    prefix = "datasets/" if repo_type == "dataset" else ""
    url = f"{HF_BASE_URL}/{prefix}{repo_id}/resolve/main/{filename}"
    download_url(url, dest)
    return dest


def download_url(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".incomplete")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    resume_from = tmp.stat().st_size if tmp.exists() else 0
    headers = {}
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"
        print_step(f"resuming {dest.name} from {resume_from / 1024**3:.2f} GiB")
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request) as response:
        status = getattr(response, "status", None)
        if resume_from and status != 206:
            print_step("server did not honor Range; restarting download")
            resume_from = 0
        length = int(response.headers.get("Content-Length", "0") or 0)
        total = length + resume_from if resume_from else length
        try:
            from tqdm.auto import tqdm
        except Exception:
            tqdm = None

        mode = "ab" if resume_from else "wb"
        with tmp.open(mode) as f:
            if tqdm is None:
                copied = resume_from
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    copied += len(chunk)
                    if total:
                        print(f"\r{dest.name}: {copied / total:.1%}", end="", flush=True)
                if total:
                    print()
            else:
                with tqdm(
                    total=total or None,
                    initial=resume_from,
                    unit="B",
                    unit_scale=True,
                    desc=dest.name,
                ) as bar:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        bar.update(len(chunk))
    tmp.replace(dest)


def clean_cfg(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if k != "_target_"}


def instantiate_model(config_path: Path, weights_path: Path, device: Any) -> Any:
    import torch

    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP

    cfg = json.loads(config_path.read_text())
    encoder_cfg = clean_cfg(cfg["encoder"])
    encoder = vit_hf_from_config(**encoder_cfg)

    predictor = ARPredictor(**clean_cfg(cfg["predictor"]))
    action_encoder = Embedder(**clean_cfg(cfg["action_encoder"]))

    def make_mlp(key: str) -> MLP:
        mlp_cfg = clean_cfg(cfg[key])
        norm_cfg = mlp_cfg.pop("norm_fn", None)
        norm_fn = torch.nn.LayerNorm
        if isinstance(norm_cfg, dict):
            target = norm_cfg.get("_target_", "")
            if target.endswith("BatchNorm1d"):
                norm_fn = torch.nn.BatchNorm1d
        elif norm_cfg is None:
            norm_fn = None
        return MLP(norm_fn=norm_fn, **mlp_cfg)

    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=make_mlp("projector"),
        pred_proj=make_mlp("pred_proj"),
    )

    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    model.to(device)
    return model


def choose_device(name: str) -> Any:
    import torch

    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        print_step("MPS requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        print_step("CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(name)


def try_stream_hf_dataset(args: argparse.Namespace) -> list[Trajectory] | None:
    """Best-effort streaming path for future HF layouts with row-level files."""
    try:
        import numpy as np
        from datasets import load_dataset
    except Exception as exc:
        print_step(f"HF dataset streaming skipped: optional datasets package unavailable ({exc})")
        return None

    try:
        stream = load_dataset(args.dataset_repo, split="train", streaming=True)
    except Exception as exc:
        print_step(f"HF dataset streaming is not available for this repo layout ({exc})")
        return None

    episodes: list[dict[str, list[Any]]] = []
    current_ep = None
    current: dict[str, list[Any]] = {}
    ep_key = None
    required = {"pixels", "action"}

    try:
        for row in stream:
            if ep_key is None:
                ep_key = "episode_idx" if "episode_idx" in row else "ep_idx" if "ep_idx" in row else None
                if ep_key is None or not required.issubset(row.keys()):
                    print_step("HF stream rows do not expose episode_idx/ep_idx plus pixels/action")
                    return None
            row_ep = int(row[ep_key])
            if current_ep is None:
                current_ep = row_ep
            if row_ep != current_ep:
                episodes.append(current)
                if len(episodes) >= args.num_trajectories:
                    break
                current_ep = row_ep
                current = {}
            for key, value in row.items():
                current.setdefault(key, []).append(value)
        if current and len(episodes) < args.num_trajectories:
            episodes.append(current)
    except Exception as exc:
        print_step(f"HF dataset streaming failed while reading rows ({exc})")
        return None

    trajectories: list[Trajectory] = []
    for episode_id, ep in enumerate(episodes[: args.num_trajectories]):
        raw_contacts = np.asarray(ep["n_contacts"]) if "n_contacts" in ep else None
        trajectories.append(
            trajectory_from_raw_episode(
                episode_id=episode_id,
                pixels=np.asarray(ep["pixels"]),
                actions=np.asarray(ep["action"]),
                states=np.asarray(ep["state"]) if "state" in ep else None,
                raw_contacts=raw_contacts,
                args=args,
                contact_prefix="hf-stream",
            )
        )
    return trajectories or None


def trajectory_from_raw_episode(
    *,
    episode_id: int,
    pixels: Any,
    actions: Any,
    states: Any | None,
    raw_contacts: Any | None,
    args: argparse.Namespace,
    contact_prefix: str,
) -> Trajectory:
    import numpy as np

    raw_len = min(len(pixels), len(actions))
    if states is not None:
        raw_len = min(raw_len, len(states))
    if raw_contacts is not None:
        raw_len = min(raw_len, len(raw_contacts))
    if args.max_model_steps is not None:
        raw_len = min(raw_len, (args.max_model_steps - 1) * args.frameskip + 1)

    model_steps = 1 + (raw_len - 1) // args.frameskip
    if model_steps <= args.history_size:
        raise RuntimeError(
            f"Episode {episode_id} is too short after frameskip={args.frameskip}: "
            f"{model_steps} model steps"
        )

    obs_idx = np.arange(model_steps) * args.frameskip
    action_rows = (model_steps - 1) * args.frameskip
    pixels_strided = pixels[obs_idx]
    actions_blocked = actions[:action_rows].reshape(model_steps - 1, -1).astype(np.float32)
    states_strided = states[obs_idx] if states is not None else None

    if raw_contacts is not None:
        contacts = contact_blocks_from_raw(
            raw_contacts[:action_rows],
            args.frameskip,
            expected_blocks=model_steps - 1,
        )
        source = f"{contact_prefix}:n_contacts"
    else:
        contacts = contact_blocks_from_state(states_strided, args.frameskip)
        source = f"{contact_prefix}:geometry-from-state"

    return Trajectory(
        episode_id=episode_id,
        pixels=pixels_strided,
        actions=actions_blocked,
        states=states_strided,
        contact_blocks=contacts,
        contact_source=source,
    )


def ensure_dataset_h5(args: argparse.Namespace) -> Path:
    if args.dataset_h5 is not None:
        path = args.dataset_h5.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"--dataset-h5 not found: {path}")
        return path

    subset_path = (
        args.cache_dir
        / "dataset"
        / f"pusht_subset_n{args.num_trajectories}_fs{args.frameskip}_seed{args.seed}.h5"
    )
    if subset_path.exists() and subset_path.stat().st_size > 0 and not args.force_redownload:
        return subset_path

    if not args.allow_full_dataset_download:
        raise RuntimeError(
            "The HF dataset repo currently contains only the large compressed "
            f"artifact {args.dataset_artifact!r}; it cannot provide a true "
            "small random-access subset over HTTP. Re-run with --dataset-h5 "
            "pointing to a local/subset .h5, or opt into the full artifact with "
            "--allow-full-dataset-download."
        )

    zst_path = hf_download(
        args.dataset_repo,
        args.dataset_artifact,
        "dataset",
        args.cache_dir,
        force=args.force_redownload,
    )
    full_h5 = decompress_dataset_artifact(zst_path, args.cache_dir / "dataset")
    materialize_subset_h5(
        full_h5=full_h5,
        subset_h5=subset_path,
        num_trajectories=args.num_trajectories,
        seed=args.seed,
        frameskip=args.frameskip,
        history_size=args.history_size,
        max_model_steps=args.max_model_steps,
        force=args.force_redownload,
    )

    if args.delete_full_after_subset:
        for path in (zst_path, full_h5):
            try:
                path.unlink()
                print_step(f"deleted {path}")
            except FileNotFoundError:
                pass

    return subset_path


def decompress_dataset_artifact(artifact_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if artifact_path.name.endswith(".h5.zst"):
        h5_path = out_dir / artifact_path.name[: -len(".zst")]
        if h5_path.exists() and h5_path.stat().st_size > 0:
            return h5_path
        print_step(f"decompressing {artifact_path.name} to {h5_path}")
        decompress_zst(artifact_path, h5_path)
        return h5_path

    if artifact_path.name.endswith(".tar.zst"):
        extract_dir = out_dir / artifact_path.name[: -len(".tar.zst")]
        extract_dir.mkdir(parents=True, exist_ok=True)
        if list(extract_dir.glob("*.h5")):
            return sorted(extract_dir.glob("*.h5"))[0]
        if shutil.which("tar") is None:
            raise RuntimeError("tar is required to extract .tar.zst dataset artifacts")
        subprocess.run(
            ["tar", "--zstd", "-xf", str(artifact_path), "-C", str(extract_dir)],
            check=True,
        )
        h5_files = sorted(extract_dir.rglob("*.h5"))
        if not h5_files:
            raise FileNotFoundError(f"no .h5 found after extracting {artifact_path}")
        return h5_files[0]

    raise ValueError(f"Unsupported dataset artifact format: {artifact_path}")


def decompress_zst(src: Path, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".incomplete")
    if shutil.which("zstd") is not None:
        subprocess.run(["zstd", "-d", "-f", "-o", str(tmp), str(src)], check=True)
        tmp.replace(dest)
        return

    try:
        import zstandard as zstd
        from tqdm.auto import tqdm
    except Exception as exc:
        raise RuntimeError(
            "Need either the zstd CLI or the Python zstandard package to "
            f"decompress {src}"
        ) from exc

    total = src.stat().st_size
    dctx = zstd.ZstdDecompressor()
    with src.open("rb") as fin, tmp.open("wb") as fout, tqdm(
        total=total, unit="B", unit_scale=True, desc=src.name
    ) as bar:
        reader = dctx.stream_reader(fin)
        while True:
            chunk = reader.read(1024 * 1024)
            if not chunk:
                break
            fout.write(chunk)
            bar.update(len(chunk))
    tmp.replace(dest)


def h5_episode_layout(h5: Any) -> tuple[Any, Any]:
    import numpy as np

    if "ep_len" in h5 and "ep_offset" in h5:
        return h5["ep_len"][:].astype(np.int64), h5["ep_offset"][:].astype(np.int64)

    ep_key = "episode_idx" if "episode_idx" in h5 else "ep_idx" if "ep_idx" in h5 else None
    if ep_key is None:
        raise KeyError("HDF5 file needs ep_len/ep_offset or episode_idx/ep_idx")
    ep = h5[ep_key][:]
    unique, starts, counts = np.unique(ep, return_index=True, return_counts=True)
    order = np.argsort(starts)
    return counts[order].astype(np.int64), starts[order].astype(np.int64)


def materialize_subset_h5(
    *,
    full_h5: Path,
    subset_h5: Path,
    num_trajectories: int,
    seed: int,
    frameskip: int,
    history_size: int,
    max_model_steps: int | None,
    force: bool,
) -> None:
    try:
        import hdf5plugin  # noqa: F401
    except Exception:
        pass
    import h5py
    import numpy as np

    if subset_h5.exists() and subset_h5.stat().st_size > 0 and not force:
        return
    subset_h5.parent.mkdir(parents=True, exist_ok=True)
    tmp = subset_h5.with_suffix(".h5.incomplete")
    if tmp.exists():
        tmp.unlink()

    print_step(f"materializing {num_trajectories} trajectory subset at {subset_h5}")
    with h5py.File(full_h5, "r") as src:
        lengths, offsets = h5_episode_layout(src)
        min_raw_len = history_size * frameskip + 1
        valid = np.flatnonzero(lengths >= min_raw_len)
        if len(valid) < num_trajectories:
            raise RuntimeError(
                f"Only {len(valid)} episodes are long enough for history_size={history_size}, "
                f"frameskip={frameskip}; need {num_trajectories}."
            )
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(valid, size=num_trajectories, replace=False))

        raw_lengths = []
        for ep in selected:
            length = int(lengths[ep])
            if max_model_steps is not None:
                length = min(length, (max_model_steps - 1) * frameskip + 1)
            raw_lengths.append(length)
        raw_lengths_arr = np.asarray(raw_lengths, dtype=np.int64)
        raw_offsets_arr = np.concatenate(([0], np.cumsum(raw_lengths_arr[:-1]))).astype(np.int64)
        total_rows = int(raw_lengths_arr.sum())

        keys = [k for k in ("pixels", "action", "state", "n_contacts", "proprio") if k in src]
        missing = [k for k in ("pixels", "action") if k not in keys]
        if missing:
            raise KeyError(f"Full HDF5 dataset missing required keys: {missing}")

        with h5py.File(tmp, "w") as dst:
            dst.create_dataset("ep_len", data=raw_lengths_arr)
            dst.create_dataset("ep_offset", data=raw_offsets_arr)
            datasets = {}
            for key in keys:
                shape = (total_rows, *src[key].shape[1:])
                kwargs = {}
                if key == "pixels":
                    kwargs.update(chunks=True, compression="lzf")
                datasets[key] = dst.create_dataset(key, shape=shape, dtype=src[key].dtype, **kwargs)

            write_pos = 0
            for ep, raw_len in zip(selected, raw_lengths):
                start = int(offsets[ep])
                end = start + int(raw_len)
                for key, ds in datasets.items():
                    ds[write_pos : write_pos + raw_len] = src[key][start:end]
                write_pos += raw_len

    tmp.replace(subset_h5)


def contact_blocks_from_raw(raw_contacts: Any, frameskip: int, expected_blocks: int | None = None) -> Any:
    import numpy as np

    contacts = np.asarray(raw_contacts)
    if contacts.ndim > 1:
        contacts = contacts.reshape(contacts.shape[0], -1).max(axis=1)
    usable = (len(contacts) // frameskip) * frameskip
    if expected_blocks is not None:
        usable = min(usable, expected_blocks * frameskip)
    blocks = contacts[:usable].reshape(-1, frameskip).max(axis=1)
    if expected_blocks is not None and len(blocks) < expected_blocks:
        blocks = np.pad(blocks, (0, expected_blocks - len(blocks)), constant_values=0)
    return blocks.astype(np.float32)


def contact_blocks_from_state(states: Any, frameskip: int) -> Any:
    import numpy as np

    if states is None:
        raise KeyError("Cannot compute geometric contacts without state")
    contacts_obs = geometric_contacts_from_state(np.asarray(states))
    if len(contacts_obs) < 2:
        return np.zeros(0, dtype=np.float32)
    blocks = np.maximum(contacts_obs[:-1], contacts_obs[1:])
    return blocks.astype(np.float32)


def geometric_contacts_from_state(states: Any, *, agent_radius: float = 15.0, margin: float = 2.0) -> Any:
    """Approximate PushT circle-vs-T contact from 7D state.

    The official dataset usually contains n_contacts; this fallback is only
    used when that column is absent. The block is approximated as the default
    T shape: two axis-aligned local rectangles rotated by block_angle.
    """
    import numpy as np

    s = np.asarray(states, dtype=np.float64)
    agent = s[:, 0:2]
    block = s[:, 2:4]
    angle = s[:, 4]
    delta = agent - block
    c = np.cos(-angle)
    si = np.sin(-angle)
    x = c * delta[:, 0] - si * delta[:, 1]
    y = si * delta[:, 0] + c * delta[:, 1]

    rects = [
        (-60.0, 60.0, -45.0, -15.0),
        (-15.0, 15.0, -15.0, 75.0),
    ]
    min_dist = np.full(len(s), np.inf, dtype=np.float64)
    for xmin, xmax, ymin, ymax in rects:
        dx = np.maximum(np.maximum(xmin - x, 0.0), x - xmax)
        dy = np.maximum(np.maximum(ymin - y, 0.0), y - ymax)
        min_dist = np.minimum(min_dist, np.sqrt(dx * dx + dy * dy))
    return min_dist <= (agent_radius + margin)


def load_h5_trajectories(args: argparse.Namespace, h5_path: Path) -> list[Trajectory]:
    try:
        import hdf5plugin  # noqa: F401
    except Exception:
        pass
    import h5py
    import numpy as np

    trajectories: list[Trajectory] = []
    with h5py.File(h5_path, "r") as h5:
        for key in ("pixels", "action"):
            if key not in h5:
                raise KeyError(f"{h5_path} missing required dataset {key!r}")
        lengths, offsets = h5_episode_layout(h5)
        min_raw_len = args.history_size * args.frameskip + 1
        valid = np.flatnonzero(lengths >= min_raw_len)
        if len(valid) < args.num_trajectories:
            raise RuntimeError(
                f"{h5_path} has {len(valid)} usable episodes, need {args.num_trajectories}"
            )
        rng = np.random.default_rng(args.seed)
        selected = np.sort(rng.choice(valid, size=args.num_trajectories, replace=False))

        for ep in selected:
            start = int(offsets[ep])
            raw_len = int(lengths[ep])
            if args.max_model_steps is not None:
                raw_len = min(raw_len, (args.max_model_steps - 1) * args.frameskip + 1)

            model_steps = 1 + (raw_len - 1) // args.frameskip
            if model_steps <= args.history_size:
                continue
            obs_rows = start + np.arange(model_steps) * args.frameskip
            action_rows = (model_steps - 1) * args.frameskip

            pixels = h5["pixels"][obs_rows]
            actions_raw = h5["action"][start : start + action_rows]
            actions = actions_raw.reshape(model_steps - 1, -1).astype(np.float32)
            states = h5["state"][obs_rows] if "state" in h5 else None

            if "n_contacts" in h5:
                raw_contacts = h5["n_contacts"][start : start + action_rows]
                contacts = contact_blocks_from_raw(
                    raw_contacts,
                    args.frameskip,
                    expected_blocks=model_steps - 1,
                )
                contact_source = "n_contacts"
            else:
                contacts = contact_blocks_from_state(states, args.frameskip)
                contact_source = "geometry-from-state"

            trajectories.append(
                Trajectory(
                    episode_id=int(ep),
                    pixels=pixels,
                    actions=actions,
                    states=states,
                    contact_blocks=contacts,
                    contact_source=contact_source,
                )
            )

    return trajectories


def preprocess_pixels(pixels: Any, device: Any, image_size: int) -> Any:
    import torch
    import torch.nn.functional as F

    x = torch.as_tensor(pixels)
    if x.ndim != 4:
        raise ValueError(f"Expected pixels as T,H,W,C or T,C,H,W, got {tuple(x.shape)}")
    if x.shape[-1] in (1, 3):
        x = x.permute(0, 3, 1, 2)
    elif x.shape[1] not in (1, 3):
        raise ValueError(f"Cannot infer channel axis for pixel shape {tuple(x.shape)}")

    try:
        from torchvision.transforms import v2 as transforms

        x = transforms.ToDtype(torch.float32, scale=True)(x)
        x = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(x)
        x = transforms.Resize(size=image_size)(x)
    except Exception:
        x = x.float()
        if x.numel() and x.max() > 2:
            x = x / 255.0
        mean = torch.tensor(IMAGENET_MEAN, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x - mean) / std
        if x.shape[-2:] != (image_size, image_size):
            x = F.interpolate(
                x,
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
    return x.to(device)


def encode_frames(model: Any, pixels: Any, device: Any, batch_size: int, image_size: int) -> Any:
    import torch

    chunks = []
    with torch.inference_mode():
        for start in range(0, len(pixels), batch_size):
            batch = preprocess_pixels(pixels[start : start + batch_size], device, image_size)
            info = model.encode({"pixels": batch.unsqueeze(0)})
            chunks.append(info["emb"].squeeze(0).detach().cpu())
    return torch.cat(chunks, dim=0)


def compute_prediction_records(
    model: Any,
    trajectory: Trajectory,
    *,
    device: Any,
    history_size: int,
    encode_batch_size: int,
    predict_batch_size: int,
    image_size: int,
    frameskip: int,
) -> list[PredictionRecord]:
    import numpy as np
    import torch

    embeddings = encode_frames(model, trajectory.pixels, device, encode_batch_size, image_size)
    actions = torch.as_tensor(trajectory.actions, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.inference_mode():
        act_emb = model.action_encoder(actions).squeeze(0).detach().cpu()

    num_steps = embeddings.shape[0]
    if num_steps <= history_size:
        return []

    target_steps = np.arange(history_size, num_steps, dtype=np.int64)
    contact_blocks = np.asarray(trajectory.contact_blocks)
    contact_present = contact_blocks > 0
    onset = int(np.argmax(contact_present)) if np.any(contact_present) else None

    records: list[PredictionRecord] = []
    with torch.inference_mode():
        for batch_start in range(0, len(target_steps), predict_batch_size):
            batch_targets = target_steps[batch_start : batch_start + predict_batch_size]
            emb_windows = []
            act_windows = []
            for target in batch_targets:
                start = int(target) - history_size
                emb_windows.append(embeddings[start:target])
                act_windows.append(act_emb[start:target])

            emb_tensor = torch.stack(emb_windows, dim=0).to(device)
            act_tensor = torch.stack(act_windows, dim=0).to(device)
            gt = embeddings[batch_targets].to(device)
            pred = model.predict(emb_tensor, act_tensor)[:, -1]
            mse = ((pred - gt) ** 2).mean(dim=1).detach().cpu().numpy()

            for target, err in zip(batch_targets, mse):
                block_idx = int(target) - 1
                n_contact = float(contact_blocks[block_idx]) if block_idx < len(contact_blocks) else 0.0
                rel = float("nan") if onset is None else float(block_idx - onset)
                records.append(
                    PredictionRecord(
                        episode_id=trajectory.episode_id,
                        model_step=int(target),
                        raw_step=int(target) * frameskip,
                        transition_block=block_idx,
                        mse=float(err),
                        n_contacts=n_contact,
                        rel_contact=rel,
                    )
                )
    return records


def run_all_predictions(
    model: Any,
    trajectories: list[Trajectory],
    args: argparse.Namespace,
    device: Any,
) -> list[PredictionRecord]:
    records: list[PredictionRecord] = []
    for i, traj in enumerate(trajectories, start=1):
        print_step(
            f"trajectory {i}/{len(trajectories)} ep={traj.episode_id} "
            f"steps={len(traj.pixels)} contact={traj.contact_source}"
        )
        records.extend(
            compute_prediction_records(
                model,
                traj,
                device=device,
                history_size=args.history_size,
                encode_batch_size=args.encode_batch_size,
                predict_batch_size=args.predict_batch_size,
                image_size=args.image_size,
                frameskip=args.frameskip,
            )
        )
    return records


def write_records_csv(records: list[PredictionRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode_id",
                "model_step",
                "raw_step",
                "transition_block",
                "mse",
                "n_contacts",
                "rel_contact",
            ],
        )
        writer.writeheader()
        for rec in records:
            writer.writerow(rec.__dict__)


def plot_by_contact_timestep(records: list[PredictionRecord], path: Path, window: int) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rel = np.asarray([r.rel_contact for r in records], dtype=np.float64)
    mse = np.asarray([r.mse for r in records], dtype=np.float64)
    mask = np.isfinite(rel) & (rel >= -window) & (rel <= window)

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160)
    if np.any(mask):
        xs = np.arange(-window, window + 1)
        means = np.full_like(xs, np.nan, dtype=np.float64)
        sems = np.full_like(xs, np.nan, dtype=np.float64)
        counts = np.zeros_like(xs, dtype=np.int64)
        rel_int = rel[mask].astype(np.int64)
        mse_masked = mse[mask]
        for i, x in enumerate(xs):
            vals = mse_masked[rel_int == x]
            counts[i] = len(vals)
            if len(vals):
                means[i] = vals.mean()
                sems[i] = vals.std(ddof=1) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
        valid = counts > 0
        ax.plot(xs[valid], means[valid], color="#2563eb", linewidth=2.0, label="mean MSE")
        ax.fill_between(
            xs[valid],
            means[valid] - sems[valid],
            means[valid] + sems[valid],
            color="#93c5fd",
            alpha=0.45,
            linewidth=0,
            label="SEM",
        )
        ax.axvline(0, color="#111827", linestyle="--", linewidth=1.2, label="contact onset")
        ax.set_xlim(-window, window)
        ax.legend(frameon=False)
    else:
        ax.text(
            0.5,
            0.5,
            "No contact onset observed in selected trajectories",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set_title("LeWM PushT One-Step Latent Error Around Contact")
    ax.set_xlabel("Model timestep relative to first contact transition")
    ax.set_ylabel("Next-latent MSE")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_contact_split(records: list[PredictionRecord], path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    mse_zero = np.asarray([r.mse for r in records if r.n_contacts == 0], dtype=np.float64)
    mse_contact = np.asarray([r.mse for r in records if r.n_contacts > 0], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(6.6, 4.8), dpi=160)
    groups = []
    labels = []
    if len(mse_zero):
        groups.append(mse_zero)
        labels.append(f"n_contacts = 0\nn={len(mse_zero)}")
    if len(mse_contact):
        groups.append(mse_contact)
        labels.append(f"n_contacts > 0\nn={len(mse_contact)}")

    if groups:
        ax.violinplot(groups, showmeans=True, showextrema=True)
        ax.set_xticks(range(1, len(labels) + 1), labels)
    else:
        ax.text(0.5, 0.5, "No prediction records", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("LeWM PushT One-Step Latent Error by Contact")
    ax.set_ylabel("Next-latent MSE")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def summarize(records: list[PredictionRecord], trajectories: list[Trajectory], args: argparse.Namespace, device: Any) -> str:
    import numpy as np

    mse = np.asarray([r.mse for r in records], dtype=np.float64)
    zero = np.asarray([r.mse for r in records if r.n_contacts == 0], dtype=np.float64)
    contact = np.asarray([r.mse for r in records if r.n_contacts > 0], dtype=np.float64)
    contact_trajs = sum(np.any(np.asarray(t.contact_blocks) > 0) for t in trajectories)
    sources = sorted(set(t.contact_source for t in trajectories))

    def fmt(arr: Any) -> str:
        if len(arr) == 0:
            return "n/a"
        return f"{arr.mean():.6g} +/- {arr.std(ddof=1):.6g}" if len(arr) > 1 else f"{arr.mean():.6g}"

    lines = [
        "# PushT LeWM Latent Contact Diagnostic",
        "",
        f"- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Model repo: `{args.model_repo}`",
        f"- Dataset repo: `{args.dataset_repo}`",
        f"- Device: `{device}`",
        f"- Trajectories evaluated: {len(trajectories)}",
        f"- Trajectories with contact onset: {contact_trajs}",
        f"- Prediction records: {len(records)}",
        f"- Contact source(s): {', '.join(sources)}",
        f"- Frameskip: {args.frameskip}",
        f"- History size: {args.history_size}",
        "",
        "## Error Summary",
        "",
        f"- Overall next-latent MSE mean +/- std: {fmt(mse)}",
        f"- `n_contacts == 0`: {fmt(zero)} across {len(zero)} predictions",
        f"- `n_contacts > 0`: {fmt(contact)} across {len(contact)} predictions",
    ]
    if len(zero) and len(contact):
        lines.append(f"- Contact/non-contact mean ratio: {contact.mean() / max(zero.mean(), 1e-12):.4g}")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `prediction_errors.csv`",
            "- `prediction_error_by_contact_timestep.png`",
            "- `prediction_error_contact_split.png`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    args = parse_args()
    validate_args(args)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch

    device = choose_device(args.device)
    print_step(f"using device: {device}")

    print_step("downloading/loading HF checkpoint files")
    config_path = hf_download(args.model_repo, "config.json", "model", args.cache_dir, force=args.force_redownload)
    weights_path = hf_download(args.model_repo, "weights.pt", "model", args.cache_dir, force=args.force_redownload)
    model = instantiate_model(config_path, weights_path, device)

    print_step("loading trajectory subset")
    trajectories = try_stream_hf_dataset(args)
    if trajectories is None:
        h5_path = ensure_dataset_h5(args)
        if args.prepare_dataset_only:
            print_step(f"dataset subset ready: {h5_path}")
            return
        print_step(f"reading HDF5 trajectories from {h5_path}")
        trajectories = load_h5_trajectories(args, h5_path)

    if len(trajectories) == 0:
        raise RuntimeError("No trajectories were loaded")

    try:
        records = run_all_predictions(model, trajectories, args, device)
    except RuntimeError as exc:
        if getattr(device, "type", str(device)) == "mps":
            print_step(f"MPS run failed ({exc}); falling back to CPU")
            device = torch.device("cpu")
            model.to(device)
            records = run_all_predictions(model, trajectories, args, device)
        else:
            raise

    if not records:
        raise RuntimeError("No prediction records were produced")

    csv_path = args.output_dir / "prediction_errors.csv"
    rel_plot = args.output_dir / "prediction_error_by_contact_timestep.png"
    split_plot = args.output_dir / "prediction_error_contact_split.png"
    summary_path = args.output_dir / "summary.md"

    print_step("writing plots and summary")
    write_records_csv(records, csv_path)
    plot_by_contact_timestep(records, rel_plot, args.contact_window)
    plot_contact_split(records, split_plot)
    summary_path.write_text(summarize(records, trajectories, args, device))

    print_step(f"done: {summary_path}")


if __name__ == "__main__":
    main()
