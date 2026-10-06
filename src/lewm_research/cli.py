"""Command-line entry points for the role-swap probe."""

import json
from pathlib import Path

import typer

app = typer.Typer(help="LeWM role-swap research tools.")


@app.command("normalization")
def normalization_cmd() -> None:
    """Fit and persist pretrained PushT action/proprio normalization."""
    from .normalization import create_pretrained_normalization

    typer.echo(json.dumps(create_pretrained_normalization(), indent=2))


@app.command("finetune")
def finetune_cmd(
    dataset: str = typer.Option(..., help="HDF5 dataset name under the work root."),
    name: str = typer.Option(..., help="Output checkpoint name."),
    init: str = typer.Option("lewm-pusht", help="Initialization checkpoint name."),
    max_steps: int = typer.Option(0, min=0),
    batch_size: int = typer.Option(32, min=1, help="Default 32 deviates from upstream 128, which exceeds this Mac\'s memory; configurable."),
    seed: int = typer.Option(0),
    device: str = typer.Option("auto"),
    log_interval: int = typer.Option(1, min=1),
    val_interval: int = typer.Option(100, min=1),
    checkpoint_interval: int = typer.Option(100, min=1),
    workers: int = typer.Option(4, min=0),
    schedule_steps: int | None = typer.Option(None, min=2, help="Immutable LR budget; defaults to max-steps. Set on both initial and resumed runs."),
    precision: str = typer.Option("auto", help="auto (bf16 on CUDA, fp32 elsewhere), fp32, or bf16."),
) -> None:
    """Fine-tune LeWM with episode-held-out validation; resume by name."""
    from .train.finetune import finetune

    typer.echo(json.dumps(finetune(dataset, name, init, max_steps, batch_size, seed, device,
                                  log_interval, val_interval, checkpoint_interval, workers, schedule_steps,
                                  precision), indent=2))


@app.command("bench-train")
def bench_train_cmd(
    dataset: str = typer.Option(...),
    device: str = typer.Option("auto"),
    steps: int = typer.Option(20, min=1),
    workers: int = typer.Option(4, min=0),
) -> None:
    """Benchmark optimizer steps at batch sizes 32, 64, and 128."""
    from .train.finetune import bench_train

    typer.echo(json.dumps(bench_train(dataset, device, steps, workers), indent=2))


@app.command("collect")
def collect_cmd(
    name: str = typer.Option(..., help="Dataset name below the work-root datasets directory."),
    episodes: int = typer.Option(..., min=1),
    steps: int = typer.Option(100, min=1),
    policy: str = typer.Option("block", help="block or mixed."),
    seed: int = typer.Option(0),
    workers: int = typer.Option(1, min=1),
    placement: str = typer.Option("clutter", help="clutter or uniform peg placement."),
) -> None:
    """Collect fixed-length PushT-Peg episodes into compressed HDF5."""
    from .data.collect import collect

    typer.echo(json.dumps(collect(name, episodes, steps, policy, seed, workers, placement), indent=2))


@app.command("smoke-eval")
def smoke_eval(
    num_eval: int = typer.Option(50, min=1, help="Number of PushT evaluations."),
    video: bool = typer.Option(False, "--video/--no-video", help="Save evaluation videos."),
    device: str = typer.Option("auto", help="auto, cuda, mps, or cpu."),
) -> None:
    """Run upstream-style pretrained PushT evaluation."""
    from .smoke import smoke_eval as evaluate

    try:
        typer.echo(json.dumps(evaluate(num_eval, video, device), indent=2))
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


@app.command()
def gates(device: str = typer.Option("auto", help="auto, cuda, mps, or cpu.")) -> None:
    """Check model load, encode, CEM, training loss, and reload parity."""
    from .gates import gates as run_gates

    typer.echo(json.dumps(run_gates(device), indent=2))


@app.command("bench-sim")
def bench_sim(device: str = typer.Option("auto", help="auto, cuda, mps, or cpu.")) -> None:
    """Benchmark PushT simulation and upstream CEM planning latency."""
    from .benchmark import bench_sim as run_benchmark

    typer.echo(json.dumps(run_benchmark(device), indent=2))


@app.command("probe-eval")
def probe_eval_cmd(
    arm: str = typer.Option(...), bases: str = typer.Option(...),
    conditions: str = typer.Option("all"), seed: int = typer.Option(42),
    budget: int = typer.Option(50), displacement_min: int = typer.Option(40),
    displacement_max: int = typer.Option(100),
    n: int = typer.Option(100, min=1), workers: int = typer.Option(4, min=1),
    population: int | None = typer.Option(None, min=2),
    iterations: int | None = typer.Option(None, min=1),
    topk: int | None = typer.Option(None, min=2), device: str = typer.Option("auto"),
    run_dir: str | None = typer.Option(None, help="Existing run directory to resume."),
    feasibility_run: str | None = typer.Option(None, help="Reference run under separate feasibility seed F."),
    approach_weight: float = typer.Option(0, min=0),
    with_target: bool | None = typer.Option(None, help="Fixed training-data T decoration; default per arm."),
    normalization: str | None = typer.Option(None, help="Frozen normalization JSON shared by all arms."),
) -> None:
    """Run matched fixed-budget episodes, resuming a specified run directory."""
    from .probe.rollout_eval import evaluate

    if seed >= 1_000_000_000:
        raise typer.BadParameter("evaluation seeds must be outside the reserved pilot range")
    typer.echo(json.dumps(evaluate(arm, bases, conditions, seed, n, run_dir,
                                  feasibility_run=feasibility_run, budget=budget,
                                  displacement_range=(displacement_min, displacement_max),
                                  workers=workers, population=population, iterations=iterations,
                                  topk=topk, device=device, approach_weight=approach_weight,
                                  with_target=with_target,
                                  normalization=json.loads(Path(normalization).read_text())
                                  if normalization else None), indent=2))


@app.command("probe-bases")
def probe_bases_cmd(n: int = typer.Option(400, min=1),
                    seed: int = typer.Option(300000000),
                    near_path_distance: float = typer.Option(55),
                    run_dir: str = typer.Option(...)) -> None:
    """Persist frozen matched construction; reject incompatible resumes."""
    from .main import prepare_bases
    typer.echo(str(prepare_bases(n, seed, near_path_distance, run_dir)))


@app.command("probe-pilot")
def probe_pilot_cmd(
    checkpoint: str = typer.Option("lewm-pusht"), workers: int = typer.Option(4, min=1),
    seed: int = typer.Option(1_000_000_000), n: int = typer.Option(10, min=1),
    reference_population: int = typer.Option(100, min=2),
    reference_iterations: int = typer.Option(10, min=1),
    lewm_population: int = typer.Option(300, min=2), lewm_iterations: int = typer.Option(30, min=1),
    device: str = typer.Option("auto"), run_dir: str | None = typer.Option(None),
) -> None:
    """Measure independent pilot timing, feasibility, and paired discordance."""
    from .probe.pilot import pilot

    typer.echo(json.dumps(pilot(checkpoint, workers, seed, n, run_dir,
                                reference_population, reference_iterations,
                                lewm_population, lewm_iterations, device), indent=2))


@app.command("probe-fidelity")
def probe_fidelity_cmd(device: str = typer.Option("auto"),
                       dataset_images: bool = typer.Option(False)) -> None:
    """Validate the controlled loop on smoke-eval's 50 no-peg expert pairs."""
    from .probe.fidelity import fidelity
    typer.echo(json.dumps(fidelity(device=device, dataset_images=dataset_images), indent=2))


@app.command("probe-calibrate")
def probe_calibrate_cmd(workers: int = typer.Option(8, min=1),
                        device: str = typer.Option("auto"),
                        run_dir: str | None = typer.Option(None)) -> None:
    """Run the independent 20-base difficulty grid and chosen pretrained arm."""
    from .probe.calibration import grid
    typer.echo(json.dumps(grid(workers=workers, device=device, run_dir=run_dir), indent=2))


@app.command("probe-readout")
def probe_readout_cmd(
    bases: str = typer.Option(...),
    checkpoints: str = typer.Option("lewm-pusht,ft_block_s0,ft_block_s1,ft_mixed_s0,ft_mixed_s1"),
    frames_per_dataset: int = typer.Option(4000, min=1), seed: int = typer.Option(42),
    device: str = typer.Option("auto"), batch_size: int = typer.Option(64, min=1),
    outer_folds: int = typer.Option(5, min=2), inner_folds: int = typer.Option(3, min=2),
    bootstrap_samples: int = typer.Option(2000, min=1), run_dir: str | None = typer.Option(None),
) -> None:
    """Nested grouped linear readouts on one common held-out frame pool."""
    from .probe.readout import run_readout
    result = run_readout(bases, checkpoints.split(","), frames_per_dataset, seed, run_dir,
                         device, batch_size, outer_folds, inner_folds, bootstrap_samples)
    typer.echo(json.dumps(result, indent=2))


@app.command("probe-abc")
def probe_abc_cmd(
    bases: str = typer.Option(...),
    checkpoints: str = typer.Option("lewm-pusht,ft_block_s0,ft_block_s1,ft_mixed_s0,ft_mixed_s1"),
    conditions: str = typer.Option("all"), n: int = typer.Option(50, min=1),
    seed: int = typer.Option(42), device: str = typer.Option("auto"),
    workers: int = typer.Option(4, min=1), population: int = typer.Option(300, min=16),
    iterations: int = typer.Option(30, min=1), topk: int = typer.Option(30, min=2),
    approach_weight: float = typer.Option(.1, min=0), batch_size: int = typer.Option(16, min=1),
    feasibility_run: str | None = typer.Option(None), bootstrap_samples: int = typer.Option(2000, min=1),
    run_dir: str | None = typer.Option(None),
) -> None:
    """Persist one physical bank per base/condition, then score every checkpoint."""
    from .probe.abc import run_abc
    result = run_abc(bases, checkpoints.split(","), conditions, n, seed, run_dir, device,
                     workers, population, iterations, topk, approach_weight, batch_size,
                     feasibility_run, bootstrap_samples)
    typer.echo(json.dumps(result, indent=2))


@app.command("probe-report")
def probe_report_cmd(
    runs: list[str] = typer.Option(..., help="Repeat --runs for each eval/readout/ABC directory; comma-separated paths also accepted."),
    feasibility_run: str | None = typer.Option(None), seed: int = typer.Option(42),
    bootstrap_samples: int = typer.Option(2000, min=1), run_dir: str | None = typer.Option(None),
    publish: bool = typer.Option(False, help="Copy the two compact result files into experiments/role_swap/results/."),
    coverage_underpowered: bool = typer.Option(False, help="Report Amendment 1's N=400 coverage power limitation."),
) -> None:
    """Write tables and frozen-bar labels from completed episode/analysis runs."""
    from .probe.report import run_report
    result = run_report([p for value in runs for p in value.split(",")], run_dir,
                        feasibility_run, seed, bootstrap_samples, publish, coverage_underpowered)
    typer.echo(json.dumps({"run_dir": result["run_dir"], "checkpoints": list(result["checkpoints"]),
                           "feasibility": result["feasibility"]}, indent=2))
