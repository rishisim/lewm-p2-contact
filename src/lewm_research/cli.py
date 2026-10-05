"""Command-line entry points for the role-swap probe."""

import json

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
    batch_size: int = typer.Option(32, min=1),
    seed: int = typer.Option(0),
    device: str = typer.Option("auto"),
    log_interval: int = typer.Option(1, min=1),
    val_interval: int = typer.Option(100, min=1),
    checkpoint_interval: int = typer.Option(100, min=1),
    workers: int = typer.Option(4, min=0),
) -> None:
    """Fine-tune LeWM with episode-held-out validation; resume by name."""
    from .train.finetune import finetune

    typer.echo(json.dumps(finetune(dataset, name, init, max_steps, batch_size, seed, device,
                                  log_interval, val_interval, checkpoint_interval, workers), indent=2))


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
