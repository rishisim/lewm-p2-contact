"""Command-line entry points for the role-swap probe."""

import json

import typer

app = typer.Typer(help="LeWM role-swap research tools.")


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
