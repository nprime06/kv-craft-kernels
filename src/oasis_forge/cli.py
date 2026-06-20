"""oasis-forge CLI.

  oasis-forge problems              list kernel problems + harvest status
  oasis-forge ledger [problem]      show attempts / best kept port
  oasis-forge solve <problem>       build prompt + launch agent (needs target box)
  oasis-forge harvest <problem>     dump golden tensors from a rollout (needs GPU box)

solve/harvest are stubs until hardware is wired; problems/ledger work now.
"""

from __future__ import annotations

from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.table import Table

from .config import PortConfig
from .ledger import Ledger
from .roofline import OASIS_DMD, budget
from .streaming import run_central_server, run_laptop_bridge

console = Console()
ROOT = Path(__file__).resolve().parents[2]


def _problems() -> list[dict]:
	out = []
	for p in sorted((ROOT / "problems").glob("*/problem.yaml")):
		data = yaml.safe_load(p.read_text())
		data["_dir"] = p.parent
		out.append(data)
	return out


@click.group()
def main() -> None:
	"""Port-and-optimize world-model kernels from B300/sm_100 to H100/sm_90."""


@main.command()
def problems() -> None:
	"""List kernel problems and whether their golden tensors have been harvested."""
	t = Table(title="oasis-h100-port problems")
	t.add_column("kernel")
	t.add_column("tier")
	t.add_column("harvested")
	t.add_column("shapes")
	for prob in _problems():
		tf = prob["_dir"] / "task_files"
		harvested = "yes" if (tf / "golden.npz").exists() else "no (stub)"
		shapes = prob.get("shapes_status", "from harvest")
		t.add_row(prob["name"], prob.get("default_tier", "structural"), harvested, str(shapes))
	console.print(t)


@main.command()
@click.argument("problem", required=False)
def ledger(problem: str | None) -> None:
	"""Show port attempts (optionally for one problem) and the best kept port."""
	cfg = PortConfig()
	led = Ledger(ROOT / cfg.ledger_path)
	rows = [a for a in led.all() if problem is None or a.problem == problem]
	if not rows:
		console.print("[yellow]no attempts logged yet[/yellow]")
		return
	t = Table(title="attempts")
	for c in ("problem", "tier", "compiled", "t1", "t2", "speedup", "approach"):
		t.add_column(c)
	for a in rows:
		t.add_row(
			a.problem,
			a.tier,
			"y" if a.compiled else "n",
			"y" if a.tier1_pass else "n",
			{True: "y", False: "n", None: "-"}[a.tier2_pass],
			f"{a.speedup:.2f}x" if a.speedup else "-",
			a.approach[:48],
		)
	console.print(t)
	for prob in {a.problem for a in rows}:
		best = led.best(prob)
		if best:
			console.print(f"[green]best {prob}: {best.speedup:.2f}x — {best.approach}[/green]")


@main.command()
@click.option("--gpus", default=1, help="H100s per player cluster")
@click.option("--chunk", default=3, help="latent frames generated per chunk")
@click.option("--window", default=21, help="cached latent frames (KV window)")
@click.option("--target-fps", default=30.0)
def roofline(gpus: int, chunk: int, window: int, target_fps: float) -> None:
	"""First-order frame-budget for the OASIS-DMD ship shapes. Shows which kernel binds and
	how the levers (GPUs, window, camera-band, FP8-FFN) close the gap to realtime."""
	console.print(f"[bold]{OASIS_DMD.name}[/bold]  chunk={chunk} latent-frames  "
		f"window={window}  target={target_fps:.0f} fps\n")
	scenarios = [
		("baseline (dense, bf16)", dict(camera_band=1.0, fp8_ffn=False)),
		("+ camera-band attn (1/3)", dict(camera_band=1 / 3, fp8_ffn=False)),
		("+ FP8 FFN/QKV", dict(camera_band=1 / 3, fp8_ffn=True)),
		("+ window 21->8", dict(camera_band=1 / 3, fp8_ffn=True, _window=8)),
	]
	t = Table(title=f"frame budget @ {gpus}x H100")
	for c in ("scenario", "per-fwd", "per-chunk", "per-frame", "fps", "vs target", "binds"):
		t.add_column(c)
	for label, kw in scenarios:
		w = kw.pop("_window", window)
		r = budget(chunk_frames=chunk, window_frames=w, n_gpus=gpus, target_fps=target_fps, **kw)
		binds = max(r.kernels, key=lambda k: k.roofline_s).name
		verdict = f"{r.gap:.0f}x slow" if r.gap > 1 else f"{1 / r.gap:.1f}x headroom"
		color = "red" if r.gap > 1 else "green"
		t.add_row(label, f"{r.per_forward_s*1000:.0f}ms", f"{r.per_chunk_s*1000:.0f}ms",
			f"{r.per_rgb_frame_s*1000:.1f}ms", f"{r.fps:.1f}",
			f"[{color}]{verdict}[/{color}]", binds)
	console.print(t)
	console.print("\n[dim]estimates from first-order FLOP/byte counts; VAE decode is a placeholder "
		"until measured. Weights bf16; attention kept bf16 (FP8-attn drift).[/dim]")


@main.command()
@click.argument("problem")
def solve(problem: str) -> None:
	"""Launch the optimizer agent for a problem. STUB: needs a target H100 box."""
	console.print(
		f"[red]solve {problem} is a stub.[/red] Wire H100Remote (src/oasis_forge/remote.py) "
		"and populate task_files via `harvest` first. See agents/agent_prompt.md."
	)


@main.command()
@click.argument("problem")
def harvest(problem: str) -> None:
	"""Dump golden input/output tensors from a rollout. STUB: needs a GPU box + model."""
	console.print(
		f"[red]harvest {problem} is a stub.[/red] Implement harvest/hooks.py against a live "
		"Oasis-500M rollout to emit task_files/golden.npz."
	)


@main.command("serve-central")
@click.option("--action-host", default="0.0.0.0", show_default=True)
@click.option("--action-port", default=7780, show_default=True)
@click.option("--client-latent-port", default=7781, show_default=True)
@click.option("--latent-height", default=28, show_default=True)
@click.option("--latent-width", default=50, show_default=True)
@click.option("--fps", default=12.0, show_default=True)
def serve_central(
	action_host: str,
	action_port: int,
	client_latent_port: int,
	latent_height: int,
	latent_width: int,
	fps: float,
) -> None:
	"""Run the central split-serving relay.

	This is the integration seam for the B300/JAX latent generator. The checked-in mode uses
	zero latents as a network/protocol smoke test until the real generator is wired.
	"""
	run_central_server(
		action_host=action_host,
		action_port=action_port,
		client_latent_port=client_latent_port,
		latent_height=latent_height,
		latent_width=latent_width,
		fps=fps,
	)


@main.command("laptop-bridge")
@click.option("--server-host", required=True, help="Central server hostname or IP.")
@click.option("--player-id", required=True, type=int)
@click.option("--server-action-port", default=7780, show_default=True)
@click.option("--listen-latent-port", default=7781, show_default=True)
@click.option("--local-action-port", default=7790, show_default=True)
@click.option("--decoder-host", default="127.0.0.1", show_default=True)
@click.option("--decoder-port", default=7777, show_default=True)
@click.option("--action-fps", default=60.0, show_default=True)
def laptop_bridge(
	server_host: str,
	player_id: int,
	server_action_port: int,
	listen_latent_port: int,
	local_action_port: int,
	decoder_host: str,
	decoder_port: int,
	action_fps: float,
) -> None:
	"""Run the laptop bridge between input, central latents, and local Metal decode."""
	run_laptop_bridge(
		server_host=server_host,
		player_id=player_id,
		server_action_port=server_action_port,
		listen_latent_port=listen_latent_port,
		local_action_port=local_action_port,
		decoder_host=decoder_host,
		decoder_port=decoder_port,
		action_fps=action_fps,
	)


if __name__ == "__main__":
	main()
