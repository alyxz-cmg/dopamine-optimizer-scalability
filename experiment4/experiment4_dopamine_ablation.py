#!/usr/bin/env python3
"""
Experiment 4 ablation:
1) Full Dopamine-2 with spectral radius reset
2) Dopamine-2 without reset
3) Dopamine-2 reset-frequency sweep

Outputs:
- metrics.csv (per-epoch records)
- summary.csv (per-condition aggregate)
- summary.json
- loss_curves.png (if matplotlib installed)

Command-Line Arguments:
python "experiment4_dopamine_ablation.py" \
  --device cuda \
  --pin-memory \
  --num-workers 2 \
  --output-dir exp4_corrected
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _rk4_step(x: np.ndarray, dt: float, f: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_lorenz(n_steps: int, dt: float = 0.01, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0) -> np.ndarray:
    x = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    def f(s: np.ndarray) -> np.ndarray:
        return np.array(
            [sigma * (s[1] - s[0]), s[0] * (rho - s[2]) - s[1], s[0] * s[1] - beta * s[2]],
            dtype=np.float64,
        )

    out = np.zeros((n_steps, 3), dtype=np.float64)
    for i in range(n_steps):
        x = _rk4_step(x, dt, f)
        out[i] = x
    return out


def simulate_rossler(n_steps: int, dt: float = 0.01, a: float = 0.2, b: float = 0.2, c: float = 5.7) -> np.ndarray:
    x = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    def f(s: np.ndarray) -> np.ndarray:
        return np.array([-s[1] - s[2], s[0] + a * s[1], b + s[2] * (s[0] - c)], dtype=np.float64)

    out = np.zeros((n_steps, 3), dtype=np.float64)
    for i in range(n_steps):
        x = _rk4_step(x, dt, f)
        out[i] = x
    return out


def make_windows(series: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(len(series) - seq_len):
        xs.append(series[i : i + seq_len])
        ys.append(series[i + seq_len])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


class VanillaRNN(nn.Module):
    def __init__(self, input_size: int = 3, hidden_size: int = 512, output_size: int = 3):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True, nonlinearity="relu")
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(x)
        return self.head(h[:, -1, :])


@dataclass
class Dopamine2Config:
    sigma: float = 1e-3
    beta_s: float = 0.95
    beta_eta: float = 0.95
    s0: float = 1e-3
    eta0: float = 1e-3
    eta_min: float = 1e-8
    eta_max: float = 5e-2
    target_spectral_radius: float = 1.0


class Dopamine2State:
    def __init__(self, model: nn.Module, cfg: Dopamine2Config):
        self.cfg = cfg
        self.s = cfg.s0
        self.eta = cfg.eta0
        self.params = [p for p in model.parameters() if p.requires_grad]

    def step(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor, loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        sig = self.cfg.sigma
        with torch.no_grad():
            noise = [torch.randn_like(p) * sig for p in self.params]
            for p, n in zip(self.params, noise):
                p.add_(n)
            perturbed_loss = loss_fn(model(x), y)
            for p, n in zip(self.params, noise):
                p.sub_(n)
            base_loss = loss_fn(model(x), y)

            rpe = float((perturbed_loss - base_loss).item())
            self.s = self.cfg.beta_s * self.s + (1.0 - self.cfg.beta_s) * abs(rpe)
            self.eta = (1.0 - self.cfg.beta_eta) * self.eta + self.cfg.beta_eta * self.s
            self.eta = max(self.cfg.eta_min, min(self.cfg.eta_max, self.eta))

            step_scale = -self.eta * rpe
            for p, n in zip(self.params, noise):
                p.add_(n, alpha=step_scale)

        return float(base_loss.item()), rpe, float(self.eta)


def spectral_radius(weight: torch.Tensor) -> float:
    # Use native eigvals on CUDA/CPU; fall back to CPU numpy for MPS.
    if weight.device.type == "mps":
        w_cpu = weight.detach().to("cpu", dtype=torch.float32).numpy()
        vals = np.linalg.eigvals(w_cpu)
        return float(np.abs(vals).max())
    vals = torch.linalg.eigvals(weight)
    return float(vals.abs().max().real.item())


def reset_spectral_radius_(model: VanillaRNN, target_radius: float) -> float:
    with torch.no_grad():
        w = model.rnn.weight_hh_l0.data
        rho = spectral_radius(w)
        if rho > 0.0 and math.isfinite(rho):
            w.mul_(target_radius / rho)
        return rho


def build_dataloader(
    attractor: str,
    seq_len: int,
    n_steps: int,
    train_split: float,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader]:
    if attractor == "lorenz":
        series = simulate_lorenz(n_steps=n_steps)
    elif attractor == "rossler":
        series = simulate_rossler(n_steps=n_steps)
    else:
        raise ValueError(f"Unknown attractor {attractor}")

    split_idx = int(len(series) * train_split)
    train_raw, val_raw = series[:split_idx], series[split_idx:]
    mu, std = train_raw.mean(axis=0, keepdims=True), train_raw.std(axis=0, keepdims=True) + 1e-8
    train_raw = (train_raw - mu) / std
    val_raw = (val_raw - mu) / std

    x_train, y_train = make_windows(train_raw, seq_len=seq_len)
    x_val, y_val = make_windows(val_raw, seq_len=seq_len)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
) -> float:
    losses = []
    non_blocking = device.type == "cuda"
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=non_blocking)
            yb = yb.to(device, non_blocking=non_blocking)
            pred = model(xb)
            losses.append(float(loss_fn(pred, yb).item()))
    return float(np.mean(losses)) if losses else float("nan")


def run_condition(
    attractor: str,
    condition: str,
    reset_every: int | None,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict], dict]:
    set_seed(seed)
    train_loader, val_loader = build_dataloader(
        attractor=attractor,
        seq_len=args.seq_len,
        n_steps=args.n_steps,
        train_split=args.train_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(args.pin_memory and device.type == "cuda"),
    )

    model = VanillaRNN(input_size=3, hidden_size=args.hidden_size, output_size=3).to(device)
    loss_fn = nn.MSELoss()
    sigma = args.sigma_lorenz if attractor == "lorenz" else args.sigma_rossler
    s0 = args.s0_lorenz if attractor == "lorenz" else args.s0_rossler
    cfg = Dopamine2Config(
        sigma=sigma,
        beta_s=args.beta_s,
        beta_eta=args.beta_eta,
        s0=s0,
        eta0=args.eta0,
        eta_min=args.eta_min,
        eta_max=args.eta_max,
        target_spectral_radius=args.target_radius,
    )
    opt_state = Dopamine2State(model, cfg)

    step_idx = 0
    rows = []
    diverged = False

    epoch_iter = range(1, args.epochs + 1)
    if args.progress and tqdm is not None:
        epoch_iter = tqdm(
            epoch_iter,
            total=args.epochs,
            desc=f"{attractor}:{condition}:seed{seed} epochs",
            leave=False,
        )

    for epoch in epoch_iter:
        epoch_losses = []
        epoch_rpes = []
        etas = []
        reset_count = 0
        pre_reset_rhos = []

        non_blocking = device.type == "cuda"
        batch_iter = train_loader
        if args.progress and tqdm is not None:
            batch_iter = tqdm(
                train_loader,
                total=len(train_loader),
                desc=f"{attractor}:{condition}:seed{seed} ep{epoch}",
                leave=False,
                position=1
            )
        for xb, yb in batch_iter:
            xb = xb.to(device, non_blocking=non_blocking)
            yb = yb.to(device, non_blocking=non_blocking)
            if reset_every is not None and reset_every > 0 and step_idx % reset_every == 0:
                pre_rho = reset_spectral_radius_(model, cfg.target_spectral_radius)
                pre_reset_rhos.append(pre_rho)
                reset_count += 1

            loss, rpe, eta = opt_state.step(model, xb, yb, loss_fn)
            step_idx += 1
            epoch_losses.append(loss)
            epoch_rpes.append(rpe)
            etas.append(eta)

            if (not math.isfinite(loss)) or loss > args.diverge_loss:
                diverged = True
                break

        val_loss = evaluate(model, val_loader, loss_fn, device=device) if not diverged else float("nan")
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        row = {
            "attractor": attractor,
            "condition": condition,
            "reset_every": reset_every if reset_every is not None else -1,
            "seed": seed,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "mean_abs_rpe": float(np.mean(np.abs(epoch_rpes))) if epoch_rpes else float("nan"),
            "mean_eta": float(np.mean(etas)) if etas else float("nan"),
            "resets_this_epoch": reset_count,
            "mean_pre_reset_rho": float(np.mean(pre_reset_rhos)) if pre_reset_rhos else float("nan"),
            "diverged": int(diverged),
        }
        rows.append(row)

        if args.progress and tqdm is not None and hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix(
                train=f"{train_loss:.4g}" if math.isfinite(train_loss) else "nan",
                val=f"{val_loss:.4g}" if math.isfinite(val_loss) else "nan",
                div=int(diverged),
            )

        if diverged:
            break

    final = rows[-1] if rows else {}
    summary = {
        "attractor": attractor,
        "condition": condition,
        "reset_every": reset_every if reset_every is not None else -1,
        "seed": seed,
        "final_epoch": int(final.get("epoch", 0)),
        "final_train_loss": float(final.get("train_loss", float("nan"))),
        "final_val_loss": float(final.get("val_loss", float("nan"))),
        "diverged": int(diverged),
    }
    
    return rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot(metrics: list[dict], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    grouped = {}
    for r in metrics:
        key = (r["attractor"], r["condition"])
        grouped.setdefault(key, []).append(r)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    for ax_idx, attractor in enumerate(["lorenz", "rossler"]):
        ax = axes[ax_idx]
        for (att, cond), rows in grouped.items():
            if att != attractor:
                continue

            rows = sorted(rows, key=lambda z: (z["seed"], z["epoch"]))
            by_seed = {}

            for rr in rows:
                by_seed.setdefault(rr["seed"], []).append(rr)
            max_epoch = max(len(v) for v in by_seed.values())
            ys = []

            for ep in range(1, max_epoch + 1):
                vals = []
                for srows in by_seed.values():
                    m = next((x for x in srows if x["epoch"] == ep), None)
                    if m is not None and math.isfinite(m["train_loss"]):
                        vals.append(m["train_loss"])
                ys.append(np.mean(vals) if vals else np.nan)
            ax.plot(range(1, max_epoch + 1), ys, label=cond)

        ax.set_title(attractor)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train MSE")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def aggregate_summary(summaries: list[dict]) -> list[dict]:
    grouped = {}
    for s in summaries:
        key = (s["attractor"], s["condition"], s["reset_every"])
        grouped.setdefault(key, []).append(s)

    rows = []
    for (att, cond, reset_every), vals in grouped.items():
        final_losses = [v["final_val_loss"] for v in vals if math.isfinite(v["final_val_loss"])]
        row = {
            "attractor": att,
            "condition": cond,
            "reset_every": reset_every,
            "num_runs": len(vals),
            "num_diverged": int(sum(v["diverged"] for v in vals)),
            "divergence_rate": float(sum(v["diverged"] for v in vals) / max(1, len(vals))),
            "mean_final_val_loss": float(np.mean(final_losses)) if final_losses else float("nan"),
            "std_final_val_loss": float(np.std(final_losses)) if final_losses else float("nan"),
        }
        rows.append(row)
    rows.sort(key=lambda r: (r["attractor"], r["condition"], r["reset_every"]))
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 4 spectral-reset ablation for Dopamine-2 on chaotic attractors.")
    p.add_argument("--output-dir", type=str, default=None, help="Output folder. Defaults to exp4_outputs/<timestamp>.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])

    # ── Training setup (Table S6) ─────────────────────────────────────────
    p.add_argument("--epochs", type=int, default=2000,help="Training epochs. Paper Table S6: 2000.")
    p.add_argument("--batch-size", type=int, default=5000, help="Batch size. Paper Table S6: 5000.")
    p.add_argument("--hidden-size", type=int, default=512, help="RNN hidden units. Paper Table S6: 512.")
    p.add_argument("--seq-len", type=int, default=32, help="Look-back window (sequence length). Paper Table S6: 32.")
    p.add_argument("--n-steps", type=int, default=5000, help="Trajectory length. Paper Section A.4: 5000 time steps.")
    p.add_argument("--train-split", type=float, default=0.8)

    # ── Dopamine-2 hyperparameters (Tables S4 & S5) ───────────────────────
    p.add_argument("--sigma-lorenz", type=float, default=1e-4, help="Perturbation scale for Lorenz. Paper Table S4: 1e-4.")
    p.add_argument("--sigma-rossler", type=float, default=1e-5, help="Perturbation scale for Rossler. Paper Table S5: 1e-5.")
    p.add_argument("--beta-s", type=float, default=0.9998, help="EMA coefficient for s. Paper Tables S4 & S5: 0.9998.")
    p.add_argument("--beta-eta", type=float, default=0.9998, help="EMA coefficient for eta. Paper Tables S4 & S5: 0.9998.")
    p.add_argument("--s0-lorenz", type=float, default=1e-4, help="Initial auxiliary variable s0 for Lorenz. Paper Table S4: 1e-4.")
    p.add_argument("--s0-rossler", type=float, default=1e-5, help="Initial auxiliary variable s0 for Rossler. Paper Table S5: 1e-5.")
    p.add_argument("--eta0", type=float, default=1e-2, help="Initial learning rate. Paper Tables S4 & S5: 1e-2.")
    p.add_argument("--eta-min", type=float, default=1e-8, help="Learning rate floor clamp.")
    p.add_argument("--eta-max", type=float, default=5e-2, help="Learning rate ceiling clamp.")
    p.add_argument("--target-radius", type=float, default=1.0, help="Target spectral radius. Paper Tables S4 & S5: 1.0.")

    # ── Ablation conditions ───────────────────────────────────────────────
    p.add_argument("--full-reset-every", type=int, default=1, help="Reset spectral radius every N steps for full_reset condition. " "Paper Algorithm 1: reset every step (1).")
    p.add_argument("--freq-list", type=int, nargs="+", default=[10, 50, 100, 500], help="Reset frequencies for sweep. Do not include 1 — that is covered " "by --full-reset-every 1. Paper ablation sweep: 10, 50, 100, 500.")

    # ── Runtime / hardware ────────────────────────────────────────────────
    p.add_argument("--diverge-loss", type=float, default=1e6, help="Loss threshold above which a run is flagged as diverged.")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (try 2-4 on stronger systems).")
    p.add_argument("--pin-memory", action="store_true", help="Enable DataLoader pinned memory (CUDA only).")
    p.add_argument("--progress", action="store_true", default=True, help="Show tqdm progress bars (requires tqdm).")

    return p.parse_args()


def resolve_device(requested: str) -> torch.device:
    req = requested.lower().strip()
    if req == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("[warn] Requested --device cuda, but CUDA is unavailable. Falling back to mps.")
            return torch.device("mps")
        print("[warn] Requested --device cuda, but CUDA is unavailable. Falling back to cpu.")
        return torch.device("cpu")
    if req == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        print("[warn] Requested --device mps, but MPS is unavailable. Falling back to cpu.")
        return torch.device("cpu")
    if req == "cpu":
        return torch.device("cpu")
    print(f"[warn] Unknown device '{requested}'. Falling back to cpu.")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"[info] Using device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path("exp4_outputs") / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict] = []
    all_summaries: list[dict] = []

    conditions = [
        ("full_reset", args.full_reset_every),
        ("no_reset", None),
    ] + [(f"freq_{k}", int(k)) for k in args.freq_list]

    run_grid = [(a, c, r, int(s)) for a in ["lorenz", "rossler"] for (c, r) in conditions for s in args.seeds]
    print(f"[info] Total runs: {len(run_grid)}")
    run_iter = run_grid
    if args.progress and tqdm is not None:
        run_iter = tqdm(run_grid, total=len(run_grid), desc="All runs", leave=True)

    for attractor, cond_name, reset_every, seed in run_iter:
                rows, summ = run_condition(
                    attractor=attractor,
                    condition=cond_name,
                    reset_every=reset_every,
                    seed=seed,
                    args=args,
                    device=device,
                )
                all_metrics.extend(rows)
                all_summaries.append(summ)
                print(
                    f"[{attractor}][{cond_name}][seed={seed}] "
                    f"epoch={summ['final_epoch']} val={summ['final_val_loss']:.6f} diverged={summ['diverged']}"
                )

    summary_rows = aggregate_summary(all_summaries)

    write_csv(out_dir / "metrics.csv", all_metrics)
    write_csv(out_dir / "per_run_summary.csv", all_summaries)
    write_csv(out_dir / "summary.csv", summary_rows)
    with (out_dir / "summary.json").open("w") as f:
        json.dump({"per_run": all_summaries, "aggregated": summary_rows}, f, indent=2)
    maybe_plot(all_metrics, out_dir / "loss_curves.png")

    print(f"\nSaved outputs to: {out_dir.resolve()}")
    print("Files: metrics.csv, per_run_summary.csv, summary.csv, summary.json, loss_curves.png (if matplotlib)")


if __name__ == "__main__":
    main()
