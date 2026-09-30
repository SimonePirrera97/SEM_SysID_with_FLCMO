#!/usr/bin/env python3
"""Estimate (Km, k0, c) by full-sequence and batched Adam BPTT."""

import argparse
import json
import time
import copy
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat


DTYPE = torch.float64
PARAMETER_SCALES = torch.tensor([2.0e-4, 2.0e-5, 2.0e-2], dtype=DTYPE)


class LevitatorModel(torch.nn.Module):
    """Forward-Euler model with dimensionless trainable parameters."""

    def __init__(self, mass: float, gravity: float, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        initial = 1.0 + 0.05 * torch.randn(3, generator=generator, dtype=DTYPE)
        self.normalized_parameters = torch.nn.Parameter(initial)
        self.register_buffer("mass", torch.tensor(mass, dtype=DTYPE))
        self.register_buffer("gravity", torch.tensor(gravity, dtype=DTYPE))
        self.register_buffer("scales", PARAMETER_SCALES.clone())

    @property
    def physical_parameters(self) -> torch.Tensor:
        return self.scales * self.normalized_parameters

    def next_gap(
        self,
        gap: torch.Tensor,
        next_gap: torch.Tensor,
        current: torch.Tensor,
        sample_time: float,
    ) -> torch.Tensor:
        magnetic_gain, magnetic_offset, friction = self.physical_parameters
        velocity = (next_gap - gap) / sample_time
        acceleration = (
            self.gravity
            - (magnetic_gain * current.square() + magnetic_offset)
            / (self.mass * gap.square())
            - friction * velocity / self.mass
        )
        return 2.0 * next_gap - gap + sample_time**2 * acceleration

    def simulate(
        self, current: torch.Tensor, initial_gap: torch.Tensor, sample_time: float
    ) -> torch.Tensor:
        # Keeping this tensor list connected is essential: no detach is allowed in BPTT.
        outputs = [initial_gap[..., 0], initial_gap[..., 1]]
        for index in range(current.shape[-1] - 2):
            outputs.append(
                self.next_gap(outputs[-2], outputs[-1], current[..., index], sample_time)
            )
        return torch.stack(outputs, dim=-1)


def normalized_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    scale = torch.std(target, dim=-1, keepdim=True).clamp_min(1e-6)
    return torch.mean(((prediction - target) / scale).square())


def train_full(
    model: LevitatorModel,
    current: torch.Tensor,
    target: torch.Tensor,
    sample_time: float,
    epochs: int,
    learning_rate: float,
    clip_norm: float,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction = model.simulate(current, target[:2], sample_time)
        loss = normalized_mse(prediction, target)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite full-sequence loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        history.append(float(loss.detach()))
        if (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"full epoch {epoch+1:5d}/{epochs}: loss={history[-1]:.6g}")
    return history


def train_batched(
    model: LevitatorModel,
    current: torch.Tensor,
    target: torch.Tensor,
    sample_time: float,
    epochs: int,
    learning_rate: float,
    clip_norm: float,
    sequence_length: int,
    batch_size: int,
    seed: int,
    validation_current=None,
    validation_target=None,
    monitor_every: int = 100,
    minimum_epochs: int = 1000,
    patience: int = 20,
    min_delta: float = 1e-5,
) -> list[float]:
    if sequence_length < 3 or sequence_length > target.numel():
        raise ValueError("invalid subsequence length")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    start_count = target.numel() - sequence_length + 1
    history = []
    best_fit = -float("inf")
    best_state = None
    stale = 0
    best_epoch = 0
    offsets = torch.arange(sequence_length)
    for epoch in range(epochs):
        starts = torch.randint(0, start_count, (batch_size,), generator=generator)
        indices = starts[:, None] + offsets[None, :]
        current_batch = current[indices]
        target_batch = target[indices]
        optimizer.zero_grad(set_to_none=True)
        prediction = model.simulate(current_batch, target_batch[:, :2], sample_time)
        loss = normalized_mse(prediction, target_batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite batched loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        history.append(float(loss.detach()))
        epoch_number = epoch + 1
        if validation_current is not None and epoch_number % monitor_every == 0:
            with torch.no_grad():
                validation_prediction = model.simulate(
                    validation_current, validation_target[:2], sample_time
                )
                error = validation_prediction - validation_target
                fit = 1.0 - torch.sum(error.square()) / torch.sum(
                    (validation_target - torch.mean(validation_target)).square()
                )
                fit_value = float(fit)
            if fit_value > best_fit + min_delta:
                best_fit = fit_value
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch_number
                stale = 0
            else:
                stale += 1
            if epoch_number >= minimum_epochs and stale >= patience:
                if best_state is not None:
                    model.load_state_dict(best_state)
                print(f"batch fatigue at epoch {epoch_number}: best FIT={best_fit:.8f} at {best_epoch}")
                break
        if (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"batch epoch {epoch+1:5d}/{epochs}: loss={history[-1]:.6g}")
    return history, best_fit, best_epoch


def scalar(data: dict, name: str) -> float:
    return float(np.asarray(data[name]).squeeze())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10_000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--initial-physical", type=float, nargs=3,
                        default=None,
                        metavar=("KM", "K0", "C"),
                        help="common physical parameter initialization")
    parser.add_argument("--method", choices=("full", "batched", "both"), default="both")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    folder = Path(__file__).resolve().parent
    data = loadmat(folder / "data" / "levitat_data.mat")
    current = torch.as_tensor(np.ravel(data["u"]), dtype=DTYPE)
    target = torch.as_tensor(np.ravel(data["y"]), dtype=DTYPE)
    validation_current = torch.as_tensor(np.ravel(data["u_validation"]), dtype=DTYPE)
    validation_target = torch.as_tensor(np.ravel(data["y_validation_true"]), dtype=DTYPE)
    if target.numel() != 1000:
        raise ValueError(f"expected 1000 samples, found {target.numel()}")
    sample_time = scalar(data, "Ts")
    true_parameters = np.array([scalar(data, key) for key in ("Km", "k0", "c")])
    settings = vars(args).copy()
    if settings["output"] is not None:
        settings["output"] = str(settings["output"])
    results = {"true_parameters": true_parameters.tolist(), "settings": settings}

    methods = ("full", "batched") if args.method == "both" else (args.method,)
    for method_index, method in enumerate(methods):
        model = LevitatorModel(scalar(data, "m"), scalar(data, "g"), args.seed + method_index)
        if args.initial_physical is not None:
            with torch.no_grad():
                model.normalized_parameters.copy_(
                    torch.as_tensor(args.initial_physical, dtype=DTYPE) / PARAMETER_SCALES
                )
        start = time.perf_counter()
        if method == "full":
            history = train_full(
                model, current, target, sample_time, args.epochs,
                args.learning_rate, args.clip_norm,
            )
        else:
            history, best_fit, best_epoch = train_batched(
                model, current, target, sample_time, args.epochs,
                args.learning_rate, args.clip_norm, args.sequence_length,
                args.batch_size, args.seed, validation_current, validation_target,
            )
        estimate = model.physical_parameters.detach().cpu().numpy()
        results[method] = {
            "estimate": estimate.tolist(),
            "relative_error": ((estimate - true_parameters) / true_parameters).tolist(),
            "final_loss": history[-1],
            "elapsed_seconds": time.perf_counter() - start,
            "loss_history": history,
            "epochs": len(history),
            "best_validation_fit": best_fit if method == "batched" else None,
            "best_validation_epoch": best_epoch if method == "batched" else None,
        }
        print(f"{method} estimate [Km, k0, c] = {estimate}")

    output = args.output or (folder / "results" / "adam_comparison.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
