#!/usr/bin/env python3
"""Generate physically consistent magnetic-levitator identification data.

The continuous-time plant is integrated with fixed-step RK4 at 100 times the
dataset sampling rate.  The saved identification data are then obtained by
resampling the fine trajectory; the identification models themselves use the
forward-Euler recurrence documented in README.md.
"""

from pathlib import Path

import numpy as np
from scipy.io import savemat


MASS = 24.197e-3  # kg
GRAVITY = 9.81  # m/s^2
MAGNETIC_GAIN = 2.10393152e-4  # N m^2 / A^2
MAGNETIC_OFFSET = 2.0e-5  # N m^2
VISCOUS_FRICTION = 2.0e-2  # N s / m
N_SAMPLES = 1000
SAMPLE_TIME = 1.0e-3  # s (1 kHz, typical of a laboratory control loop)
FINE_STEP = 1.0e-5  # s
N_FINE_PER_SAMPLE = round(SAMPLE_TIME / FINE_STEP)
EQUILIBRIUM_GAP = 29.5e-3  # m
NOISE_STD = 1.0e-6  # m (1 micrometre)
RANDOM_SEED = 4


def excitation(time: float) -> float:
    """Persistently exciting current perturbation in amperes."""
    return (
        0.025 * np.sin(2.0 * np.pi * 1.7 * time + 0.2)
        + 0.018 * np.sin(2.0 * np.pi * 4.1 * time + 1.1)
        + 0.012 * np.sin(2.0 * np.pi * 9.3 * time + 2.0)
    )


def main() -> None:
    if not np.isclose(N_FINE_PER_SAMPLE * FINE_STEP, SAMPLE_TIME):
        raise ValueError("SAMPLE_TIME must be an integer multiple of FINE_STEP")

    equilibrium_current = np.sqrt(
        (MASS * GRAVITY * EQUILIBRIUM_GAP**2 - MAGNETIC_OFFSET)
        / MAGNETIC_GAIN
    )

    # State-feedback gains place the local closed-loop poles near -8 and -10.
    acceleration_gap_gain = 2.0 * GRAVITY / EQUILIBRIUM_GAP
    acceleration_current_gain = (
        -2.0 * MAGNETIC_GAIN * equilibrium_current
        / (MASS * EQUILIBRIUM_GAP**2)
    )
    desired_position_coefficient = -80.0
    desired_velocity_coefficient = -18.0
    position_gain = (
        desired_position_coefficient - acceleration_gap_gain
    ) / acceleration_current_gain
    velocity_gain = (
        desired_velocity_coefficient + VISCOUS_FRICTION / MASS
    ) / acceleration_current_gain

    def current_command(time: float, state: np.ndarray) -> float:
        gap, velocity = state
        command = (
            equilibrium_current
            + position_gain * (gap - EQUILIBRIUM_GAP)
            + velocity_gain * velocity
            + excitation(time)
        )
        return float(np.clip(command, 0.25, 1.75))

    def dynamics(state: np.ndarray, coil_current: float) -> np.ndarray:
        gap, velocity = state
        magnetic_force = (
            MAGNETIC_GAIN * coil_current**2 + MAGNETIC_OFFSET
        ) / gap**2
        acceleration = (
            GRAVITY
            - magnetic_force / MASS
            - VISCOUS_FRICTION * velocity / MASS
        )
        return np.array([velocity, acceleration])

    def integrate_zoh(initial_state: np.ndarray, currents: np.ndarray) -> np.ndarray:
        """Fine-step RK4 integration for a sampled, open-loop current."""
        states = np.empty((len(currents), 2), dtype=np.float64)
        states[0] = initial_state
        for sample, coil_current in enumerate(currents[:-1]):
            state = states[sample]
            for _ in range(N_FINE_PER_SAMPLE):
                k1 = dynamics(state, coil_current)
                k2 = dynamics(state + 0.5 * FINE_STEP * k1, coil_current)
                k3 = dynamics(state + 0.5 * FINE_STEP * k2, coil_current)
                k4 = dynamics(state + FINE_STEP * k3, coil_current)
                state = state + FINE_STEP * (k1 + 2*k2 + 2*k3 + k4) / 6.0
            states[sample + 1] = state
        return states

    fine_count = (N_SAMPLES - 1) * N_FINE_PER_SAMPLE + 1
    state_fine = np.empty((fine_count, 2), dtype=np.float64)
    current_fine = np.empty(fine_count, dtype=np.float64)
    acceleration_fine = np.empty(fine_count, dtype=np.float64)
    state_fine[0] = [EQUILIBRIUM_GAP + 0.3e-3, 0.0]

    # A digital controller computes one command per dataset sample. The command
    # is held while the continuous plant is integrated over the sample interval.
    for sample in range(N_SAMPLES - 1):
        first = sample * N_FINE_PER_SAMPLE
        coil_current = current_command(sample * SAMPLE_TIME, state_fine[first])
        for substep in range(N_FINE_PER_SAMPLE):
            index = first + substep
            state = state_fine[index]
            current_fine[index] = coil_current
            acceleration_fine[index] = dynamics(state, coil_current)[1]
            k1 = dynamics(state, coil_current)
            k2 = dynamics(state + 0.5 * FINE_STEP * k1, coil_current)
            k3 = dynamics(state + 0.5 * FINE_STEP * k2, coil_current)
            k4 = dynamics(state + FINE_STEP * k3, coil_current)
            state_fine[index + 1] = state + FINE_STEP * (k1 + 2*k2 + 2*k3 + k4) / 6.0
    current_fine[-1] = current_command((N_SAMPLES - 1) * SAMPLE_TIME, state_fine[-1])
    acceleration_fine[-1] = dynamics(state_fine[-1], current_fine[-1])[1]

    resample = np.arange(N_SAMPLES) * N_FINE_PER_SAMPLE
    time = resample * FINE_STEP
    gap_true = state_fine[resample, 0]
    velocity_true = state_fine[resample, 1]
    acceleration_true = acceleration_fine[resample]
    coil_current = current_fine[resample]
    rng = np.random.default_rng(RANDOM_SEED)
    measured_gap = gap_true + rng.normal(0.0, NOISE_STD, N_SAMPLES)

    magnetic_force = (
        MAGNETIC_GAIN * coil_current**2 + MAGNETIC_OFFSET
    ) / gap_true**2
    friction_force = VISCOUS_FRICTION * velocity_true
    force_balance_error = (
        MASS * acceleration_true
        - (MASS * GRAVITY - magnetic_force - friction_force)
    )

    # Short open-loop validation experiment: no state feedback is used here.
    validation_count = 64
    validation_time = np.arange(validation_count) * SAMPLE_TIME
    validation_current = np.clip(
        equilibrium_current + np.array([excitation(t) for t in validation_time]),
        0.25, 1.75,
    )
    validation_state = integrate_zoh(
        np.array([EQUILIBRIUM_GAP, 0.0]), validation_current
    )
    validation_gap = validation_state[:, 0]

    limits = {
        "gap": (20e-3, 40e-3),
        "current": (0.25, 1.75),
        "velocity_abs": 0.10,
        "acceleration_abs": 3.0,
    }
    if not (limits["gap"][0] < gap_true.min() < gap_true.max() < limits["gap"][1]):
        raise RuntimeError("gap trajectory is outside the levitator operating range")
    if not (limits["current"][0] <= coil_current.min() <= coil_current.max() <= limits["current"][1]):
        raise RuntimeError("coil current is outside its allowed range")
    if np.max(np.abs(velocity_true)) > limits["velocity_abs"]:
        raise RuntimeError("velocity is implausibly large")
    if np.max(np.abs(acceleration_true)) > limits["acceleration_abs"]:
        raise RuntimeError("acceleration is implausibly large")
    if np.max(np.abs(force_balance_error)) > 1e-12:
        raise RuntimeError("continuous-time force balance is inconsistent")

    output_path = Path(__file__).resolve().parent / "data" / "levitat_data.mat"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(
        output_path,
        {
            "u": coil_current[:, None],
            "y": measured_gap[:, None],
            "y_true": gap_true[:, None],
            "velocity_true": velocity_true[:, None],
            "acceleration_true": acceleration_true[:, None],
            "time": time[:, None],
            "Ts": SAMPLE_TIME,
            "fine_step": FINE_STEP,
            "Km": MAGNETIC_GAIN,
            "k0": MAGNETIC_OFFSET,
            "c": VISCOUS_FRICTION,
            "m": MASS,
            "g": GRAVITY,
            "z_bar": EQUILIBRIUM_GAP,
            "ibar": equilibrium_current,
            "measurement_noise_std": NOISE_STD,
            "position_feedback_gain": position_gain,
            "velocity_feedback_gain": velocity_gain,
            "u_validation": validation_current[:, None],
            "y_validation_true": validation_gap[:, None],
            "validation_time": validation_time[:, None],
        },
    )

    print(f"Saved {N_SAMPLES} samples to {output_path}")
    print(f"fine step / sample time : {FINE_STEP:.1e} / {SAMPLE_TIME:.1e} s")
    print(f"gap                    : {1e3*gap_true.min():.3f} .. {1e3*gap_true.max():.3f} mm")
    print(f"current                : {coil_current.min():.3f} .. {coil_current.max():.3f} A")
    print(f"velocity               : {1e3*velocity_true.min():.3f} .. {1e3*velocity_true.max():.3f} mm/s")
    print(f"acceleration           : {acceleration_true.min():.3f} .. {acceleration_true.max():.3f} m/s^2")
    print(f"magnetic force         : {magnetic_force.min():.4f} .. {magnetic_force.max():.4f} N")
    print(f"equilibrium current    : {equilibrium_current:.4f} A")
    print(f"max force-balance error: {np.max(np.abs(force_balance_error)):.3e} N")


if __name__ == "__main__":
    main()
