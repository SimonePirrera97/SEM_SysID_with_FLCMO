# Example 4: gray-box magnetic levitator

The continuous-time data-generating model is

```
z_dot = v
v_dot = g - (Km*i^2 + k0)/(m*z^2) - (c/m)*v
```

Three parameters (`Km`, `k0`, `c`) are estimated; mass `m` and gravity `g` are
known. `generate_data.py` integrates the plant with fixed-step RK4 at 10 us and
resamples at `Ts = 1 ms` under closed-loop stabilization (the plant is
open-loop unstable), producing 1,000 training samples plus a 64-sample
open-loop validation record. Applying forward Euler and eliminating velocity
gives the order-2 implicit NIO model used for identification.

All three methods below share the initial guess
`(Km, k0, c) = (2.5e-4, 2.5e-5, 3.0e-2)`.

## FL-CMO (this paper's algorithm)

```matlab
addpath('Example4_GrayBoxMagneticLevitation', 'native')
run_optimized_tau(2e-3, "flcmo")
```

`K=1`, step `tau=2e-3`; training stops when validation BFR (checked every 100
iterations) fails to improve by `1e-5` over 20 consecutive checks, up to
`1e5` iterations, restoring the best validation point. Output:
`results/optimized_result_flcmo.mat`.

## Adam BPTT baseline

```sh
python3 Example4_GrayBoxMagneticLevitation/run_adam_comparison.py \
    --method batched --initial-physical 2.5e-4 2.5e-5 3e-2 \
    --output Example4_GrayBoxMagneticLevitation/results/adam_truncated_result.json
```

Length-128 truncated sequences, batches of 16, learning rate `1e-3`, up to
`1e4` epochs, gradient-norm clipping at 1.0, parameters optimized in
normalized coordinates. `--method full` runs the (much slower, non-converging)
full-sequence variant discussed in the text.

## Least-squares baseline

```matlab
run_ls
```

Linear, convex one-step prediction-error estimate; a closed-form solve
(`results/ls_result.mat`).

## Validation

`validation_accuracy.m` evaluates all estimates on the 64-sample open-loop
validation record (`results/validation_accuracy.mat`). `build_mex.m` rebuilds
the packed-Jacobian MEX provider (`levitator_constraints_mex.mexmaca64`); the
readable MATLAB fallback is `levitator_constraints_matlab.m`.
