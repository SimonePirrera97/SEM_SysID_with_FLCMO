"""Run Example-3 Adam baselines with 265 length-128 subsequences per batch."""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import scipy.io as sio
import tensorflow as tf
from tensorflow import keras

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "revised"
OUT.mkdir(parents=True, exist_ok=True)
ORDER, FIT_END, BATCH_SIZE, WINDOW = 3, 2800, 265, 128
# A 265x265 update processes roughly 17 times the samples of the former
# 64x64 setting. Cap update count accordingly while retaining fatigue stop.
MAX_EPOCHS, CV_MAX_EPOCHS = 4000, 4000
NNOE_MAX_EPOCHS = 5000
MIN_EPOCHS, PATIENCE, MIN_DELTA = 500, 500, 1e-8
REGULARIZATION = 1e-4
LEARNING_RATES = (3e-3, 1e-3, 3e-4)
GRIDS = {
    "NNOE_BATCHED": [(5,), (8,), (10,), (15,), (3, 3), (5, 5), (7, 7),
                     (10, 10), (3, 3, 3), (5, 5, 5), (7, 7, 7)],
    "LSTM": [(2,), (3,), (4,), (5,), (7,), (2, 2), (3, 3), (4, 4),
             (5, 5), (2, 2, 2), (3, 3, 3)],
    "GRU": [(3,), (4,), (5,), (7,), (10,), (3, 3), (4, 4), (5, 5),
            (7, 7), (2, 2, 2), (3, 3, 3)],
}
REPRESENTATIVE = {"NNOE_BATCHED": (5, 5), "LSTM": (2, 2), "GRU": (5,)}


def metrics(y, p):
    rmse = np.sqrt(np.mean((p - y) ** 2, axis=0))
    den = np.linalg.norm(y - y.mean(axis=0), axis=0)
    return rmse, 100 * (1 - np.linalg.norm(p - y, axis=0) / den)


def windows(u, y, length=WINDOW):
    # Use one full batch of evenly distributed, contiguous windows whenever
    # possible: 265 sequences, each with a 128-sample simulation horizon.
    count = min(BATCH_SIZE, len(u) - length + 1)
    starts = np.linspace(0, len(u) - length, count, dtype=int)
    return (np.stack([u[first:first+length] for first in starts]),
            np.stack([y[first:first+length] for first in starts]))


def recurrent(kind, hidden, nin, nout, seed):
    keras.utils.set_random_seed(seed)
    cls = keras.layers.LSTM if kind == "LSTM" else keras.layers.GRU
    return keras.Sequential([
        keras.Input((None, nin)),
        *(cls(width, return_sequences=True, activation="tanh") for width in hidden),
        keras.layers.TimeDistributed(keras.layers.Dense(nout)),
    ])


class BatchedNNOE(keras.Model):
    """NNOE trained by truncated free simulation on batches of subsequences."""

    def __init__(self, hidden, output_dimension, seed):
        super().__init__()
        keras.utils.set_random_seed(seed)
        reg = keras.regularizers.L2(REGULARIZATION)
        self.hidden_layers = [keras.layers.Dense(w, activation="tanh", kernel_regularizer=reg)
                              for w in hidden]
        self.output_layer = keras.layers.Dense(output_dimension, kernel_regularizer=reg)

    def call(self, regressor):
        return self.one_step(regressor)

    def one_step(self, regressor):
        value = regressor
        for layer in self.hidden_layers:
            value = layer(value)
        return self.output_layer(value)

    def simulate_window(self, u, y0):
        outputs = [y0[:, k, :] for k in range(ORDER)]
        for k in range(ORDER, WINDOW):
            y_history = [outputs[k - j] for j in range(1, ORDER + 1)]
            u_history = [u[:, k - j, :] for j in range(ORDER + 1)]
            outputs.append(self.one_step(tf.concat(y_history + u_history, axis=1)))
        return tf.stack(outputs, axis=1)


def simulate_nnoe_numpy(model, u, y0):
    y = np.zeros((len(u), y0.shape[1]), dtype=np.float32)
    y[:ORDER] = y0[:ORDER]
    weights = [(layer.kernel.numpy(), layer.bias.numpy()) for layer in model.hidden_layers]
    output_weights = (model.output_layer.kernel.numpy(), model.output_layer.bias.numpy())
    for k in range(ORDER, len(u)):
        value = np.concatenate([*(y[k-j] for j in range(1, ORDER+1)),
                                *(u[k-j] for j in range(ORDER+1))])
        for kernel, bias in weights:
            value = np.tanh(value @ kernel + bias)
        y[k] = value @ output_weights[0] + output_weights[1]
    return y


def train_batched_nnoe(hidden, u, y, learning_rate, max_epochs, seed):
    model = BatchedNNOE(hidden, y.shape[1], seed)
    xu, yu = windows(u, y)
    regressor_dimension = ORDER*y.shape[1] + (ORDER+1)*u.shape[1]
    model(tf.zeros((1, regressor_dimension), dtype=tf.float32))
    optimizer = keras.optimizers.Adam(learning_rate)
    best, best_weights, stale = np.inf, None, 0
    generator = np.random.default_rng(seed + 1000)

    @tf.function(reduce_retracing=True)
    def train_batch(ub, yb):
        with tf.GradientTape() as tape:
            prediction = model.simulate_window(ub, yb[:, :ORDER])
            data_loss = tf.reduce_mean(tf.square(prediction[:, ORDER:] - yb[:, ORDER:]))
            loss = data_loss + tf.add_n(model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return data_loss

    start = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        epoch_losses = []
        permutation = generator.permutation(len(xu))
        for first in range(0, len(xu), BATCH_SIZE):
            indexes = permutation[first:first+BATCH_SIZE]
            ub = tf.convert_to_tensor(xu[indexes])
            yb = tf.convert_to_tensor(yu[indexes])
            data_loss = train_batch(ub, yb)
            epoch_losses.append(float(data_loss))
        value = float(np.mean(epoch_losses))
        if np.isfinite(value) and value < best - MIN_DELTA:
            best, stale = value, 0
            best_weights = [w.numpy().copy() for w in model.weights]
        else:
            stale += 1
        if epoch >= MIN_EPOCHS and stale >= PATIENCE:
            break
    if best_weights is not None:
        model.set_weights(best_weights)
    return model, epoch, time.perf_counter() - start, "training loss stabilized" if epoch < max_epochs else "maximum epochs"


def train_recurrent(kind, hidden, u, y, learning_rate, max_epochs, seed, validation=False):
    model = recurrent(kind, hidden, u.shape[1], y.shape[1], seed)
    model.compile(keras.optimizers.Adam(learning_rate), "mse")
    if validation:
        xu, yu = windows(u[:FIT_END], y[:FIT_END])
        xv, yv = windows(u[FIT_END:], y[FIT_END:])
        monitor, extra = "val_loss", {"validation_data": (xv, yv)}
    else:
        xu, yu = windows(u, y)
        monitor, extra = "loss", {}
    callback = keras.callbacks.EarlyStopping(
        monitor=monitor, patience=PATIENCE, min_delta=MIN_DELTA,
        restore_best_weights=True, start_from_epoch=MIN_EPOCHS,
    )
    start = time.perf_counter()
    history = model.fit(xu, yu, batch_size=BATCH_SIZE, epochs=max_epochs,
                        verbose=0, callbacks=[callback], **extra)
    elapsed = time.perf_counter() - start
    epochs = len(history.history["loss"])
    reason = "training loss stabilized" if epochs < max_epochs else "maximum epochs"
    return model, epochs, elapsed, reason


def train(kind, hidden, u, y, learning_rate, max_epochs, seed, validation=False):
    if kind == "NNOE_BATCHED":
        end = FIT_END if validation else len(u)
        return train_batched_nnoe(hidden, u[:end], y[:end], learning_rate, max_epochs, seed)
    return train_recurrent(kind, hidden, u, y, learning_rate, max_epochs, seed, validation)


def predict(kind, model, u, y0):
    if kind == "NNOE_BATCHED":
        return simulate_nnoe_numpy(model, u, y0)
    return model(u[None], training=False).numpy()[0]


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    tr = sio.loadmat(HERE / "data" / "dataset_training.mat")
    te = sio.loadmat(HERE / "data" / "dataset_test.mat")
    u = tr["utrain"][:3500].astype("float32")
    y = tr["ytrain"][:3500, :2].astype("float32")
    ut = te["u_test"][:5000].astype("float32")
    yt = te["y_test"][:5000, :2].astype("float32")
    selection_path = OUT / "python_algorithm_selection.json"
    prior_selection = json.loads(selection_path.read_text()) if selection_path.exists() else {}
    if prior_selection and all(
            int(prior_selection[k]["batch_size"]) == BATCH_SIZE and
            int(prior_selection[k].get("sequence_length", -1)) == WINDOW
            for k in GRIDS):
        selected = prior_selection
        print(f"Reusing completed length-{WINDOW} algorithm selection.", flush=True)
    else:
        cv, selected = [], {}
        for kind in GRIDS:
            family = []
            for index, learning_rate in enumerate(LEARNING_RATES, 1):
                model, epochs, seconds, reason = train(
                    kind, REPRESENTATIVE[kind], u, y, learning_rate,
                    CV_MAX_EPOCHS, 500 + index, True)
                prediction = predict(kind, model, u[FIT_END:], y[FIT_END:FIT_END+ORDER])
                rmse, bfr = metrics(y[FIT_END:], prediction)
                row = dict(model=kind, candidate=index, learning_rate=learning_rate,
                           batch_size=BATCH_SIZE, sequence_length=WINDOW, epochs=epochs,
                           validation_rmse1=rmse[0], validation_rmse2=rmse[1],
                           validation_bfr1=bfr[0], validation_bfr2=bfr[1],
                           mean_validation_bfr=bfr.mean(), training_seconds=seconds,
                           stop_reason=reason)
                cv.append(row); family.append(row)
                print(f"CV {kind} lr={learning_rate:g}: {bfr[0]:.2f}/{bfr[1]:.2f}% ({epochs} epochs)", flush=True)
            selected[kind] = max(family, key=lambda row: row["mean_validation_bfr"])
        write_csv(OUT / "python_algorithm_cv.csv", cv)
        selection_path.write_text(json.dumps(selected, indent=2, default=float))

    results_path = OUT / "python_all_paper_models.csv"
    final_by_key = {}
    if results_path.exists():
        with results_path.open(newline="") as stream:
            for old in csv.DictReader(stream):
                final_by_key[(old["model"], int(old["paper_row"]))] = old
    for kind, architectures in GRIDS.items():
        only_family = os.environ.get("EX3_ONLY_FAMILY")
        if only_family and kind != only_family:
            continue
        setting = selected[kind]
        for row_number, hidden in enumerate(architectures, 1):
            if row_number < 3:
                continue
            model, epochs, seconds, reason = train(
                kind, hidden, u, y, setting["learning_rate"],
                NNOE_MAX_EPOCHS if kind == "NNOE_BATCHED" else MAX_EPOCHS,
                row_number, False)
            train_prediction = predict(kind, model, u, y[:ORDER])
            prediction = predict(kind, model, ut, yt[:ORDER])
            train_rmse, train_bfr = metrics(y, train_prediction)
            rmse, bfr = metrics(yt, prediction)
            row = dict(model=kind, paper_row=row_number,
                       hidden="x".join(map(str, hidden)), layers=len(hidden),
                       parameters=model.count_params(), epochs=epochs,
                       training_rmse1=train_rmse[0], training_rmse2=train_rmse[1],
                       training_bfr1=train_bfr[0], training_bfr2=train_bfr[1],
                       test_rmse1=rmse[0], test_rmse2=rmse[1],
                       test_bfr1=bfr[0], test_bfr2=bfr[1],
                       mean_test_bfr=bfr.mean(), training_seconds=seconds,
                       seconds_per_epoch=seconds/epochs, learning_rate=setting["learning_rate"],
                       batch_size=BATCH_SIZE, stop_reason=reason)
            final_by_key[(kind, row_number)] = row
            tag = kind.lower()
            if kind == "NNOE_BATCHED":
                np.savez(OUT / f"nnoe_batched_model_{row_number:02d}.npz",
                         **{f"weight_{i}": value for i, value in enumerate(model.get_weights())})
            else:
                model.save(OUT / f"{tag}_model_{row_number:02d}.keras")
            final = [final_by_key[key] for key in sorted(final_by_key)]
            write_csv(results_path, final)
            print(f"{kind} {row['hidden']}: {bfr[0]:.2f}/{bfr[1]:.2f}% ({epochs} epochs)", flush=True)
    print("Example 3 Adam methods: updated all retained paper architectures.")


if __name__ == "__main__":
    main()
