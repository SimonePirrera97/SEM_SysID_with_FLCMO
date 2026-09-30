"""Full-record, untruncated BPTT for the retained Example-3 Adam models."""
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
OUT = HERE / "results" / "full_sequence"
OUT.mkdir(parents=True, exist_ok=True)
ORDER, FIT_END = 3, 2800
REGULARIZATION = 1e-3
MAX_EPOCHS = 1000
MIN_EPOCHS, MONITOR_EVERY, FIT_PATIENCE, FIT_MIN_DELTA = 200, 10, 10, 1e-2
LEARNING_RATE = 3e-3
GRIDS = {
    "NNOE_FULL": [(3, (10,)), (4, (15,)), (6, (5, 5)),
                  (7, (7, 7)), (10, (5, 5, 5))],
    "LSTM": [(3, (4,)), (4, (5,)), (6, (2, 2)),
             (7, (3, 3)), (10, (2, 2, 2))],
    "GRU": [(3, (5,)), (4, (7,)), (6, (3, 3)),
            (7, (4, 4)), (10, (2, 2, 2))],
}
REPRESENTATIVE = {kind: rows[2][1] for kind, rows in GRIDS.items()}


def metrics(y, p):
    rmse = np.sqrt(np.mean((p-y)**2, axis=0))
    den = np.linalg.norm(y-y.mean(axis=0), axis=0)
    return rmse, 100*(1-np.linalg.norm(p-y, axis=0)/den)


def parameter_penalty(model):
    return REGULARIZATION * tf.add_n(
        [tf.reduce_sum(tf.square(v)) for v in model.trainable_variables])


def recurrent(kind, hidden, nin, nout, seed):
    keras.utils.set_random_seed(seed)
    cls = keras.layers.LSTM if kind == "LSTM" else keras.layers.GRU
    return keras.Sequential([
        keras.Input((None, nin)),
        *(cls(width, return_sequences=True, activation="tanh") for width in hidden),
        keras.layers.TimeDistributed(keras.layers.Dense(nout)),
    ])


class FullNNOE(keras.Model):
    def __init__(self, hidden, nout, seed):
        super().__init__(); keras.utils.set_random_seed(seed)
        self.hidden_layers = [keras.layers.Dense(w, activation="tanh") for w in hidden]
        self.output_layer = keras.layers.Dense(nout)

    def one_step(self, value):
        for layer in self.hidden_layers: value = layer(value)
        return self.output_layer(value)

    def simulate(self, u, y0):
        # State is [y(k-1),y(k-2),y(k-3)]; tf.scan preserves full BPTT.
        indexes=(tf.range(ORDER,tf.shape(u)[0])[:,None]
                 - tf.range(ORDER+1)[None,:])
        gathered=tf.gather(u,indexes)
        ureg=tf.reshape(gathered,(tf.shape(gathered)[0],-1))
        initial = (y0[2], y0[1], y0[0])
        def advance(state, uk):
            pred = self.one_step(tf.concat([state[0],state[1],state[2],uk],axis=0)[None])[0]
            return pred,state[0],state[1]
        states = tf.scan(advance, ureg, initializer=initial)
        return tf.concat([y0, states[0]], axis=0)


def predict(kind, model, u, y0):
    if kind == "NNOE_FULL": return model.simulate(tf.constant(u),tf.constant(y0)).numpy()
    return model(tf.constant(u[None]),training=False).numpy()[0]


def train(kind, hidden, u, y, lr, epochs, seed, max_seconds=None):
    if kind == "NNOE_FULL":
        model=FullNNOE(hidden,y.shape[1],seed)
        model.one_step(tf.zeros((1,ORDER*y.shape[1]+(ORDER+1)*u.shape[1])))
    else:
        model=recurrent(kind,hidden,u.shape[1],y.shape[1],seed)
    optimizer=keras.optimizers.Adam(lr)
    tu=tf.constant(u); ty=tf.constant(y)
    @tf.function(reduce_retracing=True)
    def step():
        with tf.GradientTape() as tape:
            pred=(model.simulate(tu,ty[:ORDER]) if kind=="NNOE_FULL"
                  else model(tu[None],training=True)[0])
            data=tf.reduce_sum(tf.square(pred-ty))
            objective=data+parameter_penalty(model)
        grad=tape.gradient(objective,model.trainable_variables)
        grad,_=tf.clip_by_global_norm(grad,100.0)
        optimizer.apply_gradients(zip(grad,model.trainable_variables))
        return objective
    best=-np.inf; weights=None; stale=0; start=time.perf_counter()
    stop_reason="maximum epochs"
    for epoch in range(1,epochs+1):
        value=float(step())
        if not np.isfinite(value):
            stop_reason="non-finite objective"
            break
        if epoch % MONITOR_EVERY == 0:
            p=predict(kind,model,u,y[:ORDER])
            den=np.linalg.norm(y-y.mean(axis=0))
            fit=100*(1-np.linalg.norm(p-y)/den)
            if np.isfinite(fit) and fit > best+FIT_MIN_DELTA:
                best=fit; stale=0; weights=[v.numpy().copy() for v in model.weights]
            else: stale+=1
            if epoch>=MIN_EPOCHS and stale>=FIT_PATIENCE:
                stop_reason="training FIT stabilized"
                break
        if max_seconds is not None and time.perf_counter()-start >= max_seconds:
            stop_reason="time cap"
            break
    if weights is not None: model.set_weights(weights)
    return model,epoch,time.perf_counter()-start,stop_reason


def write_csv(path,rows):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)


def main():
    tr=sio.loadmat(HERE/"data"/"dataset_training.mat")
    te=sio.loadmat(HERE/"data"/"dataset_test.mat")
    u=tr["utrain"][:3500].astype("float32"); y=tr["ytrain"][:3500,:2].astype("float32")
    ut=te["u_test"][:5000].astype("float32"); yt=te["y_test"][:5000,:2].astype("float32")
    # eta=3e-3 was selected for every family in the preceding Example-3
    # validation searches; only FL-CMO tau is re-searched in this experiment.
    selected={kind:{"learning_rate":LEARNING_RATE,"regularization":REGULARIZATION,
                    "sequence_length":"full","batch_size":1} for kind in GRIDS}
    (OUT/"adam_selection.json").write_text(json.dumps(selected,indent=2))
    final=[]
    for kind,models in GRIDS.items():
        lr=selected[kind]["learning_rate"]
        for paper_row,hidden in models:
            model,epochs,seconds,reason=train(kind,hidden,u,y,lr,MAX_EPOCHS,paper_row)
            ptr=predict(kind,model,u,y[:ORDER]); p=predict(kind,model,ut,yt[:ORDER])
            trm,tb=metrics(y,ptr); rm,b=metrics(yt,p)
            row=dict(model=kind,paper_row=paper_row,hidden="x".join(map(str,hidden)),layers=len(hidden),
                     parameters=sum(int(np.prod(v.shape)) for v in model.trainable_variables),epochs=epochs,training_rmse1=trm[0],training_rmse2=trm[1],
                     training_bfr1=tb[0],training_bfr2=tb[1],test_rmse1=rm[0],test_rmse2=rm[1],
                     test_bfr1=b[0],test_bfr2=b[1],mean_test_bfr=b.mean(),training_seconds=seconds,
                     seconds_per_epoch=seconds/epochs,learning_rate=lr,regularization=REGULARIZATION,
                     sequence_length=len(u),batch_size=1,stop_reason=reason)
            final.append(row); write_csv(OUT/"adam_full_sequence_models.csv",final)
            if kind=="NNOE_FULL":
                np.savez(OUT/f"nnoe_full_row_{paper_row:02d}.npz",**{f"weight_{i}":v for i,v in enumerate(model.get_weights())})
            else: model.save(OUT/f"{kind.lower()}_full_row_{paper_row:02d}.keras")
            print(f"{kind} row {paper_row} {row['hidden']}: {b[0]:.2f}/{b[1]:.2f}%, {seconds:.2f}s",flush=True)


if __name__=="__main__": main()
