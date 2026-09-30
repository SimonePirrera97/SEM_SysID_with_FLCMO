"""Retained Example-3 LSTM/GRU models with length-512 batched BPTT."""
from __future__ import annotations
import csv, os, time
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")
import numpy as np
import scipy.io as sio
import tensorflow as tf
from tensorflow import keras

HERE=Path(__file__).resolve().parent
OUT=HERE/"results"/"fatigue_common"; OUT.mkdir(parents=True,exist_ok=True)
LENGTH=512; BATCH_SIZE=64; STRIDE=8; REGULARIZATION=1e-3
LEARNING_RATE=3e-3; MAX_EPOCHS=10000
MIN_EPOCHS=200; MONITOR_EVERY=10; FIT_PATIENCE=10; FIT_MIN_DELTA=1e-2
GRIDS={
    "LSTM":[(3,(4,)),(4,(5,)),(6,(2,2)),(7,(3,3)),(10,(2,2,2))],
    "GRU":[(3,(5,)),(4,(7,)),(6,(3,3)),(7,(4,4)),(10,(2,2,2))],
}

def metrics(y,p):
    rm=np.sqrt(np.mean((p-y)**2,axis=0)); den=np.linalg.norm(y-y.mean(axis=0),axis=0)
    return rm,100*(1-np.linalg.norm(p-y,axis=0)/den)

def windows(u,y):
    starts=np.arange(0,len(u)-LENGTH+1,STRIDE)
    return np.stack([u[s:s+LENGTH] for s in starts]),np.stack([y[s:s+LENGTH] for s in starts])

def network(kind,hidden,nin,nout,seed):
    keras.utils.set_random_seed(seed)
    cls=keras.layers.LSTM if kind=="LSTM" else keras.layers.GRU
    return keras.Sequential([keras.Input((None,nin)),
        *(cls(w,return_sequences=True,activation="tanh") for w in hidden),
        keras.layers.TimeDistributed(keras.layers.Dense(nout))])

def predict(model,u): return model(tf.constant(u[None]),training=False).numpy()[0]

def train(kind,hidden,u,y,seed,learning_rate=LEARNING_RATE):
    model=network(kind,hidden,u.shape[1],y.shape[1],seed); xu,yu=windows(u,y)
    optimizer=keras.optimizers.Adam(learning_rate); rng=np.random.default_rng(seed+1000)
    full_scale=float(len(u)*y.shape[1]); best=-np.inf; weights=None; stale=0
    @tf.function(reduce_retracing=True)
    def step(ub,yb):
        with tf.GradientTape() as tape:
            p=model(ub,training=True); data=tf.reduce_mean(tf.square(p-yb))*full_scale
            reg=REGULARIZATION*tf.add_n([tf.reduce_sum(tf.square(v)) for v in model.trainable_variables])
            loss=data+reg
        g=tape.gradient(loss,model.trainable_variables); g,_=tf.clip_by_global_norm(g,100.0)
        optimizer.apply_gradients(zip(g,model.trainable_variables)); return loss
    start=time.perf_counter()
    for epoch in range(1,MAX_EPOCHS+1):
        perm=rng.permutation(len(xu))
        for first in range(0,len(xu),BATCH_SIZE):
            ix=perm[first:first+BATCH_SIZE]
            value=float(step(tf.constant(xu[ix]),tf.constant(yu[ix])))
        if not np.isfinite(value): raise RuntimeError("non-finite objective before fatigue")
        if epoch%MONITOR_EVERY==0:
            p=predict(model,u); fit=100*(1-np.linalg.norm(p-y)/np.linalg.norm(y-y.mean(axis=0)))
            if np.isfinite(fit) and fit>best+FIT_MIN_DELTA:
                best=fit; stale=0; weights=[v.numpy().copy() for v in model.weights]
            else: stale+=1
            if epoch>=MIN_EPOCHS and stale>=FIT_PATIENCE: break
    if weights is not None:model.set_weights(weights)
    if epoch>=MAX_EPOCHS:raise RuntimeError("maximum epoch cap reached without fatigue")
    return model,epoch,time.perf_counter()-start,"training FIT stabilized",len(xu)

def write(path,rows):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

def main():
    tr=sio.loadmat(HERE/"data"/"dataset_training.mat");te=sio.loadmat(HERE/"data"/"dataset_test.mat")
    u=tr["utrain"][:3500].astype("float32");y=tr["ytrain"][:3500,:2].astype("float32")
    ut=te["u_test"][:5000].astype("float32");yt=te["y_test"][:5000,:2].astype("float32")
    path=OUT/"recurrent_l512.csv"; rows=[]
    for kind,models in GRIDS.items():
        for paper_row,hidden in models:
            try:
                m,e,s,reason,nwin=train(kind,hidden,u,y,paper_row)
                lr=LEARNING_RATE
            except RuntimeError as exc:
                # A divergence is not a fatigue stop; retry once at the next
                # stable Adam rate and accept only a genuine fatigue stop.
                print(f"{kind} row {paper_row}: {exc}; retrying eta=1e-3",flush=True)
                m,e,s,reason,nwin=train(kind,hidden,u,y,paper_row,1e-3);lr=1e-3
            pt=predict(m,u);p=predict(m,ut);trm,trb=metrics(y,pt);rm,b=metrics(yt,p)
            row=dict(model=kind,horizon=LENGTH,batch_size=BATCH_SIZE,windows=nwin,paper_row=paper_row,
                hidden="x".join(map(str,hidden)),layers=len(hidden),parameters=m.count_params(),epochs=e,
                training_rmse1=trm[0],training_rmse2=trm[1],training_bfr1=trb[0],training_bfr2=trb[1],
                test_rmse1=rm[0],test_rmse2=rm[1],test_bfr1=b[0],test_bfr2=b[1],mean_test_bfr=b.mean(),
                training_seconds=s,seconds_per_epoch=s/e,learning_rate=lr,regularization=REGULARIZATION,
                stop_reason=reason)
            rows.append(row);write(path,rows);m.save(OUT/f"{kind.lower()}_l512_row_{paper_row:02d}.keras")
            print(f"{kind} L=512 row {paper_row} {row['hidden']}: {b[0]:.2f}/{b[1]:.2f}%, {e} epochs, {s:.2f}s",flush=True)
if __name__=="__main__":main()
