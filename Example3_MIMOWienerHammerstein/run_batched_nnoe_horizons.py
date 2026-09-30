"""Retained NNOE-Adam models at horizons 64 and 521 with FIT fatigue."""
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
ORDER=3; BATCH_SIZE=64; REGULARIZATION=1e-3; LEARNING_RATE=3e-3
MIN_EPOCHS=200; MONITOR_EVERY=10; FIT_PATIENCE=10; FIT_MIN_DELTA=1e-2
MAX_EPOCHS=10000
MODELS=[(3,(10,)),(4,(15,)),(6,(5,5)),(7,(7,7)),(10,(5,5,5))]
HORIZONS=(64,521)

def metrics(y,p):
    rm=np.sqrt(np.mean((p-y)**2,axis=0)); den=np.linalg.norm(y-y.mean(axis=0),axis=0)
    return rm,100*(1-np.linalg.norm(p-y,axis=0)/den)

class NNOE(keras.Model):
    def __init__(self,hidden,nout,seed):
        super().__init__(); keras.utils.set_random_seed(seed)
        self.hidden_layers=[keras.layers.Dense(w,activation="tanh") for w in hidden]
        self.output_layer=keras.layers.Dense(nout)
    def one_step(self,x):
        for layer in self.hidden_layers: x=layer(x)
        return self.output_layer(x)
    def simulate(self,u,y0):
        outputs=[y0[:,k] for k in range(ORDER)]
        length=u.shape[1]
        for k in range(ORDER,length):
            reg=tf.concat([outputs[k-j] for j in range(1,ORDER+1)]+[u[:,k-j] for j in range(ORDER+1)],axis=1)
            outputs.append(self.one_step(reg))
        return tf.stack(outputs,axis=1)

def simulate_numpy(model,u,y0):
    y=np.zeros((len(u),y0.shape[1]),np.float32); y[:ORDER]=y0[:ORDER]
    ws=[(l.kernel.numpy(),l.bias.numpy()) for l in model.hidden_layers]
    ow=(model.output_layer.kernel.numpy(),model.output_layer.bias.numpy())
    for k in range(ORDER,len(u)):
        z=np.concatenate([*(y[k-j] for j in range(1,ORDER+1)),*(u[k-j] for j in range(ORDER+1))])
        for w,b in ws: z=np.tanh(z@w+b)
        y[k]=z@ow[0]+ow[1]
    return y

def windows(u,y,length):
    # Dense overlapping coverage, capped only to avoid redundant near-duplicates.
    stride=4 if length==64 else 8
    starts=np.arange(0,len(u)-length+1,stride)
    return np.stack([u[s:s+length] for s in starts]),np.stack([y[s:s+length] for s in starts])

def train(hidden,u,y,horizon,seed,learning_rate=LEARNING_RATE):
    model=NNOE(hidden,y.shape[1],seed); dim=ORDER*y.shape[1]+(ORDER+1)*u.shape[1]
    model.one_step(tf.zeros((1,dim))); xu,yu=windows(u,y,horizon)
    optimizer=keras.optimizers.Adam(learning_rate); rng=np.random.default_rng(seed+1000)
    full_scale=float(len(u)*y.shape[1]); best=-np.inf; weights=None; stale=0
    @tf.function(reduce_retracing=True)
    def step(ub,yb):
        with tf.GradientTape() as tape:
            p=model.simulate(ub,yb[:,:ORDER]); data=tf.reduce_mean(tf.square(p-yb))*full_scale
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
            p=simulate_numpy(model,u,y[:ORDER]); fit=100*(1-np.linalg.norm(p-y)/np.linalg.norm(y-y.mean(axis=0)))
            if np.isfinite(fit) and fit>best+FIT_MIN_DELTA:
                best=fit; stale=0; weights=[v.numpy().copy() for v in model.weights]
            else: stale+=1
            if epoch>=MIN_EPOCHS and stale>=FIT_PATIENCE: break
    if weights is not None: model.set_weights(weights)
    if epoch>=MAX_EPOCHS: raise RuntimeError("maximum epoch cap reached without fatigue")
    return model,epoch,time.perf_counter()-start,"training FIT stabilized",len(xu)

def write(path,rows):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

def main():
    tr=sio.loadmat(HERE/"data"/"dataset_training.mat");te=sio.loadmat(HERE/"data"/"dataset_test.mat")
    u=tr["utrain"][:3500].astype("float32");y=tr["ytrain"][:3500,:2].astype("float32")
    ut=te["u_test"][:5000].astype("float32");yt=te["y_test"][:5000,:2].astype("float32")
    result_path=OUT/"nnoe_adam_horizons.csv"; by_key={}
    if result_path.exists():
        with result_path.open(newline="") as f:
            for old in csv.DictReader(f): by_key[(int(old["horizon"]),int(old["paper_row"]))]=old
    for horizon in HORIZONS:
        for paper_row,hidden in MODELS:
            only_horizon=os.environ.get("EX3_HORIZON")
            only_row=os.environ.get("EX3_PAPER_ROW")
            if only_horizon and horizon!=int(only_horizon): continue
            if only_row and paper_row!=int(only_row): continue
            learning_rate=float(os.environ.get("EX3_LEARNING_RATE",LEARNING_RATE))
            m,e,s,reason,nwin=train(hidden,u,y,horizon,paper_row,learning_rate)
            pt=simulate_numpy(m,u,y[:ORDER]);p=simulate_numpy(m,ut,yt[:ORDER])
            trm,trb=metrics(y,pt);rm,b=metrics(yt,p)
            row=dict(horizon=horizon,batch_size=BATCH_SIZE,windows=nwin,paper_row=paper_row,
                hidden="x".join(map(str,hidden)),layers=len(hidden),parameters=sum(int(np.prod(v.shape)) for v in m.trainable_variables),
                epochs=e,training_rmse1=trm[0],training_rmse2=trm[1],training_bfr1=trb[0],training_bfr2=trb[1],
                test_rmse1=rm[0],test_rmse2=rm[1],test_bfr1=b[0],test_bfr2=b[1],mean_test_bfr=b.mean(),
                training_seconds=s,seconds_per_epoch=s/e,learning_rate=learning_rate,regularization=REGULARIZATION,stop_reason=reason)
            by_key[(horizon,paper_row)]=row
            rows=[by_key[k] for k in sorted(by_key)]
            write(result_path,rows)
            np.savez(OUT/f"nnoe_adam_h{horizon}_row_{paper_row:02d}.npz",**{f"weight_{i}":v for i,v in enumerate(m.get_weights())})
            print(f"h={horizon} row {paper_row} {row['hidden']}: {b[0]:.2f}/{b[1]:.2f}%, {e} epochs, {s:.2f}s",flush=True)
if __name__=="__main__":main()
