"""Four additional initializations for retained LSTM, GRU and NNOE-Adam."""
from __future__ import annotations
import csv, importlib.util, os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import scipy.io as sio

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")
HERE=Path(__file__).resolve().parent; OUT=HERE/"results"/"replicates"; OUT.mkdir(parents=True,exist_ok=True)

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
rmod=load("recurrent_l512",HERE/"run_recurrent_l512.py")
nmod=load("nnoe_horizons",HERE/"run_batched_nnoe_horizons.py")
GRIDS={
 "NNOE_ADAM":[(3,(10,)),(4,(15,)),(6,(5,5)),(7,(7,7)),(10,(5,5,5))],
 "LSTM":rmod.GRIDS["LSTM"], "GRU":rmod.GRIDS["GRU"]}
PATH=OUT/"adam_replicates.csv"
FAILURES=OUT/"adam_failed_attempts.csv"

def metrics(y,p):
    rm=np.sqrt(np.mean((p-y)**2,axis=0)); den=np.linalg.norm(y-y.mean(axis=0),axis=0)
    return rm,100*(1-np.linalg.norm(p-y,axis=0)/den)
def write(rows):
    with PATH.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

def record_failure(method,replicate,seed,paper_row,hidden,learning_rate,error):
    row=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),method=method,
        replicate=replicate,initialization_seed=seed,paper_row=paper_row,
        hidden="x".join(map(str,hidden)),learning_rate=learning_rate,
        regularization=1e-3,horizon=512,batch_size=64,error=str(error))
    exists=FAILURES.exists()
    with FAILURES.open("a",newline="") as f:
        w=csv.DictWriter(f,fieldnames=row.keys())
        if not exists:w.writeheader()
        w.writerow(row)

def main():
    tr=sio.loadmat(HERE/"data"/"dataset_training.mat");te=sio.loadmat(HERE/"data"/"dataset_test.mat")
    u=tr["utrain"][:3500].astype("float32");y=tr["ytrain"][:3500,:2].astype("float32")
    ut=te["u_test"][:5000].astype("float32");yt=te["y_test"][:5000,:2].astype("float32")
    rows=[]
    if PATH.exists(): rows=list(csv.DictReader(PATH.open()))
    done={(r["method"],int(r["replicate"]),int(r["paper_row"])) for r in rows}
    for replicate in (1,2,3,4):
      for method,models in GRIDS.items():
       for paper_row,hidden in models:
        key=(method,replicate,paper_row)
        if key in done: continue
        seed=1000*replicate+paper_row
        if method=="NNOE_ADAM":
            rates=(1e-3,5e-4,3e-4) if paper_row==10 else (nmod.LEARNING_RATE,1e-3,5e-4)
            for learning_rate in rates:
                try:
                    model,epochs,seconds,reason,nwin=nmod.train(hidden,u,y,512,seed,learning_rate)
                    break
                except RuntimeError as error:
                    record_failure(method,replicate,seed,paper_row,hidden,learning_rate,error)
                    nmod.tf.keras.backend.clear_session()
            else:
                raise RuntimeError(f"all stable learning-rate fallbacks failed for {key}")
            ptr=nmod.simulate_numpy(model,u,y[:nmod.ORDER]); pred=nmod.simulate_numpy(model,ut,yt[:nmod.ORDER])
            params=sum(int(np.prod(v.shape)) for v in model.trainable_variables)
        else:
            learning_rate=rmod.LEARNING_RATE
            model,epochs,seconds,reason,nwin=rmod.train(method,hidden,u,y,seed,rmod.LEARNING_RATE)
            ptr=rmod.predict(model,u);pred=rmod.predict(model,ut);params=model.count_params()
        trm,trb=metrics(y,ptr);rm,b=metrics(yt,pred)
        row=dict(method=method,replicate=replicate,initialization_seed=seed,paper_row=paper_row,
            hidden="x".join(map(str,hidden)),layers=len(hidden),parameters=params,
            training_rmse1=trm[0],training_rmse2=trm[1],training_bfr1=trb[0],training_bfr2=trb[1],
            test_rmse1=rm[0],test_rmse2=rm[1],test_bfr1=b[0],test_bfr2=b[1],mean_test_bfr=b.mean(),
            training_seconds=seconds,epochs=epochs,seconds_per_epoch=seconds/epochs,
            learning_rate=learning_rate,regularization=1e-3,horizon=512,batch_size=64,windows=nwin,
            monitor_every=10,minimum_epochs=200,fit_patience=10,fit_min_delta=1e-2,stop_reason=reason)
        rows.append(row);write(rows)
        if method=="NNOE_ADAM":
            np.savez(OUT/f"nnoe_adam_rep{replicate}_row_{paper_row:02d}.npz",**{f"weight_{i}":v for i,v in enumerate(model.get_weights())})
        else:model.save(OUT/f"{method.lower()}_rep{replicate}_row_{paper_row:02d}.keras")
        print(f"{method} replicate {replicate} row {paper_row}: {b[0]:.2f}/{b[1]:.2f}%, {seconds:.2f}s",flush=True)
if __name__=="__main__":main()
