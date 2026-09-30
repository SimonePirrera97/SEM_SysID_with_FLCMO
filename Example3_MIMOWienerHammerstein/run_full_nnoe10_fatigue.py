"""One full-record NNOE[10] Adam fit, stopped by FIT fatigue or 15 minutes."""
from __future__ import annotations
import csv, importlib.util, os
from pathlib import Path
import numpy as np
import scipy.io as sio
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")
HERE=Path(__file__).resolve().parent;OUT=HERE/"results"/"replicates";OUT.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location("full",HERE/"run_full_sequence_baselines.py");m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def main():
 tr=sio.loadmat(HERE/"data"/"dataset_training.mat");te=sio.loadmat(HERE/"data"/"dataset_test.mat")
 u=tr["utrain"][:3500].astype("float32");y=tr["ytrain"][:3500,:2].astype("float32")
 ut=te["u_test"][:5000].astype("float32");yt=te["y_test"][:5000,:2].astype("float32")
 seed=3003;time_cap=900
 model,e,s,reason=m.train("NNOE_FULL",(10,),u,y,3e-3,10000,seed,max_seconds=time_cap)
 pt=m.predict("NNOE_FULL",model,u,y[:3]);p=m.predict("NNOE_FULL",model,ut,yt[:3]);trm,trb=m.metrics(y,pt);rm,b=m.metrics(yt,p)
 row=dict(method="NNOE_ADAM_FULL",initialization_seed=seed,paper_row=3,hidden="10",layers=1,parameters=172,
  training_rmse1=trm[0],training_rmse2=trm[1],training_bfr1=trb[0],training_bfr2=trb[1],test_rmse1=rm[0],test_rmse2=rm[1],
  test_bfr1=b[0],test_bfr2=b[1],mean_test_bfr=b.mean(),training_seconds=s,epochs=e,seconds_per_epoch=s/e,
  learning_rate=3e-3,regularization=1e-3,sequence_length=3500,batch_size=1,monitor_every=10,minimum_epochs=200,
  fit_patience=10,fit_min_delta=1e-2,time_cap_seconds=time_cap,stop_reason=reason)
 with (OUT/"nnoe10_full_timecap900.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=row);w.writeheader();w.writerow(row)
 np.savez(OUT/"nnoe10_full_timecap900.npz",**{f"weight_{i}":v for i,v in enumerate(model.get_weights())})
 print(f"Full NNOE[10]: {b[0]:.2f}/{b[1]:.2f}%, {e} epochs, {s:.2f}s, {reason}",flush=True)
if __name__=="__main__":main()
