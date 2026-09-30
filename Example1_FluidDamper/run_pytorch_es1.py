"""PyTorch NNARX/NNOE baselines for Example 1 with common MATLAB weights."""
from __future__ import annotations
import argparse, csv, time
from pathlib import Path
import numpy as np
import scipy.io as sio
import torch
from torch import nn

ORDER, WIDTH, SPLIT = 4, 8, 1600
REGULARIZATION = 1e-3
torch.set_default_dtype(torch.float64)

class Model(nn.Module):
    def __init__(self, theta: np.ndarray):
        super().__init__()
        i=0; ny=WIDTH*ORDER; nu=WIDTH*(ORDER+1)
        # Match nnoe_from_vector.m: output-history and input-history blocks.
        wy=theta[i:i+ny].reshape((ORDER,WIDTH),order="F").T; i+=ny
        wu=theta[i:i+nu].reshape((ORDER+1,WIDTH),order="F").T; i+=nu
        w1=np.concatenate((wy,wu),axis=1)
        b1=theta[i:i+WIDTH]; i+=WIDTH
        w2=theta[i:i+WIDTH].reshape(1,-1); i+=WIDTH
        b2=theta[i:i+1]
        self.w1=nn.Parameter(torch.from_numpy(w1.copy()))
        self.b1=nn.Parameter(torch.from_numpy(b1.copy()))
        self.w2=nn.Parameter(torch.from_numpy(w2.copy()))
        self.b2=nn.Parameter(torch.from_numpy(b2.copy()))
    def forward(self,x):
        return (torch.tanh(x@self.w1.T+self.b1)@self.w2.T+self.b2).squeeze(-1)

def regressors(u,y):
    x=[]
    for k in range(ORDER,len(u)):
        x.append(np.r_[y[k-1:k-ORDER-1:-1] if k>ORDER else y[k-1::-1],u[k:k-ORDER-1:-1] if k>ORDER else u[k::-1]])
    return torch.tensor(np.asarray(x)),torch.tensor(y[ORDER:])

def simulate(model,u,y0,grad=False):
    u=torch.as_tensor(u); history=[torch.as_tensor(v) for v in y0[:ORDER]]
    context=torch.enable_grad() if grad else torch.no_grad()
    with context:
        for k in range(ORDER,len(u)):
            phi=torch.stack(history[-ORDER:][::-1]+[u[j] for j in range(k,k-ORDER-1,-1)])
            history.append(model(phi.unsqueeze(0))[0])
        return torch.stack(history)

def metrics(y,p):
    y=y[ORDER:]; p=p[ORDER:]; e=p-y
    return float(np.sqrt(np.mean(e*e))),float(100*(1-np.linalg.norm(e)/np.linalg.norm(y-y.mean())))

def train_nnarx(theta,u,y,lr,batch,epochs,timeout_seconds):
    model=Model(theta); x,target=regressors(u,y)
    opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=REGULARIZATION); best=np.inf; stale=0; start=time.perf_counter()
    generator=torch.Generator().manual_seed(2026); timed_out=False
    for epoch in range(1,epochs+1):
        for idx in torch.randperm(len(x),generator=generator).split(batch):
            opt.zero_grad(); loss=torch.mean((model(x[idx])-target[idx])**2); loss.backward(); opt.step()
        value=float(torch.mean((model(x)-target)**2))
        if value < best-1e-8: best=value; stale=0
        else: stale+=1
        if time.perf_counter()-start >= timeout_seconds: timed_out=True; break
        if stale>=40: break
    return model,epoch,time.perf_counter()-start,timed_out

def train_nnoe(theta,u,y,lr,epochs,timeout_seconds=None):
    model=Model(theta); target=torch.tensor(y); opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=REGULARIZATION)
    best=np.inf; best_state=None; stale=0; start=time.perf_counter(); timed_out=False
    for epoch in range(1,epochs+1):
        opt.zero_grad(); pred=simulate(model,u,y[:ORDER],grad=True)
        loss=torch.mean((pred[ORDER:]-target[ORDER:])**2); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),10.0); opt.step(); value=float(loss)
        if np.isfinite(value) and value < best-1e-7:
            best=value; stale=0; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else: stale+=1
        if timeout_seconds is not None and time.perf_counter()-start >= timeout_seconds:
            timed_out=True; break
        if stale>=40: break
    if best_state is not None: model.load_state_dict(best_state)
    return model,epoch,time.perf_counter()-start,timed_out

def train_nnoe_batched(theta,u,y,lr,sequence_length,epochs,timeout_seconds):
    """Truncated simulation-error training on measured-initialized subsequences."""
    model=Model(theta); opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=REGULARIZATION)
    starts=list(range(0,len(u)-ORDER,sequence_length-ORDER))
    best=np.inf; best_state=None; stale=0; start=time.perf_counter(); timed_out=False
    for epoch in range(1,epochs+1):
        losses=[]
        for first in starts:
            last=min(first+sequence_length,len(u))
            if last-first <= ORDER: continue
            target=torch.tensor(y[first:last]); opt.zero_grad()
            pred=simulate(model,u[first:last],y[first:first+ORDER],grad=True)
            loss=torch.mean((pred[ORDER:]-target[ORDER:])**2); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),10.0); opt.step()
            losses.append(float(loss))
            if time.perf_counter()-start >= timeout_seconds:
                timed_out=True; break
        value=float(np.mean(losses))
        if np.isfinite(value) and value < best-1e-7:
            best=value; stale=0; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else: stale+=1
        if timed_out or stale>=40: break
    if best_state is not None: model.load_state_dict(best_state)
    return model,epoch,time.perf_counter()-start,timed_out

def write(path,rows):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

def report(log_path, operation, result):
    line=f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {operation} | result: {result}"
    print(line,flush=True)
    with log_path.open("a",encoding="utf-8") as f:
        f.write(line+"\n")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--method",choices=("nnarx","nnoe","nnoe_batched"),required=True)
    ap.add_argument("--data",type=Path,required=True); ap.add_argument("--initializations",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True); ap.add_argument("--timeout-seconds",type=float,default=60.0)
    ap.add_argument("--runs",type=int,required=True); ap.add_argument("--log",type=Path,required=True)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    if a.timeout_seconds <= 0: raise ValueError("--timeout-seconds must be positive")
    d=sio.loadmat(a.data); init=sio.loadmat(a.initializations)["theta0"]
    u=d["u_train"].ravel(); y=d["y_train"].ravel(); uv=d["u_valid"].ravel(); yv=d["y_valid"].ravel()
    if not 1 <= a.runs <= init.shape[1]: raise ValueError("--runs exceeds available common initializations")
    report(a.log,f"Started PyTorch {a.method.upper()}",f"{a.runs} common initializations, regularization={REGULARIZATION:.6g}")
    if a.method=="nnarx": settings=((1e-3,32),(5e-4,32),(1e-3,64),(5e-4,64))
    elif a.method=="nnoe": settings=((1e-3,0),(5e-4,0),(1e-4,0))
    else: settings=tuple((lr,length) for length in (32,64,128) for lr in (1e-3,5e-4,1e-4))
    cv=[]
    for j,(lr,batch) in enumerate(settings,1):
        if a.method=="nnarx": model,epochs,seconds,timed_out=train_nnarx(init[:,0],u[:SPLIT],y[:SPLIT],lr,batch,300,a.timeout_seconds)
        elif a.method=="nnoe": model,epochs,seconds,timed_out=train_nnoe(init[:,0],u[:SPLIT],y[:SPLIT],lr,250,a.timeout_seconds)
        else: model,epochs,seconds,timed_out=train_nnoe_batched(init[:,0],u[:SPLIT],y[:SPLIT],lr,batch,250,a.timeout_seconds)
        p=simulate(model,u[SPLIT:],y[SPLIT:SPLIT+ORDER]).numpy(); rmse,fit=metrics(y[SPLIT:],p)
        cv.append(dict(candidate=j,learning_rate=lr,batch_size=batch,validation_RMSE=rmse,validation_FIT=fit,epochs=epochs,seconds=seconds,timedOut=timed_out,timeoutSeconds=a.timeout_seconds))
        report(a.log,f"PyTorch {a.method.upper()} hyperparameter candidate {j}",f"learning_rate={lr:.6g}, batch_size={batch}, validation RMSE={rmse:.6g}, FIT={fit:.6g}%, epochs={epochs}, time={seconds:.6g} s, timed_out={timed_out}")
    write(a.out/f"{a.method}_pytorch_validation.csv",cv); selected=max(cv,key=lambda r:r["validation_FIT"])
    report(a.log,f"PyTorch {a.method.upper()} hyperparameter selection",f"candidate={selected['candidate']}, learning_rate={selected['learning_rate']:.6g}, batch_size={selected['batch_size']}, validation FIT={selected['validation_FIT']:.6g}%")
    rows=[]
    for run in range(a.runs):
        timed_out=False
        if a.method=="nnarx": model,epochs,seconds,timed_out=train_nnarx(init[:,run],u,y,selected["learning_rate"],selected["batch_size"],1000,a.timeout_seconds)
        elif a.method=="nnoe": model,epochs,seconds,timed_out=train_nnoe(init[:,run],u,y,selected["learning_rate"],1000,a.timeout_seconds)
        else: model,epochs,seconds,timed_out=train_nnoe_batched(init[:,run],u,y,selected["learning_rate"],int(selected["batch_size"]),1000,a.timeout_seconds)
        p=simulate(model,uv,yv[:ORDER]).numpy(); rmse,fit=metrics(yv,p)
        rows.append(dict(run=run+1,testRMSE=rmse,testFIT=fit,seconds=seconds,epochs=epochs,learningRate=selected["learning_rate"],batchSize=selected["batch_size"],timedOut=timed_out,timeoutSeconds=a.timeout_seconds))
        report(a.log,f"PyTorch {a.method.upper()} identification from initialization {run+1}",f"test RMSE={rmse:.6g}, FIT={fit:.6g}%, epochs={epochs}, time={seconds:.6g} s, timed_out={timed_out}")
    write(a.out/f"{a.method}_pytorch_runs.csv",rows)
    report(a.log,f"Computed means for PyTorch {a.method.upper()}",f"RMSE={np.mean([r['testRMSE'] for r in rows]):.6g}, FIT={np.mean([r['testFIT'] for r in rows]):.6g}%, time={np.mean([r['seconds'] for r in rows]):.6g} s")
    report(a.log,f"Completed PyTorch {a.method.upper()} results",f"{len(rows)} rows retained for aggregate CSV")

if __name__=="__main__": main()
