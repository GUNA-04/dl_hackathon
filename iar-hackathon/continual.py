import numpy as np,torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
from iar import compute_prototypes,compute_risk,allocate_budget,uniform_budget
class SimpleHead(nn.Module):
 def __init__(self,d,n):super().__init__();self.fc=nn.Linear(d,n)
 def forward(self,x):return self.fc(x)
def lab(labels,idx,classes):m={int(c):i for i,c in enumerate(classes)};return np.array([m[int(c)] for c in labels[idx]],dtype=np.int64)
def expand(old,d,n,dev):
 m=SimpleHead(d,n).to(dev)
 if old is not None:
  with torch.no_grad():m.fc.weight[:old.fc.out_features].copy_(old.fc.weight);m.fc.bias[:old.fc.out_features].copy_(old.fc.bias)
 return m
def train(m,e,idx,y,cs,epochs,dev):
 dl=DataLoader(TensorDataset(torch.from_numpy(e[idx]).float(),torch.from_numpy(lab(y,idx,cs))),128,shuffle=True);o=optim.SGD(m.parameters(),lr=.05,momentum=.9,weight_decay=1e-4);m.train()
 for _ in range(epochs):
  for x,z in dl:o.zero_grad();l=nn.functional.cross_entropy(m(x.to(dev)),z.to(dev));l.backward();o.step()
def ev(m,e,idx,y,cs,dev):
 with torch.no_grad():return float((m(torch.from_numpy(e[idx]).float().to(dev)).argmax(1)==torch.from_numpy(lab(y,idx,cs)).to(dev)).float().mean())
def ev_taskwise(m,e,idx,y,task_classes,seen,dev):
 """Task-incremental eval: restrict logits to THIS task's class columns, then argmax.
 seen = full ordered class list the head was trained with (column order).
 task_classes = the classes belonging to the task being scored."""
 col={int(c):i for i,c in enumerate(seen)}
 cols=torch.tensor([col[int(c)] for c in task_classes],device=dev)
 with torch.no_grad():
  logits=m(torch.from_numpy(e[idx]).float().to(dev))[:,cols]
  local=logits.argmax(1)
  true=torch.from_numpy(lab(y,idx,list(task_classes))).to(dev)
  return float((local==true).float().mean())
def run_continual_learning(e,y,tasks,epochs=20,device="cpu",replay_buffer=None,method="naive",tau=1.,seed=42):
 torch.manual_seed(seed);rng=np.random.default_rng(seed);n=len(tasks);R=np.full((n,n),np.nan,np.float32);m=None;seen=[];rows=[]
 for t in range(n):
  info=tasks[t];seen=sorted(seen+info["classes"]);m=expand(m,e.shape[1],len(seen),device);r=np.empty(0,dtype=np.int64) if replay_buffer is None else replay_buffer.sample();train(m,e,np.concatenate([info["train_idx"],r]),y,seen,epochs,device)
  if replay_buffer is not None:
   cand=np.concatenate([r,info["train_idx"]])
   if method in {"iar", "iar_random"} and t<n-1:
    # INVERTED ALLOCATION: we measured r = -0.75, i.e. classes MOST similar to the
    # incoming task forget LEAST (they are implicitly rehearsed by the overlapping
    # features). So we spend the buffer on the LEAST similar (isolated) classes.
    # Note: -risk is passed to allocate_budget only; `rows` still records the TRUE
    # similarity so risk_scores.csv keeps reporting the real r = -0.75 finding.
    nxt=tasks[t+1]["classes"];risk=compute_risk(seen,nxt,compute_prototypes(e,y,seen+nxt));q=allocate_budget(-risk,replay_buffer.capacity,temp=tau);sel="random" if method=="iar_random" else "herding";rows += [{"transition":t+1,"class_id":int(c),"predicted_risk":float(v)} for c,v in zip(seen,risk)]
   else:q=uniform_budget(len(seen),replay_buffer.capacity);sel="random" if method=="random" else "herding"
   replay_buffer.update(cand,seen,q,sel,rng)
  for j in range(t+1):R[t,j]=ev_taskwise(m,e,tasks[j]["val_idx"],y,tasks[j]["classes"],seen,device)
 return R,rows
def run_naive(*a,**k):return run_continual_learning(*a,method="naive",**k)
def run_joint_training(e,y,tasks,epochs=20,device="cpu",seed=42):
 cs=sorted(c for z in tasks.values() for c in z["classes"]);idx=np.concatenate([z["train_idx"] for z in tasks.values()]);m=SimpleHead(e.shape[1],len(cs)).to(device);train(m,e,idx,y,cs,epochs,device);R=np.full((len(tasks),len(tasks)),np.nan,np.float32)
 for j,z in tasks.items():R[-1,j]=ev_taskwise(m,e,z["val_idx"],y,z["classes"],cs,device)
 return R,[]
