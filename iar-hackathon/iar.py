"""Interference-aware allocation from frozen prototypes."""
import numpy as np
def compute_prototypes(embeddings,labels,class_ids):
 out={}
 for c in class_ids:
  p=embeddings[labels==c].mean(0);out[int(c)]=(p/(np.linalg.norm(p)+1e-8)).astype(np.float32)
 return out
def compute_risk(old_classes,new_classes,prototypes):
 if not old_classes:return np.empty(0,dtype=np.float32)
 return (np.stack([prototypes[c] for c in old_classes])@np.stack([prototypes[c] for c in new_classes]).T).max(1)
def allocate_budget(risk,total_budget,floor=1,temp=1.0):
 n=len(risk)
 if not n:return np.empty(0,dtype=np.int64)
 if total_budget<n*floor:q=np.zeros(n,dtype=np.int64);q[np.argsort(-risk)[:total_budget]]=1;return q
 remaining=total_budget-n*floor;z=(risk-risk.max())/max(temp,1e-6);p=np.exp(z);p/=p.sum();raw=p*remaining;extra=np.floor(raw).astype(int)
 for i in np.argsort(-(raw-extra))[:remaining-extra.sum()]:extra[i]+=1
 return extra+floor
def uniform_budget(n_classes,total_budget):return allocate_budget(np.zeros(n_classes),total_budget)
