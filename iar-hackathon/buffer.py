"""Index-based replay buffers."""
import numpy as np
class ReplayBuffer:
 def __init__(self,embeddings,labels,capacity=100,temp=1.0):self.embeddings,self.labels,self.capacity,self.temp=embeddings,labels,capacity,temp;self.samples=np.empty(0,dtype=np.int64)
 def sample(self):return self.samples.copy()
 def update(self,new_indices,classes,quotas,selection,rng):
  picks=[]
  for c,k in zip(classes,quotas):
   candidates=np.unique(np.concatenate([self.samples[self.labels[self.samples]==c],new_indices[self.labels[new_indices]==c]]))
   if len(candidates) and k>0:
    if selection=="random":picks.append(rng.choice(candidates,min(int(k),len(candidates)),replace=False))
    else:
     x=self.embeddings[candidates].copy();p=x.mean(0);p/=np.linalg.norm(p)+1e-8;x/=np.linalg.norm(x,axis=1,keepdims=True)+1e-8;picks.append(candidates[np.argsort(-(x@p))[:min(int(k),len(candidates))]])
  self.samples=np.concatenate(picks).astype(np.int64) if picks else np.empty(0,dtype=np.int64)
class RandomBuffer(ReplayBuffer):pass
class HerdingBuffer(ReplayBuffer):pass
class IARBuffer(ReplayBuffer):pass
