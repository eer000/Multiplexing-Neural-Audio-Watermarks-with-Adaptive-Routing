import torch
import torch.nn.functional as F
from .models import local_noise_limit
def constrain(x,d,mode):
    p=F.avg_pool1d(x.square(),641,1,320)
    if mode=="allocation":
        # Smooth energy gate; an energy proxy, not a voiced-speech classifier.
        gate=(p/(p.mean(-1,keepdim=True).clamp_min(1e-12))).sqrt().clamp(max=1)
        d=d*gate
    strong=d if mode=='off' else local_noise_limit(x,d,30)
    if mode in ("quiet_only","active_only"):
        quiet=p<=torch.quantile(p,.2,dim=-1,keepdim=True)
        strong=torch.where(quiet if mode=="quiet_only" else ~quiet,strong,d)
    d=strong
    return d*(x.square().mean(-1,keepdim=True)*10**(-26/10)/d.square().mean(-1,keepdim=True).clamp_min(1e-12)).sqrt().clamp(max=1)
