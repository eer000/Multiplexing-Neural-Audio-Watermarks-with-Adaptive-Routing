"""AudioSeal pretrained detector adaptation; independent immutable experiment dirs."""
import argparse,json,random,hashlib,time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from audioseal import AudioSeal
from speechtokenizer import SpeechTokenizer
from audioseal_st.models import local_noise_limit,GatedLongContextProxy
from audioseal_st.quiet_methods_v2 import constrain
def set_trainable(generator, mode):
 for p in generator.parameters():p.requires_grad=False
 for module in [generator.decoder,generator.msg_processor]:
  for p in module.parameters():p.requires_grad=True
from audioseal_st.metrics import operating_point

def main():
 ap=argparse.ArgumentParser(description="Feature-constrained joint adaptation of pretrained AudioSeal")
 for flag in ['manifest','data-root','output','init-generator','proxy-checkpoint','sptk-config','sptk-checkpoint']:
  ap.add_argument('--'+flag,type=Path,required=True)
 ap.add_argument('--steps',type=int,default=600);ap.add_argument('--save-every',type=int,default=100);ap.add_argument('--smoke',action='store_true')
 a=ap.parse_args();a.mode='latent_joint';out=a.output;out.mkdir(parents=True,exist_ok=False)
 torch.set_num_threads(2);torch.backends.cudnn.enabled=False;torch.manual_seed(20260915);rng=random.Random(20260915)
 joint=a.mode in ['joint','latent_joint'];old=a.mode=='det_old';steps=4 if a.smoke else a.steps;every=4 if a.smoke else a.save_every
 split=json.loads(a.manifest.read_text())
 assert set(split)=={'train','val','cal','test'}, 'Manifest requires train/val/cal/test path lists'
 groups={k:[a.data_root/p for p in split[k]] for k in split}
 assert all(groups.values())
 flat=[str(p.resolve()) for ps in groups.values() for p in ps];assert len(flat)==len(set(flat)), 'Splits overlap'
 train,val,cal,test=[groups[k] for k in ['train','val','cal','test']]
 if a.smoke:train,val,cal,test=train[:8],val[:4],cal[:4],test[:4]
 gen=AudioSeal.load_generator('audioseal_wm_16bits').cuda().eval();base=AudioSeal.load_generator('audioseal_wm_16bits').cuda().eval()
 ckpt=a.init_generator
 gen.load_state_dict(torch.load(ckpt,map_location='cuda',weights_only=True))
 det=AudioSeal.load_detector('audioseal_detector_16bits').cuda().eval();original_det=AudioSeal.load_detector('audioseal_detector_16bits').cuda().eval()
 st=SpeechTokenizer.load_from_checkpoint(str(a.sptk_config),str(a.sptk_checkpoint)).cuda().eval()
 for model in [gen,base,original_det,st]:
  for p in model.parameters():p.requires_grad=False
 if joint:set_trainable(gen,'decoder_msg')
 proxy=None
 if joint:
  proxy=GatedLongContextProxy().cuda().eval();proxy.load_state_dict(torch.load(a.proxy_checkpoint,map_location='cuda',weights_only=True))
  for p in proxy.parameters():p.requires_grad=False
 def modelhash(model):
  h=hashlib.sha256()
  for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
  return h.hexdigest()
 sthash=modelhash(st);ghash=modelhash(gen)
 meta=dict(mode=a.mode,smoke=a.smoke,steps=steps,init_generator=str(ckpt),init_generator_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),init_detector='audioseal_detector_16bits',split={k:[str(p) for p in v] for k,v in dict(train=train,val=val,cal=cal,test=test).items()},seed=20260915,seconds=4,lr_detector=1e-5,lr_generator=2e-6,quality='old local26+cap26; new strong30+cap26+attenuation to original AudioSeal residual power',protocol='Validation selection and calibration speakers disjoint; test-clean separate speakers. All negative ST branches match positive channel. Empirical test operating point descriptive only; calibrated test FPR measured, not assumed.',code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (out/'manifest.json').write_text(json.dumps(meta,indent=2))
 def stcall(x):return st.decode(st.encode(x))[...,:x.shape[-1]]
 def wm(x,msg,target):
  d=gen.get_watermark(x,16000,message=msg)
  if old:
   d=local_noise_limit(x,d,26);d=d*(x.square().mean(-1,keepdim=True)*10**(-2.6)/d.square().mean(-1,keepdim=True).clamp_min(1e-12)).sqrt().clamp(max=1)
  else:d=constrain(x,d,'calibrated')
  y=(x+d).clamp(-1,1)
  if not old:
   d=y-x;y=x+d*(target/d.square().mean(-1,keepdim=True).clamp_min(1e-12)).sqrt().clamp(max=1)
  return y
 def score(model,x):return model(x,sample_rate=16000)[0][:,1].mean(-1)
 cache={}
 torch.manual_seed(20260916)  # Same messages/noise across methods, independent of proxy initialization.
 with torch.no_grad():
  for split,paths in dict(train=train,val=val,cal=cal,test=test).items():
   items=[]
   for i,p in enumerate(paths):
    ar,sr=sf.read(p,dtype='float32',always_2d=True);assert sr==16000
    x=torch.from_numpy(ar[:64000].mean(1)).view(1,1,-1).cuda();msg=torch.randint(0,2,(1,16),device='cuda')
    target=((x+base.get_watermark(x,16000,message=msg)).clamp(-1,1)-x).square().mean(-1,keepdim=True)
    y=wm(x,msg,target);n=torch.randn_like(x);n=n/n.square().mean().sqrt().clamp_min(1e-12)*(y-x).square().mean().sqrt()
    z=(x+n).clamp(-1,1)
    d=dict(x=x,msg=msg,target=target,y=y,cx=stcall(x),cy=stcall(y),noise=stcall(z))
    items.append({k:v.cpu() for k,v in d.items()})
    if (i+1)%32==0:print(json.dumps(dict(cache=split,done=i+1,total=len(paths))),flush=True)
   cache[split]=items
   torch.save(items,out/(split+'_cache.pt'))
 def batch(items):return {k:torch.cat([t[k] for t in items]).cuda() for k in items[0]}
 def evaluate(split,step):
  rows=[]
  with torch.no_grad():
   for item in cache[split]:
    t=batch([item]);y=wm(t['x'],t['msg'],t['target']) if joint else t['y'];cy=stcall(y) if joint else t['cy']
    row=dict(positive=float(score(det,cy)),negative=float(score(det,t['cx'])),noise_negative=float(score(det,t['noise'])),original_positive=float(score(original_det,cy)),original_negative=float(score(original_det,t['cx'])),snr=float(10*torch.log10(t['x'].square().mean()/(y-t['x']).square().mean().clamp_min(1e-12))))
    if split in ['val','test']:
     from pesq import pesq
     from pystoi import stoi
     xnp=t['x'].flatten().cpu().numpy();ynp=y.flatten().cpu().numpy();row['pesq']=float(pesq(16000,xnp,ynp,'wb'));row['stoi']=float(stoi(xnp,ynp,16000))
    rows.append(row)
  (out/(split+'_scores_'+str(step)+'.json')).write_text(json.dumps(rows))
  return rows
 def summarize(rows,threshold):
  return dict(tpr=float(np.mean([r['positive']>threshold for r in rows])),fpr=float(np.mean([r['negative']>threshold for r in rows])),noise_fpr=float(np.mean([r['noise_negative']>threshold for r in rows])),threshold=float(threshold),n=len(rows),empirical=operating_point([r['positive'] for r in rows],[r['negative'] for r in rows]),original_detector_empirical=operating_point([r['original_positive'] for r in rows],[r['original_negative'] for r in rows]),quality={k:float(np.mean([r[k] for r in rows])) for k in ['snr','pesq','stoi'] if k in rows[0]})
 def threshold(rows):
  vals=sorted(r['negative'] for r in rows);return vals[-(int(.01*len(vals))+1)]
 optD=torch.optim.AdamW(det.parameters(),lr=1e-5)
 optG=torch.optim.AdamW([p for p in gen.parameters() if p.requires_grad],lr=2e-6) if joint else None
 # Validate against independent calibration at each selection stage; test only initial and selected final.
 cal0=evaluate('cal',0);val0=evaluate('val',0);test0=evaluate('test',0)
 initial=summarize(test0,threshold(cal0));(out/'initial_test.json').write_text(json.dumps(initial,indent=2))
 best=-1;history=[];start=time.time()
 for step in range(1,steps+1):
  t=batch([cache['train'][rng.randrange(len(cache['train']))] for _ in range(2)])
  det.eval();gen.eval();is_st=step%4!=0
  if joint:
   y=wm(t['x'],t['msg'],t['target'])
   with torch.no_grad():actual=stcall(y)
   surrogate=proxy(y);rec=actual+surrogate-surrogate.detach()
  else:y=t['y'];rec=t['cy']
  pos=rec if is_st else y;neg=t['cx'] if is_st else t['x']
  # Adapt pretrained AudioSeal detector; equal total positive and negative weight.
  ps=score(det,pos.detach());ns=score(det,neg);hs=score(det,t['noise'])
  lossD=F.binary_cross_entropy(ps.clamp(1e-6,1-1e-6),torch.ones_like(ps))+.5*F.binary_cross_entropy(ns.clamp(1e-6,1-1e-6),torch.zeros_like(ns))+.5*F.binary_cross_entropy(hs.clamp(1e-6,1-1e-6),torch.zeros_like(hs))
  optD.zero_grad();lossD.backward();torch.nn.utils.clip_grad_norm_(det.parameters(),5);optD.step()
  lossG=0.
  if joint:
   for p in det.parameters():p.requires_grad=False
   pscore=score(det,pos);cs=score(det,y)
   lossG=F.binary_cross_entropy(pscore.clamp(1e-6,1-1e-6),torch.ones_like(pscore))+.25*F.binary_cross_entropy(cs.clamp(1e-6,1-1e-6),torch.ones_like(cs))
   # Preserve accepted audio as a perceptual anchor, in addition to hard residual constraints.
   lossG=lossG+10*((y-t['y']).square().mean()/t['x'].square().mean().clamp_min(1e-8))
   window=torch.hann_window(512,device='cuda');xs=torch.stft(t['x'].squeeze(1),512,128,window=window,return_complex=True).abs().square();rs=torch.stft((y-t['x']).squeeze(1),512,128,window=window,return_complex=True).abs().square()
   ref=F.avg_pool2d(xs.unsqueeze(1),(5,3),1,(2,1)).squeeze(1)+.01*xs.mean()+1e-8;lossG=lossG+10*(rs/ref).mean()
   if a.mode=='latent_joint':
    with torch.no_grad():ex=st.encoder(t['x']);ec=st.encoder(t['cx'])
    before=st.encoder(y)-ex;after=st.encoder(rec)-ec
    # Exploratory continuous-feature survival objective; real quantized ST supplies forward values.
    target=before.detach();lossG=lossG+.1*(1-F.cosine_similarity(after.flatten(1),target.flatten(1),dim=1,eps=1e-6)).mean()
   optG.zero_grad();lossG.backward();torch.nn.utils.clip_grad_norm_([p for p in gen.parameters() if p.requires_grad],5);optG.step()
   for p in det.parameters():p.requires_grad=True
  if step%10==0:print(json.dumps(dict(step=step,lossD=float(lossD),lossG=float(lossG),elapsed=time.time()-start)),flush=True)
  if step%every==0 or step==steps:
   cr=evaluate('cal',step);vr=evaluate('val',step);summary=summarize(vr,threshold(cr));history.append(dict(step=step,validation=summary))
   state=dict(generator=gen.state_dict(),detector=det.state_dict(),optimizerD=optD.state_dict(),optimizerG=optG.state_dict() if joint else None,step=step,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=rng.getstate())
   torch.save(state,out/('stage_'+str(step)+'.pt'));torch.save(det.state_dict(),out/('detector_'+str(step)+'.pt'));torch.save(gen.state_dict(),out/('generator_'+str(step)+'.pt'))
   criterion=summary['tpr'] if summary['fpr']<=.02 and summary['noise_fpr']<=.02 else -1
   if criterion>best or not (out/'selected.pt').exists():best=criterion;torch.save(dict(generator=gen.state_dict(),detector=det.state_dict(),step=step,threshold=threshold(cr)),out/'selected.pt')
   (out/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(history[-1]),flush=True)
 sel=torch.load(out/'selected.pt',map_location='cuda',weights_only=True);gen.load_state_dict(sel['generator']);det.load_state_dict(sel['detector'])
 result=summarize(evaluate('test','selected'),sel['threshold']);result.update(complete=True,selected_step=sel['step'],initial_test=initial,generator_unchanged=modelhash(gen)==ghash,sptk_unchanged=modelhash(st)==sthash,steps=steps)
 assert result['sptk_unchanged'];assert joint or result['generator_unchanged']
 (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=="__main__":main()
