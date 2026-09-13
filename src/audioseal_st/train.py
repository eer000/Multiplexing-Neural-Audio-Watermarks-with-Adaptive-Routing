"""Proxy-gradient adaptation. Set CUDA_VISIBLE_DEVICES to select a GPU.

True SpeechTokenizer supplies forward values; a frozen proxy supplies gradients.
Experimental loss settings require validation and listening review before use.
"""
import argparse, json, random, time, hashlib
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F


def enable_sptk_ste(st):
    st.eval()
    for layer in st.quantizer.vq.layers:
        layer.train()
        layer._codebook.eval()
    for module in st.modules():
        if isinstance(module, torch.nn.RNNBase):
            module.train()


def set_trainable(generator, mode):
    for p in generator.parameters(): p.requires_grad=False
    for module in (generator.decoder, generator.msg_processor):
        for p in module.parameters(): p.requires_grad=True


def det_forward(det, x, sr):
    result=det(x, sample_rate=sr)
    return result[0],result[1],None


def main():
    from audioseal import AudioSeal
    from speechtokenizer import SpeechTokenizer
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--sptk-config',type=Path,required=True)
    ap.add_argument('--sptk-checkpoint',type=Path,required=True)
    ap.add_argument('--proxy-checkpoint',type=Path,required=True)
    ap.add_argument('--init-checkpoint',type=Path,help='Generator weights only; optimizer and sampling restart from seed.')
    ap.add_argument('--train-samples',type=int,default=2048)
    ap.add_argument('--val-samples',type=int,default=64)
    ap.add_argument('--steps',type=int,default=1500)
    ap.add_argument('--eval-every',type=int,default=250)
    ap.add_argument('--seconds',type=float,default=4)
    ap.add_argument('--seed',type=int,default=20260915)
    ap.add_argument('--lr',type=float,default=2e-5)
    ap.add_argument('--mse-weight',type=float,default=30)
    ap.add_argument('--snr-weight',type=float,default=.1)
    ap.add_argument('--snr-reference',type=float,default=18)
    ap.add_argument('--residual-mode',choices=['raw','shaped','capped'],default='capped')
    ap.add_argument('--minimum-snr',type=float,default=22)
    ap.add_argument('--spectral-ratio',type=float,default=.2)
    ap.add_argument('--presence-weight',type=float,default=1)
    ap.add_argument('--bit-weight',type=float,default=.2)
    ap.add_argument('--clean-weight',type=float,default=.5)
    ap.add_argument('--hard-bit-weight',type=float,default=0)
    ap.add_argument('--high-weight',type=float,default=200)
    ap.add_argument('--low-weight',type=float,default=200)
    ap.add_argument('--quiet-weight',type=float,default=20)
    ap.add_argument('--masking-weight',type=float,default=0)
    ap.add_argument('--post-noise-weight',type=float,default=0)
    ap.add_argument('--adapter',action='store_true')
    args=ap.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(args.seed);rng=random.Random(args.seed)
    files=sorted(args.data_root.rglob('*.flac'))
    speakers=sorted({p.stem.split('-')[0] for p in files});rng.shuffle(speakers)
    valsp=set(speakers[:24]);rng.shuffle(files)
    train=[];val=[]
    for p in files:
        target=val if p.stem.split('-')[0] in valsp else train
        limit=args.val_samples if target is val else args.train_samples
        if len(target)>=limit:continue
        info=sf.info(p)
        if info.samplerate==16000 and info.duration>=args.seconds:target.append(p)
        if len(train)==args.train_samples and len(val)==args.val_samples:break
    def tensor(p):
        x,sr=sf.read(p,dtype='float32',always_2d=True)
        return torch.from_numpy(x[:int(16000*args.seconds)].mean(1)).view(1,1,-1).cuda()
    tr=[tensor(p) for p in train];va=[tensor(p) for p in val]
    gen=AudioSeal.load_generator('audioseal_wm_16bits').cuda().eval()
    if args.init_checkpoint:
        gen.load_state_dict(torch.load(args.init_checkpoint,map_location='cuda',weights_only=True))
    if args.adapter:
        from .models import ResidualAdapter
        # Isolate adapter initialization from fixed validation-message RNG.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            gen=ResidualAdapter(gen).cuda()
    def watermark(x, msg):
        residual=gen.get_watermark(x,16000,message=msg)
        if args.residual_mode=='shaped':
            from .models import shape_residual
            residual=shape_residual(x,residual,args.minimum_snr,args.spectral_ratio)
        elif args.residual_mode=='capped':
            signal=x.square().mean(-1,keepdim=True)
            noise=residual.square().mean(-1,keepdim=True).clamp_min(1e-12)
            scale=(signal*10**(-args.minimum_snr/10)/noise).clamp_min(1e-12).sqrt().clamp(max=1)
            residual=residual*scale
        return residual
    det=AudioSeal.load_detector('audioseal_detector_16bits').cuda().eval()
    st=SpeechTokenizer.load_from_checkpoint(str(args.sptk_config),str(args.sptk_checkpoint)).cuda().eval()
    assert st.sample_rate==16000
    if not args.adapter:set_trainable(gen,'decoder_msg')
    for model in [det,st]:
        for p in model.parameters():p.requires_grad=False
    def statehash(model):
        h=hashlib.sha256()
        for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    frozenhash=statehash(st)
    proxy=None
    if args.proxy_checkpoint:
        from .models import GatedLongContextProxy
        proxy=GatedLongContextProxy().cuda().eval()
        proxy.load_state_dict(torch.load(args.proxy_checkpoint,map_location='cuda',weights_only=True))
        for p in proxy.parameters():p.requires_grad=False
    params=[p for p in gen.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=args.lr,weight_decay=1e-5)
    messages=[torch.randint(0,2,(1,16),device='cuda') for x in va]
    noise_rng=torch.Generator(device='cuda').manual_seed(args.seed+901)
    manifest=dict(train=[str(p.relative_to(args.data_root)) for p in train],val=[str(p.relative_to(args.data_root)) for p in val],val_speakers=sorted(valsp),
        seed=args.seed,steps=args.steps,seconds=args.seconds,initialization='original AudioSeal',objective='ST presence BCE + 0.2 ST message BCE + 0.5 clean message BCE + 30 MSE + 0.1 per-example 18dB SNR hinge',
        sptk_hash_before=frozenhash,arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    manifest['initialization']=str(args.init_checkpoint) if args.init_checkpoint else 'original AudioSeal'
    manifest['optimizer_restart']=True
    manifest['objective']='True ST forward / frozen proxy backward; loss weights and residual constraints recorded in arguments.'
    if args.init_checkpoint:
        manifest['initialization_sha256']=hashlib.sha256(args.init_checkpoint.read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    def artifact_losses(x, residual):
        spectrum=torch.fft.rfft(residual,dim=-1)
        energy=spectrum.abs().square()
        weights=torch.ones_like(energy);weights[...,1:-1]=2
        energy=energy*weights/residual.shape[-1]**2
        frequencies=torch.fft.rfftfreq(residual.shape[-1],1/16000,device=x.device)
        signal=x.square().mean().clamp_min(1e-8)
        high=energy[...,frequencies>=3400].sum()/signal
        low=energy[...,frequencies<150].sum()/signal
        q=(1-F.avg_pool1d(x.abs(),513,stride=1,padding=256)/.01).clamp(0,1)
        quiet=(q*residual.square()).mean()/signal
        return high,low,quiet
    def evaluate():
        gen.eval();det.eval();st.eval();rows=[];neg=[]
        with torch.no_grad():
            for x,msg in zip(va,messages):
                y=(x+watermark(x,msg)).clamp(-1,1)
                rec=st.decode(st.encode(y))[...,:x.shape[-1]]
                cleanrec=st.decode(st.encode(x))[...,:x.shape[-1]]
                presence,bits,_=det_forward(det,rec,16000)
                npres=det_forward(det,cleanrec,16000)[0][:,1].mean()
                snr=10*torch.log10(x.square().mean()/(y-x).square().mean().clamp_min(1e-12))
                high,low,quiet=artifact_losses(x,y-x)
                row=dict(presence=float(presence[:,1].mean()),bit_acc=float(((bits>=.5)==msg.bool()).float().mean()),snr=float(snr),high_energy_ratio=float(high),low_energy_ratio=float(low),quiet_energy_ratio=float(quiet))
                if len(rows)<8:
                    from pesq import pesq
                    row['pesq']=float(pesq(16000,x.flatten().cpu().numpy(),y.flatten().cpu().numpy(),'wb'))
                rows.append(row)
                neg.append(float(npres))
        threshold=max(neg)
        return dict(rows=rows,threshold=threshold,tpr=sum(r['presence']>threshold for r in rows)/len(rows),
            presence_mean=np.mean([r['presence'] for r in rows]).item(),bit_acc=np.mean([r['bit_acc'] for r in rows]).item(),snr=np.mean([r['snr'] for r in rows]).item(),pesq_8=float(np.mean([r['pesq'] for r in rows if 'pesq' in r])),high_energy_ratio=float(np.mean([r['high_energy_ratio'] for r in rows])),low_energy_ratio=float(np.mean([r['low_energy_ratio'] for r in rows])))
    start=time.time();history=[dict(step=0,eval=evaluate())];best=-1
    print(json.dumps(history[-1]),flush=True)
    for step in range(1,args.steps+1):
        gen.train();det.train();st.eval()
        x=tr[rng.randrange(len(tr))];msg=torch.randint(0,2,(1,16),device='cuda')
        y=(x+watermark(x,msg)).clamp(-1,1)
        with torch.no_grad():actual=st.decode(st.encode(y))[...,:x.shape[-1]]
        surrogate=proxy(y)
        rec=actual+surrogate-surrogate.detach()
        presence,bits,_=det_forward(det,rec,16000);cbits=det_forward(det,y,16000)[1]
        mse=(y-x).square().mean();snr=10*torch.log10(x.square().mean().clamp_min(1e-12)/mse.clamp_min(1e-12))
        bit_losses=F.binary_cross_entropy(bits,msg.float(),reduction='none')
        bit_loss=bit_losses.mean()+args.hard_bit_weight*torch.topk(bit_losses,4,dim=-1).values.mean()
        loss=args.presence_weight*F.binary_cross_entropy(presence[:,1],torch.ones_like(presence[:,1]))+args.bit_weight*bit_loss+args.clean_weight*F.binary_cross_entropy(cbits,msg.float())+args.mse_weight*mse+args.snr_weight*F.relu(args.snr_reference-snr)
        if args.post_noise_weight:
            noise=torch.randn(rec.shape,device=rec.device,generator=noise_rng)
            noise=noise/noise.square().mean().sqrt().clamp_min(1e-12)*x.square().mean().sqrt()*.1
            npres,nbits,_=det_forward(det,(rec+noise).clamp(-1,1),16000)
            robust=F.binary_cross_entropy(npres[:,1],torch.ones_like(npres[:,1]))+args.bit_weight*F.binary_cross_entropy(nbits,msg.float())
            loss=loss+args.post_noise_weight*robust
        high,low,quiet=artifact_losses(x,y-x)
        loss=loss+args.high_weight*high+args.low_weight*low+args.quiet_weight*quiet
        if args.masking_weight:
            window=torch.hann_window(512,device=x.device)
            xs=torch.stft(x.squeeze(1),512,128,window=window,return_complex=True).abs().square()
            rs=torch.stft((y-x).squeeze(1),512,128,window=window,return_complex=True).abs().square()
            reference=F.avg_pool2d(xs.unsqueeze(1),(5,3),stride=1,padding=(2,1)).squeeze(1)+.01*xs.mean()+1e-8
            loss=loss+args.masking_weight*(rs/reference).mean()
        opt.zero_grad();loss.backward();grad=torch.nn.utils.clip_grad_norm_(params,5.0);opt.step()
        if step%10==0:print(json.dumps(dict(step=step,loss=float(loss),grad=float(grad),seconds=time.time()-start)),flush=True)
        if step%args.eval_every==0 or step==args.steps:
            info=evaluate();torch.save(gen.state_dict(),out/f'step_{step}.pt');history.append(dict(step=step,eval=info));print(json.dumps(history[-1]),flush=True)
            torch.save(dict(generator=gen.state_dict(),optimizer=opt.state_dict(),step=step,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=rng.getstate(),noise_rng=noise_rng.get_state()),out/f'training_state_{step}.pt')
            score=info['tpr']
            if score>best:best=score;torch.save(gen.state_dict(),out/'best.pt')
            (out/'history.json').write_text(json.dumps(history,indent=2))
            with torch.no_grad():
                for sample,(x,msg) in enumerate(zip(va[:3],messages[:3])):
                    y=(x+watermark(x,msg)).clamp(-1,1)
                    sf.write(out/f'step_{step}_sample_{sample}.wav',y.flatten().cpu().numpy(),16000,subtype='PCM_16')
                    if step==args.eval_every:sf.write(out/f'clean_sample_{sample}.wav',x.flatten().cpu().numpy(),16000,subtype='PCM_16')
    assert frozenhash==statehash(st),'Tokenizer state drifted!'
    (out/'complete.json').write_text(json.dumps(dict(complete=True,sptk_unchanged=True,elapsed_seconds=time.time()-start),indent=2))

if __name__=='__main__': main()
