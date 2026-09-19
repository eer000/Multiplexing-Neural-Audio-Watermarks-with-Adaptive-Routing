"""Embed using the released generator and its exact residual constraints."""
import argparse
from pathlib import Path
import json
import soundfile as sf
import torch
import torchaudio
from audioseal import AudioSeal
from .quiet_methods_v2 import constrain

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--weights',type=Path,default=Path('weights'))
    p.add_argument('--message',default='0101010101010101')
    p.add_argument('--device',default='cpu')
    a=p.parse_args()
    if len(a.message)!=16 or set(a.message)-{'0','1'}:p.error('message must contain 16 binary digits')
    if a.output.exists():p.error('output already exists')
    torch.set_num_threads(2)
    wave,sr=sf.read(a.input,dtype='float32',always_2d=True)
    x=torch.from_numpy(wave.mean(1)).view(1,1,-1).to(a.device)
    if sr!=16000:x=torchaudio.functional.resample(x,sr,16000)
    g=AudioSeal.load_generator('audioseal_wm_16bits').to(a.device).eval()
    base=AudioSeal.load_generator('audioseal_wm_16bits').to(a.device).eval()
    g.load_state_dict(torch.load(a.weights/'generator.pt',map_location=a.device,weights_only=True))
    msg=torch.tensor([[int(v) for v in a.message]],device=a.device)
    with torch.inference_mode():
        d=constrain(x,g.get_watermark(x,16000,message=msg),'calibrated')
        d=(x+d).clamp(-1,1)-x
        db=(x+base.get_watermark(x,16000,message=msg)).clamp(-1,1)-x
        d=d*(db.square().mean(-1,keepdim=True)/d.square().mean(-1,keepdim=True).clamp_min(1e-12)).sqrt().clamp(max=1)
        y=x+d
    a.output.parent.mkdir(parents=True,exist_ok=True)
    sf.write(a.output,y.flatten().cpu().numpy(),16000,subtype='PCM_16')
    print(json.dumps({'output':str(a.output),'sample_rate':16000,'message':a.message}))

if __name__=='__main__':main()
