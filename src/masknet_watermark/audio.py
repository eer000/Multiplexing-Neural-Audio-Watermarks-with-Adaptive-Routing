"""Arbitrary-duration mono WAV embedding and timestamped detection."""
import argparse,json
from pathlib import Path
import soundfile as sf
import torch
from .inference import MaskNet

def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode", choices=["embed", "detect"])
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--operating-point")
    a=p.parse_args()
    target=Path(a.output)
    if target.exists():
        raise FileExistsError("Choose a new output path")
    samples,rate=sf.read(a.input,dtype="float32",always_2d=True)
    if samples.shape[1]!=1:
        raise ValueError("Convert input to mono first")
    model=MaskNet(a.model,operating_point=a.operating_point)
    wave=torch.from_numpy(samples[:,0]).reshape(1,1,-1)
    if a.mode=="embed":
        marked=model.embed_audio(wave,rate)
        sf.write(target,marked.numpy().reshape(-1),rate,subtype="FLOAT")
    else:
        result=model.score_audio(wave,rate)
        target.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")

if __name__=="__main__":
    main()
