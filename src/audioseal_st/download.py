"""Download the versioned release assets and verify SHA-256."""
import argparse,hashlib,json,urllib.request
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('weights'));p.add_argument('--training',action='store_true');p.add_argument('--masknet',action='store_true');a=p.parse_args()
    url='https://raw.githubusercontent.com/eer000/Multiplexing-Neural-Audio-Watermarks-with-Adaptive-Routing/v0.2.0-latent600/weights.json'
    with urllib.request.urlopen(url) as f:meta=json.load(f)
    names=['generator.pt','detector.pt']
    if a.training:names+=['initial_generator.pt','proxy.pt']
    if a.masknet:names+=['masknet.pt']
    a.output.mkdir(parents=True,exist_ok=True)
    for name in names:
        target=a.output/name;expected=meta['files'][name]['sha256']
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest()!=expected:raise ValueError('Existing file has different hash: '+str(target))
            continue
        tmp=target.with_suffix('.download')
        urllib.request.urlretrieve(meta['base_url']+name,tmp)
        if hashlib.sha256(tmp.read_bytes()).hexdigest()!=expected:tmp.unlink();raise ValueError('SHA-256 mismatch: '+name)
        tmp.replace(target);print(target)

if __name__=='__main__':main()
