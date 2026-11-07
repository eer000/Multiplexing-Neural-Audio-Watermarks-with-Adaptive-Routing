"""Generate this model's fixed paired single-attack evaluation audio."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import soundfile as sf
from .channels import CONDITIONS
from .transforms import Channels,digest

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--codec-assets',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    meta=json.loads((Path(args.model)/'model.json').read_text())
    source=Path(args.manifest);manifest=json.loads(source.read_text())
    if meta.get('format')!='masknet-presence-inference-v3' or manifest.get('model_id')!=meta['model_id'] or manifest.get('role')!='evaluation':
        raise ValueError('This watermark needs its own evaluation manifest')
    grouped={};speaker_hashes=set(meta['calibration_speaker_SHA256'])
    for record in manifest['records']:
        if record['condition']!='NA' or record['label'] not in (0,1):
            raise ValueError('Provide this model\'s unattacked clean/marked pairs')
        if hashlib.sha256(str(record['speaker_id']).encode()).hexdigest() in speaker_hashes:
            raise ValueError('Calibration and evaluation speakers overlap')
        pair=grouped.setdefault(record['sample_id'],{})
        if record['label'] in pair:raise ValueError('Duplicate source/label')
        pair[record['label']]=record
    if not grouped or any(set(v)!={0,1} for v in grouped.values()):raise ValueError('Complete clean/marked pairs are required')
    output=Path(args.output);output.mkdir(exist_ok=False)
    channels=Channels(json.loads(Path(args.codec_assets).read_text()));rows=[];provenance=[]
    try:
        for index,(sample_id,pair) in enumerate(grouped.items()):
            if str(pair[0]['speaker_id'])!=str(pair[1]['speaker_id']):raise ValueError('Speaker mismatch')
            waves={}
            for label,record in pair.items():
                wave,sr=sf.read(source.parent/record['path'],dtype='float32',always_2d=True)
                if sr!=16000 or wave.shape!=(96000,1):raise ValueError('Sources must be six-second mono 16 kHz audio')
                waves[label]=wave[:,0]
            base_seed=int(pair[0].get('seed',int(hashlib.sha256(sample_id.encode()).hexdigest()[:8],16)))
            for condition in CONDITIONS:
                name=condition['id'];seed=base_seed+int(hashlib.sha256(name.encode()).hexdigest()[:8],16)
                transformed={label:channels.apply(wave,condition,seed,waves[0]) for label,wave in waves.items()}
                reference=waves[0]
                if condition['family']=='CROP':
                    size=int(len(reference)*condition['parameters']['retain_fraction'])
                    start=int(np.random.default_rng(seed).integers(0,len(reference)-size+1));reference=reference[start:start+size]
                folder=output/name;folder.mkdir(exist_ok=True);refname=f'{index:06d}_reference.wav';sf.write(folder/refname,reference,16000,subtype='FLOAT')
                for label,wave in transformed.items():
                    filename=f'{index:06d}_{label}.wav';sf.write(folder/filename,wave,16000,subtype='FLOAT')
                    rows.append({'sample_id':sample_id,'speaker_id':str(pair[0]['speaker_id']),'condition':name,'label':label,'path':name+'/'+filename,'reference_path':name+'/'+refname,'paired_seed':seed})
                    provenance.append({'sample_id':sample_id,'condition':name,'label':label,'samples':len(wave),'wave_SHA256':hashlib.sha256(wave.tobytes()).hexdigest(),'file_SHA256':digest(folder/filename)})
        result={'model_id':meta['model_id'],'role':'evaluation','conditions':[v['id'] for v in CONDITIONS],'records':rows}
        (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
        (output/'channel_provenance.json').write_text(json.dumps({'input_manifest_SHA256':digest(source),'model_manifest_SHA256':digest(Path(args.model)/'model.json'),'conditions':CONDITIONS,'codec_states':channels.identities,'ffmpeg_version':channels.ffmpeg_version,'waveforms':provenance,'paired_seed_rule':'source seed plus first 8 SHA256 hex digits of condition ID','sample_rate':16000},indent=2)+'\n')
    finally:channels.unload()

if __name__=='__main__':main()
