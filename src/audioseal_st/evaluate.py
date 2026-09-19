"""Evaluate single AudioSeal and A_T/PerTh multiplexing through actual ST.

Manifest: {"files": [{"path": "relative/file.flac", "split": "eval", "message": [0,...]}]}.
Allowed splits: cal, eval. Paths resolve under --data-root. Messages must be fixed.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from audioseal import AudioSeal
from speechtokenizer import SpeechTokenizer

from .metrics import any_operating_point, calibrated_any, operating_point
from .models import MaskNet, audioseal_outputs, true_st, shape_residual


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return 'inf' if value > 0 else '-inf'
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--audioseal-checkpoint', type=Path, required=True)
    ap.add_argument('--detector-checkpoint',type=Path,required=True)
    ap.add_argument('--sptk-config', type=Path, required=True)
    ap.add_argument('--sptk-checkpoint', type=Path, required=True)
    ap.add_argument('--mask-checkpoint', type=Path)
    ap.add_argument('--mask-config', type=Path)
    ap.add_argument('--seconds', type=float, default=6)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--single-only', action='store_true')
    ap.add_argument('--attacks', default='NA,ST', help='Comma-separated single attacks: NA,ST,lowpass,gaussian,mp3')
    args = ap.parse_args()
    from .attacks import transform
    attacks=args.attacks.split(',')
    if not attacks or any(k not in {'NA','ST','lowpass','gaussian','mp3'} for k in attacks):
        ap.error('Unsupported single attack')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(20260913)
    entries = json.loads(args.manifest.read_text())['files']
    assert entries and all(e['split'] in {'cal', 'eval'} for e in entries)
    paths = [e['path'] for e in entries]
    if len(set(paths)) != len(paths):
        raise ValueError('Duplicate source paths in manifest')
    base = AudioSeal.load_generator('audioseal_wm_16bits').to(args.device).eval()
    adapted = AudioSeal.load_generator('audioseal_wm_16bits').to(args.device).eval()
    adapted.load_state_dict(torch.load(args.audioseal_checkpoint, map_location=args.device, weights_only=True))
    detector = AudioSeal.load_detector('audioseal_detector_16bits').to(args.device).eval()
    detector.load_state_dict(torch.load(args.detector_checkpoint,map_location=args.device,weights_only=True))
    original_detector=AudioSeal.load_detector('audioseal_detector_16bits').to(args.device).eval()
    st = SpeechTokenizer.load_from_checkpoint(str(args.sptk_config), str(args.sptk_checkpoint)).to(args.device).eval()
    pm = None
    if not args.single_only:
        import perth
        pm = perth.PerthImplicitWatermarker(device=args.device)
        pm.perth_net.eval()
    mask = None
    if args.mask_checkpoint:
        if pm is None or not args.mask_config:
            raise ValueError('Mask evaluation requires PerTh and explicit mask config')
        mask = MaskNet(**json.loads(args.mask_config.read_text())).to(args.device).eval()
        mask.load_state_dict(torch.load(args.mask_checkpoint, map_location=args.device, weights_only=True))
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata['hashes'] = {key: sha256(getattr(args, key)) for key in
                          ['manifest', 'audioseal_checkpoint', 'detector_checkpoint', 'mask_checkpoint', 'mask_config', 'sptk_config', 'sptk_checkpoint'] if getattr(args,key) is not None}
    metadata['code_sha256'] = sha256(__file__)
    metadata['protocol'] = 'Attacks applied independently, not composed; Gaussian noise normalized per input at 20dB with paired seed. First channel averaged to mono; resampled to 16kHz; crop/pad to requested duration; clip [-1,1]; all RVQ layers; clean negatives undergo same ST'
    (args.output/'manifest.json').write_text(json.dumps(metadata, indent=2))

    def scores(x, message=None):
        presence, bits = audioseal_outputs(detector, x)
        a = float(presence.mean())
        result = {'A': a, 'A_original':float(audioseal_outputs(original_detector,x)[0].mean())}
        if pm is not None:
            p = pm.get_watermark(x.flatten().cpu().numpy(), sample_rate=16000, round=False)
            if isinstance(p, tuple):
                p = p[0]
            result['P'] = float(np.asarray(p).mean())
        if message is not None:
            result['bit_acc'] = float(((bits >= .5) == message.bool()).float().mean())
        if not all(np.isfinite(v) for v in result.values()):
            raise ValueError('Nonfinite detector score')
        return result

    started = time.time()
    rows = []
    with (args.output/'scores.jsonl').open('w') as f, torch.inference_mode():
        for i, e in enumerate(entries):
            wave, sr = sf.read(args.data_root/e['path'], dtype='float32', always_2d=True)
            x = torch.from_numpy(wave.mean(1)).view(1, 1, -1).to(args.device)
            if sr != 16000:
                x = torchaudio.functional.resample(x, sr, 16000)
            n = int(16000*args.seconds)
            original_length = x.shape[-1]
            x = torch.nn.functional.pad(x[..., :n], (0, max(0, n-x.shape[-1])))
            row = dict(path=e['path'], split=e['split'], original_samples=original_length,
                       clean={k: scores(transform(x,k,st,seed=20260913+i)) for k in attacks}, methods={})
            if e['split'] == 'eval':
                if len(e['message']) != 16 or not set(e['message']) <= {0, 1}:
                    raise ValueError('Each evaluation item requires 16 fixed binary message bits')
                msg = torch.tensor([e['message']], device=args.device)
                row['message'] = e['message']
                da = adapted.get_watermark(x, 16000, message=msg)
                from .quiet_methods_v2 import constrain
                da=constrain(x,da,'calibrated')
                da=(x+da).clamp(-1,1)-x
                db=(x+base.get_watermark(x,16000,message=msg)).clamp(-1,1)-x
                da=da*(db.square().mean(-1,keepdim=True)/da.square().mean(-1,keepdim=True).clamp_min(1e-12)).sqrt().clamp(max=1)
                waves = {}
                if pm is not None:
                    px = np.asarray(pm.apply_watermark(x.flatten().cpu().numpy(), sample_rate=16000), dtype=np.float32).reshape(-1)
                    px = np.pad(px[:n], (0, max(0, n-len(px))))
                    dp = torch.from_numpy(px).view(1, 1, -1).to(args.device)-x
                    # PerTh residual reused only for explicitly requested multiplexing.
                waves['AT_a1'] = x+da
                if mask is not None:
                    m = mask(x)
                    waves['MaskNet_AT_P'] = x+m[:, :1]*da+m[:, 1:2]*dp
                    row['mask'] = {str(j): dict(min=float(m[:, j].min()), max=float(m[:, j].max()), mean=float(m[:, j].mean())) for j in range(2)}
                for name, raw in waves.items():
                    y = raw.clamp(-1, 1)
                    row['methods'][name] = dict(**{k:scores(transform(y,k,st,seed=20260913+i),msg) for k in attacks},
                        snr=float(10*torch.log10(x.square().mean().clamp_min(1e-12)/(y-x).square().mean().clamp_min(1e-12))),
                        clip_fraction=float((raw.abs() > 1).float().mean()))
                    from pesq import pesq
                    from pystoi import stoi
                    clean_np=x.flatten().cpu().numpy();water_np=y.flatten().cpu().numpy()
                    row['methods'][name]['pesq']=float(pesq(16000,clean_np,water_np,'wb'))
                    row['methods'][name]['stoi']=float(stoi(clean_np,water_np,16000))
            f.write(json.dumps(row)+'\n')
            f.flush()
            rows.append(row)
            if (i+1) % 16 == 0:
                print(json.dumps(dict(done=i+1, total=len(entries), seconds=time.time()-started)), flush=True)
    ev, cal = ([r for r in rows if r['split'] == s] for s in ('eval', 'cal'))
    report = dict(complete=True, n_eval=len(ev), n_cal=len(cal), results=[], elapsed_seconds=time.time()-started)
    for attack in attacks:
        columns = ['A'] if pm is None else ['A', 'P']
        neg = np.array([[r['clean'][attack][c] for c in columns] for r in ev])
        for name in ev[0]['methods']:
            pos = np.array([[r['methods'][name][attack][c] for c in columns] for r in ev])
            result = dict(method=name, attack=attack, detectors={c: operating_point(pos[:, j], neg[:, j]) for j, c in enumerate(columns)},
                          snr_mean=float(np.mean([r['methods'][name]['snr'] for r in ev])),
                          bit_acc=float(np.mean([r['methods'][name][attack]['bit_acc'] for r in ev])))
            if pm is not None:
                result['empirical_any'] = any_operating_point(pos, neg)
                if cal:
                    cn = np.array([[r['clean'][attack][c] for c in columns] for r in cal])
                    result['independent_calibration_any'] = calibrated_any(pos, neg, cn)
            from .metrics import _lowest_threshold
            assert cal, 'Independent calibration is required'
            result['independent_calibration_detectors']={}
            for c in (['A','A_original'] if pm is None else ['A','P','A_original']):
                cn=np.array([r['clean'][attack][c] for r in cal])
                en=np.array([r['clean'][attack][c] for r in ev])
                ep=np.array([r['methods'][name][attack][c] for r in ev])
                threshold=_lowest_threshold(cn,int(.01*len(cn)))
                result['independent_calibration_detectors'][c]=dict(tpr=float((ep>=threshold).mean()),fpr=float((en>=threshold).mean()),calibration_fpr=float((cn>=threshold).mean()),threshold=float(threshold))
            result['quality']={k:float(np.mean([r['methods'][name][k] for r in ev])) for k in ['snr','pesq','stoi']}
            report['results'].append(result)
    (args.output/'summary.json').write_text(json.dumps(json_safe(report), indent=2, allow_nan=False))
    print(json.dumps(json_safe(report), indent=2), flush=True)


if __name__ == '__main__':
    main()
