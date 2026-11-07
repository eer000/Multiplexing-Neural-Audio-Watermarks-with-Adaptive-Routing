"""Evaluate only the watermark loaded through this package's frozen bundle API."""
import argparse,collections,json,math,random,hashlib,time,statistics,itertools
from pathlib import Path
import soundfile as sf
import torch
from .inference import MaskNet

def wilson(successes, total):
    if not total:
        return None
    z = 1.959963984540054
    rate = successes / total
    denom = 1 + z*z / total
    middle = (rate + z*z/(2*total)) / denom
    half = z*math.sqrt(rate*(1-rate)/total + z*z/(4*total*total)) / denom
    return [0. if successes == 0 else max(0., middle-half), 1. if successes == total else min(1., middle+half)]

def speaker_interval(records, label, resamples=5000):
    groups = collections.defaultdict(lambda: [0, 0])
    for row in records:
        if row["label"] == label:
            groups[row["speaker_id"]][0] += int(row["detected"])
            groups[row["speaker_id"]][1] += 1
    groups = list(groups.values())
    if not groups:
        return None
    rng = random.Random(202611070000)
    rates = []
    for _ in range(resamples):
        sample = [groups[rng.randrange(len(groups))] for _ in groups]
        rates.append(sum(v[0] for v in sample) / sum(v[1] for v in sample))
    rates.sort()
    return {"interval": [rates[int(.025*(resamples-1))], rates[int(.975*(resamples-1))]],
            "speaker_N": len(groups), "resamples": resamples}

def roc(records):
    positive = sum(v["label"] == 1 for v in records)
    negative = sum(v["label"] == 0 for v in records)
    if not positive or not negative:
        return {"status": "missing_positive_or_negative", "positive_N": positive, "negative_N": negative}
    ordered = sorted(records, key=lambda v: v["detection_margin"], reverse=True)
    tp, fp = 0, 0
    points = [{"threshold": None, "TPR": 0., "FPR": 0.}]
    for threshold, group in itertools.groupby(ordered, key=lambda v: v["detection_margin"]):
        rows = list(group)
        tp += sum(v["label"] == 1 for v in rows)
        fp += sum(v["label"] == 0 for v in rows)
        points.append({"threshold": threshold, "TPR": tp/positive, "FPR": fp/negative})
    return {"status": "complete", "positive_N": positive, "negative_N": negative,
            "points": points, "fixed_decision_margin": 0., "descriptive_no_threshold_adoption": True}

def evaluate(model, manifest, base, with_quality=False):
    if manifest.get("model_id") != model.metadata["model_id"]:
        raise ValueError("Evaluation manifest identifies a different model")
    if manifest.get("role") != "evaluation":
        raise ValueError("A separate evaluation manifest is required")
    conditions = manifest["conditions"]
    if not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("Condition names must be unique")
    calibration_speakers = set(model.metadata.get("calibration_speakers", []))
    calibration_hashes = set(model.metadata.get("calibration_speaker_SHA256", []))
    if not calibration_speakers and not calibration_hashes:
        raise ValueError("Calibration speaker identities are required")
    seen = set()
    source_speakers = {}
    grouped = collections.defaultdict(list)
    wave_records = {}
    timings = []
    for record in manifest["records"]:
        if record["condition"] not in conditions:
            raise ValueError("Unregistered evaluation condition")
        speaker = str(record["speaker_id"])
        if speaker in calibration_speakers or hashlib.sha256(speaker.encode()).hexdigest() in calibration_hashes:
            raise ValueError("Calibration and evaluation speakers overlap")
        if record["label"] not in (0, 1):
            raise ValueError("Expected clean=0 or marked=1 labels")
        prior = source_speakers.setdefault(record["sample_id"], speaker)
        if prior != speaker:
            raise ValueError("One source has conflicting speaker identities")
        identity = (record["sample_id"], record["condition"], record["label"])
        if identity in seen:
            raise ValueError("Repeated evaluation source/condition/label")
        seen.add(identity)
        audio, sr = sf.read(Path(base) / record["path"], dtype="float32", always_2d=True)
        if audio.shape[1] != 1:
            raise ValueError("Evaluation audio must be mono")
        started = time.perf_counter()
        result = model.score(torch.from_numpy(audio[:, 0]).reshape(1, 1, -1), sr)
        timings.append(time.perf_counter() - started)
        if with_quality:
            wave_records[identity] = (audio[:, 0], sr, record)
        grouped[record["condition"]].append({**record, **result})
    points = []
    for condition in conditions:
        records = grouped[condition]
        pos = [v for v in records if v["label"] == 1]
        neg = [v for v in records if v["label"] == 0]
        tp = sum(v["detected"] for v in pos)
        fp = sum(v["detected"] for v in neg)
        points.append({"condition": condition, "positive_N": len(pos), "negative_N": len(neg),
                       "tp": tp, "fp": fp, "TPR": tp/len(pos) if pos else None,
                       "actual_FPR": fp/len(neg) if neg else None,
                       "TPR_Wilson95": wilson(tp, len(pos)), "FPR_Wilson95": wilson(fp, len(neg)),
                       "TPR_speaker_bootstrap95": speaker_interval(records, 1),
                       "FPR_speaker_bootstrap95": speaker_interval(records, 0),
                       "speaker_N": len({v["speaker_id"] for v in records}),
                       "status": "complete" if pos and neg else "missing_positive_or_negative"})
    valid = [v for v in points if v["TPR"] is not None]
    quality_records = []
    if with_quality:
        from .quality import measure, summarize
        for (sample_id, condition, label), (audio, sr, record) in wave_records.items():
            if label != 1:
                continue
            negative = wave_records.get((sample_id, condition, 0))
            if negative is None:
                continue
            incremental = measure(negative[0], audio, sr)
            total = None
            if record.get("reference_path"):
                reference, reference_sr = sf.read(Path(base)/record["reference_path"], dtype="float32", always_2d=True)
                if reference_sr != sr or reference.shape != (len(audio), 1):
                    raise ValueError("Total-quality reference must align with the marked audio")
                total = measure(reference[:, 0], audio, sr)
            quality_records.append({"sample_id": sample_id, "speaker_id": record["speaker_id"],
                "condition": condition, "incremental": incremental, "total": total,
                "peak": float(abs(audio).max()), "fraction_outside_unit": float((abs(audio)>1).mean())})
        quality_summary = summarize(quality_records)
    else:
        quality_summary = None
    return {"model_id": model.metadata["model_id"], "calibration_id": model.metadata["calibration_id"],
            "query_ids": model.query_ids, "thresholds": model.thresholds.cpu().tolist(),
            "operating_point": model.operating_point,
            "fusion": "Any over the frozen query set", "points": points,
            "mean_TPR_measured_conditions": sum(v["TPR"] for v in valid)/len(valid) if valid else None,
            "mean_condition_N": len(valid), "registered_condition_N": len(conditions),
            "thresholds_fitted_on_evaluation": False,
            "ROC": {condition: roc(grouped[condition]) for condition in conditions},
            "quality": quality_summary, "quality_records": quality_records,
            "inference_cost": {"score_N": len(timings), "first_score_seconds": timings[0] if timings else None,
                "warm_score_seconds_median": statistics.median(timings[1:]) if len(timings)>1 else None,
                "score_seconds_sum": sum(timings)},
            "records": dict(grouped)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--operating-point", default=None)
    parser.add_argument("--quality", action="store_true")
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    result = evaluate(MaskNet(args.model, args.device, args.operating_point), json.loads(manifest_path.read_text()), manifest_path.parent, args.quality)
    output = Path(args.output)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)

if __name__ == "__main__":
    main()
