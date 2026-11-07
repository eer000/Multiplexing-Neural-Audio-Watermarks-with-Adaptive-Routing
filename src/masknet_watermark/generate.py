"""Generate paired clean/marked records using this package's fixed model only."""
import argparse
import hashlib
import json
from pathlib import Path
import soundfile as sf
import torch
from .inference import MaskNet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    model = MaskNet(args.model)
    manifest = Path(args.sources)
    sources = json.loads(manifest.read_text())
    if sources.get("model_id") != model.metadata["model_id"] or sources.get("role") != "evaluation":
        raise ValueError("A separate evaluation manifest for this model is required")
    records = sources["sources"]
    ids = [v["sample_id"] for v in records]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Evaluation source IDs must be unique")
    calibration = set(model.metadata["calibration_speaker_SHA256"])
    if any(hashlib.sha256(str(v["speaker_id"]).encode()).hexdigest() in calibration for v in records):
        raise ValueError("Calibration and evaluation speakers overlap")
    output = Path(args.output)
    output.mkdir(exist_ok=False)
    paired = []
    for index, record in enumerate(records):
        audio, sample_rate = sf.read(manifest.parent/record["path"], dtype="float32", always_2d=True)
        if sample_rate != 16000 or audio.shape != (96000, 1):
            raise ValueError("Each source must be six seconds of mono audio at 16 kHz")
        marked = model.embed(torch.from_numpy(audio[:, 0]).reshape(1, 1, -1), sample_rate)
        for label, values in [(0, audio[:, 0]), (1, marked.cpu().numpy().reshape(-1))]:
            name = f"{index:06d}_{label}.wav"
            sf.write(output/name, values, sample_rate, subtype="FLOAT")
            paired.append({"sample_id": record["sample_id"], "speaker_id": str(record["speaker_id"]),
                           "condition": "NA", "label": label, "path": name,
                           "reference_path": f"{index:06d}_0.wav"})
    (output/"manifest.json").write_text(json.dumps({"model_id": model.metadata["model_id"],
        "role": "evaluation", "conditions": ["NA"], "records": paired}, indent=2)+"\n")

if __name__ == "__main__":
    main()
