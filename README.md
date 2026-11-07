# Multiplexing Neural Audio Watermarks with Adaptive Routing

MaskNet embeds a watermark in speech and detects its presence with a fixed key. The model combines an audio embedder, a shared evidence encoder, a temporal masking readout and a native keyed readout. Detection uses a calibrated Any rule over the two readouts.

## Installation

Python 3.10+, PyTorch 2.10.0 and a CUDA GPU are required.

```bash
git clone https://github.com/eer000/Multiplexing-Neural-Audio-Watermarks-with-Adaptive-Routing.git MaskNet
cd MaskNet
pip install -e .
```

Download the files from [MaskNet on Hugging Face](https://huggingface.co/eeryyq/MaskNet/tree/main) into a local `weights` directory: `MaskNet_embedding.pt`, `MaskNet_evidence.pt`, `MaskNet_readout.pt` and `model.json`.

## Inference

Inputs are 16 kHz mono audio. Output length matches the input, including for files longer than six seconds.

```python
import soundfile as sf
import torch
from masknet_watermark import MaskNet

model = MaskNet("weights")
audio, sr = sf.read("speech.wav", dtype="float32")
wave = torch.from_numpy(audio).reshape(1, 1, -1)
marked = model.embed_audio(wave, sr)
sf.write("marked.wav", marked.numpy().reshape(-1), sr, subtype="FLOAT")

result = model.score_audio(marked, sr)
for segment in result["segments"]:
    print(segment["start_seconds"], segment["end_seconds"], segment["detected"])
```

The same workflow is available from the command line:

```bash
python -m masknet_watermark.audio embed --model weights --input speech.wav --output marked.wav
python -m masknet_watermark.audio detect --model weights --input marked.wav --output detections.json
```

Embedding uses six-second windows with one-second overlap and smooths the watermark residual at window boundaries. Detection returns timestamped per-window scores and decisions; tail padding is recorded in the result. The released false-positive operating points apply to each analysis window. Whole-file aggregation requires its own calibration.

The low-level `embed` and `score` methods support six-second embedding and 0.6-, 1.5- or six-second detection. `embed_audio` and `score_audio` handle file duration automatically. Weights are verified by checksum when loaded.

## Evaluation

The evaluation code evaluates MaskNet on paired clean and marked speech. Install the quality metrics with `pip install -e ".[evaluation]"`; codec channels additionally use `.[codec-evaluation]`, `ffmpeg` and the assets described in `codec_assets.example.json`.

```bash
python -m masknet_watermark.generate --model weights --sources sources.json --output paired
python -m masknet_watermark.perturb --model weights --manifest paired/manifest.json --codec-assets codecs.json --output attacked
python -m masknet_watermark.evaluate --model weights --manifest attacked/manifest.json --output results.json --quality
```

Use six-second sources for this benchmark. `sources.json` has `model_id: "MaskNet"`, `role: "evaluation"` and a `sources` list containing `sample_id`, `speaker_id` and `path`. Evaluation speakers are separate from calibration speakers. Results include detection counts, actual false positives, sample sizes, confidence intervals, ROC and quality measurements, with missing values recorded explicitly.

## License

Apache License 2.0. Component attribution is provided in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
