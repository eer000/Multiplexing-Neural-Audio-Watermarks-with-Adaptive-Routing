# Multiplexing Neural Audio Watermarks with Adaptive Routing

Code for AudioSeal adaptation to SpeechTokenizer and adaptive routing of AudioSeal + PerTh with MaskNet.

This repository currently focuses on **training code and evaluation entry points**. It does not claim a verified reproduction of the paper's numerical results.

## Components

| Component | Status |
|---|---|
| AudioSeal → SpeechTokenizer adaptation | Experimental implementation |
| Adapted AudioSeal single-attack evaluation | Implemented |
| MaskNet network | Implemented |
| Adapted AudioSeal + PerTh through MaskNet evaluation | Implemented |
| Manuscript-aligned MaskNet training | Explicit placeholder |

## Installation

Python 3.10+ and a compatible PyTorch runtime are required.

```bash
pip install -e .
pip install -e '.[multiplex]'  # for PerTh routing evaluation
```

MP3 evaluation additionally requires `ffmpeg` with libmp3lame. `requirements-tested.txt` records the historical research environment; platform compatibility may vary.

## Train AudioSeal for SpeechTokenizer

```bash
python -m audioseal_st.train \
  --data-root /path/to/training_audio \
  --proxy-checkpoint /path/to/proxy.pt \
  --sptk-config /path/to/tokenizer_config.json \
  --sptk-checkpoint /path/to/tokenizer.pt \
  --residual-mode capped --minimum-snr 29 \
  --output runs/adaptation
```

The generator decoder and message processor are updated. The native detector and tokenizer are frozen; real tokenizer outputs supply forward values and a frozen proxy supplies gradients. The current loader expects 16 kHz FLAC and speaker-prefixed filenames of the form `speaker-chapter-utterance.flac`. Parameters are an experimental recipe, not an endorsed final configuration.

## Evaluate adapted AudioSeal

Populate the manifest schema in `configs/eval.example.json` with local paths and fixed binary messages.

```bash
python -m audioseal_st.evaluate \
  --manifest configs/eval.example.json --data-root /path/to/evaluation_audio \
  --audioseal-checkpoint /path/to/adapted_generator.pt \
  --sptk-config /path/to/tokenizer_config.json \
  --sptk-checkpoint /path/to/tokenizer.pt \
  --single-only --attacks ST \
  --residual-mode capped --minimum-snr 29 --output runs/evaluation
```

`--attacks NA,ST,lowpass,gaussian,mp3` evaluates attacks independently, never as a chain. Defaults: 3.4 kHz low-pass biquad, 20 dB Gaussian noise relative to each input, and MP3 at 32 kbps. Match residual constraints to training.

## MaskNet and multiplexed evaluation

MaskNet produces two time-varying weights for the adapted AudioSeal and PerTh residuals:

```text
output = clip(clean + mask_A * residual_A + mask_P * residual_P, -1, 1)
```

Architecture and output activation are explicit in `configs/mask.example.json`. The manuscript-aligned training entry point, `python -m audioseal_st.train_mask`, is not implemented yet and exits with a clear message.

```bash
python -m audioseal_st.evaluate \
  --manifest configs/eval.example.json --data-root /path/to/evaluation_audio \
  --audioseal-checkpoint /path/to/adapted_generator.pt \
  --sptk-config /path/to/tokenizer_config.json \
  --sptk-checkpoint /path/to/tokenizer.pt \
  --mask-checkpoint /path/to/mask.pt --mask-config configs/mask.example.json \
  --attacks ST --residual-mode capped --minimum-snr 29 \
  --output runs/multiplexed_evaluation
```

This evaluates the AudioSeal/PerTh pair and its component references, additive baseline, and MaskNet routing. Other watermark families are outside this release.

## Output and interpretation

- Per-clip scores, configuration hashes, and aggregate detection/SNR results are saved.
- TPR@FPR≤1% uses transformed clean negatives for each single attack.
- Joint Any optimizes two thresholds under one shared false-positive budget. It is not simultaneous survival or payload recovery.
- Empirical ROC statistics are not independently calibrated deployment thresholds.
- The evaluation loader accepts generator-format AudioSeal parameters; experimental residual adapters need a separate loader.
- Public PESQ/STOI reporting and manuscript-aligned MaskNet training remain incomplete.

## Checks

```bash
python scripts/evaluate.py --config configs/example.json --dry-run
PYTHONPATH=src python -m unittest discover -s tests -v
```

The dry run checks configuration only and leaves metric fields null. It never substitutes for real inference. Automated checks do not establish paper-level reproduction.

## License

License selection is pending. Third-party dependencies retain their respective terms.
