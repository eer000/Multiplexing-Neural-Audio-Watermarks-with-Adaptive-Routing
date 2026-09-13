# Multiplexing Neural Audio Watermarks with Adaptive Routing

Research code for SpeechTokenizer-aware AudioSeal adaptation and MaskNet routing of adapted AudioSeal + PerTh.

**Initial research release. No validated final weights, datasets, or verified paper results are bundled.** Paths in the examples are placeholders. Some modules remain explicitly incomplete.

## What is available

| Component | Status |
|---|---|
| AudioSeal adaptation to SpeechTokenizer | Research implementation: real ST forward, frozen proxy backward; decoder/message processor fine-tuning |
| Adapted AudioSeal single-attack evaluation | Implemented; original AudioSeal reference; NA, ST, low-pass, Gaussian noise, MP3 |
| MaskNet network | Implemented; architecture/activation explicitly configured |
| MaskNet routing of adapted AudioSeal + PerTh | Checkpoint-based evaluation implemented; validated routing weights pending |
| Manuscript-aligned MaskNet training | Placeholder; `train_mask` exits explicitly |
| Proxy training and final checkpoint downloads | Pending release |
| Paper results / citation metadata | Pending verification |

No other watermark methods or datasets are required by this release. PerTh is needed only for multiplexing. A single-method comparison is not a broad watermark benchmark.

## Install

Use Python 3.10+ and install a compatible PyTorch runtime for your hardware, then:

```bash
pip install -e .                  # AudioSeal-only evaluation/training
pip install -e '.[multiplex]'     # additionally enables PerTh multiplexing
```

MP3 evaluation requires `ffmpeg` with libmp3lame on PATH. Historical runtime versions are recorded in `requirements-tested.txt`; this file is not a guarantee for every operating system. No model-dependent end-to-end release validation has been completed yet.

## 1. Fine-tune AudioSeal for ST

Supply your own audio and checkpoint paths:

```bash
python -m audioseal_st.train \
  --data-root /path/to/train-clean \
  --proxy-checkpoint /path/to/proxy.pt \
  --sptk-config /path/to/st_config.json \
  --sptk-checkpoint /path/to/st.pt \
  --output runs/adapt \
  --residual-mode capped --minimum-snr 29
```

The current data loader uses LibriSpeech-style FLAC filenames to create speaker-disjoint train/validation splits. Data is not distributed. This is an experimental training objective, not a claim of exact manuscript reproduction or successful adaptation. The proxy checkpoint format is `GatedLongContextProxy` (96 channels, 12 blocks). Detector and tokenizer weights are frozen. Every evaluation interval saves generator weights; optimizer/random states are also saved, but full resume support is pending.

## 2. Evaluate adapted AudioSeal

Replace the example audio path in `configs/eval.example.json`; add actual files with fixed 16-bit messages. A useful evaluation requires enough positive and negative samples; the one-entry example is only a schema illustration.

```bash
python -m audioseal_st.evaluate \
  --manifest configs/eval.example.json --data-root /path/to/audio \
  --audioseal-checkpoint /path/to/adapted_generator.pt \
  --sptk-config /path/to/st_config.json --sptk-checkpoint /path/to/st.pt \
  --single-only --attacks ST \
  --residual-mode capped --minimum-snr 29 --output runs/eval_st
```

For separate single-attack results, use `--attacks NA,ST,lowpass,gaussian,mp3`. These are **not composed attacks**. Low-pass is a 3.4 kHz biquad, noise is 20 dB relative to each input, and MP3 is 32 kbps. These settings are initial public defaults, not a claimed reproduction of all historical benchmark settings.

## 3. MaskNet method

`audioseal_st.models.MaskNet` produces two time-varying weights from mono input. The routed audio is:

```text
watermarked = clip(clean + mask_A * adapted_AudioSeal_residual
                         + mask_P * PerTh_residual, -1, 1)
```

The example configuration is five convolution layers, 128 hidden channels, kernel size 15, and ReLU output. Checkpoints must match the explicit configuration; no assumption is made about undocumented historical weights. The manuscript-aligned training entry point is reserved as `python -m audioseal_st.train_mask` and currently exits with a clear pending-implementation message.

## 4. Evaluate adapted AudioSeal + PerTh through MaskNet

```bash
python -m audioseal_st.evaluate \
  --manifest configs/eval.example.json --data-root /path/to/audio \
  --audioseal-checkpoint /path/to/adapted_generator.pt \
  --sptk-config /path/to/st_config.json --sptk-checkpoint /path/to/st.pt \
  --mask-checkpoint /path/to/mask.pt --mask-config configs/mask.example.json \
  --attacks ST --residual-mode capped --minimum-snr 29 \
  --output runs/eval_mask
```

This reports the component references, simple additive combination, and MaskNet combination for the same AudioSeal/PerTh pair. No unrelated watermark combinations are evaluated. Only generator-format AudioSeal checkpoints are accepted; experimental residual-adapter checkpoints need a future explicit loader.

## Metrics and interpretation

- Primary metric: empirical TPR at FPR ≤ 1%, with clean negatives undergoing the same single attack.
- Joint Any: exact joint threshold search with one shared false-positive budget; not two independent 1% budgets.
- Outputs: per-clip `scores.jsonl`, configuration/checkpoint hashes, `summary.json` with detection, bit accuracy and SNR.
- Full public PESQ/STOI reporting is pending. Existence detection is not message recovery.
- Empirical ROC uses evaluation labels; it is not an independently calibrated deployment threshold.
- Match residual shaping/energy settings to training. A weight file alone does not specify the full inference method.

## Check the placeholder without models

```bash
python scripts/evaluate.py --config configs/example.json --dry-run
```

This only validates configuration. All metric fields are null; it never pretends to run model inference. Numerical metric tests are under `tests/`.

## License and releases

License selection is pending. Third-party models/code/data are not bundled and retain their own terms. Validated weights, paper metadata, reproduction tables, and a tagged paper release will be added after verification.
