# Multiplexing Neural Audio Watermarks with Adaptive Routing

AudioSeal adaptation for SpeechTokenizer, with feature-constrained generator–detector fine-tuning and AudioSeal–PerTh routing through MaskNet.

**Current release: [Feature-constrained AudioSeal, step 600](https://github.com/eer000/Multiplexing-Neural-Audio-Watermarks-with-Adaptive-Routing/releases/tag/v0.2.0-latent600).** The release includes the adapted generator and detector, the initialization and proxy used for fine-tuning, and the historical MaskNet routing checkpoint.

## Install and download

Use Python 3.10+ with PyTorch and TorchAudio built for the same platform. Training requires CUDA. MP3 evaluation requires FFmpeg with libmp3lame.

```bash
pip install -e '.[multiplex]'
python -m audioseal_st.download --output weights --training --masknet
```

Downloads are versioned and verified against the SHA-256 hashes in [weights.json](weights.json). AudioSeal's upstream base generator is loaded by its official package. For training or ST evaluation, obtain `config.json` and `SpeechTokenizer.pt` from the [official 16 kHz HuBERT-average SpeechTokenizer model](https://huggingface.co/fnlp/SpeechTokenizer/tree/main/speechtokenizer_hubert_avg), and place them in `tokenizer/`.

## Embed a watermark

```bash
python -m audioseal_st.infer --input speech.wav --output watermarked.wav --weights weights
```

The output is mono 16 kHz PCM. The default message is `0101010101010101`; `--message` accepts any 16 binary digits. This release is evaluated for watermark presence; presence detection and payload recovery are separate metrics.

Inference applies the same constraints as training: a smooth 40 ms input-energy-dependent residual limit with parameter 30 dB, a global 26 dB cap, and attenuation to at most the original AudioSeal residual power for the same audio and message. The local limiter is a pointwise envelope constraint, rather than a guaranteed window-wise SNR bound.

## Fine-tune

Create `train_manifest.json` with four non-overlapping path lists named `train`, `val`, `cal`, and `test`, relative to `audio/`. Use speaker-disjoint groups. Inputs must be mono-compatible 16 kHz audio of at least four seconds; the first four seconds are used.

```bash
python -m audioseal_st.train \
  --manifest train_manifest.json --data-root audio \
  --init-generator weights/initial_generator.pt \
  --proxy-checkpoint weights/proxy.pt \
  --sptk-config tokenizer/config.json \
  --sptk-checkpoint tokenizer/SpeechTokenizer.pt \
  --steps 600 --save-every 100 --output runs/feature_constraint
```

The recipe starts from the supplied adapted AudioSeal generator and the official pretrained AudioSeal detector. It updates the generator decoder/message processor and the detector, while freezing SpeechTokenizer and the proxy. Real ST supplies forward outputs; the proxy supplies backward gradients. A cosine constraint encourages the input-induced change in continuous ST encoder features to remain aligned after ST reconstruction. The loss also anchors generated audio to the initial low-noise output and penalizes spectral masking violations. Detector training includes identically processed clean negatives and energy-matched ordinary-noise negatives. One quarter of steps also use the unattacked positive and clean negative branch.

The detector learning rate is 1e-5, generator learning rate 2e-6, and batch size 2. Each stage saves both models, optimizer state, and random state. The released models are the step-600 pair from an 800-step experiment. The supplied initialization supports this adaptation stage directly. Training history and the published stage are distinct from validation-based checkpoint selection performed by a new run.

## Evaluate single attacks

Create `eval_manifest.json` with a `files` array. Each entry contains `path` (relative to `audio/`), `split` (`cal` or `eval`), and `message` (16 binary integers). Use separate calibration and evaluation speakers and fixed messages. Calibration entries supply clean negatives; evaluation entries supply paired clean and watermarked audio.

```bash
python -m audioseal_st.evaluate \
  --manifest eval_manifest.json --data-root audio \
  --audioseal-checkpoint weights/generator.pt \
  --detector-checkpoint weights/detector.pt \
  --sptk-config tokenizer/config.json \
  --sptk-checkpoint tokenizer/SpeechTokenizer.pt \
  --single-only --seconds 4 \
  --attacks NA,ST,lowpass,gaussian,mp3 --output runs/single
```

Attacks are independent: no attack, all-layer ST reconstruction, 3.4 kHz biquad low-pass, 20 dB Gaussian noise, and 32 kbps MP3. Gaussian noise uses the same per-clip seed across compared waveforms and is scaled relative to each input.

For historical MaskNet + PerTh routing, omit `--single-only` and add:

```bash
--mask-checkpoint weights/masknet.pt --mask-config configs/masknet.json
```

The router has three convolutional layers, 32 hidden channels, kernel size 15, and sigmoid output. It mixes the constrained AudioSeal and PerTh residuals before clipping. This historical router is distributed for direct evaluation, and is not retrained by the AudioSeal adaptation command.

Reports contain raw per-clip scores, artifact hashes, SNR/PESQ/STOI, original and adapted AudioSeal detector scores, and PerTh scores. Primary detection results use independent calibration. Joint Any uses a fixed 0.5% calibration budget per detector; its test false-positive rate is measured. Empirical test-set ROC statistics are supplied separately.

## Released-checkpoint measurements

1,000 calibration clips and 500 evaluation clips from LibriSpeech test-clean, with disjoint speakers and excluding the preceding 128-clip experiment. Quality compares watermarked audio with its original waveform before attack. All rows use independently calibrated, attack-specific thresholds.

| Single attack | Adapted AudioSeal TPR / FPR | Historical MaskNet + PerTh Any TPR / FPR |
|---|---:|---:|
| None | 96.0% / 2.0% | 100% / 6.4% |
| SpeechTokenizer | 70.4% / 1.8% | 59.8% / 6.4% |
| Low-pass | 83.8% / 1.2% | 100% / 5.8% |
| Gaussian noise | 99.6% / 1.4% | 100% / 4.6% |
| MP3 | 54.4% / 3.4% | 100% / 7.6% |

| Embedding | SNR | PESQ | STOI |
|---|---:|---:|---:|
| Adapted AudioSeal | 29.12 dB | 4.014 | 0.9952 |
| Historical MaskNet + PerTh | 15.33 dB | 3.923 | 0.9888 |

These are measurements of this released adaptation, distinct from the paper's benchmark. The observed ST false-positive rate is 1.8%, above the 1% calibration target. [Machine-readable results](results/latent600.json) retain the measured rates and thresholds. Joint Any means at least one detector accepts, rather than simultaneous survival of both payloads.

## Checks and attribution

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

AudioSeal and its pretrained weights are MIT licensed. See [third-party notices](THIRD_PARTY_NOTICES.md). Other dependencies retain their respective terms. No additional license grant is made for this repository's original code and artifacts.
