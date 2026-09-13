"""Validate a pipeline configuration; real inference is intentionally unimplemented."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('outputs/dry_run.json'))
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding='utf-8'))
        if config['sample_rate'] != 16000:
            raise ValueError('sample_rate must be 16000')
        if config['attack'] != 'speech_tokenizer':
            raise ValueError('This scaffold specifies a single SpeechTokenizer attack')
        if not 0 < config['target_fpr'] < 1:
            raise ValueError('target_fpr must lie between 0 and 1')
        for key in ('audio_manifest', 'audioseal_generator', 'audioseal_detector',
                    'speech_tokenizer_config', 'speech_tokenizer_checkpoint'):
            if not isinstance(config['paths'][key], str) or not config['paths'][key].strip():
                raise ValueError(f'Expected a nonempty path for {key}')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(f'Invalid configuration: {exc}')
    if not args.dry_run:
        parser.exit(2, 'Real evaluation is not implemented yet. Use --dry-run to check the scaffold.\n')
    result = {
        'status': 'dry_run_only',
        'model_inference_executed': False,
        'configuration': config,
        'planned_stages': ['load_audio', 'embed_watermark', 'speech_tokenizer',
                           'detect_positive_and_clean_negative', 'compute_metrics'],
        'metrics': {'tpr_at_target_fpr': None, 'observed_fpr': None,
                    'snr': None, 'pesq': None, 'stoi': None},
        'note': 'Configuration checked only. No audio, models, or experimental results.'
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f'Dry run completed: {args.output}. No experimental metrics computed.')


if __name__ == '__main__':
    main()
