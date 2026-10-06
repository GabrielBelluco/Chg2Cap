"""Manifest-driven RSCC training; original Chg2Cap modules, no automatic test access.

Large runs require explicit opt-in. This entry point never downloads data/weights.
"""

import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.RSCC.rscc import RSCCDataset, build_vocab, sha256_file, validate_splits
from model.model_encoder import Encoder, AttentiveEncoder
from model.model_decoder import DecoderTransformer
from captioning.common import assert_disjoint
from captioning.losses import batch_loss, validation_loss
from captioning.common import check_image_hashes, write_json
from captioning.text_metrics import repetition
from captioning.resume import validate_resume, restore_training, require_disk_space


def generate_rows(encoder, attentive, decoder, dataset, indices, vocab):
    encoder.eval()
    attentive.eval()
    decoder.eval()
    inverse = {i: w for w, i in vocab.items()}
    output = []
    with torch.no_grad():
        for i in indices:
            before, after, _, _, _, _, name = dataset[i]
            a, b = encoder(before.unsqueeze(0).cuda(), after.unsqueeze(0).cuda())
            a, b = attentive(a, b)
            ids = decoder.sample(a, b, k=1)
            words = [inverse[j] for j in ids if j not in (0, 2, 3)]
            row = dataset.records[i]
            output.append({'id': name, 'split': row['split'], 'before': row['before'], 'after': row['after'],
                           'reference': row['caption'], 'prediction': ' '.join(words), 'generated_token_ids': ids,
                           'ended_with_eos': ids[-1] == 3, 'output_cap_reached': ids[-1] != 3,
                           'repetition': repetition(words)})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--max-train-seconds', type=float, default=1200)
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--generation-limit', type=int, default=128)
    parser.add_argument('--allow-large-run', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--resume', type=Path, help='Resume a full-epoch best.pth into a NEW output directory')
    parser.add_argument('--skip-final-generation', action='store_true', help='Save weights/results without generating every validation caption at the end')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new run directory')
    if not 1 <= args.epochs <= 1000 or not 120 <= args.max_train_seconds <= 604800:
        parser.error('Require 1..1000 epochs and 120..604800 seconds')
    if not 1 <= args.batch_size <= 32 or not 1 <= args.patience <= 1000 or not 8 <= args.generation_limit <= 551:
        parser.error('Invalid batch size, patience or output limit')
    rows = [json.loads(l) for l in (args.prepared / 'manifest.jsonl').read_text().splitlines()]
    validate_splits(rows)
    assert_disjoint(rows)
    if (len(rows) > 1000 or args.max_train_seconds > 1800 or args.epochs > 50) and not args.allow_large_run:
        parser.error('Large run requires explicit --allow-large-run')
    vocab = json.loads((args.prepared / 'vocab.json').read_text())
    audit = json.loads((args.prepared / 'audit.json').read_text())
    if vocab != build_vocab([r for r in rows if r['split'] == 'train']):
        parser.error('Vocabulary is not exclusively derived from training')
    # Test rows participate in overlap checks only. Their images and captions are not evaluated.
    active = [r for r in rows if r['split'] in ('train', 'val')]
    check_image_hashes(active, args.image_root)
    datasets = {s: RSCCDataset(args.prepared / 'manifest.jsonl', args.image_root, vocab, s, audit['max_length'])
                for s in ('train', 'val')}
    resume_state = None
    resume_sha256 = None
    if args.resume:
        if args.resume.name != 'best.pth':
            parser.error('Resume currently accepts only best.pth, not last/temporary checkpoints')
        resume_state = torch.load(args.resume, map_location='cpu', weights_only=True)
        resume_cfg = {'manifest_sha256': sha256_file(args.prepared / 'manifest.jsonl'),
                      'vocab_sha256': sha256_file(args.prepared / 'vocab.json'),
                      'max_length': audit['max_length'], 'batch_size': args.batch_size, 'seed': args.seed}
        validate_resume(resume_state, vocab, resume_cfg, args.epochs)
        old_history = json.loads((args.resume.parent / 'history.json').read_text())
        saved_epoch = next(h for h in old_history if h['epoch'] == resume_state['epoch'])
        if not saved_epoch['full_epoch'] or saved_epoch['steps'] != resume_state['steps'] or saved_epoch['val_loss'] != resume_state['validation_loss']:
            parser.error('Source checkpoint/history mismatch or partial epoch')
        for path in ('model/model_encoder.py', 'model/model_decoder.py', 'data/RSCC/rscc.py'):
            if sha256_file(path) != resume_state['config']['code_sha256'][path]:
                parser.error('Resume model/loader/loss source changed: ' + path)
        # New checkpoints must match the active loss/optimizer implementation.
        for path in ('captioning/losses.py', 'captioning/resume.py'):
            expected = resume_state['config']['code_sha256'].get(path)
            if expected and sha256_file(path) != expected:
                parser.error('Resume implementation changed: ' + path)
        # Historical checkpoints require their original source files for hash checks.
        if 'captioning/losses.py' not in resume_state['config']['code_sha256']:
            for path in ('scripts/run_rscc_pilot.py', 'scripts/train_rscc.py'):
                expected = resume_state['config']['code_sha256'].get(path)
                if not expected or sha256_file(path) != expected:
                    parser.error('Legacy resume source missing or changed: ' + path)
        resume_sha256 = sha256_file(args.resume)
        require_disk_space(args.output.parent, args.resume.stat().st_size, copies=3)
    if args.check_only:
        print(json.dumps({'status': 'inputs_validated_no_training', 'train': len(datasets['train']),
                          'val': len(datasets['val']), 'test_evaluated': False, 'vocab_size': len(vocab),
                          'resume_epoch': resume_state['epoch'] if resume_state else None}))
        return
    if not torch.cuda.is_available():
        parser.error('GPU not accessible from this execution environment')
    cached = Path(torch.hub.get_dir()) / 'checkpoints/resnet101-63fe2227.pth'
    if not cached.is_file():
        parser.error('Cached ResNet101 weights required; downloads disabled')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(datasets['val'], batch_size=args.batch_size, shuffle=False, num_workers=0)
    args.output.mkdir(parents=True)
    code = ['captioning/train.py', 'captioning/resume.py', 'captioning/losses.py',
            'captioning/common.py', 'captioning/text_metrics.py',
            'data/RSCC/rscc.py', 'model/model_encoder.py', 'model/model_decoder.py']
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    cfg.update({'torch': str(torch.__version__), 'rocm': torch.version.hip, 'gpu': torch.cuda.get_device_name(0),
                'max_length': audit['max_length'], 'vocab_size': len(vocab), 'learning_rate': .0001,
                'architecture': 'Original ResNet101 + 3 attentive layers + 1 decoder layer, 8 heads, dropout .1',
                'initialization': 'Cached ImageNet ResNet frozen/eval; attentive encoder and decoder random; no diagnostic checkpoint.',
                'selection': 'Minimum cross-entropy on all validation rows; patience measured in epochs.',
                'test_policy': 'Never evaluated by this script.', 'image_size': 256,
                'normalization': 'ImageNet RGB', 'optimizer': 'Adam', 'grad_clip': 1.,
                'manifest_sha256': sha256_file(args.prepared / 'manifest.jsonl'),
                'vocab_sha256': sha256_file(args.prepared / 'vocab.json'),
                'resnet_weights_sha256': sha256_file(cached), 'code_sha256': {p: sha256_file(p) for p in code}})
    if resume_state:
        cfg.update({'initialization': 'Restored model, Adam, Torch/CUDA RNG and shuffle generator from source best checkpoint.',
                    'resume_sha256': resume_sha256, 'resume_epoch': resume_state['epoch'],
                    'resume_step': resume_state['steps'], 'budget_scope': 'New session only; historical updates excluded from elapsed time.'})
    write_json(args.output / 'config.json', cfg)
    for path in code:
        dest = args.output / 'source_snapshot' / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    for name in ('manifest.jsonl', 'vocab.json', 'audit.json'):
        shutil.copy2(args.prepared / name, args.output / name)
    encoder = Encoder('resnet101').cuda().eval()
    encoder.fine_tune(False)
    attentive = AttentiveEncoder(3, [16, 16, 2048], 8, 512, 2048, .1).cuda()
    decoder = DecoderTransformer(2048, 2048, len(vocab), audit['max_length'], vocab, 8, 1, .1).cuda()
    decoder.max_lengths = args.generation_limit
    params = list(attentive.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=.0001)
    history, best, stale, step = [], math.inf, 0, 0
    start_epoch = 1
    checkpoint_bytes = args.resume.stat().st_size if args.resume else 0
    if resume_state:
        restore_training(resume_state, (encoder, attentive, decoder), optimizer, generator)
        start_epoch = resume_state['epoch'] + 1
        best, step = resume_state['validation_loss'], resume_state['steps']
        shutil.copy2(args.resume, args.output / 'best.pth')
        print(json.dumps({'status': 'resumed', 'next_epoch': start_epoch, 'restored_steps': step,
                          'max_train_seconds_this_session': args.max_train_seconds}), flush=True)
        del resume_state
    initial_step = step
    start = time.monotonic()
    reason = 'epoch_limit'

    def save_checkpoint(path, epoch, val_loss):
        nonlocal checkpoint_bytes
        if checkpoint_bytes:
            require_disk_space(path.parent, checkpoint_bytes)
        state = {'encoder_dict': encoder.state_dict(), 'encoder_trans_dict': attentive.state_dict(),
                 'decoder_dict': decoder.state_dict(), 'optimizer_dict': optimizer.state_dict(),
                 'epoch': epoch, 'steps': step, 'validation_loss': val_loss, 'config': cfg, 'vocab': vocab,
                 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                 'data_generator_rng': generator.get_state(), 'full_epoch': history[-1]['full_epoch']}
        tmp = path.with_suffix('.tmp')
        torch.save(state, tmp)
        tmp.replace(path)
        checkpoint_bytes = path.stat().st_size

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            attentive.train()
            decoder.train()
            total, count, batches = 0., 0, 0
            for batch in train_loader:
                if time.monotonic() - start > args.max_train_seconds - 75:
                    reason = 'time_budget'
                    break
                optimizer.zero_grad(set_to_none=True)
                loss, n = batch_loss(encoder, attentive, decoder, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize()
                step += 1
                batches += 1
                total += loss.item() * n
                count += n
                if step % 25 == 0:
                    progress = {'epoch': epoch, 'steps': step, 'last_loss': loss.item(), 'wall_seconds': time.monotonic() - start}
                    write_json(args.output / 'progress.json', progress)
                    print(json.dumps(progress), flush=True)
            if not batches:
                break
            val_loss = validation_loss(encoder, attentive, decoder, val_loader)
            entry = {'epoch': epoch, 'steps': step, 'batches': batches, 'full_epoch': batches == len(train_loader),
                     'train_loss': total / count, 'val_loss': val_loss, 'wall_seconds': time.monotonic() - start}
            history.append(entry)
            write_json(args.output / 'history.json', history)
            print(json.dumps(entry), flush=True)
            if val_loss < best:
                best, stale = val_loss, 0
                save_checkpoint(args.output / 'best.pth', epoch, val_loss)
            else:
                stale += 1
            if epoch in (1, 3, 5) or epoch % 5 == 0:
                probes = generate_rows(encoder, attentive, decoder, datasets['val'], range(min(4, len(datasets['val']))), vocab)
                write_json(args.output / f'val-probes-epoch-{epoch:03d}.json', probes)
            if stale >= args.patience:
                reason = 'validation_patience'
                break
            if reason == 'time_budget':
                break
        if history:
            save_checkpoint(args.output / 'last.pth', history[-1]['epoch'], history[-1]['val_loss'])
    except BaseException:
        reason = 'error_or_interrupt'
        raise
    finally:
        elapsed = time.monotonic() - start
        write_json(args.output / 'training_budget.json', {'wall_seconds_including_validation_and_saving': elapsed,
                                                        'limit_seconds': args.max_train_seconds, 'steps': step, 'stop_reason': reason})
    if not (args.output / 'best.pth').is_file():
        raise RuntimeError('No checkpoint produced')
    del optimizer, params
    state = torch.load(args.output / 'best.pth', map_location='cpu', weights_only=True)
    attentive.load_state_dict(state['encoder_trans_dict'], strict=True)
    decoder.load_state_dict(state['decoder_dict'], strict=True)
    selected_epoch, selected_step = state['epoch'], state['steps']
    del state
    predictions = []
    if not args.skip_final_generation:
        predictions = generate_rows(encoder, attentive, decoder, datasets['val'], range(len(datasets['val'])), vocab)
        write_json(args.output / 'validation_predictions.json', predictions)
    result = {'status': 'completed', 'training_steps': step, 'epochs_executed': len(history),
              'selected_epoch': selected_epoch, 'selected_step': selected_step, 'best_validation_loss': best,
              'training_wall_seconds': elapsed, 'stop_reason': reason, 'test_evaluated': False,
              'session_training_steps': step - initial_step, 'final_generation_skipped': args.skip_final_generation,
              'validation_predictions': len(predictions), 'checkpoint_sha256': sha256_file(args.output / 'best.pth'),
              'last_checkpoint_sha256': sha256_file(args.output / 'last.pth') if (args.output / 'last.pth').exists() else None,
              'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated()}
    write_json(args.output / 'result.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
