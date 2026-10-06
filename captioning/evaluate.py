"""Current experiment inference; test requires a protocol frozen after validation."""

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from data.RSCC.rscc import RSCCDataset, sha256_file, validate_splits, resolve_image
from captioning.common import assert_disjoint
from captioning.common import check_image_hashes, write_json
from captioning.train import generate_rows
from captioning.text_metrics import metrics
from model.model_encoder import Encoder, AttentiveEncoder
from model.model_decoder import DecoderTransformer


def select_rows(rows, scope, per_event=2, seed=17, include_earlier=False, split='val'):
    """Choose before generation; deterministic under manifest reordering."""
    if split not in ('val', 'test'):
        raise ValueError('Only validation or reserved test allowed')
    if split == 'test' and (scope != 'all' or include_earlier):
        raise ValueError('Reserved test requires all pairs and no validation probes')
    val = [r for r in rows if r['split'] == split]
    if not val or len({r['id'] for r in val}) != len(val):
        raise ValueError('Validation is empty or contains duplicate IDs')
    if scope == 'all':
        return sorted(val, key=lambda r: r['id'])
    if scope != 'sample' or per_event < 1:
        raise ValueError('Invalid selection')
    chosen = {}
    for event in sorted({r['event'] for r in val}):
        candidates = [r for r in val if r['event'] == event]
        if len(candidates) < per_event:
            raise ValueError('Not enough validation rows in event: ' + event)
        ranked = sorted(candidates, key=lambda r: hashlib.sha256(
            f"evaluation:{seed}:{r['id']}".encode()).hexdigest())
        chosen.update((r['id'], r) for r in ranked[:per_event])
    if include_earlier:
        # These are the same first four validation records used by train_rscc.py.
        chosen.update((r['id'], r) for r in val[:4])
    return sorted(chosen.values(), key=lambda r: r['id'])


def validate_checkpoint(state, vocab, manifest_hash, vocab_hash, audit, run):
    if state['vocab'] != vocab:
        raise ValueError('Checkpoint vocabulary mismatch')
    cfg = state['config']
    for key, expected in [('manifest_sha256', manifest_hash),
                          ('vocab_sha256', vocab_hash), ('max_length', audit['max_length'])]:
        if cfg[key] != expected:
            raise ValueError('Checkpoint mismatch: ' + key)
    for path in ('model/model_encoder.py', 'model/model_decoder.py', 'data/RSCC/rscc.py'):
        if sha256_file(path) != cfg['code_sha256'][path]:
            raise ValueError('Source changed since training: ' + path)
    history = json.loads((run / 'history.json').read_text())
    entry = next((h for h in history if h['epoch'] == state['epoch']), None)
    if entry is None or not entry['full_epoch'] or entry['steps'] != state['steps']:
        raise ValueError('Checkpoint not matched to a complete epoch')
    if entry['val_loss'] != state['validation_loss']:
        raise ValueError('Checkpoint loss/history mismatch')


def render_html(path, selected, outputs, info, image_root, split='val'):
    """Read-only local gallery: images remain at their existing locations."""
    esc = html.escape
    maps = {label: {r['id']: r for r in rows} for label, rows in outputs.items()}
    pieces = ['<!doctype html><html lang="pt-BR"><meta charset="utf-8">',
              '<title>Comparação RSCC — validação</title>',
              '<style>body{font:17px sans-serif;max-width:1150px;margin:30px auto;padding:12px;line-height:1.5}'
              'article{border-top:2px solid #aaa;margin:32px 0} .images{display:flex;gap:15px}'
              'figure{margin:0;width:50%}img{width:100%;max-width:512px}pre{white-space:pre-wrap}</style>',
              '<h1>Comparação de legendas — validação</h1>'
              '<p>Sem treinamento e sem avaliação do teste. A referência também pode conter erros. '
              'Julgue mudança visível, omissão, repetição, inversão temporal e afirmação sem suporte. '
              'Registrar “inconclusivo” quando necessário. Esta página não salva anotações.</p>']
    for row in selected:
        pieces.append('<article><h2>' + esc(row['id']) + '</h2><div class="images">')
        for field, phase, title in [('before', 'pre', 'Antes'), ('after', 'post', 'Depois')]:
            image_path = resolve_image(image_root, row[field], phase)
            from urllib.parse import quote
            src = quote(os.path.relpath(image_path, path.parent), safe='/')
            pieces.append(f'<figure><figcaption>{title}</figcaption><img loading="lazy" src="{esc(src)}" alt="{title}"></figure>')
        pieces.append('</div><h3>Referência</h3><p>' + esc(row['caption']) + '</p>')
        for label, mapping in maps.items():
            prediction = mapping.get(row['id'])
            if prediction:
                pieces.append(f'<h3>{esc(label)} — época {info[label]["epoch"]}</h3><p>' +
                              esc(prediction['prediction']) + '</p>')
        pieces.append('</article>')
    rendered = '\n'.join(pieces) + '</html>'
    if split == 'test':
        rendered = rendered.replace('— validação', '— teste reservado').replace(
            'Sem treinamento e sem avaliação do teste.',
            'Teste reservado com checkpoint fixado na validação. Sem treinamento.')
    path.write_text(rendered, encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, default=Path('output/rscc/ebd-full-v1'))
    p.add_argument('--best', type=Path, default=Path('output/rscc/ebd-resume-v1'))
    p.add_argument('--prepared', type=Path, default=Path('data/RSCC/prepared/ebd-full-v1'))
    p.add_argument('--image-root', type=Path, default=Path('data/RSCC/full/images'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scope', choices=['sample', 'all'], default='sample')
    p.add_argument('--per-event', type=int, default=2)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--include-earlier-probes', action='store_true')
    p.add_argument('--generation-limit', type=int, default=551)
    p.add_argument('--max-eval-seconds', type=int, default=3600)
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--models', choices=['both', 'best'], default='both')
    p.add_argument('--allow-test', action='store_true')
    p.add_argument('--protocol', type=Path, help='Required frozen protocol for a new test')
    args = p.parse_args()
    is_test = args.split == 'test'
    if is_test and (not args.allow_test or args.models != 'best' or args.scope != 'all'
                    or args.include_earlier_probes or not args.protocol):
        p.error('Test requires --allow-test --models best --scope all --protocol; no probes')
    frozen = None
    if is_test:
        from captioning.protocol import check
        frozen = check(args.protocol, args.best, args.prepared, args.generation_limit)
    if args.output.exists():
        p.error('Output already exists; choose a NEW directory')
    if not 1 <= args.per_event <= 100 or not 8 <= args.generation_limit <= 551 or args.max_eval_seconds < 60:
        p.error('Invalid selection, generation limit or budget')
    start = time.monotonic()
    manifest = args.prepared / 'manifest.jsonl'
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    validate_splits(rows)
    assert_disjoint(rows)
    selected = select_rows(rows, args.scope, args.per_event, args.seed, args.include_earlier_probes, args.split)
    if is_test and len(selected) != frozen['test_pairs']:
        raise ValueError('Test coverage differs from frozen manifest')
    vocab = json.loads((args.prepared / 'vocab.json').read_text())
    audit = json.loads((args.prepared / 'audit.json').read_text())
    if args.generation_limit > audit['max_length']:
        p.error('Generation limit exceeds model capacity')
    manifest_hash = sha256_file(manifest)
    vocab_hash = sha256_file(args.prepared / 'vocab.json')
    check_image_hashes(selected, args.image_root)
    info = {}
    for label, run in [('baseline', args.baseline), ('best', args.best)]:
        if args.models == 'best' and label != 'best':
            continue
        checkpoint = run / 'best.pth'
        digest = sha256_file(checkpoint)
        result_file = run / 'result.json'
        if result_file.exists() and json.loads(result_file.read_text())['checkpoint_sha256'] != digest:
            raise ValueError('Checkpoint checksum differs from training result')
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        validate_checkpoint(state, vocab, manifest_hash, vocab_hash, audit, run)
        if is_test and (state['epoch'] != frozen['epoch'] or digest != frozen['checkpoint_sha256']):
            raise ValueError('Checkpoint differs from frozen validation')
        info[label] = {'epoch': state['epoch'], 'steps': state['steps'],
                       'checkpoint': str(checkpoint.resolve()), 'sha256': digest,
                       'training_validation_loss': state['validation_loss']}
        del state
    plan = {'status': 'inputs_validated_no_inference', 'scope': args.scope,
            'pairs': len(selected), 'models': info, 'test_evaluated': False,
            'generation_limit': args.generation_limit, 'selection_seed': args.seed,
            'per_event': args.per_event, 'include_earlier_probes': args.include_earlier_probes,
            'manifest_sha256': manifest_hash, 'vocab_sha256': vocab_hash,
            'max_eval_seconds': args.max_eval_seconds,
            'budget_scope': 'Soft wall budget from process preparation; checked between images. One generation/save can overrun.',
            'warning': 'Validation used for checkpoint selection; text metrics are not factual accuracy. Sample is exploratory.',
            'script_sha256': sha256_file(Path(__file__))}
    plan['split'] = args.split
    from captioning.protocol import code_hashes
    plan['code_sha256'] = code_hashes()
    plan['test_authorized'] = is_test
    plan['frozen_protocol'] = frozen
    plan['warning'] = ('Reserved EBD test; checkpoint and greedy generation fixed before evaluation. '
                       'Do not tune on these results. Text metrics do not establish visual correctness.' if is_test else
                       'Validation used for checkpoint selection; text metrics are not factual accuracy.'
                       + (' Sample is exploratory.' if args.scope == 'sample' else ' Full validation.'))
    print(json.dumps(plan, indent=2), flush=True)
    if args.check_only:
        return
    if not torch.cuda.is_available():
        p.error('GPU unavailable; run in the normal ROCm terminal')
    cached = Path(torch.hub.get_dir()) / 'checkpoints/resnet101-63fe2227.pth'
    if not cached.is_file():
        p.error('Cached ResNet101 required; this evaluation must not download weights')
    torch.set_num_threads(4)
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args.output.mkdir(parents=True)
    write_json(args.output / 'protocol.json', plan)
    write_json(args.output / 'selected.json', selected)
    dataset = RSCCDataset(manifest, args.image_root, vocab, args.split, audit['max_length'])
    lookup = {r['id']: i for i, r in enumerate(dataset.records)}
    outputs = {}
    try:
        for label in info:
            if time.monotonic() - start >= args.max_eval_seconds:
                raise TimeoutError('Evaluation budget reached before loading next model')
            state = torch.load(info[label]['checkpoint'], map_location='cpu', weights_only=True)
            encoder = Encoder('resnet101').cuda().eval()
            attentive = AttentiveEncoder(3, [16, 16, 2048], 8, 512, 2048, .1).cuda().eval()
            decoder = DecoderTransformer(2048, 2048, len(vocab), audit['max_length'], vocab, 8, 1, .1).cuda().eval()
            for model, key in [(encoder, 'encoder_dict'), (attentive, 'encoder_trans_dict'), (decoder, 'decoder_dict')]:
                model.load_state_dict(state[key], strict=True)
                model.requires_grad_(False)
            del state, model
            decoder.max_lengths = args.generation_limit
            outputs[label] = []
            with (args.output / f'{label}-predictions.jsonl').open('x', encoding='utf-8') as stream:
                for n, row in enumerate(selected, 1):
                    if time.monotonic() - start >= args.max_eval_seconds:
                        raise TimeoutError('Evaluation budget reached; partial outputs preserved')
                    prediction = generate_rows(encoder, attentive, decoder, dataset, [lookup[row['id']]], vocab)[0]
                    prediction['event'] = row['event']
                    outputs[label].append(prediction)
                    stream.write(json.dumps(prediction, ensure_ascii=False) + '\n')
                    stream.flush()
                    print(json.dumps({'model': label, 'epoch': info[label]['epoch'], 'done': n,
                                      'total': len(selected), 'seconds': round(time.monotonic() - start, 1)}), flush=True)
            write_json(args.output / f'{label}-predictions.json', outputs[label])
            del encoder, attentive, decoder
            torch.cuda.empty_cache()
        scores = {label: metrics(predictions) for label, predictions in outputs.items()}
        for label, predictions in outputs.items():
            scores[label]['distinct_caption_fraction'] = len({r['prediction'] for r in predictions}) / len(predictions)
            scores[label]['epoch'] = info[label]['epoch']
        report = {'status': 'completed', 'test_evaluated': is_test, 'pairs': len(selected),
                  'models': scores, 'wall_seconds': time.monotonic() - start,
                  'warning': plan['warning'], 'unmeasured': ['METEOR', 'CIDEr-D', 'visual correctness']}
        write_json(args.output / 'metrics.json', report)
        render_html(args.output / 'comparison.html', selected, outputs, info, args.image_root, args.split)
        summary = ['# Avaliação EBD — geração livre', '',
                   f'Comparação em {len(selected)} pares de validação; teste não avaliado.', '',
                   '| Modelo / época | BLEU-4 | ROUGE-L | Repetição consecutiva | Trigramas repetidos |',
                   '| --- | ---: | ---: | ---: | ---: |']
        for label, score in scores.items():
            summary.append(f'| {label} / {score["epoch"]} | {score["BLEU_1_to_4"][3]:.5f} | '
                           f'{score["ROUGE_L"]:.5f} | {100*score["mean_adjacent_repeat_fraction"]:.2f}% | '
                           f'{100*score["mean_repeated_trigram_fraction"]:.2f}% |')
        summary.extend(['', 'BLEU/ROUGE estão na escala 0–1. Repetição é média por legenda, não taxa de erro.',
                        'A validação já participou da seleção do checkpoint; não é teste independente.',
                        'A amostra pequena é exploratória. Métricas textuais não comprovam correção visual.',
                        'METEOR e CIDEr-D não foram medidos. Confira as imagens em comparison.html.'])
        if is_test:
            summary[2] = f'Avaliação do checkpoint fixado da época {info["best"]["epoch"]} em {len(selected)} pares de teste reservado.'
            summary = [line for line in summary if not line.startswith(('A validação já', 'A amostra pequena'))]
            summary.extend(['Não ajustar checkpoint ou geração com base neste teste.',
                            'METEOR, ST5-SCS e BERTScore serão calculados separadamente nas predições salvas.'])
        elif args.scope == 'all':
            summary = [line.replace('A amostra pequena é exploratória. ', '') for line in summary]
        (args.output / 'RESUMO.md').write_text('\n'.join(summary) + '\n', encoding='utf-8')
        write_json(args.output / 'status.json', {'status': 'completed', 'pairs_per_model': len(selected)})
        print(json.dumps(report, indent=2), flush=True)
    except BaseException as exc:
        write_json(args.output / 'status.json', {'status': 'incomplete', 'reason': type(exc).__name__,
                   'detail': str(exc), 'completed_pairs': {k: len(v) for k, v in outputs.items()},
                   'test_evaluated': is_test and any(outputs.values()), 'wall_seconds': time.monotonic() - start})
        raise


if __name__ == '__main__':
    main()
