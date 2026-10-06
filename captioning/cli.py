"""Interface corrente Chg2Cap/EBD. Fluxo histórico preservado em rscc.py.

Somente biblioteca padrão: ajuda e --dry-run não importam Torch nem usam GPU.
Os processos filhos usam ambientes separados para não modificar venv_rocm.
Consulte README.md para o protocolo, saídas e comandos completos.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def local(value):
    """Caminhos relativos sempre partem do repositório, não do terminal atual."""
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def read(path):
    return json.loads(Path(path).read_text())


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/captioning_ebd.json')
    sub = p.add_subparsers(dest='action', required=True)
    for name in ('train', 'validation', 'test', 'metrics'):
        s = sub.add_parser(name)
        s.add_argument('--output', required=True, help='Diretório NOVO; nunca sobrescreve')
        s.add_argument('--dry-run', action='store_true', help='Mostrar comandos sem executar ou criar arquivos')
        if name == 'train':
            s.add_argument('--hours', type=float, required=True, help='Orçamento de treino em horas (limite suave)')
            s.add_argument('--resume', help='best.pth de origem; omitir inicia treino novo')
            s.add_argument('--check-only', action='store_true', help='Validar entradas sem treinar')
        elif name == 'metrics':
            s.add_argument('--evaluation', required=True, help='Pasta de uma avaliação concluída')
            s.add_argument('--allow-test', action='store_true')
        else:
            s.add_argument('--run', required=True, help='Pasta de treino com best.pth e history.json')
            s.add_argument('--hours', type=float, default=2, help='Limite suave da geração; métricas têm orçamento separado')
            s.add_argument('--check-only', action='store_true', help='Validar entradas/checkpoint sem gerar')
            s.add_argument('--basic-only', action='store_true', help='Somente geração, BLEU, ROUGE e repetição')
            if name == 'validation':
                s.add_argument('--baseline', help='Pasta opcional de outro treino para comparação na validação')
            else:
                s.add_argument('--allow-test', action='store_true', help='Confirmar acesso ao teste do protocolo congelado')
                s.add_argument('--protocol', required=True, help='JSON congelado após validação')
    return p


def generation_command(a, c):
    """Traduz a interface curta para os scripts auditados, sem mudar o algoritmo."""
    cmd = [str(local(c['training_python'])), '-B', '-u']
    common = ['--prepared', str(local(c['prepared'])), '--image-root', str(local(c['image_root'])),
              '--output', str(local(a.output))]
    if a.action == 'train':
        cmd += ['-m', 'captioning.train', *common, '--epochs', str(c['epochs']),
                '--patience', str(c['patience']), '--batch-size', str(c['batch_size']),
                '--seed', str(c['seed']), '--max-train-seconds', str(a.hours * 3600),
                '--generation-limit', str(c['generation_limit']), '--allow-large-run',
                '--skip-final-generation']
        # Treino seleciona por perda na validação, sem inferência no teste.
        # Separar geração evita que ela consuma o orçamento reservado ao treino.
        if a.resume:
            cmd += ['--resume', str(local(a.resume))]
    else:
        cmd += ['-m', 'captioning.evaluate', *common, '--split', 'test' if a.action == 'test' else 'val',
                '--scope', 'all', '--models', 'both' if getattr(a, 'baseline', None) else 'best',
                '--best', str(local(a.run)),
                '--baseline', str(local(getattr(a, 'baseline', None) or a.run)),
                '--generation-limit', str(c['generation_limit']), '--max-eval-seconds', str(int(a.hours * 3600))]
        # Novos testes exigem os artefatos e parâmetros congelados na validação.
        if a.action == 'test':
            cmd += ['--allow-test', '--protocol', str(local(a.protocol))]
    if a.check_only:
        cmd += ['--check-only']
    return cmd


def metric_commands(c, evaluation, output, labels, split):
    commands = []
    for label in labels:
        common = ['--predictions', str(evaluation / f'{label}-predictions.json'), '--split', split]
        if split == 'test':
            common += ['--allow-test']
        commands.append([str(local(c['meteor_python'])), '-B', '-u', '-m', 'captioning.meteor',
                         *common, '--output', str(output / f'{label}-meteor.json')])
        for metric, model, dtype, batch, seconds in [('st5-scs', 'st5_model', 'float16', 1, 7200),
                                                    ('bertscore', 'bert_model', 'float32', 4, 3600)]:
            commands.append([str(local(c['semantic_python'])), '-B', '-u', '-m', 'captioning.semantic',
                             *common, '--model-path', str(local(c[model])), '--metric', metric,
                             '--device', 'cuda', '--dtype', dtype, '--batch-size', str(batch),
                             '--threads', '4', '--bert-layers', '17', '--max-seconds', str(seconds),
                             '--output', str(output / f'{label}-{metric}.json')])
    return commands


def run(command):
    print('\n' + shlex.join(command), flush=True)
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    # Mesmo terminal: Ctrl+C chega aos processos. Nenhum monitor em segundo plano.
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def evaluation_info(evaluation):
    metrics = read(evaluation / 'metrics.json')
    protocol = read(evaluation / 'protocol.json')
    if metrics.get('status') != 'completed':
        raise ValueError('Geração incompleta: não consolidar como avaliação completa')
    split = protocol.get('split', 'val')
    if split not in ('val', 'test'):
        raise ValueError('Split inválido')
    labels = list(metrics['models'])
    reference_ids = None
    references = None
    for label in labels:
        rows = read(evaluation / f'{label}-predictions.json')
        ids = [r['id'] for r in rows]
        if not ids or len(ids) != len(set(ids)) or len(ids) != metrics['pairs']:
            raise ValueError('Contagem ou IDs inválidos')
        if any(r['split'] != split for r in rows):
            raise ValueError('Predições misturam divisões')
        if reference_ids is not None and ids != reference_ids:
            raise ValueError('Modelos não foram avaliados nos mesmos pares/ordem')
        texts = [r['reference'] for r in rows]
        if references is not None and texts != references:
            raise ValueError('Referências diferentes entre modelos')
        reference_ids = ids
        references = texts
    return split, labels, metrics


def summarize(evaluation, output):
    split, labels, base = evaluation_info(evaluation)
    lines = [f'# RSCC — {split}', '', f"Pares: {base['pairs']}. Métricas em escala 0–1.", '',
             '| Modelo | BLEU-4 | ROUGE-L | METEOR-NLTK | ST5-SCS | BERTScore F1 | Trigramas repetidos |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for label in labels:
        source = (evaluation / f'{label}-predictions.json').read_bytes()
        ids = [r['id'] for r in json.loads(source)]
        reports = {}
        for name in ('meteor', 'st5-scs', 'bertscore'):
            d = read(output / f'{label}-{name}.json')
            if d['source_sha256'] != hashlib.sha256(source).hexdigest() or [r['id'] for r in d['per_pair']] != ids:
                raise ValueError('Relatório não corresponde às predições')
            if d['test_evaluated'] != (split == 'test'):
                raise ValueError('Relatório com divisão incorreta')
            if name != 'meteor' and d['status'] != 'completed':
                raise ValueError('Métrica incompleta')
            reports[name] = d
        m = base['models'][label]
        values = [m['BLEU_1_to_4'][3], m['ROUGE_L'], reports['meteor']['METEOR_NLTK'],
                  reports['st5-scs']['means']['st5_scs'], reports['bertscore']['means']['f1'],
                  m['mean_repeated_trigram_fraction']]
        lines.append('| ' + label + ' | ' + ' | '.join(f'{v:.6f}' for v in values) + ' |')
        # Truncamento é uma limitação do avaliador, não uma falha na geração.
    lines += ['', 'METEOR usa NLTK; ST5 usa GPU FP16; BERTScore padrão usa RoBERTa-large camada 17 em FP32.',
              'Consulte os JSONs para truncamentos, versões, tempos e valores por par.',
              'Sem CIDEr-D ou taxa de acerto visual. Similaridade textual não comprova correção factual.',
              'A validação participa da seleção; o teste não deve orientar novos ajustes.']
    with (output / 'RESUMO.md').open('x') as f:
        f.write('\n'.join(lines) + '\n')
    print('\nResumo:', output / 'RESUMO.md')


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    c = read(local(a.config))
    output = local(a.output)
    if output.exists():
        p.error('A pasta de saída já existe. Escolha uma nova; nada será sobrescrito.')
    if hasattr(a, 'hours') and not 1/30 <= a.hours <= 168:
        p.error('--hours deve estar entre 2 minutos e 168 horas')
    if a.action == 'test' and not a.allow_test:
        p.error('Teste requer --allow-test; não use seus resultados para escolher o modelo')
    commands = []
    if a.action == 'metrics':
        evaluation = local(a.evaluation)
        split, labels, _ = evaluation_info(evaluation)
        if split == 'test' and not a.allow_test:
            p.error('Métricas do teste requerem --allow-test')
        metric_output = output
        use_metrics = True
    else:
        commands.append(generation_command(a, c))
        evaluation, metric_output = output, output / 'scores'
        split = 'test' if a.action == 'test' else 'val'
        labels = ['baseline', 'best'] if getattr(a, 'baseline', None) else ['best']
        use_metrics = a.action != 'train' and not a.basic_only and not a.check_only
    if use_metrics:
        commands += metric_commands(c, evaluation, metric_output, labels, split)
    if a.dry_run:
        for cmd in commands:
            print(shlex.join(cmd))
        return
    # Falhar antes da geração longa se faltar algum ambiente ou peso semântico.
    for cmd in commands:
        if not Path(cmd[0]).is_file():
            p.error('Ambiente não encontrado: ' + cmd[0])
    if use_metrics:
        for key in ('st5_model', 'bert_model'):
            if not (local(c[key]) / 'model.safetensors').is_file():
                p.error('Peso local ausente: ' + c[key])
    if a.action != 'metrics':
        run(commands.pop(0))
    if use_metrics:
        evaluation_info(evaluation)
        metric_output.mkdir(parents=True, exist_ok=False)
        with (metric_output / 'workflow.json').open('x') as f:
            json.dump({'configuration': c, 'evaluation': str(evaluation), 'commands': commands}, f, indent=2)
        for cmd in commands:
            run(cmd)
        summarize(evaluation, metric_output)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f'Erro: {exc}', file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print('\nInterrompido. Arquivos anteriores preservados; verifique saídas parciais.', file=sys.stderr)
        sys.exit(130)
