"""Preparação e execução manual Qwen × Chg2Cap. Ajuda/dry-run: somente stdlib.

Veja README.md. Nada instala ou baixa implicitamente: setup e download são
etapas explícitas; todas as gerações são offline e o teste exige freeze.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
LABEL = 'qwen'
PINNED = {'torch': '2.4.0+rocm6.1', 'torchvision': '0.19.0+rocm6.1',
          'transformers': '4.51.3', 'accelerate': '1.6.0',
          'huggingface-hub': '0.30.2', 'tokenizers': '0.21.1',
          'safetensors': '0.5.3', 'numpy': '1.26.4', 'Pillow': '12.0.0'}


def local(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def new_path(path):
    if path.exists() or path.is_symlink():
        raise ValueError(f'Saída já existe; escolha outro nome: {path}')


def configuration(path):
    c = read(local(path))
    if c['model_id'] != 'Qwen/Qwen2.5-VL-3B-Instruct' or not re.fullmatch(r'[0-9a-f]{40}', c['revision']):
        raise ValueError('Esta implementação exige Qwen2.5-VL-3B e revisão de 40 caracteres')
    if c['dtype'] not in ('float16','float32') or c['attention'] not in ('eager', 'sdpa'):
        raise ValueError('Precisão suportada: float16/float32; atenção: eager/sdpa')
    if c['batch_size'] != 1 or c['do_sample'] is not False or c['num_beams'] != 1:
        raise ValueError('Protocolo implementado: lote 1, geração greedy sem amostragem')
    if c['image_size'] not in (256, 512) or not 4*28*28 <= c['min_pixels'] <= c['max_pixels'] <= 512*512:
        raise ValueError('Resolução/pixels inválidos')
    if not isinstance(c['max_new_tokens'], int) or not 8 <= c['max_new_tokens'] <= 1024:
        raise ValueError('max_new_tokens deve estar entre 8 e 1024')
    if not isinstance(c['prompt'], str) or not c['prompt'].strip() or any(x in c['prompt'] for x in ('{', '}')):
        raise ValueError('Prompt literal não vazio; não admite interpolação de metadados')
    if not 1 <= c['threads'] <= 16 or not isinstance(c['seed'], int) or c['repetition_penalty'] <= 0:
        raise ValueError('Threads/seed/penalidade inválidos')
    return c


def manifest(c):
    path = local(c['prepared']) / 'manifest.jsonl'
    if digest(path) != c['manifest_sha256']:
        raise ValueError('Manifesto diferente do experimento Chg2Cap')
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    from data.RSCC.rscc import validate_splits
    validate_splits(rows)
    if dict(Counter(r['split'] for r in rows)) != c['expected_counts']:
        raise ValueError('Contagens das divisões diferentes do esperado')
    for row in rows:
        if row['source'] != 'EBD' or not row['caption'].strip():
            raise ValueError('Apenas EBD com referência original é aceito')
        for field in ('before', 'after'):
            p = Path(row[field])
            if p.is_absolute() or '..' in p.parts or not re.fullmatch(r'[0-9a-f]{64}', row[field+'_sha256']):
                raise ValueError('Caminho/hash inválido no manifesto')
    return rows


def select(rows, action, limit=None):
    split = 'test' if action == 'test' else 'val'
    selected = sorted((r for r in rows if r['split'] == split), key=lambda r: r['id'])
    if action == 'smoke':
        return selected[:2]
    if limit is not None:
        if action != 'validation' or not 1 <= limit <= len(selected):
            raise ValueError('Limite de pares permitido apenas na validação')
        return selected[:limit]
    return selected


def compare_inputs(rows, saved, exact=True):
    old = {r['id']: r for r in saved}
    if len(old) != len(saved) or (exact and set(old) != {r['id'] for r in rows}):
        raise ValueError('IDs não correspondem ao conjunto comparado')
    for row in rows:
        b = old.get(row['id'])
        if b is None or any(b[k] != row[k] for k in ('split', 'before', 'after')) or b['reference'] != row['caption']:
            raise ValueError('Par/referência diferente do Chg2Cap: ' + row['id'])


def repetition(tokens):
    # Exact convention from scripts/diagnose_rscc_repetition.py; no Torch import.
    words = [t for t in tokens if any(c.isalnum() for c in t)]
    adjacent = sum(a == b for a, b in zip(words, words[1:]))
    trigrams = [tuple(words[i:i+3]) for i in range(max(0, len(words)-2))]
    return {'words': len(words), 'adjacent_repeat_fraction': adjacent/max(1, len(words)-1),
            'repeated_trigram_fraction': 1-len(set(trigrams))/len(trigrams) if trigrams else 0.}


def normalize_prediction(text):
    from data.RSCC.rscc import tokenize
    # Raw prose is retained separately; lexical convention matches Chg2Cap.
    return ' '.join(tokenize(text)[1:-1])


def messages(prompt):
    # Only image placeholders and fixed prompt. PIL pixels go to processor separately.
    return [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'image'},
                                        {'type': 'text', 'text': prompt}]}]


def source_hashes(c):
    files = [str(p.relative_to(ROOT)) for p in sorted((ROOT/'captioning').glob('*.py'))]
    files += [c['requirements'], c['constraints'], c['metrics_config'],
              'data/RSCC/rscc.py', 'eval_func/bleu/bleu.py', 'eval_func/bleu/bleu_scorer.py',
              'eval_func/rouge/rouge.py']
    return {str(p): digest(local(p)) for p in files}


def versions():
    result = {name: importlib.metadata.version(name) for name in PINNED}
    for name, expected in PINNED.items():
        if result[name] != expected:
            raise ValueError(f'Versão inesperada: {name}={result[name]}, esperado {expected}')
    result['python'] = '.'.join(map(str, sys.version_info[:3]))
    names={d.metadata['Name'] for d in importlib.metadata.distributions() if d.metadata['Name']}
    result['all_packages']={name:importlib.metadata.version(name) for name in sorted(names,key=str.lower)}
    if sys.version_info[:2] != (3, 12):
        raise ValueError('Ambiente deve usar Python 3.12')
    return result


def model_inventory(c, verify=True):
    directory = local(c['model_dir'])
    inventory = read(directory / 'download.json')
    if inventory.get('status') != 'completed' or inventory['model_id'] != c['model_id'] or inventory['revision'] != c['revision']:
        raise ValueError('Pesos locais não correspondem ao modelo/revisão')
    if not inventory.get('files'):
        raise ValueError('Inventário de pesos vazio')
    index = read(directory / 'model.safetensors.index.json')
    required = set(index['weight_map'].values()) | {'config.json', 'tokenizer.json',
                'tokenizer_config.json', 'preprocessor_config.json', 'model.safetensors.index.json'}
    if not required <= set(inventory['files']):
        raise ValueError('Inventário não cobre todos os shards/arquivos obrigatórios')
    for name, item in inventory['files'].items():
        p = directory / name
        if Path(name).name != name or not p.is_file() or p.stat().st_size != item['bytes']:
            raise ValueError('Arquivo local inválido: ' + name)
        if verify and digest(p) != item['sha256']:
            raise ValueError('Peso/arquivo local modificado: ' + name)
    return inventory


def validate_evaluation(c, directory, require_full=False):
    status = read(directory / 'status.json')
    protocol = read(directory / 'protocol.json')
    rows = read(directory / 'qwen-predictions.json')
    if status['status'] != 'completed' or status['completed_pairs'] != len(rows):
        raise ValueError('Geração incompleta')
    if protocol['configuration'] != c or protocol['code_sha256'] != source_hashes(c):
        raise ValueError('Configuração/código mudou desde a geração')
    if digest(directory / 'qwen-predictions.json') != status['predictions_sha256']:
        raise ValueError('Predições foram modificadas')
    if not rows or len({r['id'] for r in rows}) != len(rows) or any(not r['raw_prediction'].strip() for r in rows):
        raise ValueError('IDs duplicados ou legenda vazia')
    split = protocol['split']
    expected = sorted((r for r in manifest(c) if r['split'] == split), key=lambda r:r['id'])
    matched=expected if require_full else [r for r in expected if r['id'] in {p['id'] for p in rows}]
    compare_inputs(matched, rows)
    if [r['id'] for r in matched] != [r['id'] for r in rows]:
        raise ValueError('Ordem das predições diferente dos IDs selecionados')
    if require_full and len(rows) != c['expected_counts'][split]:
        raise ValueError('Congelamento exige validação completa')
    if any(r['split'] != split or r['model_id'] != c['model_id'] or r['model_revision'] != c['revision'] for r in rows):
        raise ValueError('Identificação/split das predições inválidos')
    return protocol, rows


def frozen(c, path):
    f = read(path)
    stored = f.pop('freeze_sha256', None)
    if stored != object_digest(f) or f['status'] != 'frozen_before_test':
        raise ValueError('Protocolo congelado inválido/modificado')
    if f['configuration'] != c or f['code_sha256'] != source_hashes(c):
        raise ValueError('Configuração/código difere do protocolo congelado')
    # Bind to original validation evidence, not just a boolean flag.
    validation = local(f['validation'])
    protocol, rows = validate_evaluation(c, validation, require_full=True)
    if protocol['split'] != 'val' or any(r['output_cap_reached'] for r in rows):
        raise ValueError('Validação congelada inválida ou truncada')
    if digest(validation/'qwen-predictions.json') != f['validation_predictions_sha256']:
        raise ValueError('Evidência da validação mudou')
    f['freeze_sha256'] = stored
    return f


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/captioning_qwen.json')
    s = p.add_subparsers(dest='action', required=True)
    for name in ('setup', 'download', 'check', 'smoke', 'validation', 'freeze', 'test', 'metrics', 'compare'):
        a = s.add_parser(name)
        a.add_argument('--dry-run', action='store_true', help='Somente plano; sem import de modelos/rede/arquivos novos')
        if name in ('setup', 'download'):
            a.add_argument('--execute', action='store_true', help='Autorizar explicitamente instalação/download no terminal')
        if name == 'check':
            a.add_argument('--verify-images', action='store_true', help='Hash de todas as imagens de validação; pode demorar')
        if name in ('smoke', 'validation', 'test'):
            a.add_argument('--output', required=True, help='Pasta nova')
            a.add_argument('--seconds', type=int, default=600 if name=='smoke' else 7200,
                           help='Timeout externo rígido; SIGTERM, depois SIGKILL em até 5s')
            if name == 'validation': a.add_argument('--limit', type=int, help='Piloto; omitir usa todos os 1824 pares')
        if name in ('test',):
            a.add_argument('--allow-test', action='store_true')
            a.add_argument('--protocol', required=True)
        if name == 'freeze':
            a.add_argument('--validation', required=True)
            a.add_argument('--output', required=True, help='Arquivo JSON novo de protocolo congelado')
        if name in ('metrics', 'compare'):
            a.add_argument('--evaluation', required=True)
            a.add_argument('--output', required=True, help='Pasta nova')
            a.add_argument('--allow-test', action='store_true')
            if name == 'metrics':
                a.add_argument('--seconds', type=int, default=7200, help='Timeout externo por avaliador')
            else:
                a.add_argument('--scores', required=True, help='Pasta das métricas Qwen concluídas')
    return p


def offline_env():
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
               PYTHONDONTWRITEBYTECODE='1')
    return env


def bounded(command, seconds, env=None):
    """Kill the process group on timeout/Ctrl+C; preserve partial evidence."""
    import signal
    process = subprocess.Popen(command, cwd=ROOT, env=env or offline_env(), start_new_session=True)
    try:
        returncode = process.wait(timeout=seconds)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        os.killpg(process.pid, signal.SIGTERM)
        try: process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def worker_command(a, c):
    python = local(c['environment']) / 'bin/python'
    cmd = [str(python), '-B', '-u', str(ROOT/'captioning/qwen_worker.py'), '--config', str(local(a.config)),
           a.action]
    for name in ('output', 'seconds', 'limit', 'protocol'):
        value = getattr(a, name, None)
        if value is not None:
            cmd += ['--'+name.replace('_','-'), str(local(value)) if name in ('output','protocol') else str(value)]
    if getattr(a, 'allow_test', False): cmd += ['--allow-test']
    return cmd


def plan(a, c):
    if a.action in ('setup', 'download'):
        return {'action':a.action, 'execute_required':True, 'environment':str(local(c['environment'])),
                'model_id':c['model_id'], 'revision':c['revision'], 'model_dir':str(local(c['model_dir'])),
                'requirements':read_text_lines(local(c['requirements'])), 'no_execution':True}
    if a.action=='metrics':
        from captioning import cli as rscc
        split=read(local(a.evaluation)/'protocol.json')['split']
        if split=='test' and not a.allow_test: raise ValueError('Métricas de teste exigem --allow-test')
        cmds=rscc.metric_commands(read(local(c['metrics_config'])),local(a.evaluation),local(a.output),['qwen'],split)
        return {'action':'metrics','commands':[shlex.join(cmd) for cmd in cmds],
                'basic_metrics':'Mesmo BLEU/ROUGE/repetição do Chg2Cap; sem carregar modelos',
                'timeout_per_command_seconds':a.seconds,'no_execution':True}
    if a.action in ('smoke','validation','test'):
        return {'command':shlex.join(worker_command(a,c)), 'timeout_seconds':a.seconds,
                'split':'test' if a.action=='test' else 'val',
                'pairs':2 if a.action=='smoke' else (getattr(a,'limit',None) or c['expected_counts']['test' if a.action=='test' else 'val']),
                'configuration':c, 'no_execution':True}
    return {'action':a.action,'configuration':c,'arguments':vars(a),'no_execution':True}


def read_text_lines(path):
    return [l for l in path.read_text().splitlines() if l and not l.startswith('#')]


def main(argv=None):
    a = parser().parse_args(argv)
    c = configuration(a.config)
    if getattr(a,'seconds',60) < 1 or (a.action=='smoke' and a.seconds>600):
        raise ValueError('Orçamento inválido; smoke limitado a 600 segundos')
    if getattr(a,'limit',None) is not None and not 1 <= a.limit <= c['expected_counts']['val']:
        raise ValueError('Limite inválido')
    if a.action=='test' and not a.allow_test:
        raise ValueError('Teste exige --allow-test e protocolo congelado')
    if hasattr(a,'output'): new_path(local(a.output))
    if a.dry_run:
        # Test still checks its immutable gate; no model or validation weights loaded.
        if a.action=='test': frozen(c,local(a.protocol))
        print(json.dumps(plan(a,c),indent=2,ensure_ascii=False)); return
    if a.action in ('setup','download') and not a.execute:
        raise ValueError('Use --dry-run para planejar ou --execute para instalar/baixar explicitamente')
    if a.action=='setup': setup(c,a.config)
    elif a.action=='download':
        bounded(worker_command(a,c), 3600, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    elif a.action=='check': check(c,a.verify_images)
    elif a.action=='freeze': freeze(c,a)
    elif a.action in ('smoke','validation','test'):
        if a.action=='test': frozen(c,local(a.protocol))
        command=worker_command(a,c)
        print(shlex.join(command),flush=True)
        try: bounded(command,a.seconds)
        except BaseException as exc:
            out=local(a.output)
            if out.is_dir() and not (out/'status.json').exists():
                write_new(out/'status.json',{'status':'incomplete','reason':type(exc).__name__,
                    'detail':str(exc),'partial_predictions':'qwen-predictions.jsonl','finished_at':now()})
            raise
    elif a.action=='metrics': run_metrics(c,a)
    elif a.action=='compare': compare(c,a)


def setup(c,config_path):
    source=local(c['source_environment']).resolve()
    dest=local(c['environment']).resolve()
    new_path(dest)
    if dest.is_relative_to(source) or source.is_relative_to(dest):
        raise ValueError('Ambiente novo não pode se sobrepor ao ambiente de origem')
    sites=list((source/'lib').glob('python3.12/site-packages'))
    if len(sites)!=1: raise ValueError('Origem deve conter site-packages Python 3.12')
    source_python=source/'bin/python'
    # Metadata only, no GPU access. Refuse unexpected ROCm before mutation.
    source_versions={d.metadata['Name'].lower():d.version for d in importlib.metadata.distributions(path=[str(sites[0])])}
    if source_versions.get('torch')!=PINNED['torch'] or source_versions.get('torchvision')!=PINNED['torchvision']:
        raise ValueError('Torch/torchvision da origem não corresponde ao ROCm conferido')
    req=local(c['requirements']); con=local(c['constraints'])
    bounded([str(source_python),'-B','-m','venv',str(dest)],60)
    site=dest/'lib/python3.12/site-packages'
    with (site/'rocm_source.pth').open('x') as f: f.write(str(sites[0])+'\n')
    write_new(dest/'setup-plan.json',{'configuration':c,'source_environment':str(source),
        'source_versions':source_versions,'requirements_sha256':digest(req),'constraints_sha256':digest(con),
        'created_at':now(),'status':'installation_started','sharing':'read-only imports via .pth'})
    python=str(dest/'bin/python')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PIP_REQUIRE_VIRTUALENV='true',PIP_DISABLE_PIP_VERSION_CHECK='1')
    bounded([python,'-B','-m','pip','install','--only-binary=:all:', '--index-url','https://pypi.org/simple',
             '-r',str(req),'-c',str(con),'--report',str(dest/'pip-report.json')],1800,env)
    bounded([python,'-B','-m','pip','check'],60,env)
    bounded([python,'-B',str(ROOT/'captioning/qwen_worker.py'),'--config',str(local(config_path)),'environment'],60)


def check(c, all_images=False):
    rows=manifest(c)
    chosen=select(rows,'validation') if all_images else select(rows,'smoke')
    from data.RSCC.rscc import resolve_image
    for row in chosen:
        for field,phase in [('before','pre'),('after','post')]:
            if digest(resolve_image(local(c['image_root']),row[field],phase))!=row[field+'_sha256']:
                raise ValueError('Imagem difere do manifesto: '+row['id'])
    print(json.dumps({'status':'inputs_checked_no_model_load','counts':c['expected_counts'],
        'image_pairs_hashed':len(chosen),'environment_exists':(local(c['environment'])/'bin/python').is_file(),
        'weights_inventory_exists':(local(c['model_dir'])/'download.json').is_file(),
        'gpu_compatibility':'not_tested'},indent=2))


def freeze(c,a):
    directory=local(a.validation)
    protocol,rows=validate_evaluation(c,directory,require_full=True)
    if protocol['split']!='val' or protocol['action']!='validation' or any(r['output_cap_reached'] for r in rows):
        raise ValueError('Freeze requer validação completa, sem saídas truncadas; ajuste somente na validação')
    inventory=model_inventory(c)
    if protocol['model_inventory_sha256']!=object_digest(inventory):
        raise ValueError('Pesos diferentes dos usados na validação')
    value={'status':'frozen_before_test','frozen_at':now(),'configuration':c,
        'code_sha256':source_hashes(c),'versions':protocol['versions'],
        'model_inventory_sha256':object_digest(inventory),'validation':str(directory.resolve()),
        'validation_predictions_sha256':digest(directory/'qwen-predictions.json'),
        'test_policy':'All fixed EBD test IDs; do not tune from test; no references/metadata in model input.'}
    value['freeze_sha256']=object_digest(value)
    target=local(a.output);target.parent.mkdir(parents=True,exist_ok=True);write_new(target,value)
    print('Protocolo congelado:',target)


def basic_metrics(rows):
    # Same evaluator classes and exact asymmetric normalization as historical Chg2Cap.
    from data.RSCC.rscc import tokenize
    from eval_func.bleu.bleu import Bleu
    from eval_func.rouge.rouge import Rouge
    refs=[[' '.join(tokenize(r['reference'])[1:-1])] for r in rows]
    hyps=[[r['prediction']] for r in rows]
    bleu,by_bleu=Bleu(4).compute_score(refs,hyps)
    rouge,by_rouge=Rouge().compute_score(refs,hyps)
    score={'pairs':len(rows),'BLEU_1_to_4':bleu,'ROUGE_L':float(rouge),
        'mean_adjacent_repeat_fraction':sum(r['repetition']['adjacent_repeat_fraction'] for r in rows)/len(rows),
        'mean_repeated_trigram_fraction':sum(r['repetition']['repeated_trigram_fraction'] for r in rows)/len(rows),
        'mean_output_whitespace_tokens':sum(len(r['prediction'].split()) for r in rows)/len(rows),
        'output_cap_reached':sum(r['output_cap_reached'] for r in rows),
        'distinct_caption_fraction':len({r['prediction'] for r in rows})/len(rows)}
    pairs=[{'id':r['id'],'BLEU_1_to_4':[float(s[i]) for s in by_bleu],
            'ROUGE_L':float(by_rouge[i]),'repetition':r['repetition']} for i,r in enumerate(rows)]
    return score,pairs


def run_metrics(c,a):
    directory=local(a.evaluation);out=local(a.output)
    protocol,rows=validate_evaluation(c,directory)
    split=protocol['split']
    if split=='test' and not a.allow_test: raise ValueError('Métricas de teste exigem --allow-test')
    from captioning import cli as rscc
    metric_config=read(local(c['metrics_config']))
    commands=rscc.metric_commands(metric_config,directory,out,['qwen'],split)
    for command in commands:
        if not Path(command[0]).is_file(): raise ValueError('Ambiente de métrica ausente: '+command[0])
    for key in ('st5_model','bert_model'):
        if not (local(metric_config[key])/'model.safetensors').is_file(): raise ValueError('Peso semântico local ausente')
    out.mkdir(parents=True,exist_ok=False)
    write_new(out/'workflow.json',{'configuration':metric_config,'commands':commands,
        'predictions_sha256':digest(directory/'qwen-predictions.json'),'evaluator_code_sha256':source_hashes(c)})
    try:
        # Reuse existing ROCm Python only for numpy-backed historical scorers; no model/GPU.
        cmd=[str(local(metric_config['training_python'])),'-B',str(ROOT/'captioning/qwen_worker.py'),
             '--config',str(local(a.config)),'basic','--evaluation',str(directory),'--output',str(out)]
        bounded(cmd,a.seconds)
        for command in commands:
            print(shlex.join(command),flush=True);bounded(command,a.seconds)
        rscc.summarize(out,out)
        write_new(out/'status.json',{'status':'completed','pairs':len(rows),'finished_at':now(),
            'reports_sha256':{name:digest(out/name) for name in ['metrics.json','qwen-meteor.json','qwen-st5-scs.json','qwen-bertscore.json']}})
    except BaseException as exc:
        write_new(out/'status.json',{'status':'incomplete','reason':type(exc).__name__,'detail':str(exc)})
        raise


def verified_scores(directory, rows, raw_hash, label='qwen'):
    ids=[r['id'] for r in rows]
    values={}
    for metric in ('meteor','st5-scs','bertscore'):
        report=read(directory/f'{label}-{metric}.json')
        if report['source_sha256']!=raw_hash or [r['id'] for r in report['per_pair']]!=ids:
            raise ValueError('Score não corresponde às predições: '+metric)
        if report.get('status','completed')!='completed' or report['test_evaluated']!=(rows[0]['split']=='test'):
            raise ValueError('Score incompleto ou split incorreto')
        values[metric]=report
    return values


def compare(c,a):
    directory=local(a.evaluation);out=local(a.output);scores=local(a.scores)
    protocol,rows=validate_evaluation(c,directory,require_full=True)
    split=protocol['split']
    if split=='test' and not a.allow_test: raise ValueError('Comparação de teste exige --allow-test')
    score_status=read(scores/'status.json')
    if score_status['status']!='completed' or score_status['pairs']!=len(rows):
        raise ValueError('Métricas Qwen incompletas')
    for name,hashed in score_status['reports_sha256'].items():
        if digest(scores/name)!=hashed: raise ValueError('Relatório de métricas foi modificado: '+name)
    qreports=verified_scores(scores,rows,digest(directory/'qwen-predictions.json'))
    base_path=local(c['chg2cap_test' if split=='test' else 'chg2cap_validation'])
    old=read(base_path);old_by_id={r['id']:r for r in old}
    compare_inputs([dict(r,caption=r['reference']) for r in rows],old)
    baseline_scores = local(c['chg2cap_test_scores' if split=='test' else 'chg2cap_validation_scores'])
    breports = verified_scores(baseline_scores, old, digest(base_path), label='best')
    base=read(base_path.parent/'metrics.json')['models']['best']
    qbasic=read(scores/'metrics.json')['models']['qwen']
    if read(scores/'status.json')['status']!='completed': raise ValueError('Métricas Qwen incompletas')
    # Basic artifact is hash bound; refuse stale or edited metrics.
    if read(scores/'workflow.json')['predictions_sha256']!=digest(directory/'qwen-predictions.json'):
        raise ValueError('Métricas básicas de outra geração')
    def summary(b,reports):
        return {'BLEU_4':b['BLEU_1_to_4'][3],'ROUGE_L':b['ROUGE_L'],
            'METEOR_NLTK':reports['meteor']['METEOR_NLTK'],
            'ST5_SCS':reports['st5-scs']['means']['st5_scs'],
            'BERTScore_F1':reports['bertscore']['means']['f1'],
            'repeated_trigrams':b['mean_repeated_trigram_fraction']}
    summaries={'Chg2Cap':summary(base,breports),'Qwen2.5_VL_3B':summary(qbasic,qreports)}
    pair_maps={model:{m:{r['id']:r for r in rep['per_pair']} for m,rep in reports.items()}
               for model,reports in [('qwen',qreports),('chg2cap',breports)]}
    basic_maps={'qwen':{r['id']:r for r in read(scores/'metrics.json')['per_pair']},
        'chg2cap':{r['id']:r for r in basic_metrics(old)[1]}}
    if any(set(mapping)!={r['id'] for r in rows} for mapping in basic_maps.values()):
        raise ValueError('Métricas básicas não correspondem aos pares')
    pairs=[]
    for row in rows:
        value={'id':row['id'],'split':split,'event':row['event'],'reference':row['reference'],
               'qwen_prediction':row['prediction'],'chg2cap_prediction':old_by_id[row['id']]['prediction']}
        for model in pair_maps:
            value[model]={m:pair_maps[model][m][row['id']] for m in pair_maps[model]}
            value[model]['basic']=basic_maps[model][row['id']]
        value['delta_qwen_minus_chg2cap']={
            'ROUGE_L':value['qwen']['basic']['ROUGE_L']-value['chg2cap']['basic']['ROUGE_L'],
            'METEOR_NLTK':value['qwen']['meteor']['meteor']-value['chg2cap']['meteor']['meteor'],
            'ST5_SCS':value['qwen']['st5-scs']['st5_scs']-value['chg2cap']['st5-scs']['st5_scs'],
            'BERTScore_F1':value['qwen']['bertscore']['f1']-value['chg2cap']['bertscore']['f1']}
        pairs.append(value)
    out.mkdir(parents=True,exist_ok=False)
    write_new(out/'comparison.json',{'status':'completed','split':split,'pairs':len(rows),
        'qwen_predictions_sha256':digest(directory/'qwen-predictions.json'),
        'chg2cap_predictions_sha256':digest(base_path),'models':summaries,'per_pair':pairs})
    lines=['# Chg2Cap × Qwen — '+split,'',f'{len(rows)} pares; scores em escala 0–1.',
        '', '| Modelo | BLEU-4 | ROUGE-L | METEOR-NLTK | ST5-SCS | BERTScore F1 | Trigramas repetidos |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for model,values in summaries.items(): lines.append('| '+model+' | '+' | '.join(f'{v:.6f}' for v in values.values())+' |')
    lines+=['','Qwen zero-shot e Chg2Cap treinado no EBD possuem pré-treinamento e custos diferentes.',
        'Apenas imagens entram no modelo. Referências sintéticas podem conter informação de metadados.',
        'Resolução de entrada registrada no protocolo; pixels efetivos Qwen constam das predições.',
        'ST5 FP16, BERTScore RoBERTa-large camada 17 FP32, METEOR NLTK; não reproduzem integralmente o artigo.',
        'Consulte JSONs para tempos, memória, falhas e truncamentos dos avaliadores.',
        'Similaridade textual não é acerto visual. Não usar o teste para novos ajustes.']
    (out/'RESUMO.md').write_text('\n'.join(lines)+'\n')
    gallery(out/'comparison.html',c,pairs)
    print('Comparação:',out/'RESUMO.md')


def gallery(path,c,rows):
    import html
    from urllib.parse import quote
    from data.RSCC.rscc import resolve_image
    selected={r['id']:r for r in manifest(c)}
    esc=html.escape
    lines=['<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>Chg2Cap × Qwen</title>',
        '<style>body{font:17px sans-serif;max-width:1100px;margin:auto;padding:20px}article{border-top:1px solid #aaa;padding:20px 0}.images{display:flex;gap:15px}img{width:48%}</style>',
        '<h1>Chg2Cap × Qwen</h1><p>Revisão visual: mudanças visíveis, omissões, inversão temporal, repetição e afirmações sem suporte. Referências também podem conter erros. Sem taxa automática de acerto.</p>']
    for row in rows:
        record=selected[row['id']]; lines+=['<article><h2>'+esc(row['id'])+'</h2><div class="images">']
        for field,phase in [('before','pre'),('after','post')]:
            src=quote(os.path.relpath(resolve_image(local(c['image_root']),record[field],phase),path.parent),safe='/')
            lines.append(f'<img loading="lazy" src="{esc(src)}" alt="{phase}">')
        lines.append('</div>')
        for name in ('reference','chg2cap_prediction','qwen_prediction'):
            lines.append('<h3>'+esc(name)+'</h3><p>'+esc(row[name])+'</p>')
        lines.append('</article>')
    path.write_text('\n'.join(lines)+'</html>\n')


if __name__=='__main__':
    try: main()
    except (ValueError,FileNotFoundError,KeyError,subprocess.SubprocessError) as exc:
        print('Erro: '+str(exc),file=sys.stderr);sys.exit(1)
    except KeyboardInterrupt:
        print('Interrompido; saídas parciais preservadas.',file=sys.stderr);sys.exit(130)
