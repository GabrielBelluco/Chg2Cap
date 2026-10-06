"""Offline semantic metrics on saved captions, using local model weights.

BERTScore is the standard token-alignment metric, NOT RSCC's mean-pooled cosine.
ST5 uses mean(abs(cosine**3)), matching the inspected RSCC formula.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
from pathlib import Path
import time

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ.setdefault('HF_HOME', str(Path(sys.prefix) / 'hf-cache'))
os.environ.setdefault('MPLCONFIGDIR', str(Path(sys.prefix) / 'mpl-cache'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--model-path', type=Path, required=True)
    p.add_argument('--metric', choices=['bertscore', 'st5-scs'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                   help='cuda is also the PyTorch device name for AMD ROCm')
    p.add_argument('--dtype', choices=['float32','float16'], default='float32')
    p.add_argument('--max-seconds', type=int, default=3600)
    p.add_argument('--bert-layers', type=int, default=17)
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--allow-test', action='store_true')
    a = p.parse_args()
    if a.split == 'test' and not a.allow_test: p.error('Test requires --allow-test')
    if a.output.exists(): p.error('Output exists')
    if min(a.batch_size, a.threads, a.max_seconds, a.bert_layers) < 1: p.error('Positive limits required')
    if not a.model_path.is_dir(): p.error('Download weights into a local directory first')
    raw = a.predictions.read_bytes(); rows = json.loads(raw)
    if not rows or len({r['id'] for r in rows}) != len(rows): p.error('Empty or duplicate IDs')
    if any(r.get('split') != a.split for r in rows): p.error('Rows do not match requested split')
    if any(not isinstance(r.get(k), str) for r in rows for k in ['prediction','reference']): p.error('Missing text')
    if a.check_only:
        print(json.dumps({'status':'inputs_checked_not_model_load_tested','pairs':len(rows)})); return
    import torch
    torch.set_num_threads(a.threads)
    if a.device == 'cuda' and not torch.cuda.is_available():
        p.error('GPU unavailable in this environment')
    if a.dtype == 'float16' and (a.device != 'cuda' or a.metric != 'st5-scs'):
        p.error('float16 currently supported only for ST5 on GPU')
    if a.device == 'cuda': torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    if a.metric == 'bertscore':
        from bert_score import BERTScorer
        model = BERTScorer(model_type=str(a.model_path.resolve()), num_layers=a.bert_layers,
                           device=a.device, idf=False, rescale_with_baseline=False,
                           batch_size=a.batch_size, use_fast_tokenizer=False)
        tokenizer = model._tokenizer
        max_length = tokenizer.model_max_length
        # Check the installed model before starting the full BERTScore evaluation.
        _, _, smoke = model.score(['the road is flooded', 'a cat sleeps on a sofa'],
                                   ['the road is flooded', 'the road is flooded'])
        if not torch.isfinite(smoke).all() or not (float(smoke[0]) > .99 and float(smoke[0]) > float(smoke[1])):
            raise ValueError('BERTScore identity/difference self-test failed')
        print('BERTScore self-test passed', flush=True)
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(str(a.model_path.resolve()), device=a.device, local_files_only=True,
                                    model_kwargs={'torch_dtype':getattr(torch,a.dtype)})
        tokenizer = model.tokenizer; max_length = model.max_seq_length
    report = {'metric':a.metric, 'parameters':vars(a).copy(),
              'source_sha256':hashlib.sha256(raw).hexdigest(), 'test_evaluated':a.split == 'test',
              'versions':{n:importlib.metadata.version(n) for n in ['torch','transformers','sentence-transformers','bert-score']},
              'max_token_length':max_length, 'truncated_texts':0, 'per_pair':[]}
    report['parameters'] = {k:str(v) if isinstance(v,Path) else v for k,v in report['parameters'].items()}
    for start in range(0,len(rows),a.batch_size):
        if time.monotonic()-started >= a.max_seconds: break
        batch = rows[start:start+a.batch_size]
        c=[r['prediction'] for r in batch]; refs=[r['reference'] for r in batch]
        report['truncated_texts'] += sum(len(tokenizer.encode(t, truncation=False)) > max_length for t in c+refs)
        if a.metric=='bertscore':
            precision, recall, f1=model.score(c,refs)
            values=[{'precision':float(x),'recall':float(y),'f1':float(z)} for x,y,z in zip(precision,recall,f1)]
            if not all(math.isfinite(v) for row in values for v in row.values()):
                raise ValueError('Nonfinite BERTScore; stopping evaluation')
        else:
            x=model.encode(c,normalize_embeddings=True,batch_size=a.batch_size)
            y=model.encode(refs,normalize_embeddings=True,batch_size=a.batch_size)
            import numpy as np
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError('Nonfinite embeddings; do not use this precision configuration')
            values=[{'st5_scs':float(abs(v)**3)} for v in (x.astype('float32')*y.astype('float32')).sum(axis=1).clip(-1,1)]
        report['per_pair'].extend(dict(id=r['id'],**v) for r,v in zip(batch,values))
        print(json.dumps({'done':len(report['per_pair']),'total':len(rows),'seconds':round(time.monotonic()-started,1)}),flush=True)
    report['status']='completed' if len(report['per_pair'])==len(rows) else 'incomplete_time_limit'
    report['wall_seconds']=time.monotonic()-started
    if a.device == 'cuda':
        report['gpu_name'] = torch.cuda.get_device_name(0)
        report['peak_gpu_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['peak_gpu_reserved_bytes'] = torch.cuda.max_memory_reserved()
    if report['status']=='completed':
        report['means']={k:sum(r[k] for r in report['per_pair'])/len(rows) for k in values[0]}
    with a.output.open('x') as f: json.dump(report,f,indent=2)
    print(report['status'])
    if report['status'] != 'completed':
        raise SystemExit(2)


if __name__=='__main__': main()
