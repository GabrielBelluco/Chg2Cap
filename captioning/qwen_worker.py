"""Worker manual Qwen: imports pesados somente após selecionar operação.

Nunca usado por --dry-run. Geração offline, safetensors, sem código remoto.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from captioning import qwen as q


def environment(c):
    installed=q.versions()
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    import torchvision
    if torch.version.hip is None or not Path(torch.__file__).resolve().is_relative_to(q.local(c['source_environment']).resolve()):
        raise ValueError('Torch não vem do ambiente ROCm de origem')
    value={'status':'imports_checked_gpu_not_tested','versions':installed,'torch_path':torch.__file__,
           'hip_version':torch.version.hip,'checked_at':q.now()}
    q.write_new(q.local(c['environment'])/'environment.json',value)
    print(json.dumps(value,indent=2))


def download(c):
    from huggingface_hub import HfApi, snapshot_download
    dest=q.local(c['model_dir']);q.new_path(dest)
    info=HfApi().model_info(c['model_id'],revision=c['revision'],files_metadata=True)
    if info.sha!=c['revision']:raise ValueError('Revisão retornada difere da fixada')
    siblings=[s for s in info.siblings if s.rfilename.endswith(('.json','.txt','.safetensors')) or s.rfilename=='LICENSE']
    expected_bytes=sum(s.size or 0 for s in siblings)
    import shutil
    if shutil.disk_usage(dest.parent if dest.parent.exists() else q.ROOT).free < expected_bytes+3*1024**3:
        raise ValueError('Espaço insuficiente para pesos e margem de 3 GiB')
    dest.mkdir(parents=True,exist_ok=False)
    q.write_new(dest/'download-plan.json',{'status':'started','model_id':c['model_id'],
        'revision':c['revision'],'expected_download_bytes':expected_bytes,'started_at':q.now()})
    print(f'Download explícito: {expected_bytes} bytes, revisão {info.sha}',flush=True)
    snapshot_download(repo_id=c['model_id'],revision=c['revision'],local_dir=str(dest),
        allow_patterns=[s.rfilename for s in siblings],max_workers=2)
    files={}
    for s in siblings:
        path=dest/s.rfilename
        if not path.is_file() or (s.size is not None and path.stat().st_size!=s.size):
            raise ValueError('Arquivo baixado ausente/incompleto: '+s.rfilename)
        hashed=q.digest(path)
        # Hub LFS hash is the SHA256 of contents; check where available.
        lfs=getattr(s,'lfs',None)
        expected=getattr(lfs,'sha256',None) if lfs else None
        if isinstance(lfs,dict): expected=lfs.get('sha256')
        if expected and expected!=hashed:raise ValueError('Checksum remoto divergente: '+s.rfilename)
        files[s.rfilename]={'bytes':path.stat().st_size,'sha256':hashed,'hub_sha256':expected}
    q.write_new(dest/'download.json',{'status':'completed','model_id':c['model_id'],
        'revision':info.sha,'files':files,'download_bytes':sum(x['bytes'] for x in files.values()),
        'finished_at':q.now()})
    q.model_inventory(c,verify=False)
    print('Pesos concluídos e inventariados. Nenhuma inferência realizada.')


def basic(c,a):
    directory=q.local(a.evaluation);out=q.local(a.output)
    protocol,rows=q.validate_evaluation(c,directory)
    score,pairs=q.basic_metrics(rows)
    q.write_new(out/'metrics.json',{'status':'completed','pairs':len(rows),'models':{'qwen':score},
        'source_sha256':q.digest(directory/'qwen-predictions.json'),'per_pair':pairs})
    q.write_new(out/'protocol.json',{'split':protocol['split'],'model_id':c['model_id']})
    # Read-only bridge to existing rscc.summarize; label always qwen, never best.
    (out/'qwen-predictions.json').symlink_to((directory/'qwen-predictions.json').resolve())


def infer(c,a):
    out=q.local(a.output);q.new_path(out)
    manifest=q.manifest(c);selected=q.select(manifest,a.action,getattr(a,'limit',None))
    split='test' if a.action=='test' else 'val'
    frozen=q.frozen(c,q.local(a.protocol)) if a.action=='test' else None
    out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();predictions=[];failure=None
    try:
        installed=q.versions()
        inventory=q.model_inventory(c)
        if frozen and (installed!=frozen['versions'] or q.object_digest(inventory)!=frozen['model_inventory_sha256']):
            raise ValueError('Ambiente/pesos diferem do protocolo congelado')
        import torch
        from PIL import Image
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, GenerationConfig
        if not torch.cuda.is_available() or torch.version.hip is None:
            raise ValueError('GPU ROCm indisponível; execute no terminal normal')
        torch.set_num_threads(c['threads']);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.cuda.reset_peak_memory_stats()
        from data.RSCC.rscc import resolve_image
        protocol={'status':'generation_started','action':a.action,'split':split,'pairs':len(selected),
            'configuration':c,'versions':installed,'code_sha256':q.source_hashes(c),
            'model_inventory_sha256':q.object_digest(inventory),'started_at':q.now(),
            'gpu_name':torch.cuda.get_device_name(0),'hip_version':torch.version.hip,
            'seconds_limit':a.seconds,'generation_input':'PIL pixels before/after and literal prompt only',
            'input_resampling':'PIL RGB resize BICUBIC, then Qwen smart_resize',
            'frozen_protocol_sha256':frozen['freeze_sha256'] if frozen else None}
        q.write_new(out/'protocol.json',protocol)
        processor=AutoProcessor.from_pretrained(str(q.local(c['model_dir'])),local_files_only=True,
            trust_remote_code=False,min_pixels=c['min_pixels'],max_pixels=c['max_pixels'],use_fast=False)
        model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(q.local(c['model_dir'])),
            local_files_only=True,trust_remote_code=False,use_safetensors=True,
            torch_dtype=getattr(torch,c['dtype']),attn_implementation=c['attention'],device_map={'':'cuda:0'}).eval()
        model.requires_grad_(False)
        # Override model sampling defaults completely, record effective configuration.
        gen=GenerationConfig(do_sample=False,num_beams=1,max_new_tokens=c['max_new_tokens'],
            repetition_penalty=c['repetition_penalty'],use_cache=True,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
            bos_token_id=model.generation_config.bos_token_id)
        q.write_new(out/'effective-generation.json',gen.to_dict())
        with (out/'qwen-predictions.jsonl').open('x',encoding='utf-8') as stream,torch.inference_mode():
            for n,row in enumerate(selected,1):
                if time.monotonic()-started>=a.seconds:raise TimeoutError('Orçamento de geração atingido')
                tick=time.monotonic();images=[]
                for field,phase in [('before','pre'),('after','post')]:
                    path=resolve_image(q.local(c['image_root']),row[field],phase)
                    if q.digest(path)!=row[field+'_sha256']:raise ValueError('Imagem alterada: '+row['id'])
                    with Image.open(path) as image:
                        images.append(image.convert('RGB').resize((c['image_size'],c['image_size']),Image.Resampling.BICUBIC))
                text=processor.apply_chat_template(q.messages(c['prompt']),tokenize=False,add_generation_prompt=True)
                inputs=processor(text=[text],images=images,padding=True,return_tensors='pt')
                if inputs['image_grid_thw'].shape[0]!=2:raise ValueError('Processador não produziu duas imagens')
                grids=inputs['image_grid_thw'].tolist()
                token_count=inputs['input_ids'].shape[1]
                image_token_id=model.config.image_token_id
                image_tokens=int((inputs['input_ids']==image_token_id).sum().item())
                max_context=model.config.text_config.max_position_embeddings if hasattr(model.config,'text_config') else model.config.max_position_embeddings
                if token_count+c['max_new_tokens']>max_context:raise ValueError('Contexto excedido; não truncar silenciosamente')
                inputs=inputs.to('cuda:0')
                torch.cuda.synchronize();gpu_tick=time.monotonic()
                ids=model.generate(**inputs,generation_config=gen)
                torch.cuda.synchronize();generation_seconds=time.monotonic()-gpu_tick
                generated=ids[0,token_count:].tolist()
                if not generated:raise ValueError('Geração vazia')
                raw=processor.tokenizer.decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                if not raw.strip():raise ValueError('Legenda vazia')
                eos=gen.eos_token_id if isinstance(gen.eos_token_id,list) else [gen.eos_token_id]
                ended=generated[-1] in eos
                normalized=q.normalize_prediction(raw)
                prediction={'id':row['id'],'split':split,'before':row['before'],'after':row['after'],
                    'event':row['event'],'reference':row['caption'],'prediction':normalized,'raw_prediction':raw,
                    'model_id':c['model_id'],'model_revision':c['revision'],'generated_token_ids':generated,
                    'generated_tokens':len(generated),'input_tokens':token_count,'image_tokens':image_tokens,
                    'ended_with_eos':ended,'output_cap_reached':not ended and len(generated)>=c['max_new_tokens'],
                    'stop_reason':'eos' if ended else 'length_limit','repetition':q.repetition(normalized.split()),
                    'image_grid_thw':grids,'effective_image_hw':[[g[1]*processor.image_processor.patch_size,g[2]*processor.image_processor.patch_size] for g in grids],
                    'generation_seconds':generation_seconds,'pair_wall_seconds':time.monotonic()-tick,
                    'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved()}
                predictions.append(prediction);stream.write(json.dumps(prediction,ensure_ascii=False)+'\n');stream.flush()
                del inputs,ids,images
                print(json.dumps({'done':n,'total':len(selected),'seconds':round(time.monotonic()-started,1)}),flush=True)
        q.write_new(out/'qwen-predictions.json',predictions)
    except BaseException as exc:
        failure=exc
        q.write_new(out/'failure.json',{'type':type(exc).__name__,'detail':str(exc),
            'pair_id':selected[len(predictions)]['id'] if len(predictions)<len(selected) else None,
            'completed_pairs':len(predictions),'finished_at':q.now()})
        raise
    finally:
        value={'status':'incomplete' if failure else 'completed','completed_pairs':len(predictions),
            'requested_pairs':len(selected),'wall_seconds':time.monotonic()-started,
            'finished_at':q.now(),'test_evaluated':split=='test' and bool(predictions),
            'output_cap_reached':sum(r['output_cap_reached'] for r in predictions),
            'failure_count':1 if failure else 0}
        if not failure:value['predictions_sha256']=q.digest(out/'qwen-predictions.json')
        if predictions:
            value['mean_pair_seconds']=sum(r['pair_wall_seconds'] for r in predictions)/len(predictions)
            value['generation_seconds']=sum(r['generation_seconds'] for r in predictions)
            value['estimated_full_split_seconds_excluding_load_and_metrics']=value['mean_pair_seconds']*c['expected_counts'][split]
            value['peak_gpu_allocated_bytes']=max(r['peak_gpu_allocated_bytes'] for r in predictions)
            value['peak_gpu_reserved_bytes']=max(r['peak_gpu_reserved_bytes'] for r in predictions)
        q.write_new(out/'status.json',value)
        print(json.dumps(value,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    sub=p.add_subparsers(dest='action',required=True)
    sub.add_parser('download');sub.add_parser('environment')
    b=sub.add_parser('basic');b.add_argument('--evaluation',required=True);b.add_argument('--output',required=True)
    for name in ('smoke','validation','test'):
        a=sub.add_parser(name);a.add_argument('--output',required=True);a.add_argument('--seconds',type=int,required=True)
        if name=='validation':a.add_argument('--limit',type=int)
        if name=='test':a.add_argument('--protocol',required=True);a.add_argument('--allow-test',action='store_true')
    a=p.parse_args();c=q.configuration(a.config)
    # Offline guarantee also holds when worker is invoked directly.
    if a.action!='download':os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
    if a.action=='test' and not a.allow_test:p.error('Teste requer --allow-test')
    if a.action=='environment':environment(c)
    elif a.action=='download':download(c)
    elif a.action=='basic':basic(c,a)
    else:
        if not 1<=a.seconds or (a.action=='smoke' and a.seconds>600):p.error('Orçamento inválido')
        infer(c,a)


if __name__=='__main__':main()
