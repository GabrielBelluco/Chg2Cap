"""Contract tests: no network, GPU, model imports or installation."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from captioning import qwen as q


class QwenTests(unittest.TestCase):
    def setUp(self):
        self.c=q.configuration('configs/captioning_qwen.json')

    def fixture(self,root):
        c=copy.deepcopy(self.c)
        rows=[]
        for split in ('train','val','test'):
            rows.append({'id':split,'split':split,'source':'EBD','scene_id':split,'event':'SECRET_EVENT',
                'before':f'EBD/SECRET_EVENT/images/{split}_pre.png',
                'after':f'EBD/SECRET_EVENT/images/{split}_post.png','caption':'SECRET_REFERENCE',
                'before_sha256':'a'*64,'after_sha256':'b'*64})
        prepared=root/'prepared';prepared.mkdir()
        p=prepared/'manifest.jsonl';p.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
        c.update(prepared=str(prepared),manifest_sha256=q.digest(p),expected_counts={'train':1,'val':1,'test':1})
        val=root/'val';val.mkdir()
        prediction=dict(rows[1],reference=rows[1]['caption'],prediction='road flooded',raw_prediction='road flooded.',
            model_id=c['model_id'],model_revision=c['revision'],output_cap_reached=False)
        q.write_new(val/'qwen-predictions.json',[prediction])
        versions=dict(q.PINNED,python='3.12.3')
        protocol={'configuration':c,'code_sha256':q.source_hashes(c),'split':'val','action':'validation',
            'versions':versions,'model_inventory_sha256':q.object_digest({'test':'inventory'})}
        q.write_new(val/'protocol.json',protocol)
        q.write_new(val/'status.json',{'status':'completed','completed_pairs':1,
            'predictions_sha256':q.digest(val/'qwen-predictions.json')})
        return c,rows,val

    def test_help_and_worker_import_no_models(self):
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as error:
            q.main(['--help'])
        self.assertEqual(error.exception.code,0)
        code = "from captioning import qwen, qwen_worker; import sys; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
        result = subprocess.run([sys.executable, '-B', '-S', '-c', code],
                                cwd=q.ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_dry_runs_no_subprocess_or_files(self):
        modules_before = set(sys.modules)
        with tempfile.TemporaryDirectory() as tmp,patch.object(q,'bounded') as execute:
            for action in ('setup','download','check','smoke','validation','freeze','compare'):
                output=Path(tmp)/action
                args=[action,'--dry-run']
                if action in ('smoke','validation','freeze','compare'):args+=['--output',str(output)]
                if action=='freeze':args+=['--validation',str(Path(tmp)/'absent')]
                if action=='compare':args+=['--evaluation',str(Path(tmp)/'absent'),'--scores',str(Path(tmp)/'scores')]
                with contextlib.redirect_stdout(io.StringIO()):q.main(args)
                self.assertFalse(output.exists())
            execute.assert_not_called()
        self.assertFalse({'torch', 'transformers'} & (set(sys.modules) - modules_before))

    def test_download_and_install_require_explicit_execute(self):
        for action in ('setup','download'):
            with self.assertRaisesRegex(ValueError,'execute'),patch.object(q,'bounded') as execute:q.main([action])
            execute.assert_not_called()

    def test_existing_output_and_symlink_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'existe'):q.main(['smoke','--output',tmp,'--dry-run'])
            link=Path(tmp)/'broken';link.symlink_to(Path(tmp)/'missing')
            with self.assertRaises(ValueError):q.new_path(link)

    def test_test_requires_authorization_and_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=['test','--output',str(Path(tmp)/'out'),'--protocol',str(Path(tmp)/'missing'),'--dry-run']
            with self.assertRaisesRegex(ValueError,'allow-test'):q.main(args)
            with self.assertRaises(FileNotFoundError):q.main(args+['--allow-test'])

    def test_only_validation_can_subset(self):
        rows=[{'id':'b','split':'val'},{'id':'a','split':'val'},{'id':'t','split':'test'}]
        self.assertEqual([r['id'] for r in q.select(rows,'smoke')],['a','b'])
        self.assertEqual([r['id'] for r in q.select(rows,'validation',1)],['a'])
        with self.assertRaises(ValueError):q.select(rows,'test',1)

    def test_prompt_contains_no_record_or_file_metadata(self):
        msg=q.messages(self.c['prompt'])
        self.assertEqual(msg[0]['content'][:2],[{'type':'image'},{'type':'image'}])
        serialized=json.dumps(msg)
        for forbidden in ('SECRET_EVENT','SECRET_REFERENCE','EBD/','file://','before_sha256'):
            self.assertNotIn(forbidden,serialized)

    def test_manifest_hash_and_reference_pairing(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,rows,val=self.fixture(Path(tmp))
            self.assertEqual(len(q.manifest(c)),3)
            q.compare_inputs([rows[1]],q.read(val/'qwen-predictions.json'))
            bad=q.read(val/'qwen-predictions.json');bad[0]['reference']='changed'
            with self.assertRaisesRegex(ValueError,'referência'):q.compare_inputs([rows[1]],bad)
            with (Path(c['prepared'])/'manifest.jsonl').open('a') as f:f.write('\n')
            with self.assertRaisesRegex(ValueError,'Manifesto'):q.manifest(c)

    def test_freeze_and_reject_config_code_or_evidence_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,_,val=self.fixture(Path(tmp));out=Path(tmp)/'frozen.json'
            a=q.parser().parse_args(['freeze','--validation',str(val),'--output',str(out)])
            with patch.object(q,'model_inventory',return_value={'test':'inventory'}),contextlib.redirect_stdout(io.StringIO()):q.freeze(c,a)
            self.assertEqual(q.frozen(c,out)['status'],'frozen_before_test')
            changed=dict(c,max_new_tokens=512)
            with self.assertRaisesRegex(ValueError,'difere'):q.frozen(changed,out)
            with patch.object(q,'source_hashes',return_value={}):
                with self.assertRaisesRegex(ValueError,'difere'):q.frozen(c,out)
            (val/'qwen-predictions.json').write_text('[]')
            with self.assertRaises(ValueError):q.frozen(c,out)

    def test_freeze_refuses_caps_or_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,_,val=self.fixture(Path(tmp));out=Path(tmp)/'freeze.json'
            a=q.parser().parse_args(['freeze','--validation',str(val),'--output',str(out)])
            rows=q.read(val/'qwen-predictions.json');rows[0]['output_cap_reached']=True
            (val/'qwen-predictions.json').write_text(json.dumps(rows))
            status=q.read(val/'status.json');status['predictions_sha256']=q.digest(val/'qwen-predictions.json')
            (val/'status.json').write_text(json.dumps(status))
            with self.assertRaisesRegex(ValueError,'truncadas'):q.freeze(c,a)
            status['status']='incomplete';(val/'status.json').write_text(json.dumps(status))
            with self.assertRaisesRegex(ValueError,'incompleta'):q.freeze(c,a)
            self.assertFalse(out.exists())

    def test_prediction_normalization_and_repetition(self):
        self.assertEqual(q.normalize_prediction('Roads, roofs. Why?'),'Roads , roofs Why')
        r=q.repetition(['road',',','road','road','road'])
        self.assertEqual(r['adjacent_repeat_fraction'],1)
        self.assertEqual(r['repeated_trigram_fraction'],.5)

    def test_metrics_preserve_existing_evaluator_parameters(self):
        from captioning import cli as rscc
        c=q.read(q.local(self.c['metrics_config']))
        commands=rscc.metric_commands(c,Path('/input'),Path('/output'),['qwen'],'test')
        self.assertTrue(all('--allow-test' in cmd for cmd in commands))
        self.assertIn('qwen-predictions.json',commands[0][commands[0].index('--predictions')+1])
        self.assertEqual(commands[1][commands[1].index('--dtype')+1],'float16')
        self.assertEqual(commands[2][commands[2].index('--dtype')+1],'float32')
        self.assertEqual(commands[2][commands[2].index('--bert-layers')+1],'17')

    def test_metrics_dry_run_and_score_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c,_,val=self.fixture(root);out=root/'scores'
            cp=root/'config.json';q.write_new(cp,c)
            with contextlib.redirect_stdout(io.StringIO()) as log,patch.object(q,'bounded') as execute:
                q.main(['--config',str(cp),'metrics','--evaluation',str(val),'--output',str(out),'--dry-run'])
            self.assertIn('captioning.semantic',log.getvalue());execute.assert_not_called();self.assertFalse(out.exists())
            out.mkdir();rows=q.read(val/'qwen-predictions.json')
            for name in ('meteor','st5-scs','bertscore'):
                q.write_new(out/f'qwen-{name}.json',{'source_sha256':'wrong','per_pair':[{'id':'val'}],
                    'status':'completed','test_evaluated':False})
            with self.assertRaisesRegex(ValueError,'não corresponde'):q.verified_scores(out,rows,'correct')

    def test_model_inventory_checks_required_shards_and_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c=dict(self.c,model_dir=str(root))
            names=['config.json','tokenizer.json','tokenizer_config.json','preprocessor_config.json','model.safetensors.index.json','model.safetensors']
            for name in names:
                (root/name).write_text(json.dumps({'weight_map':{'weight':'model.safetensors'}}) if 'index' in name else '{}')
            inv={'status':'completed','model_id':c['model_id'],'revision':c['revision'],
                'files':{name:{'bytes':(root/name).stat().st_size,'sha256':q.digest(root/name)} for name in names}}
            q.write_new(root/'download.json',inv)
            self.assertEqual(q.model_inventory(c),inv)
            (root/'model.safetensors').write_text('[]')
            with self.assertRaisesRegex(ValueError,'modificado'):q.model_inventory(c)
            inv['files'].pop('model.safetensors');(root/'download.json').write_text(json.dumps(inv))
            with self.assertRaisesRegex(ValueError,'shards'):q.model_inventory(c)

    def test_precision_configuration_and_invalid_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'config.json';c=dict(self.c,dtype='float32')
            q.write_new(path,c);self.assertEqual(q.configuration(path)['dtype'],'float32')
            c['do_sample']=True;path.write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError,'greedy'):q.configuration(path)

    def test_external_timeout_really_stops_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker=Path(tmp)/'marker'
            code=f'import time; time.sleep(2); open({str(marker)!r},"w").write("unexpected")'
            with self.assertRaises(subprocess.TimeoutExpired):q.bounded([sys.executable,'-B','-c',code],.05)
            self.assertFalse(marker.exists())


if __name__=='__main__':unittest.main()
