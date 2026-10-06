"""Small synthetic integration fixtures; no model/GPU/network or real evaluation."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from captioning import qwen as q


class MetricTests(unittest.TestCase):
    def test_exact_existing_scorers_and_no_model_import(self):
        import sys
        modules_before = set(sys.modules)
        from data.RSCC.rscc import tokenize
        from eval_func.bleu.bleu import Bleu
        from eval_func.rouge.rouge import Rouge
        rows=[{'id':'one','reference':'The road is flooded.','prediction':'The road is flooded',
               'repetition':q.repetition('The road is flooded'.split()),'output_cap_reached':False},
              {'id':'two','reference':'Buildings, roads and fields.','prediction':'roads and fields',
               'repetition':q.repetition('roads and fields'.split()),'output_cap_reached':False}]
        with contextlib.redirect_stdout(io.StringIO()):
            actual,pairs=q.basic_metrics(rows)
            refs=[[' '.join(tokenize(r['reference'])[1:-1])] for r in rows]
            hyps=[[r['prediction']] for r in rows]
            expected_bleu,_=Bleu(4).compute_score(refs,hyps)
            expected_rouge,_=Rouge().compute_score(refs,hyps)
        self.assertEqual(actual['BLEU_1_to_4'],expected_bleu)
        self.assertEqual(actual['ROUGE_L'],expected_rouge)
        self.assertEqual([p['id'] for p in pairs],['one','two'])
        import sys
        self.assertFalse({'torch', 'transformers'} & (set(sys.modules) - modules_before))

    def test_completed_metrics_summary_paired_comparison_and_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c=copy.deepcopy(q.configuration('configs/captioning_qwen.json'))
            prepared=root/'prepared';prepared.mkdir();image_root=root/'images'
            records=[]
            for split in ['train','val','test']:
                row={'id':split,'split':split,'source':'EBD','scene_id':split,'event':'synthetic',
                    'before':f'EBD/synthetic/images/{split}_pre.png','after':f'EBD/synthetic/images/{split}_post.png',
                    'caption':'The road is flooded.','before_sha256':'a'*64,'after_sha256':'b'*64}
                records.append(row)
                for name in ['before','after']:
                    p=image_root/row[name];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture')
            path=prepared/'manifest.jsonl';path.write_text('\n'.join(json.dumps(r) for r in records))
            c.update(prepared=str(prepared),image_root=str(image_root),manifest_sha256=q.digest(path),
                expected_counts={'train':1,'val':1,'test':1})
            evaluation=root/'evaluation';evaluation.mkdir();scores=root/'scores';scores.mkdir()
            old=root/'chg2cap';old.mkdir()
            c['chg2cap_validation']=str(old/'best-predictions.json')
            row=records[1]
            pred=dict(row,reference=row['caption'],prediction='The road is flooded',raw_prediction='The road is flooded.',
                repetition=q.repetition('The road is flooded'.split()),output_cap_reached=False,
                model_id=c['model_id'],model_revision=c['revision'])
            q.write_new(evaluation/'qwen-predictions.json',[pred]);q.write_new(old/'best-predictions.json',[pred])
            code_hashes=q.source_hashes(c)
            q.write_new(evaluation/'protocol.json',{'configuration':c,'code_sha256':code_hashes,
                'split':'val','action':'validation'})
            q.write_new(evaluation/'status.json',{'status':'completed','completed_pairs':1,
                'predictions_sha256':q.digest(evaluation/'qwen-predictions.json')})
            spec=importlib.util.spec_from_file_location('worker',q.ROOT/'captioning/qwen_worker.py')
            worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
            a=q.parser().parse_args(['metrics','--evaluation',str(evaluation),'--output',str(scores)])
            with contextlib.redirect_stdout(io.StringIO()):worker.basic(c,a)
            base=q.basic_metrics([pred])[0]
            q.write_new(old/'metrics.json',{'models':{'best':base}})
            metrics=root/'baseline-scores';metrics.mkdir(parents=True)
            c['chg2cap_validation_scores']=str(metrics)
            code_hashes=q.source_hashes(c)
            protocol=q.read(evaluation/'protocol.json');protocol['configuration']=c
            (evaluation/'protocol.json').write_text(json.dumps(protocol))
            names={'meteor':'meteor-all-best-v1.json','st5-scs':'st5-all-best-gpu-fp16-v2.json',
                'bertscore':'bertscore-all-best-gpu-fp32-v1.json'}
            for name in names:
                value={'source_sha256':q.digest(evaluation/'qwen-predictions.json'),
                    'status':'completed','test_evaluated':False,'METEOR_NLTK':.4,
                    'means':{'st5_scs':.6,'f1':.8},'per_pair':[{'id':'val','meteor':.4,'st5_scs':.6,'f1':.8}]}
                q.write_new(scores/f'qwen-{name}.json',value)
                value['source_sha256']=q.digest(old/'best-predictions.json')
                q.write_new(metrics/f'best-{name}.json',value)
            q.write_new(scores/'workflow.json',{'predictions_sha256':q.digest(evaluation/'qwen-predictions.json')})
            report_names=['metrics.json','qwen-meteor.json','qwen-st5-scs.json','qwen-bertscore.json']
            q.write_new(scores/'status.json',{'status':'completed','pairs':1,
                'reports_sha256':{n:q.digest(scores/n) for n in report_names}})
            from captioning import cli as rscc
            with contextlib.redirect_stdout(io.StringIO()):rscc.summarize(scores,scores)
            self.assertIn('| qwen |',(scores/'RESUMO.md').read_text())
            out=root/'comparison'
            a=q.parser().parse_args(['compare','--evaluation',str(evaluation),'--scores',str(scores),'--output',str(out)])
            with patch.object(q,'ROOT',root),patch.object(q,'source_hashes',return_value=code_hashes),contextlib.redirect_stdout(io.StringIO()):
                q.compare(c,a)
            report=q.read(out/'comparison.json')
            self.assertEqual(report['pairs'],1)
            self.assertEqual(report['per_pair'][0]['delta_qwen_minus_chg2cap'],
                {'ROUGE_L':0,'METEOR_NLTK':0,'ST5_SCS':0,'BERTScore_F1':0})
            self.assertIn('qwen_prediction',(out/'comparison.html').read_text())
            # Modified score must not be silently promoted to a comparison.
            (scores/'qwen-meteor.json').write_text('{}')
            a.output=str(root/'bad')
            with patch.object(q,'ROOT',root),patch.object(q,'source_hashes',return_value=code_hashes):
                with self.assertRaisesRegex(ValueError,'modificado'):q.compare(c,a)
            self.assertFalse((root/'bad').exists())


if __name__=='__main__':unittest.main()
