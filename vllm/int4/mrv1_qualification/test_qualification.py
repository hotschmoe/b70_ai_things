import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent

def load(name):
    spec=importlib.util.spec_from_file_location(name,HERE/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
probe=load('agent_probe');gate=load('gate');prepare=load('prepare')

class Qualification(unittest.TestCase):
    def test_real_probe_payloads_and_strict_gate(self):
        captures=[]
        def request(base,path,*args,**kw):
            return json.dumps({'data':[{'id':'hotschmoe-dd'}]}) if path=='/v1/models' else 'metrics\n'
        def stream(base,payload,*args,**kw):
            captures.append(copy.deepcopy(payload))
            sku=next(m['content'].split('SKU ')[1].rstrip('.') for m in reversed(payload['messages']) if m['role']=='user')
            _,index,turn=sku.split('-')
            if payload['tool_choice']=='auto':
                msg={'role':'assistant','content':None,'tool_calls':[{'id':'fixture-call','type':'function','function':{'name':'lookup_stock','arguments':json.dumps({'sku':sku})}}]};finish='tool_calls'
            else:msg={'role':'assistant','content':str(730+int(index)*10+int(turn))};finish='stop'
            return {'choices':[{'message':msg,'finish_reason':finish}], 'usage':{'completion_tokens':4,'prompt_tokens_details':{'cached_tokens':1600}}}
        with tempfile.TemporaryDirectory() as tmp:
            for i,(temp,seed) in enumerate([(0,42),(.7,43)]):
                out=Path(tmp)/str(i)
                argv=['probe','--model','hotschmoe-dd','--out',str(out),'--shared-cache','--stream','--cache-namespace',f'round-{i}','--temperature',str(temp),'--seed',str(seed),'--records','360']
                with patch.object(sys,'argv',argv),patch.object(probe,'request',request),patch.object(probe,'stream_request',stream):self.assertEqual(probe.main(),0)
                self.assertTrue(gate.agent(out)['passed'])
                values=captures[-32:]
                self.assertEqual({p['cache_salt'] for p in values},{f'round-{i}'})
                self.assertEqual({p['temperature'] for p in values},{temp})
                self.assertEqual({p['seed'] for p in values},{seed})
                data=json.loads((out/'results.json').read_text());original=copy.deepcopy(data)
                for row in data[0]['rows']:row['response']['usage']['prompt_tokens_details']['cached_tokens']=0
                (out/'results.json').write_text(json.dumps(data))
                with self.assertRaises(AssertionError):gate.agent(out)
                data=copy.deepcopy(original);data[0]['rows'][1]['response']['choices'][0]['message']['content']='731'
                (out/'results.json').write_text(json.dumps(data))
                with self.assertRaises(AssertionError):gate.agent(out)
                data=copy.deepcopy(original);data[0]['rows'][0]['response']['choices'][0]['finish_reason']='length'
                (out/'results.json').write_text(json.dumps(data))
                with self.assertRaises(AssertionError):gate.agent(out)
    def test_frozen_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan=json.loads(prepare.prepare(Path(tmp)/'fresh').read_text())
            self.assertIn(prepare.IMAGE,plan['server'])
            self.assertEqual(plan['server'][-2:],['--tp-host-trace','phase-mrv1'])
            probes=[j for j in plan['jobs'] if '--cache-namespace' in j['command']]
            self.assertEqual(len(probes),6)
            self.assertEqual(len({j['command'][j['command'].index('--cache-namespace')+1] for j in probes}),6)
            self.assertTrue(all(j['command'][j['command'].index('--records')+1]=='360' for j in probes))
            self.assertEqual(len([j for j in plan['jobs'] if j['name'].endswith('-quality')]),8)

if __name__=='__main__':unittest.main()
