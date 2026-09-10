"""No HTTP, GPU or key-file reads: inject temporary response fixtures."""
import json
import re
import threading
import unittest
import smoke_frontdoor as smoke


class SmokeTests(unittest.TestCase):
    def test_both_auth_schemes_and_four_concurrent_answers(self):
        key='temporary-fixture';seen=[];lock=threading.Lock();barrier=threading.Barrier(4)
        def request(path,kind,token,payload):
            with lock:seen.append((path,kind))
            if token!=key:return 401,'{}'
            if path=='/v1/models':
                return 200,json.dumps({'data':[{'id':i,'max_model_len':200000} for i in [smoke.STABLE,smoke.ALIAS]]})
            self.assertEqual(payload['model'],smoke.STABLE)
            self.assertFalse(payload['chat_template_kwargs']['enable_thinking'])
            barrier.wait(timeout=5)
            n=int(re.search(r'137 \+ (\d+)',payload['messages'][0]['content'])[1])
            return 200,json.dumps({'model':smoke.STABLE,'choices':[{'message':{'content':str(137+n)},'finish_reason':'stop'}]})
        report=smoke.run('fixture',key,request)
        self.assertTrue(report['passed']);self.assertEqual(len(report['checks']),9)
        self.assertEqual(sum(path=='/v1/chat/completions' for path,kind in seen),4)
        self.assertEqual({r['auth_kind'] for r in report['checks'] if r['name'].startswith('concurrent')},{'bearer','x-api-key'})

    def test_wrong_identity_or_context_skips_inference(self):
        for models in [[{'id':smoke.ALIAS,'max_model_len':200000},{'id':smoke.STABLE,'max_model_len':200000}],
                       [{'id':smoke.STABLE,'max_model_len':100000},{'id':smoke.ALIAS,'max_model_len':100000}]]:
            def request(path,kind,token,payload):
                self.assertEqual(path,'/v1/models')
                return (200,json.dumps({'data':models})) if token=='fixture' else (401,'{}')
            report=smoke.run('fixture','fixture',request)
            self.assertFalse(report['passed']);self.assertTrue(report['inference_skipped'])

if __name__=='__main__':unittest.main()
