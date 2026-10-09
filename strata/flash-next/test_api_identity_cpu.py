#!/usr/bin/env python3
"""Exact-GGUF API/token transport and real CPU disk identity fixture; no model/GPU."""
from pathlib import Path
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import urllib.request
import urllib.error

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path('/mnt/vm_8tb/github/strata')
IMAGE='sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7'
RECEIPT=ROOT/'strata/flash-next/api-identity-cpu-receipt.json'


def main():
    lock=json.loads((ROOT/'strata/flash-next/model-lock.json').read_text())
    first=next(f for f in lock['files'] if f['path'].endswith('00001-of-00004.gguf'))
    gguf=ROOT/lock['destination']/first['path']
    with tempfile.TemporaryDirectory(prefix='strata-api-identity-cpu-') as temp:
        work=Path(temp)
        for name in subprocess.check_output(['git','ls-files'],cwd=SOURCE,text=True).splitlines():
            src=SOURCE/name
            if src.is_file():
                dst=work/name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(src.read_bytes())
        patches=sorted((ROOT/'strata/flash-next/patches').glob('*.patch'))
        for patch in patches:
            subprocess.run(['git','apply','--check',str(patch)],cwd=work,check=True)
            subprocess.run(['git','apply',str(patch)],cwd=work,check=True)
        sys.path[:0]=[str(work),str(work/'tools')]
        tokenizer=importlib.import_module('strata_tokenizer')
        server=importlib.import_module('serve.server')
        identity_api=importlib.import_module('serve.artifact_identity')
        pack=work/'fixture-pack'
        exported=tokenizer.extract(gguf,pack)
        tok_dir=pack/'tokenizer'
        original=tokenizer.Tokenizer.from_gguf(gguf)
        vocab=json.loads((tok_dir/'vocab.json').read_text())
        tokens=[None]*len(vocab)
        for text,index in vocab.items(): tokens[index]=text
        settings=json.loads((tok_dir/'tokenizer.json').read_text())
        runtime_tok=tokenizer.Tokenizer(tokens,(tok_dir/'merges.txt').read_text().split('\n'),
            json.loads((tok_dir/'token_type.json').read_text()),settings['pre'],settings['special_ids'])
        assert runtime_tok.tokens==original.tokens and runtime_tok.token_types==original.token_types
        assert runtime_tok.pre==original.pre and runtime_tok.special_ids==original.special_ids
        assert runtime_tok.ranks==original.ranks
        # A protocol-only subprocess records the exact GEN IDs delivered by the
        # actual StrataEngine transport. It performs no inference or GPU work.
        fake=work/'protocol-only-engine.py'
        consumed=work/'consumed.jsonl'
        reply=runtime_tok.encode('OK',parse_special=True)+runtime_tok.encode('<|im_end|>',parse_special=True)
        fake.write_text('#!'+sys.executable+'\n'+
            'import json,sys,os\n'+
            'print("READY 8192 stop",flush=True)\n'+
            'for line in sys.stdin:\n'+
            ' if line.startswith("GEN "):\n'+
            '  ids=[int(x) for x in line.split()[-1].split(",")]\n'+
            '  with open('+repr(str(consumed))+',"a") as f: f.write(json.dumps({"ids":ids,"identity":os.environ.get("STRATA_ARTIFACT_IDENTITY_SHA256")})+"\\n")\n'+
            '  for token in '+repr(reply)+': print("T "+str(token),flush=True)\n'+
            '  print("DONE '+str(len(reply))+' "+str(len(ids))+" 0 0 stop",flush=True)\n'+
            ' elif line.startswith("QUIT"): break\n')
        fake.chmod(0o755)
        alias=lock['research_alias']+'-Strata-SYCL-nativeHC-v1'
        args=['--pack',str(pack),'--max-context','8192']
        env=dict(os.environ)
        # Match the fixture's effective child environment, then inject the
        # manifest digest through the same field used by the production server.
        env['STRATA_SYCL_NATIVE_HC']='1'
        manifest={'schema':1,'primary_model_name':'hotschmoe-dd','research_alias':alias,
            'model':{'repo':lock['repo'],'revision':lock['revision'],'artifact':'UD-Q4_K_XL',
                     'tokenizer_gguf':str(gguf),'tokenizer_gguf_sha256':first['sha256']},
            'tokenizer_files':{n:identity_api.sha256_file(tok_dir/n) for n in identity_api.FILES},
            'runtime':{'exe_sha256':identity_api.sha256_file(fake),'args':args,
                       'env':identity_api.runtime_environment(env),'python_versions':identity_api.python_contract(),
                       'python_sources':identity_api.source_contract()}}
        path=work/'identity.json';path.write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
        cfg={'artifact_identity_manifest':str(path),'model_name':'hotschmoe-dd','aliases':[alias]}
        identity=identity_api.load_identity(cfg,tok_dir)
        identity_api.validate_runtime(identity,str(fake),args,env)
        env[identity_api.IDENTITY_ENV]=identity['sha256']
        def guard(exe=str(fake),actual_args=args,actual_env=env):
            identity_api.validate_runtime(identity,exe,actual_args,actual_env)
        engine=server.StrataEngine(str(fake),args,env=env,identity_guard=guard)
        template=server.ChatTemplate(tok_dir/'chat_template.jinja')
        svc=server.Service(engine,runtime_tok,template,model_name='hotschmoe-dd')
        svc.set_aliases([alias]);svc.artifact_identity=identity['metadata']
        http=server.Server(('127.0.0.1',0),server.make_handler(svc))
        thread=threading.Thread(target=http.serve_forever,daemon=True);thread.start()
        url='http://127.0.0.1:'+str(http.server_address[1])
        cases=[]
        try:
            model_data=json.load(urllib.request.urlopen(url+'/v1/models',timeout=10))
            assert [m['id'] for m in model_data['data']]==['hotschmoe-dd',alias]
            assert model_data['data'][1]['alias_of']=='hotschmoe-dd'
            assert all(m['meta']['artifact_identity']==identity['metadata'] for m in model_data['data'])
            for n,(model,text) in enumerate([('hotschmoe-dd','Reply briefly: 2 + 2.'),
                                            (alias,'Unicode: \u65e5\u672c\u8a9e, caf\u00e9, \u0627\u0644\u0639\u0631\u0628\u064a\u0629.')]):
                messages=[{'role':'user','content':text}]
                kwargs={'enable_thinking':False}
                expected=svc.encode_prompt(messages,None,kwargs)
                rendered=svc.render_prompt(messages,None,kwargs)
                assert expected==original.encode(rendered,parse_special=True)
                request={'model':model,'messages':messages,'max_tokens':8,'temperature':0,
                         'chat_template_kwargs':kwargs,'stream':False}
                req=urllib.request.Request(url+'/v1/chat/completions',data=json.dumps(request).encode(),
                    headers={'Content-Type':'application/json'})
                result=json.load(urllib.request.urlopen(req,timeout=20))
                assert result['model']==model
                records=[json.loads(row) for row in consumed.read_text().splitlines()]
                assert records[-1]['ids']==expected and records[-1]['identity']==identity['sha256']
                cases.append({'served_id':model,'messages':messages,'token_ids':expected,
                              'prompt_sha256':hashlib.sha256(rendered.encode()).hexdigest(),
                              'transport_ids_equal':True})
            before=len(consumed.read_text().splitlines())
            bad={'model':'wrong-quant','messages':[{'role':'user','content':'hello'}],'max_tokens':1}
            try:
                urllib.request.urlopen(urllib.request.Request(url+'/v1/chat/completions',data=json.dumps(bad).encode(),
                    headers={'Content-Type':'application/json'}),timeout=10)
                raise AssertionError('unknown served ID accepted')
            except urllib.error.HTTPError as e: assert e.code==400
            assert len(consumed.read_text().splitlines())==before
            rejects=[]
            def reject(label,fn):
                try: fn()
                except (ValueError,FileNotFoundError): rejects.append(label)
                else: raise AssertionError(label+' was accepted')
            reject('wrong primary',lambda:identity_api.load_identity({**cfg,'model_name':'other'},tok_dir))
            reject('wrong alias',lambda:identity_api.load_identity({**cfg,'aliases':['other']},tok_dir))
            reject('changed launch arguments',lambda:identity_api.validate_runtime(identity,str(fake),args+['--no-ple'],env))
            reject('changed math environment',lambda:identity_api.validate_runtime(identity,str(fake),args,{**env,'STRATA_GR_V3':'1'}))
            data=fake.read_bytes();fake.write_bytes(data+b'# changed binary\n')
            reject('changed executable before reload',guard);fake.write_bytes(data)
            tpl=(tok_dir/'chat_template.jinja').read_bytes();(tok_dir/'chat_template.jinja').write_bytes(tpl+b'\n')
            reject('changed template before reload',guard);(tok_dir/'chat_template.jinja').write_bytes(tpl)
            (tok_dir/'chat_template.jinja').unlink()
            reject('missing template cannot fall back',lambda:identity_api.load_identity(cfg,tok_dir))
            (tok_dir/'chat_template.jinja').write_bytes(tpl)
        finally:
            http.shutdown();http.server_close();thread.join(timeout=10);engine.close()
        # Compile and execute the actual host disk session implementation. New
        # source identity must reject a file written under the former identity.
        (work/'state_identity.cpp').write_text(r'''
#include "strata/core/api_state_identity.hpp"
#include "strata/core/conversation_file.hpp"
#include <cassert>
#include <cstdio>
#include <filesystem>
using namespace strata::core;
uint64_t current(std::string& err) {
    std::string contract; assert(sycl_api_state_identity(contract,err));
    SessionConfig c;c.engine_version="fixture|"+contract;c.backend="sycl";
    return session_config_fingerprint(c);
}
int main() {
    std::string error; unsetenv("STRATA_SYCL_NATIVE_HC");unsetenv("STRATA_ARTIFACT_IDENTITY_SHA256");
    const uint64_t old=current(error);
    const std::string digest(64,'a');setenv("STRATA_ARTIFACT_IDENTITY_SHA256",digest.c_str(),1);
    setenv("STRATA_SYCL_NATIVE_HC","1",1);const uint64_t native=current(error);assert(native!=old);
    setenv("STRATA_GR_V3","1",1);assert(current(error)!=native);unsetenv("STRATA_GR_V3");
    SavedConversation state;for(size_t i=0;i<state.geometry.size();++i)state.geometry[i]=100+i;
    state.layer_lo=0;state.layer_hi=48;state.live.ids={101,102};state.live.gdn={1,2};state.live.ple={3};
    const std::string file="/work/stale-session.bin";size_t size=0;
    assert(session_file_write(file,state,{123,old},size,error));
    SavedConversation restored;
    assert(session_file_read(file,{123,old},restored,size,error));
    assert(!session_file_read(file,{123,native},restored,size,error));
    std::string output;unsetenv("STRATA_ARTIFACT_IDENTITY_SHA256");
    assert(!sycl_api_state_identity(output,error));
    setenv("STRATA_ARTIFACT_IDENTITY_SHA256","bad",1);assert(!sycl_api_state_identity(output,error));
    std::printf("PASS actual CPU session-file roundtrip, stale identity refusal and native digest guards\n");
}
''')
        command=('g++ -std=c++20 -O1 -fsanitize=address,undefined -fno-sanitize-recover=all -I/work/sycl/include -I/work/include '
                 '/work/state_identity.cpp /work/src/core/conversation_file.cpp '
                 '/work/src/core/conversation_memory.cpp -o /work/state_identity && /work/state_identity')
        subprocess.run(['docker','run','--rm','--network','none','--user',f'{os.getuid()}:{os.getgid()}',
            '--entrypoint','/bin/bash','-v',str(work)+':/work',IMAGE,'-lc',command],check=True)
        receipt={'scope':'CPU actual GGUF export/renderer/API/GEN transport to protocol-only child; no model inference/GPU',
            'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip(),
            'patch_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in patches},
            'tokenizer_gguf_sha256':first['sha256'],'model_revision':lock['revision'],
            'manifest_sha256':identity['sha256'],'tokenizer_files':manifest['tokenizer_files'],
            'model_list':model_data,'cases':cases,'negative_cases':rejects,
            'actual_session_file_stale_identity_refused':True,'host_sanitizers_clean':True,
            'python_versions':manifest['runtime']['python_versions'],'python_sources':manifest['runtime']['python_sources'],
            'gpu_model_consumption_verified':False,'prefix_state_fidelity_verified':False}
        RECEIPT.write_text(json.dumps(receipt,indent=2,ensure_ascii=True)+'\n')
        print('PASS exact GGUF tokenizer/template export, two API aliases and actual GEN token transport;7 identity negatives')
        print('Receipt:',RECEIPT)
        print('No model inference, device discovery, GPU execution, real prefix-state restore or serving qualification.')


if __name__=='__main__':main()
