import ast,pathlib,torch,json
p=pathlib.Path(__file__).with_name('kv_fp8_scale_oracle.py');tree=ast.parse(p.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='verify_exact_cache_bytes');scope={'torch':torch};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p),'exec'),scope);check=scope[fn.name]
slots=torch.tensor([9,2,8,0,5]);k=torch.tensor([.125,56.,-56.,0.,.000244140625]).reshape(5,1,1).half();v=torch.tensor([.5,-112.,112.,0.,.00048828125]).reshape(5,1,1).half()
kc=torch.zeros(3,4,1,1,dtype=torch.uint8);vc=torch.zeros_like(kc)
for slot,kbyte,vbyte in zip(slots.tolist(),[0x38,0x7e,0xfe,0,1],[0x40,0xfe,0x7e,0,1]):kc[slot//4,slot%4]=kbyte;vc[slot//4,slot%4]=vbyte
check(kc,vc,k,v,slots,4,.125,.25)
cases=['manual IEEE E4M3 bytes, distinct scales, permuted locations']
for label,which,slot in [('wrong-active-K','k',9),('wrong-active-V','v',2),('modified-unused-slot','k',1)]:
 a=kc.clone();b=vc.clone();(a if which=='k' else b)[slot//4,slot%4]^=1
 try:check(a,b,k,v,slots,4,.125,.25)
 except AssertionError:cases.append('reject '+label)
 else:raise AssertionError(label+' not rejected')
try:check(kc,vc,k,v,slots,4,.25,.125)
except AssertionError:cases.append('reject swapped K/V scales')
else:raise AssertionError('wrong scales not rejected')
print(json.dumps(dict(passed=True,cases=cases,gpu_used=False),indent=2))
