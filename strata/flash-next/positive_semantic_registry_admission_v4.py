"""NEW equivalent nonregistry V2 source preparation, semantic current-entry gate."""
from pathlib import Path
from types import SimpleNamespace
import batch_api_cache_positive_v2 as original
import registry_c137_batch37_entry_association_v3 as registry
ROOT=original.ROOT;HERE=original.HERE

def registry_gate(alias):
 path=ROOT/'evals/configs/models.yaml';proof=registry.association(path);ids=registry.parse(path.read_bytes()) if hasattr(registry,'parse') else None
 from strict_registry_yaml_v3 import parse
 models=parse(path.read_bytes())['models'];original.require(sum(row['served_model_id']==alias for row in models)==1,'Actual current accepted positive entry mustoccur once')
 return {'path':str(path),'sha256':original.sha(path),'served_model_id':alias,'semantic_entry_preservation':proof,'scope':'NEW semantic current165 entry gate; frozenV2 wholeglobal predicate NOT claimed'}

def _namespace():
 ns=dict(vars(original));proxy=SimpleNamespace(**vars(original.contract));proxy.registry_gate=registry_gate;ns['contract']=proxy;return ns

def admit(plan):
 source=Path(original.__file__).read_text();start=source.index('def manifest_binding(plan):');end=source.index('\ndef prepare(',start);body=source[start:end];needle="require(contract.registry_gate(plan['research_alias'])==plan['registry_binding'],'Actual NEW alias registry association differs')";original.require(body.count(needle)==1,'Exact new semantic registry integration failed')
 # This is a NEW equivalently checked source view, never original validator PASS.
 ns=_namespace();exec(compile(body,str(original.__file__)+'[NEW-semantic-registry-V4]','exec'),ns);return ns['manifest_binding'](plan)

def prepare(a):
 source=Path(original.__file__).read_text();start=source.index('def prepare(a):');end=source.index('\ndef run(',start);ns=_namespace();ns['manifest_binding']=admit;exec(compile(source[start:end],str(original.__file__)+'[NEW-semantic-registry-prepare-V4]','exec'),ns);return ns['prepare'](a)
