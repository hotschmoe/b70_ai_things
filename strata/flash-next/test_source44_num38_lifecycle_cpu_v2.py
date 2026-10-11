"""Actual-shaped UR/L0Q8 metadata only; no actual allocation or model."""
import ast,copy,unittest
from pathlib import Path
from unittest.mock import patch
import source44_num38_lifecycle_v2 as life
import source44_num38_fields_v2 as fields

def fixture(size=7048460):
    pointers=[0x2000000+i*0x400000 for i in range(10)];tiers=[1,2]*5
    def allocation(kind,address,extent):return '<--- %s(.hContext = 0x111, .size = %d, .ppMem = 0x2 (0x%x)) -> UR_RESULT_SUCCESS;\n'%(kind,extent,address)
    text=allocation('urUSMDeviceAlloc',0x1000,size)+'L0Q8 allocation stage=0 pointer=0x1000 bytes=%d owner_queue=verifier_cs\n'%size
    for address,tier in zip(pointers,tiers):text+=allocation('urUSMDeviceAlloc'if tier==1 else'urUSMHostAlloc',address,3072000)
    text+='L0Q8 frame pid=123 request=1 metadata=frame.json\n'
    text+='L0Q8 release_begin stage=0 pointer=0x1000 bytes=%d\n'%size
    text+='<--- urUSMFree(.hContext = 0x111, .pMem = 0x1000) -> UR_RESULT_SUCCESS;\nL0Q8 release_returned stage=0\n'
    return text,pointers,tiers

class Controls(unittest.TestCase):
    def test_new_extent_live_owner_and_release(self):
        text,pointers,tiers=fixture();namespace=life.ports()
        result=namespace['live_blob_bindings'](text,123,1,pointers,tiers)
        self.assertTrue(result['passed']);self.assertEqual(len(result['source_live_ranges']),10)
        audit=namespace['audit_text'](text);self.assertTrue(audit['passed'],audit['errors']);self.assertEqual(audit['owners'][0]['bytes'],7048460)
        with self.assertRaises(ValueError):life.original.live_blob_bindings(text,123,1,pointers,tiers)

    def test_old_extent_and_foreign_context_refused(self):
        text,pointers,tiers=fixture(7023304);namespace=life.ports()
        with self.assertRaises(ValueError):namespace['live_blob_bindings'](text,123,1,pointers,tiers)
        self.assertFalse(namespace['audit_text'](text)['passed'])
        text,pointers,tiers=fixture();wrong=text.replace('.hContext = 0x111, .pMem','.hContext = 0x222, .pMem')
        self.assertFalse(namespace['audit_text'](wrong)['passed'])

    def test_real_actual38_contract_sum_and_collector_route(self):
        rows=[{'bytes':1}for _ in range(37)]+[{'bytes':7048204-37}]
        with patch.object(fields,'field_contract',return_value=rows):self.assertEqual(life.extent(),(7048204,7048460))
        rows[-1]['bytes']-=25156
        with patch.object(fields,'field_contract',return_value=rows):
            with self.assertRaises(ValueError):life.extent()
        self.assertIs(fields.port().__globals__['live_blob_bindings'],life.live_blob_bindings)

    def test_extent_port_undo_exact(self):
        source=ast.parse(Path(life.original.__file__).read_bytes());original=[copy.deepcopy(n)for n in source.body if isinstance(n,ast.FunctionDef)and n.name in('audit_text','live_blob_bindings')]
        # Undo the actual producer of the executable derived functions.
        changed=copy.deepcopy(life.ports()['_ported_AST'])
        class Undo(ast.NodeTransformer):
            def visit_Name(self,n):
                return ast.copy_location(ast.Constant({'FRAME_BYTES_44':7023048,'TARGET_BYTES_44':7023304}[n.id]),n)if n.id in('FRAME_BYTES_44','TARGET_BYTES_44')else n
            def visit_Constant(self,n):
                return ast.copy_location(ast.Constant(n.value.replace('complete38field','complete33field').replace('single7048460B','single7023304B')),n)if type(n.value)is str else n
        self.assertEqual([ast.dump(n,include_attributes=False)for n in original],[ast.dump(Undo().visit(n),include_attributes=False)for n in changed])

if __name__=='__main__':unittest.main()
