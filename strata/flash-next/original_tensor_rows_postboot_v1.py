"""Original decoded rows; only source admission uses explicit postboot proof."""
import hashlib
from pathlib import Path
from original_first_gdn_layer_v1 import OriginalTensorRows,source_metadata_contract
from original_gguf_postboot_reference_v1 import OriginalGgufPostboot

class BoundOriginalFull48RowsPostboot(OriginalTensorRows):
    def __init__(self,manifest,identity,*,model_association):
        self.source_identity_sha256=hashlib.sha256(Path(identity).read_bytes()).hexdigest()
        self.reader=OriginalGgufPostboot(manifest,identity,model_association=model_association)
        self.source_contract=source_metadata_contract(self.reader.files[0]['metadata'])
        self.postboot_admission={'old_current_stat_gate_passed':False,
                                'current_runtime_qualified':False,
                                'current_identity_sha256':model_association['current_identity_sha256']}
