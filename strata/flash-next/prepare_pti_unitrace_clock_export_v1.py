"""Pure exact-source patch reconstruction; never builds or queries a GPU."""
from pathlib import Path
import difflib
import hashlib
import json
import struct
ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/mnt/vm_8tb/b70/build/pti-unitrace-source-6c0d6b0b-v1')
HEADER=ROOT/'strata/flash-next/pti_unitrace_clock_export_v1.h'
ZE='tools/unitrace/src/levelzero/ze_collector.h'
TIMER='tools/unitrace/src/unitimer.h'
NEW_HEADER='tools/unitrace/src/pti_unitrace_clock_export_v1.h'
ORIGINAL={ZE:'6ee0130e2b03918a5e4bb5d310dacb94e5ba06bf67c21a23d3134dc406aa9122',TIMER:'44b529c2ad00957d0253bdbecb84791b55a6796c8587469adee76da7c08c9185'}
def require(v,m):
 if not v:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def replace_once(s,a,b):require(s.count(a)==1,'exact source anchor');return s.replace(a,b)
def changes():
 for n,h in ORIGINAL.items():require(sha(BASE/n)==h,'pinned PTI source changed '+n)
 z=(BASE/ZE).read_text();t=(BASE/TIMER).read_text()
 z=replace_once(z,'#include "unitimer.h"','#include "unitimer.h"\n#include "../pti_unitrace_clock_export_v1.h"')
 initial='''            desc.host_time_origin_ = host_time;'''
 initial_add='''
            if (b70_pti_clock_v1::Enabled()) b70_pti_clock_v1::Emit("device_origin", {
              {"device_handle", reinterpret_cast<uintptr_t>(device)}, {"descriptor_id", static_cast<uint64_t>(desc.id_)},
              {"host_ns", host_time}, {"device_ticks", ticks}, {"timer_frequency_hz", desc.device_timer_frequency_},
              {"kernel_valid_bits", props.kernelTimestampValidBits}, {"global_valid_bits", props.timestampValidBits},
              {"timer_mask", desc.device_timer_mask_}, {"ns_per_cycle_ieee754_u64", b70_pti_clock_v1::DoubleBits(desc.device_ns_per_cycle_)},
              {"uuid_word0_native_u64", b70_pti_clock_v1::UuidWord(props.uuid.id)}, {"uuid_word1_native_u64", b70_pti_clock_v1::UuidWord(props.uuid.id + 8)},
              {"epoch_origin_ns", UniTimer::GetEpochTime(0)}});'''
 z=replace_once(z,initial,initial+initial_add)
 # Subdevice descriptors have the same original frequency/mask fields.
 sub='''                sub_desc.host_time_origin_ = host_time;'''
 # The initial anchor does not match the differently indented subdevice line after exact replacement.
 require(z.count(sub)==1,'subdevice source anchor')
 sub_add=initial_add.replace('            if','                if').replace('reinterpret_cast<uintptr_t>(device)','reinterpret_cast<uintptr_t>(sub_devices[j])').replace('desc.','sub_desc.')
 z=replace_once(z,sub,sub+sub_add)
 finish='''    start = command->submit_time_ + time_shift;
    end = start + duration;'''
 record='''
    if (b70_pti_clock_v1::Enabled()) b70_pti_clock_v1::Emit("instance_normalization", {
      {"instance_id", command->instance_id_}, {"command_id", command->kernel_command_id_},
      {"device_handle", reinterpret_cast<uintptr_t>(command->device_)}, {"command_list_handle", reinterpret_cast<uintptr_t>(command->command_list_)},
      {"queue_handle", reinterpret_cast<uintptr_t>(command->queue_)}, {"engine_ordinal", command->engine_ordinal_}, {"engine_index", command->engine_index_},
      {"submitter_tid", command->tid_}, {"append_host_ns", command->append_time_}, {"submit_host_ns", command->submit_time_},
      {"submit_device_ticks", command->submit_time_device_}, {"raw_kernel_start_ticks", ts.global.kernelStart}, {"raw_kernel_end_ticks", ts.global.kernelEnd},
      {"timer_frequency_hz", device_freq}, {"timer_mask", device_mask}, {"ns_per_cycle_ieee754_u64", b70_pti_clock_v1::DoubleBits(ns_per_cycle)},
      {"normalized_start_host_ns", start}, {"normalized_end_host_ns", end}, {"epoch_origin_ns", UniTimer::GetEpochTime(0)},
      {"host_time_origin_ns", command->host_time_origin_}, {"graph_command", static_cast<uint64_t>(command->graph_command_)}});'''
 z=replace_once(z,finish,finish+record)
 t=replace_once(t,'#include "utils.h"','#include "utils.h"\n#include "pti_unitrace_clock_export_v1.h"')
 end='''    static uint64_t GetEpochTime(uint64_t systime) {'''
 # End of StartUniTimer: preserve both original loops and only append output.
 old='''        }
    }

    static uint64_t GetEpochTime(uint64_t systime) {'''
 addition='''        }
        if (b70_pti_clock_v1::Enabled()) b70_pti_clock_v1::Emit("host_clock_origin", {
          {"epoch_origin_ns", epoch_start_time_},
          {"monotonic_raw_minus_boot_i64_bits", static_cast<uint64_t>(boot_monotonic_raw_time_diff_)},
          {"monotonic_raw_minus_tai_i64_bits", static_cast<uint64_t>(tai_monotonic_raw_time_diff_)}});
    }

    static uint64_t GetEpochTime(uint64_t systime) {'''
 t=replace_once(t,old,addition)
 return {ZE:z,TIMER:t,NEW_HEADER:HEADER.read_text()}
def patch_text():
 out=[]
 for name,body in changes().items():
  old=(BASE/name).read_text()if(BASE/name).exists()else ''
  out.extend(difflib.unified_diff(old.splitlines(True),body.splitlines(True),fromfile='a/'+name if old else '/dev/null',tofile='b/'+name))
 return ''.join(out)
def normalize(row):
 """CPU mirror of the unchanged C++ GetHostTime inputs, no tolerance."""
 for k,v in row.items():require(type(v)is int and 0<=v<(1<<64),'typed u64 normalization input '+k)
 bits=row['timer_mask'];scale=struct.unpack('=d',struct.pack('=Q',row['ns_per_cycle_ieee754_u64']))[0]
 require(row['timer_frequency_hz']>0 and scale==float(1000000000)/float(row['timer_frequency_hz']),'exact frequency-derived normalization scalar')
 start=row['raw_kernel_start_ticks']&bits;end=row['raw_kernel_end_ticks']&bits;submit=row['submit_device_ticks']&bits
 shift=int((start-submit if start>submit else bits-submit+1+start)*scale)
 dur=int((end-start if start<=end else bits-start+1+end)*scale)
 return row['submit_host_ns']+shift,row['submit_host_ns']+shift+dur
if __name__=='__main__':print(patch_text(),end='')
