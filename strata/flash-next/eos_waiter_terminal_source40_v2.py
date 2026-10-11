"""Source40 EOS precedence guard; frozen waiter proof and rawERROR retained."""
import eos_waiter_terminal_adjudication40_v1 as original

def source_guard(association):
 if association['actual_yielded_pinned_eos']is True and association['actual_client_cancelled']is False:
  if association['actual_native_terminal']['finish']!='stop':raise ValueError('Actual source40 noncancelled EOS must native-stop, never cancel/length waiver')
 return association
class Terminals(original.Terminals):
 def association(self,call):return source_guard(super().association(call))
def adjudicate(events,observer_proof):
 value=original.adjudicate(events,observer_proof)
 for association in value['actual_API_terminal_associations']:source_guard(association)
 return {**value,'actual_source40_EOS_precedence_verified':True,'frozen_waiter_adjudicator_bytes_changed':False}
