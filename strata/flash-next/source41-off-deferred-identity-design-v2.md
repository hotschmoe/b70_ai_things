# Failed identity attempts remain failed

CONFIG -> Frozen V1 is held: a failed closing predicate could be retried after
repairing its fixture and appear successful. V2 keeps all original predicate,
source, PID and caller fresh-scan gates. Every attempted close records start,
finish and error, and every exception latches into the operation error.

COMMAND -> Use source41_off_deferred_identity_v2.DeferredOffIdentity in a future
reviewed caller. A later actual valid scan may close/retire the identity scope,
but require_closed continues to reject an operation with any prior failure.

RESULT -> Eleven CPU controls include the original seven and hash, ordering,
page and source failure followed by valid retry. Each retry records its actual
success while the first failure remains latched. No actual original predicates,
model payload, full scan, GPU or active actor is executed by these controls.

VERDICT -> Source-only unqualified composition proposal. V1 and all active
actors remain unchanged. This module is not integrated or runtime authority.
