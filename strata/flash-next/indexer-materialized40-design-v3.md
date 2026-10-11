# Isolated materialized indexer V3

CONFIG -> unchanged V1 C++ leaf, original four-row raw/gamma fixture, source40
archive and flags, eighteen frames across all three public append APIs. V1 and V2 are
preserved and held. This successor changes only the operation witness and its
producer/reader routing; model, quantization, arithmetic and reference are unchanged.

COMMAND -> Root prepares with prepare_indexer_materialized40_inputs_v2.py and
executes qualify_indexer_materialized40_v2.py using the same explicit arguments
as V1. No compiler or device execution by the source agent.

RESULT -> Each owned actor admits all original semantic predicates once, then
recollects complete ByteEpoch bytes on every reuse. The producer labels actual
predevice, postrun and postproof boundaries. The public reader independently
admits the original predicates and compares its complete current byte roster to
every original boundary. Calls are typed integers; entry/ready plus sequential
current-use labels account for every call. All required role labels occur exactly
once, with postproof last. Owner PID is the original report producer PID.

The reader joins compile EOF <= predevice start/end <= runtime command start,
runtime EOF <= postrun start/end <= GPU terminal, and post-full4 finish <=
post-page epoch <= final postproof start/end <= report finish. Inherited strict
command receipt, container terminal, health, journal, full4, page and artifact
checks run before these joins. Entry/ready alone cannot qualify the byte proof.

VERDICT -> Source/CPU only. Isolated storage qualification requires actual root
execution. Full model fidelity, model semantic fix and compiler cause remain false.

CPU command:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=strata/flash-next python3 -m unittest test_indexer_materialized40_cpu_v1 test_indexer_materialized40_cpu_v3 -q

25 controls cover the unchanged storage/lifecycle baseline, complete role receipt
joins, missing seals, typed count/owner, label order, repeated roles, changed bytes,
compile/runtime/final chronology and the actual derived producer role call order.

V3 additionally requires exactly six calls and the complete successful source
roster: entry, ready, current-use-2, current-use-3:predevice,
current-use-4:postrun, current-use-5, current-use-6:postproof. Controls reject
consistently renumbered witnesses that omit either generic fixture seal, and
six-call witnesses that move role labels to other source call positions.
