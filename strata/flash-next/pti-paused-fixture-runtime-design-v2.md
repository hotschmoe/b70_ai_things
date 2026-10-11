# Concrete compiler-header repair

CONFIG -> Preserve failed fixture V1 and its original raw evidence. New V2
uses the header spelled by actual SDK43/44 dpct/graph.hpp:
sycl/ext/oneapi/experimental/graph.hpp. The fixture body is otherwise identical.

COMMAND -> probe_pti_paused_regular_fixture_v2.py retains the original owned
no-device compile and card0 OFF/ON runtime, selecting only the new fixture
source and new plan. validate_pti_paused_regular_fixture_v2.py selects that
producer and source. The inside runtime and clock/ownership/cleanup checks
remain byte-identical. ROOT is mounted read-only for sources; the fixture does
not read model payload. No no-model-mount claim is made.

RESULT -> Two focused CPU checks prove exact include-only C++ derivation and
actual compile-recipe selection. No compiler, GPU or actual runtime was
executed by the author.

VERDICT -> Source repair only; root must execute the fresh compile and loaded
OFF/ON fixture. No model coverage, calibration, bandwidth or latency claim.
