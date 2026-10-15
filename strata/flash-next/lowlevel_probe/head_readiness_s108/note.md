# S108 one-pass historical head-readiness extraction

CONFIG -> Exact retained S95 trace269853744B, SHA256
1139c4c8cc9a4b0737a6c57e67a8b717151ec4276714fa355fb48e3cd9ab3583.
Use reviewed cycles5..16, unique first-C2/two-select/first-following-Q8 anchors.
Raw process metadata uniquely binds primary GPU pid4294952351 to
DEVICE<b70s4dayz>[Intel(R) Arc(TM) Pro B70 Graphics]0:b:0:0#0. Actual metadata
spelling, all process/thread records and exact source pins are retained.
No clock correction, CPU-steady or cross-card join is applied.

COMMAND -> bin/cpu-run bash head-readiness-s108/run.sh, exact absolute command
in command.json. Stdlib JSON, no library install or new framework. ONE raw
read, SHA256 over those in-memory bytes, ONE JSON parse; no second raw scan.
All complete primary-GPU events overlapping the12 anchor intervals are kept,
including potential boundary crossers/unexpected names. IDs validate anchors;
no missing events or roles are inferred from ID gaps. No GPU or new trace.

RESULT -> Actual terminal0, CPU lease released after7s; wall6.76s, user5.43s,
sys1.12s, maxRSS1502896KiB. Read0.482s/hash0.135s/JSON3.435s/bounded scan0.999s.
1663171raw events,173582complete primary-GPU events;132retained events,
exactly11 in each reviewed interval. All required anchors match retained exact
id/name/pid/tid/start/duration. Two-C2/one-C6 cycle structure is required from
the retained reviewed records; the second C2 lies outside these short extracts.

All12 have the source-consistent ordinal sequence:
C2 head -> sampler -> top-probability -> sampler -> top-probability ->
select -> select -> mapped step/position copy -> token embedding ->
embedding RMS -> embedding Q8. No additional/crossing event occurs inside
these12 specific intervals. Owner/buffer bindings are not independently logged;
owner order is source-informed, and this is not an arbitrary-window classifier.

Historical instrumented medians across12 nested cycles (microseconds):
C2 head service1202.2395; head-end to first-select start254.625;
two sampler/probability pairs' service sum243.175; last probability-end to
first-select start2.125; two selectors' service sum33.6975; second-select-end
to mapped-copy start21.875 and to embedding start25.125. Head-end through
first-Q8 end325.625; copy/embed/RMS/Q8 service sum12.6025.

VERDICT -> The former apparent254us head-to-select gap contains substantial
measured sampler/probability work. The bounded offline pass resolves the
missing operation coverage without another GPU profile. The21.875us interval
after select is a gap in collected operation coverage, not proven physical
idle, data starvation, a removable wait, or a host-causality measurement.
Service sums, interval spans and whole-request wall time are different scopes.
This is historical S95 with component graphs; it establishes no S106 timing,
queue-fed-ahead behavior, speed/stability gain, or attainable saving.

Dev-loop -> Original broad S95 analysis took33.181s; this single targeted pass
closed in6.76s and emits only132events plus metadata. Retain these bounded
outputs for future agents; no raw reparse/retry, GPU test or candidate selected.
