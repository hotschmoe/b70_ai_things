CONFIG
Source40 server3166f70f yields EOS before its background waiter consumes BDONE.
The original waiter sets held/busy but never updates self.last. Historical V6
GeneratorExit/error and stale admission snapshot remain unchanged and failed;
there is no retroactive waiter retirement evidence for that run.

COMMAND
Install a new output-only observer using the exact original release method code,
a local Thread factory and a same-return/same-exception queue-get proxy in the
original wait closure. No global Thread/Queue monkeypatch, self.last update,
request/scheduling/math change or inserted device wait is made. Retain actual
Thread objects. After phase clients finish, a separate metadata request publishes
proof(calls), with actual ident/native_id/start/alive status and exact selected
queue/waiter counts. Pending threads never qualify. Process errors remain latched.

RESULT
18 CPU controls pass, including execution of the exact pinned original waiter
code on tiny queues. Full original native grammar accepts a late owned BDONE
only with source/call/PID/generation/queue/thread retirement proof. Missing or
foreign terminal, token count, unknown error, live thread and changed busy array
fail. A busy snapshot attributed to a newer owner requires actual distinct BADM1;
it never claims the current slot remains idle. Canceled clients preserve cancel
scope and cannot borrow the pinned-EOS GeneratorExit exception.

VERDICT
The named chronology adjudication changes one original grammar predicate and
preserves all raw engine_end error/self.last records. Source-derived busy=False
transition is separate from the later observed snapshot and actual Thread exit.
The future V7 producer must independently publish and rejoin proof packet/ACK,
phase/source ownership, raw native lines and every client/tokenizer text result.
No actual GPU/API/cache/math success is claimed by this source/CPU preparation.
