# Same-recipe observed CPU retry V2 lifecycle correction

CONFIG -> Unchanged e633 screen and reviewed0b4fa9e0 passive observerV3.
Original unexecuted wrapperV1 remains preserved. Peer review found idle work
outside protected cleanup/report scope and observer timeout escaping finally.
COMMAND -> Root19 CPU controls PASS (14observer +5wrapper), including synthetic
owned process retirement timeout and protected idle/signal-handler ordering.
RESULT -> Idle is now an owned Popen inside try/finally, with signal handlers
installed first. Every spawned child is joined to actual terminal before
exclusion release/report. A30s normal retirement timeout records failure and
continues the owned join while retaining exclusion. Screen keeps all Docker
cleanup ownership. No kill or memory/inference recipe change. Full current
source guards and both final readers remain required. No runtime executed.
VERDICT -> New source-only generation awaiting peer review before actual
idle30 plus unchanged all16 continuation. No causal attribution/model math/
API/cache/latency/shelf qualification from orchestration success.
