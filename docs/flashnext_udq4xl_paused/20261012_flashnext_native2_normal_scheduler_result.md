# Default-scheduler decode during incoming prefill, 2026-10-12

CONFIG -> Current52 pair24/24 native2, defaultdecode_share0.5, mirror0,
no heavy capture. A P2048/GEN512; C P2048/GEN64 arrives after actual A256.
COMMAND -> Producer28018 PASS742s; independent18588 PASS.
RESULT -> Both complete512/64 histories/finish exact versus serial;
63 actual completed rows2 events, no raw capture, full4/source/all8/
normalexit/EOF/ownedretirement/health/collective/kernel close pass.
Native long TTFT25.986s, gapsP50 47.416/P95 73.206/max9740.653ms;
incoming TTFT22.035s, completion27.121s, gapsP50 72.100/P95 77.308/
max683.602ms. During incoming request, longP50 54.843/P95 77.043ms.
Top long gaps4640.466,4266.993,9740.653ms. Last gap straddles
incomingfirsttoken:9.057140s before firsttoken and~0.6835s after.
Producer4e80a8e5f44aaa8f9abe70270742d63fdae0b8a1c29f6ea836b8d744dfda0f88; publicbf91278a04a78b7c53934c7a11b0a0337d30d6b5ca94038f25785602d5ba0dd2.
VERDICT -> Real bounded CLI tail-latency evidence, not HTTP/population
stability or causal serial-vs-native speedup. Savedlogs/source identify
read_part's q>=b0 skip across1536pin/2041/2047 logical boundaries;
existing singlechunk parts never yielddecode. No physical idle claim.
GEN source explicitly T1 withoutMTP; no throwaway spec2-row diagnosis.

| Native role | TTFT s | Completion s | Gap P50 ms | Gap P95 ms | Maximum gap ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| ongoing long decode | 25.986 | 70.634 | 47.416 | 73.206 | 9740.653 |
| incoming prefill and decode | 22.035 | 27.121 | 72.100 | 77.308 | 683.602 |

The incoming request was sent12.128s after the long request's first token,
and emitted its first token34.163s after that first token. The long request
then had three large gaps: token258 from12.179->16.820s; token298 from
18.759->23.026s; token343 from25.106->34.846s. All times are client-side
receipts and sends in one host monotonic domain; they are not device clocks.

Within the9.741s gap, incoming PP1536/2041/2047 and first-token/admission
events are observed before the first two-row completion. Source
sycl/src/program/generate.cpp10473 skips decode after the final chunk of
each read_part. The1536 checkpoint and subsequent <=512-token logical
parts therefore permit consecutive prefill work without a decode turn.
This is a concrete scheduling lead. It does not prove device idle, assign
the entire gap to one operation, or quantify an unexecuted fix.

Serial controls have different GEN execution and process/cache temperature;
their timing is retained but not interpreted as a causal native speed gain.
Percentiles use linear interpolation at(n-1)*p. One bounded run is not a
latency distribution across independent launches. Natural EOS was enabled;
these four histories ended at512/64 length caps.

Evidence root: /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/
f17-source37-20261010/source52-native2-fairness-v1-run-v1.
Public reader: source52-native2-fairness-v1-public-v1.json in F17.
