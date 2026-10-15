Thanks for checking.

1. The handback bug was fixed in 1fce5591, committed after your comment: carry_leg/add_carry retain the optional slot counts and add the final solo DONE figures. The actual-helper regression covers mixed legs, missing/None fields and explicit zero. The real SYCL handback oracle gave 40/46 + 304/371 = 344/417 at both the server and API.
2. 91b02a5c updates docs/BATCHING.md with the optional accepted/offered fields, their slot-leg scope and legacy behavior. It also notes non-MTP 0/0 and that verification-accepted drafts need not all be emitted.
3. Agreed on keeping one change per PR. That commit removes restore-payload telemetry from this PR; it is parked separately. The draft emitters and server accounting remain unchanged.

Rechecked six targeted CPU tests, all passing (unittest 1.937 seconds; CPU-only container 11.236 seconds). No CUDA/SYCL rebuild or GPU rerun for these documentation/removal-only changes; CUDA remains inspected rather than built on this host.
