# GPU test list for the cutover window (run in one window, in order; each <= 3 min unless noted)

Owner of ordering and gates: worker INT. Each item: command, expected time, PASS gate. Sections per item: KV, SAMP, VIS, API, INT.
Final steps (INT): dd_functional.py through 18080 with image fixture; switch units; rollback if any gate fails.
