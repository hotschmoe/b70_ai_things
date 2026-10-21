#!/usr/bin/env python3
"""Compare early/late kernel sums in O4's already-segmented Ld trace.

The trusted local pickle contains {device_name: [list_of_kernel_tuples]}.
Each tuple is (start_us, end_us, kernel_name). Only modal-count graph
segments are compared; tracing changes absolute timings.
"""

import argparse
import collections
import json
import pickle
import statistics


def average_kernels(steps):
    totals = collections.defaultdict(float)
    for step in steps:
        for start, end, name in step:
            totals[name] += end - start
    return {name: value / len(steps) for name, value in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("steps_pickle")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.steps_pickle, "rb") as handle:
        devices = pickle.load(handle)
    output = {"source": args.steps_pickle, "devices": []}
    for device, segments in devices.items():
        mode = collections.Counter(map(len, segments)).most_common(1)[0][0]
        steps = [segment for segment in segments if len(segment) == mode]
        if len(steps) < 144:
            raise ValueError("Need at least 144 complete graph segments")
        early, late = steps[8:72], steps[-64:]
        before, after = average_kernels(early), average_kernels(late)
        kernels = [dict(name=name, early_us=before.get(name, 0),
                        late_us=after.get(name, 0),
                        delta_us=after.get(name, 0) - before.get(name, 0))
                   for name in before.keys() | after.keys()]
        kernels.sort(key=lambda row: -row["delta_us"])
        item = dict(device=device, modal_kernels=mode, complete_steps=len(steps),
                    early_indices=[8, 72], late_indices=[len(steps)-64, len(steps)],
                    early_span_us=statistics.mean(s[-1][1]-s[0][0] for s in early),
                    late_span_us=statistics.mean(s[-1][1]-s[0][0] for s in late),
                    kernels=kernels)
        output["devices"].append(item)
        print(json.dumps({key: value for key, value in item.items()
                          if key != "kernels"}))
        for row in kernels[:12]:
            print(json.dumps(row))
    with open(args.output, "w", encoding="ascii") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
