#!/usr/bin/env python3
"""Two near-full chats, overwrite both slots, then restore under concurrent load.

The server must already hold bin/gpu-run's lease. Engine logs must also show
parking/restore; cached_tokens proves reuse, not that a conversation was parked.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from bench_serving import get, send


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18085")
    parser.add_argument("--target", type=int, default=199600)
    parser.add_argument("--parking", action="store_true")
    parser.add_argument("--resume-from", type=Path,
                        help="Continue the parking test using saved restore-A/B requests")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = args.base_url.rstrip("/")
    identity = get(base, "/v1/models")
    (args.output / "models.json").write_text(json.dumps(identity, indent=2) + "\n")
    if identity["data"][0]["id"] != "hotschmoe-dd" or identity["data"][0]["meta"]["n_ctx"] < args.target:
        raise RuntimeError("Wrong model name or context capacity")
    unit = "The history of computing is long and full of surprising turns. "
    count = (args.target-180)//12
    chats = []
    codes = []
    for label in "ABC":
        needles = [f"{label}-COBALT-7291", f"{label}-RIVER-5183", f"{label}-ORBIT-9462"]
        prompt = f"[Independent long conversation {label}] "
        previous = 0
        for part, fraction in enumerate([0.1, 0.5, 0.95]):
            position = int(count*fraction)
            prompt += unit*(position-previous) + f"\nDeployment code {part+1} is {needles[part]}.\n"
            previous = position
        prompt += unit*(count-previous) + "\nReturn only deployment codes 1, 2 and 3, in order."
        chats.append([dict(role="user", content=prompt)])
        codes.append(needles)

    def record(name, index, messages, restore=False):
        row = send(base, "hotschmoe-dd", messages, 96)
        (args.output / (name + ".json")).write_text(json.dumps(row, indent=2, ensure_ascii=True) + "\n")
        if not all(code in row["text"] for code in codes[index]):
            raise RuntimeError(name + " failed retrieval: " + row["text"])
        if row["usage"]["prompt_tokens"] < args.target-1000:
            raise RuntimeError("Prompt too short to qualify the requested context")
        if restore:
            if row["timings"]["cache_n"] < args.target-1000 or row["timings"]["prompt_n"] > 128:
                raise RuntimeError("Restored conversation recomputed its long prefix")
        print(json.dumps(dict(name=name, ttft_s=row["ttft_s"], wall_s=row["wall_s"],
                              usage=row["usage"], timings=row["timings"], text=row["text"])), flush=True)
        return row

    start = time.perf_counter()
    if args.resume_from:
        for i, label in enumerate("AB"):
            row = json.loads((args.resume_from / ("restore-" + label + ".json")).read_text())
            chats[i] = row["request"]["messages"] + [dict(role="assistant", content=row["text"]),
                dict(role="user", content="Return the same three deployment codes one more time, in order.")]
        first = None
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = list(pool.map(lambda i: record("initial-" + "AB"[i], i, chats[i]), range(2)))
    if args.parking or args.resume_from:
        if first is not None:
            for i in range(2):
                chats[i] += [dict(role="assistant", content=first[i]["text"]),
                             dict(role="user", content="Return those same three deployment codes again, in order.")]
        # One lone chat may run on the parent session, leaving both batch slots
        # intact. Two long answers force the batch path and overwrite both slots.
        def displace(label):
            prompt = f"[Slot displacement {label}] " + unit*330 + (
                "\nImplement a complete typed Python binary search tree with insert, delete, "
                "in-order iteration, balancing, and thorough tests. Explain the invariants "
                "and provide full code and tests. Write at least 1000 tokens.")
            row = send(base, "hotschmoe-dd", [dict(role="user", content=prompt)], 1500)
            (args.output / ("displace-" + label + ".json")).write_text(json.dumps(row, indent=2) + "\n")
            if row["usage"]["completion_tokens"] < 512:
                raise RuntimeError("Displacement response too short to exercise batching")
            print(json.dumps(dict(name="displace-"+label, usage=row["usage"], timings=row["timings"])), flush=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(displace, "CD"))
        slots = get(base, "/slots")
        (args.output / "displaced-slots.json").write_text(json.dumps(slots, indent=2) + "\n")
        if len(slots) != 2 or any(slot["n_prompt_tokens"] > 10000 for slot in slots):
            raise RuntimeError("Both long conversations were not displaced from the GPU slots")
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda i: record("restore-" + "AB"[i], i, chats[i], True), range(2)))
    summary = dict(passed=True, parking=args.parking or bool(args.resume_from), target=args.target,
                   elapsed_s=time.perf_counter()-start, health=get(base, "/health"))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
