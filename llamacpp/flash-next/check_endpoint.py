#!/usr/bin/env python3
"""Bounded CPU-side client. Screening only; does not establish model quality.

Run only against a server launched under the project GPU lease. This client
never launches a server or executes generated source. The required operator
attestation concerns the effective server configuration, not a request option.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="ascii")


def artifact(path):
    data = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": digest(data), "bytes": len(data)}


def check(case, content):
    text = content.strip()
    if case["check"] == "exact":
        return text == case["expected"]
    if case["check"] == "keywords":
        return all(word in text.lower() for word in case["words"])
    if case["check"] == "json":
        try:
            value = json.loads(text)
            return (value == case["expected"] and type(value.get("count")) is int
                    and all(type(v) is int for v in value.get("items", [])))
        except (ValueError, AttributeError, TypeError):
            return False
    if case["check"] == "safe_add_ast":
        try:
            parsed = ast.parse(text)
            expected = ast.parse("def add(a, b):\n    return a + b\n")
            if ast.dump(parsed, include_attributes=False) != ast.dump(expected, include_attributes=False):
                return False
            # Interpret only the already matched expression using integer inputs.
            # Never compile, eval, exec, import, or spawn model-generated code.
            expression = parsed.body[0].body[0].value
            for a, b, expected_result in [(2, 3, 5), (-7, 4, -3), (0, 0, 0), (999, 1, 1000)]:
                env = {"a": a, "b": b}
                result = env[expression.left.id] + env[expression.right.id]
                if result != expected_result:
                    return False
            return True
        except (ValueError, SyntaxError, RecursionError):
            return False
    raise ValueError("Unknown check: " + case["check"])


def exchange(url, payload, output, name, timeout):
    body = None if payload is None else json.dumps(payload, ensure_ascii=True).encode("ascii")
    save(output / (name + ".request.json"), {"url": url, "payload": payload,
         "body_sha256": None if body is None else digest(body)})
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(16 * 1024 * 1024 + 1)
            status = response.status
    except urllib.error.HTTPError as error:
        raw = error.read(16 * 1024 * 1024 + 1)
        status = error.code
    # ASCII JSON envelope preserves the original response bytes losslessly.
    import base64
    save(output / (name + ".response.json"), {"status": status,
         "raw_base64": base64.b64encode(raw).decode("ascii"),
         "sha256": digest(raw), "elapsed_seconds_diagnostic_only": time.monotonic() - started})
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError("Response exceeded bounded capture size")
    if status != 200:
        raise ValueError("HTTP status " + str(status))
    return json.loads(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, help="Base URL, e.g. http://127.0.0.1:18080")
    parser.add_argument("--output", required=True, type=Path, help="New evidence directory")
    parser.add_argument("--model-lock", required=True, type=Path)
    parser.add_argument("--runtime-receipt", required=True, type=Path)
    parser.add_argument("--confirm-no-speculation", required=True, action="store_true",
                        help="Attest effective server config disables speculation")
    parser.add_argument("--rounds", type=int, choices=[1, 2], default=2)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--suite", type=Path, default=Path(__file__).with_name("baseline_suite.json"))
    args = parser.parse_args()
    if not 0 < args.timeout <= 3600:
        parser.error("timeout must be >0 and <=3600 seconds per request")
    base = args.endpoint.rstrip("/")
    parsed = urllib.parse.urlparse(base)
    if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.path not in ("", "/v1"):
        parser.error("endpoint must be an HTTP(S) base URL, optionally ending /v1")
    if base.endswith("/v1"):
        base = base[:-3]
    suite = json.loads(args.suite.read_text())
    provenance = {"model_lock": artifact(args.model_lock), "runtime_receipt": artifact(args.runtime_receipt),
                  "suite": artifact(args.suite), "client": artifact(Path(__file__))}
    args.output.mkdir(parents=True, exist_ok=False)
    for label, path in [("model-lock", args.model_lock), ("runtime-receipt", args.runtime_receipt)]:
        import base64
        save(args.output / (label + ".snapshot.json"), {"raw_base64": base64.b64encode(path.read_bytes()).decode("ascii")})
    result = {"status": "screening_only", "promoted": False, "authority_established": False,
              "provenance": provenance, "speculation_disabled_operator_attestation": True,
              "runtime_settings_independently_verified": False,
              "rounds": args.rounds, "checks": [], "passed": False,
              "limits": ["Keyword prose check is not semantic evaluation", "No performance claim",
                         "Response text hashes are not token-ID parity", "No fresh-server or concurrency qualification",
                         "Model lock is linked and copied, not independently verified against loaded weights"]}
    try:
        identity = exchange(base + "/v1/models", None, args.output, "models-before", args.timeout)
        ids = [entry["id"] for entry in identity.get("data", [])]
        if not ids or ids[0] != "hotschmoe-dd":
            raise ValueError("Expected first served model ID hotschmoe-dd; got " + repr(ids))
        previous = {}
        for round_index in range(args.rounds):
            for case in suite["cases"]:
                name = "round%d-%s" % (round_index + 1, case["id"])
                payload = {"model": "hotschmoe-dd", "messages": [{"role": "user", "content": case["prompt"]}],
                           "temperature": 0, "seed": 42, "max_tokens": 512, "stream": False,
                           "cache_prompt": False}
                response = exchange(base + "/v1/chat/completions", payload, args.output, name, args.timeout)
                choice = response["choices"][0]
                content = choice["message"].get("content")
                if not isinstance(content, str):
                    raise ValueError("Missing textual assistant content")
                hashed = digest(content.encode("utf-8"))
                row = {"case": case["id"], "round": round_index + 1, "output_sha256": hashed,
                       "check_passed": check(case, content), "finish_reason": choice.get("finish_reason"),
                       "repeat_equal": None if case["id"] not in previous else previous[case["id"]] == hashed,
                       "response_model": response.get("model"), "usage": response.get("usage")}
                previous[case["id"]] = hashed
                result["checks"].append(row)
                save(args.output / "screening-result.json", result)
        after = exchange(base + "/v1/models", None, args.output, "models-after", args.timeout)
        if [entry["id"] for entry in after.get("data", [])] != ids:
            raise ValueError("Served model identity list changed during screen")
        result["passed"] = all(row["check_passed"] and row["repeat_equal"] is not False
                               and row["finish_reason"] == "stop" and row["response_model"] == "hotschmoe-dd"
                               for row in result["checks"])
    except Exception as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        save(args.output / "screening-result.json", result)
    print(json.dumps({"passed": result["passed"], "status": result["status"],
                      "result": str(args.output / "screening-result.json")}, ensure_ascii=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
