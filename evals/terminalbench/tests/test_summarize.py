from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from evals.terminalbench.summarize import summarize_job, summarize_trajectory


def event(message: dict[str, Any], event_type: str = "message") -> dict[str, Any]:
    return {"type": event_type, "message": message}


def assistant(
    stop: str,
    content: list[dict[str, Any]] | None = None,
    raw: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return event(
        {
            "role": "assistant",
            "content": content or [],
            "stopReason": stop,
            "rawStopReason": raw,
            "errorMessage": error,
        }
    )


def tool_call(
    call_id: str,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return {"type": "toolCall", "id": call_id, "name": name, "arguments": arguments}


def tool_result(call_id: str, is_error: bool) -> dict[str, Any]:
    return event(
        {
            "role": "toolResult",
            "toolCallId": call_id,
            "toolName": "fixture",
            "content": [],
            "isError": is_error,
        }
    )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=True) + "\n" for record in records),
        encoding="ascii",
    )


class TrajectorySummaryTest(unittest.TestCase):
    def test_session_is_primary_and_records_length_and_unique_tools(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            trial = Path(temp)
            call = tool_call("same-id", "bash", {"command": "cat /app/package.json"})
            write_jsonl(
                trial / "agent/pi/sessions/session.jsonl",
                [
                    assistant("toolUse", [call], raw="tool_calls"),
                    tool_result("same-id", False),
                    assistant("toolUse", [call], raw="tool_calls"),
                    assistant("length", raw="length"),
                ],
            )
            write_jsonl(
                trial / "agent/pi.txt",
                [
                    event(
                        {
                            "role": "assistant",
                            "content": [],
                            "stopReason": "stop",
                            "rawStopReason": "stop",
                        },
                        "message_end",
                    )
                ],
            )

            summary = summarize_trajectory(trial, None)

            self.assertEqual(summary["source"], "session_jsonl")
            self.assertEqual(summary["final_stop_reason"], "length")
            self.assertEqual(summary["raw_stop_reason"], "length")
            self.assertTrue(summary["length_stop"])
            self.assertFalse(summary["normal_completion"])
            self.assertEqual(summary["tool_count"], 1)

    def test_timeout_is_harbor_state_and_last_edit_test_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            trial = Path(temp)
            write_jsonl(
                trial / "agent/pi/sessions/session.jsonl",
                [
                    assistant(
                        "toolUse",
                        [tool_call("edit-1", "write", {"path": "/app/scripts/release.ts"})],
                        raw="tool_calls",
                    ),
                    tool_result("edit-1", False),
                    assistant(
                        "toolUse",
                        [tool_call("test-early", "bash", {"command": "cd /app && bun run release"})],
                        raw="tool_calls",
                    ),
                    tool_result("test-early", False),
                    assistant(
                        "toolUse",
                        [
                            tool_call(
                                "edit-2",
                                "edit",
                                {"path": "/app/scripts/release.ts", "oldText": "a", "newText": "b"},
                            )
                        ],
                        raw="tool_calls",
                    ),
                    tool_result("edit-2", False),
                    assistant(
                        "toolUse",
                        [tool_call("test-final", "bash", {"command": "cd /app && bun run release"})],
                        raw="tool_calls",
                    ),
                    tool_result("test-final", True),
                ],
            )
            exception = {
                "exception_type": "AgentTimeoutError",
                "exception_message": "Agent execution timed out after 1800.0 seconds",
            }

            summary = summarize_trajectory(trial, exception)

            self.assertEqual(summary["final_stop_reason"], "toolUse")
            self.assertEqual(summary["raw_stop_reason"], "tool_calls")
            self.assertTrue(summary["timeout"])
            self.assertEqual(summary["timeout_type"], "AgentTimeoutError")
            self.assertFalse(summary["length_stop"])
            self.assertEqual(summary["tool_count"], 4)
            self.assertEqual(summary["confirmed_edit_count"], 2)
            self.assertTrue(summary["edit_occurred"])
            self.assertEqual(summary["post_edit_test"]["attempt_count"], 1)
            self.assertEqual(summary["post_edit_test"]["status"], "failed")
            self.assertFalse(summary["post_edit_test"]["passed"])
            self.assertEqual(
                summary["post_edit_test"]["evidence"][0]["tool_call_id"],
                "test-final",
            )

    def test_edit_detection_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            trial = Path(temp)
            write_jsonl(
                trial / "agent/pi/sessions/session.jsonl",
                [
                    assistant(
                        "toolUse",
                        [tool_call("failed-write", "write", {"path": "/app/src/main.py"})],
                    ),
                    tool_result("failed-write", True),
                    assistant(
                        "toolUse",
                        [
                            tool_call(
                                "shell-edit",
                                "bash",
                                {"command": "cd /app && sed -i 's/a/b/' src/main.py"},
                            )
                        ],
                    ),
                    tool_result("shell-edit", False),
                ],
            )

            summary = summarize_trajectory(trial, None)

            self.assertEqual(summary["edit_attempt_count"], 1)
            self.assertEqual(summary["confirmed_edit_count"], 0)
            self.assertFalse(summary["edit_occurred"])
            self.assertFalse(summary["edit_classification_complete"])
            self.assertEqual(
                summary["ambiguous_shell_tool_call_ids"],
                ["shell-edit"],
            )
            self.assertEqual(summary["post_edit_test"]["status"], "none")

    def test_stderr_redirection_is_not_a_shell_edit(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            trial = Path(temp)
            write_jsonl(
                trial / "agent/pi/sessions/session.jsonl",
                [
                    assistant(
                        "toolUse",
                        [
                            tool_call(
                                "test-only",
                                "bash",
                                {"command": "cd /app && bun run release 2>&1 | tail -5"},
                            )
                        ],
                    ),
                    tool_result("test-only", False),
                ],
            )

            summary = summarize_trajectory(trial, None)

            self.assertFalse(summary["edit_occurred"])
            self.assertTrue(summary["edit_classification_complete"])

    def test_pi_text_fallback_ignores_truncated_tail(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            trial = Path(temp)
            pi_path = trial / "agent/pi.txt"
            pi_path.parent.mkdir(parents=True)
            complete = event(
                {
                    "role": "assistant",
                    "content": [],
                    "stopReason": "length",
                    "rawStopReason": "length",
                },
                "message_end",
            )
            pi_path.write_text(
                json.dumps(complete, ensure_ascii=True) + "\n" + '{"type":"turn_end",',
                encoding="ascii",
            )

            summary = summarize_trajectory(trial, None)

            self.assertEqual(summary["source"], "pi_txt")
            self.assertFalse(summary["parse_complete"])
            self.assertEqual(summary["final_stop_reason"], "length")
            self.assertTrue(summary["length_stop"])


class JobSummaryActivityTest(unittest.TestCase):
    def test_job_aggregates_activity_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            job = Path(temp) / "job"
            normal = job / "normal"
            length = job / "length"
            for trial, stop, raw in (
                (normal, "stop", "stop"),
                (length, "length", "length"),
            ):
                write_jsonl(
                    trial / "agent/pi/sessions/session.jsonl",
                    [assistant(stop, raw=raw)],
                )
                result = {
                    "task_name": f"terminal-bench/{trial.name}",
                    "finished_at": "2026-08-29T00:00:01Z",
                    "started_at": "2026-08-29T00:00:00Z",
                    "exception_info": None,
                }
                trial.mkdir(parents=True, exist_ok=True)
                (trial / "result.json").write_text(
                    json.dumps(result, ensure_ascii=True) + "\n",
                    encoding="ascii",
                )

            summary = summarize_job(job)

            self.assertEqual(summary["tasks"], 2)
            self.assertEqual(summary["normal_completions"], 1)
            self.assertEqual(summary["length_stops"], 1)
            self.assertEqual(summary["timeouts"], 0)
            self.assertEqual(summary["tool_calls"], 0)
            self.assertEqual(summary["edit_trials"], 0)
            self.assertEqual(len(summary["task_results"]), 2)
            self.assertIn("trajectory", summary["task_results"][0])


if __name__ == "__main__":
    unittest.main()
