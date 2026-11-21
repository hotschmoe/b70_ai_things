"""CPU-only render check of the Inkling 2 Mini chat template (from the GGUF metadata, HF API copy).
Usage: python3 -I check_template.py [chat_template.jinja]
Python jinja2 lacks llama.cpp's {% generation %} tag and the tojson(sort_keys, separators) kwargs, so both are shimmed.
When the server is up, the authoritative check is POST /apply-template (see README.md)."""
import json, os, re, sys
import jinja2

here = os.path.dirname(os.path.abspath(__file__))
src = open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "chat_template.jinja")).read()
src = re.sub(r"\{%-?\s*(end)?generation\s*-?%\}", "", src)


def tojson(v, sort_keys=False, separators=None, indent=None, **kw):
    return json.dumps(v, sort_keys=sort_keys, separators=separators or (", ", ": "), ensure_ascii=False)


env = jinja2.Environment(undefined=jinja2.StrictUndefined if False else jinja2.Undefined)
env.filters["tojson"] = tojson
env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(RuntimeError(m))
tpl = env.from_string(src)
tools = [{"type": "function", "function": {"name": "bash", "description": "run a shell command",
          "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
msgs = [
    {"role": "system", "content": "You are a coding agent."},
    {"role": "user", "content": "List files in /tmp."},
    {"role": "assistant", "content": "", "reasoning_content": "I should run ls.",
     "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "bash", "arguments": {"command": "ls /tmp", "timeout": 5}}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "a.txt\nb.txt"},
]
for eff in ("xhigh", "high", None):
    kw = {"reasoning_effort": eff} if eff else {}
    out = tpl.render(messages=msgs, tools=tools, **kw)
    print("=== reasoning_effort=%s" % eff); print(out); print()
    assert 'Thinking effort level: ' + ("0.99" if eff == "xhigh" else "0.9") in out
    assert '<|message_model|>bash<|content_invoke_tool_json|>{"name":"bash","args":{"command":"ls /tmp","timeout":5}}<|end_message|>' in out
    assert '<|message_system|>tool_declare<|content_xml|>' in out
print("OK: tool-call format = <|message_model|>NAME<|content_invoke_tool_json|>{\"name\":..,\"args\":{..}}<|end_message|>")
