#!/usr/bin/env python3
"""Read-only qualification tracing around the frozen API's real engine boundary.

No tokenizer/template, sampling, engine command, generated token or state changes.
"""
import hashlib
import json
import os
import struct
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, '/src')
from serve import server

TRACE = Path(os.environ['B70_C1_TRACE'])
LOCK = threading.Lock()
LOCAL = threading.local()
COUNTER = 0


def token_sha(ids):
    return hashlib.sha256(b''.join(struct.pack('<I', int(i)) for i in ids)).hexdigest()


def emit(row):
    with LOCK:
        with TRACE.open('a') as out:
            out.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + '\n')


encode_original = server.Service.encode_prompt


def encode(self, messages, tools, kwargs):
    ids = encode_original(self, messages, tools, kwargs)
    LOCAL.prompt = {'ids': list(ids), 'sha256': token_sha(ids), 'messages': messages,
                    'tools': tools, 'template_kwargs': kwargs}
    return ids


original = server.StrataEngine.generate


def generate(self, ids, max_new, sampling, cancel, embeddings=None):
    global COUNTER
    with LOCK:
        COUNTER += 1
        call = COUNTER
    prompt = getattr(LOCAL, 'prompt', None)
    sent = list(ids)
    started = time.time()
    pid = getattr(self.proc, 'pid', None)
    emit({'kind': 'engine_begin', 'call': call, 'epoch': started, 'engine_pid': pid,
          'submitted_ids': sent, 'submitted_ids_sha256': token_sha(sent), 'max_new': max_new,
          'rendered_prompt': prompt, 'rendered_matches_submitted': bool(prompt and prompt['ids'] == sent),
          'sampling': sampling, 'embeddings': embeddings})
    generated = []
    error = None
    try:
        for token in original(self, ids, max_new, sampling, cancel, embeddings):
            if isinstance(token, int):
                generated.append(token)
            yield token
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        done = dict(self.last)
        consumed = done.get('prompt_read')
        reused = done.get('reused', 0)
        emit({'kind': 'engine_end', 'call': call, 'epoch': time.time(), 'engine_pid': pid,
              'engine_done': done, 'generated_ids': generated, 'generated_ids_sha256': token_sha(generated),
              'full_prompt_consumed': isinstance(consumed, int) and consumed + reused == len(sent),
              'error': error, 'cancelled': cancel.is_set(), 'elapsed_s': time.time()-started})


close_original = server.StrataEngine.close


def close(self):
    child = self.proc
    started = time.time()
    error = None
    try:
        return close_original(self)
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        code = child.poll() if child is not None else None
        emit({'kind': 'engine_close', 'engine_pid': getattr(child, 'pid', None),
              'epoch': time.time(), 'elapsed_s': time.time()-started,
              'exit_code': code, 'clean_exit': child is not None and code == 0,
              'error': error})


server.StrataEngine.close = close
server.Service.encode_prompt = encode
server.StrataEngine.generate = generate
raise SystemExit(server.main())
