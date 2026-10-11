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
from c1_trace_contract import pinned_eos, completion_facts

sys.path.insert(0, '/src')
from serve import server

TRACE = Path(os.environ['B70_C1_TRACE'])
LOCK = threading.Lock()
LOCAL = threading.local()
COUNTER = 0
CFG = json.loads(Path(sys.argv[sys.argv.index("--config")+1]).read_text())
EOS_IDS = pinned_eos(CFG)


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


parse_done_original = server.StrataEngine._parse_done


def parse_done(self, line):
    result = parse_done_original(self, line)
    self._b70_done_sequence = getattr(self, '_b70_done_sequence', 0) + 1
    self._b70_done_observation = {'sequence': self._b70_done_sequence,
        'call': getattr(LOCAL, 'call', None), 'raw': line, 'done': dict(self.last)}
    LOCAL.done_observation=dict(self._b70_done_observation)
    LOCAL.done_count=getattr(LOCAL,'done_count',0)+1
    return result


server.StrataEngine._parse_done = parse_done
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
    before_done = getattr(self, '_b70_done_sequence', 0)
    LOCAL.call=call
    LOCAL.done_observation=None
    LOCAL.done_count=0
    emit({'kind': 'engine_begin', 'call': call, 'epoch': started, 'engine_pid': pid,
          'submitted_ids': sent, 'submitted_ids_sha256': token_sha(sent), 'max_new': max_new,
          'rendered_prompt': prompt, 'rendered_matches_submitted': bool(prompt and prompt['ids'] == sent),
          'sampling': sampling, 'embeddings': embeddings, 'pinned_eos_ids': EOS_IDS})
    generated, error, closed = [], None, False
    iterator = original(self, ids, max_new, sampling, cancel, embeddings)
    try:
        for token in iterator:
            if isinstance(token, int):
                generated.append(token)
            yield token
    except GeneratorExit:
        closed = True
        # Explicitly close the underlying real iterator before reading DONE.
        # Its existing finally block sends STOP/drains when a consumer ends on EOS.
        # This prevents stale prior-request DONE from qualifying a close.
        try:
            iterator.close()
        except BaseException as exc:
            error = {'type': type(exc).__name__, 'message': str(exc)}
            raise
        raise
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        observation = getattr(LOCAL, 'done_observation', None)or{}
        fresh = getattr(LOCAL,'done_count',0)==1 and observation.get('call')==call
        done = dict(observation.get('done')or{})
        facts = completion_facts(done, generated, len(sent), EOS_IDS, bool(fresh), cancel.is_set())
        if closed and not facts['normal_eos_close'] and error is None:
            error = {'type': 'GeneratorExit', 'message': 'Consumer closed without a fresh complete pinned-EOS stop'}
        elif error is None and not facts['completion_valid']:
            error = {'type': 'IncompleteEngineCompletion', 'message': 'Fresh DONE/consumption/generated-count gate failed'}
        emit({'kind': 'engine_end', 'call': call, 'epoch': time.time(), 'engine_pid': pid,
              'engine_done': done, 'generated_ids': generated, 'generated_ids_sha256': token_sha(generated),
              **facts, 'pinned_eos_ids': EOS_IDS, 'consumer_closed': closed,
              'done_observation': observation if fresh else None,
              'error': error, 'cancelled': cancel.is_set(), 'elapsed_s': time.time()-started})
        LOCAL.call=None
        LOCAL.done_observation=None
        LOCAL.done_count=0


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
