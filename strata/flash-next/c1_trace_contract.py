"""CPU-testable strict completion contract shared by C1 tracing and screening."""
import hashlib
import json
from pathlib import Path


def pinned_eos(cfg):
    manifest = json.loads(Path(cfg['artifact_identity_manifest']).read_text())
    path = Path(cfg['tokenizer']) / 'tokenizer.json'
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest['tokenizer_files']['tokenizer.json']:
        raise ValueError('C1 trace tokenizer settings differ from pinned artifact')
    settings = json.loads(raw)
    eos = settings['special_ids']['tokenizer.ggml.eos_token_id']
    if type(eos) is not int or not 0 <= eos < settings['vocab_size']:
        raise ValueError('C1 trace requires the pinned tokenizer EOS ID')
    return [eos]


def completion_facts(done, generated, prompt_length, eos_ids, fresh_done, cancelled):
    count = done.get('generated')
    read, reused, total = done.get('prompt_read'), done.get('reused'), done.get('prompt_tokens')
    consumed = (type(read) is int and type(reused) is int and type(total) is int and
                0 <= read <= prompt_length and 0 <= reused <= prompt_length and
                total == prompt_length and read + reused == prompt_length)
    count_matches = type(count) is int and count >= 0 and count == len(generated)
    terminal_eos = bool(generated) and generated[-1] in eos_ids and not any(t in eos_ids for t in generated[:-1])
    complete = fresh_done and not cancelled and consumed and count_matches and done.get('finish') in ['stop', 'length']
    normal_close = complete and done.get('finish') == 'stop' and terminal_eos
    return {'full_prompt_consumed': consumed, 'generated_count_matches_done': count_matches,
            'terminal_pinned_eos': terminal_eos, 'fresh_done_observed': fresh_done,
            'completion_valid': bool(complete), 'normal_eos_close': bool(normal_close)}


def trace_end_accepted(row):
    return (row.get('error') is None and row.get('cancelled') is False and
            row.get('completion_valid') is True and row.get('fresh_done_observed') is True and
            row.get('full_prompt_consumed') is True and row.get('generated_count_matches_done') is True and
            bool(row.get('generated_ids')) and
            (not row.get('consumer_closed') or row.get('normal_eos_close') is True))
