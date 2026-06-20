#!/usr/bin/env python3
# AutoRound W4A8-int8 quant driver (corrected for auto_round 0.13.1 API; the repo's 10_*.sh used a
# dead API). Builds the W4A8 scheme by hand: int4 sym group-128 weights + per-token dynamic int8
# activations -> exports compressed-tensors (llm_compressor) -> serves via XPUW4A8IntLinearKernel.
# Env: SRC, OUT, ITERS, NSAMPLES, SEQLEN.
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound
from auto_round.schemes import QuantizationScheme

SRC = os.environ["SRC"]; OUT = os.environ["OUT"]
ITERS = int(os.environ.get("ITERS", "200"))
NSAMPLES = int(os.environ.get("NSAMPLES", "128"))
SEQLEN = int(os.environ.get("SEQLEN", "2048"))
print(f"[autoround-w4a8] src={SRC} out={OUT} iters={ITERS} nsamples={NSAMPLES} seqlen={SEQLEN}", flush=True)

# int4 sym group-128 weights + per-token dynamic int8 activations (symmetric) -> W4A8-int8.
W4A8 = QuantizationScheme(bits=4, group_size=128, sym=True, data_type="int",
                          act_bits=8, act_dynamic=True, act_sym=True, act_data_type="int")

tok = AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(SRC, torch_dtype="auto", trust_remote_code=True)
ar = AutoRound(model, tok, scheme=W4A8, iters=ITERS, nsamples=NSAMPLES, seqlen=SEQLEN,
               device_map="xpu", format="llm_compressor")
ar.quantize_and_save(output_dir=OUT, format="llm_compressor")
print("QUANT_DONE", OUT, flush=True)
