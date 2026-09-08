#!/usr/bin/env python3
"""API server entry for an explicit, inherited calibration hook."""
import runpy

if __name__ == '__main__':
    runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')
