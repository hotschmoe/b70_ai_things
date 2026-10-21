"""Read-only config proof for local benchmark collection; no env fields."""
from fastapi import Request
from pydantic import TypeAdapter
from vllm.config import VllmConfig


def install():
    from vllm.entrypoints import serve

    if getattr(serve, '_x5_config_info', False):
        return
    original = serve.register_vllm_serve_api_routers
    adapter = TypeAdapter(VllmConfig)

    def public(value):
        if isinstance(value, dict):
            return {k: ('<redacted>' if k.lower() in
                        ('hf_token', 'api_key', 'access_token', 'password') and v else public(v))
                    for k, v in value.items()}
        if isinstance(value, list):
            return [public(v) for v in value]
        return value

    def register(app):
        original(app)

        @app.get('/server_info')
        async def config_info(request: Request, config_format: str = 'json'):
            config = adapter.dump_python(request.app.state.vllm_config,
                                         mode='json', fallback=str)
            return {'vllm_config': public(config)}

    serve.register_vllm_serve_api_routers = register
    serve._x5_config_info = True
