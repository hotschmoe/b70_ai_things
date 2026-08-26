"""Harbor Pi adapter for local SGLang reasoning models.

Harbor's built-in Pi adapter emits a minimal custom-model definition. Pi then
classifies that model as non-reasoning and clamps ``--thinking xhigh`` to off.
This adapter supplies the Qwen-compatible reasoning metadata needed by the
Ornith and Qwen models used in this repository.
"""

from typing import Any, override

from harbor.agents.installed.pi import Pi
from harbor.agents.model_connection import ResolvedModelConnection


class SglangReasoningPi(Pi):
    """Pi configured for a Qwen-style SGLang chat-completions endpoint."""

    def __init__(
        self,
        *args: Any,
        context_window: int = 65536,
        max_tokens: int = 16384,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if context_window <= 0:
            raise ValueError("context_window must be positive")
        if max_tokens <= 0 or max_tokens >= context_window:
            raise ValueError("max_tokens must be positive and below context_window")
        self._context_window = context_window
        self._max_tokens = max_tokens

    @override
    def _build_custom_models_json(
        self,
        access: ResolvedModelConnection,
        model_id: str,
    ) -> dict[str, Any] | None:
        models_json = super()._build_custom_models_json(access, model_id)
        if models_json is None:
            return None

        provider = next(iter(models_json["providers"].values()))
        provider["compat"] = {
            "supportsStore": False,
            "supportsDeveloperRole": False,
            "supportsReasoningEffort": False,
            "supportsUsageInStreaming": True,
            "supportsFinishReason": True,
            "maxTokensField": "max_tokens",
            "thinkingFormat": "qwen-chat-template",
            "supportsStrictMode": False,
        }
        provider["models"] = [
            {
                "id": model_id,
                "name": model_id,
                "reasoning": True,
                # Pi exposes xhigh only when the model maps it explicitly. The
                # Qwen chat template has an on/off control, so xhigh means the
                # model's native thinking mode, not a server-side effort tier.
                "thinkingLevelMap": {"off": None, "xhigh": "xhigh"},
                "input": ["text"],
                "contextWindow": self._context_window,
                "maxTokens": self._max_tokens,
                "cost": {
                    "input": 0,
                    "output": 0,
                    "cacheRead": 0,
                    "cacheWrite": 0,
                },
            }
        ]
        return models_json
