from .vllm import VLLMModelInterface, VLLMModelConfig
from .openai import OpenAIModelInterface, OpenAIModelConfig
# from .claude import ClaudeModelInterface, ClaudeModelConfig
# from .gemini import GeminiModelInterface, GeminiModelConfig
# from .routerapi import RouterAPIModelInterface, RouterAPIModelConfig
# from .together import TogetherModelInterface, TogetherModelConfig

REGISTERED_MODEL = {
    "vllm": {
        "model_cls": VLLMModelInterface,
        "config_cls": VLLMModelConfig,
    },
    "openai": {
        "model_cls": OpenAIModelInterface,
        "config_cls": OpenAIModelConfig
    },
    # "claude": {
    #     "model_cls": ClaudeModelInterface,
    #     "config_cls": ClaudeModelConfig
    # },
    # "gemini": {
    #     "model_cls": GeminiModelInterface,
    #     "config_cls": GeminiModelConfig
    # },
    # # "routerapi": {
    # #     "model_cls": RouterAPIModelInterface,
    # #     "config_cls": RouterAPIModelConfig
    # # },
    # "together": {
    #     "model_cls": TogetherModelInterface,
    #     "config_cls": TogetherModelConfig
    # }
}

try:
    from .claude import ClaudeModelInterface, ClaudeModelConfig
    REGISTERED_MODEL["claude"] = {
        "model_cls": ClaudeModelInterface,
        "config_cls": ClaudeModelConfig
    }
except ImportError:
    pass

try:
    from .gemini import GeminiModelInterface, GeminiModelConfig
    REGISTERED_MODEL["gemini"] = {
        "model_cls": GeminiModelInterface,
        "config_cls": GeminiModelConfig
    }
except ImportError:
    pass

try:
    from .together import TogetherModelInterface, TogetherModelConfig
    REGISTERED_MODEL["together"] = {
        "model_cls": TogetherModelInterface,
        "config_cls": TogetherModelConfig
    }
except ImportError:
    pass