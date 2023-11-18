from importlib import import_module
from typing import Any, Dict

from annomatic.config.base import ModelConfig


class ConfigFactory:
    CONFIG_PATH = "annomatic.config.base"

    CONFIG_CLASSES = {
        "openai": "OpenAiConfig",
        "huggingface": "HuggingFaceConfig",
        "vllm": "VllmConfig",
    }

    @staticmethod
    def _create(
        model_type: str,
        config_classes: Dict[str, str],
        config_path: str,
        **kwargs,
    ):
        try:
            module = import_module(config_path)
            config_class = getattr(
                module,
                config_classes.get(model_type.lower(), "unknown_model"),
            )
            return config_class(**kwargs)
        except Exception:
            available_models = ", ".join(config_classes.keys())
            raise ValueError(
                f"Unknown model: {model_type}."
                f" Available models: {available_models}",
            )

    @staticmethod
    def create(model_type: str, **kwargs) -> ModelConfig:
        return ConfigFactory._create(
            model_type,
            ConfigFactory.CONFIG_CLASSES,
            ConfigFactory.CONFIG_PATH,
            **kwargs,
        )


class BenchmarkConfigFactory(ConfigFactory):
    CONFIG_PATH = "annomatic.config.benchmark"
    CONFIG_CLASSES = {
        "openai": "OpenAiBenchmarkConfig",
        "huggingface": "HuggingFaceBenchmarkConfig",
        "vllm": "VllmBenchmarkConfig",
    }

    @staticmethod
    def create(model_type: str, **kwargs) -> ModelConfig:
        return BenchmarkConfigFactory._create(
            model_type,
            BenchmarkConfigFactory.CONFIG_CLASSES,
            BenchmarkConfigFactory.CONFIG_PATH,
            **kwargs,
        )
