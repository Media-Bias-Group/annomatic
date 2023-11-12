from annomatic.llm.base import ModelConfig
from annomatic.llm.huggingface.config import (
    HuggingFaceBenchmarkConfig,
    HuggingFaceConfig,
)
from annomatic.llm.openai.config import OpenAiBenchmarkConfig, OpenAiConfig
from annomatic.llm.vllm.config import VllmBenchmarkConfig, VllmConfig


class ConfigFactory:
    CONFIG_CLASSES = {
        "openai": OpenAiConfig,
        "huggingface": HuggingFaceConfig,
        "vllm": VllmConfig,
    }

    @staticmethod
    def create_config(model_type: str, **kwargs) -> ModelConfig:
        return ConfigFactory._create_config(
            model_type,
            ConfigFactory.CONFIG_CLASSES,
            **kwargs,
        )

    @staticmethod
    def _create_config(model_type: str, config_classes, **kwargs):
        config_class = config_classes.get(model_type)
        if config_class is not None:
            return config_class(**kwargs)
        else:
            raise ValueError(f"Unknown model: {model_type}")


class BenchmarkConfigFactory(ConfigFactory):
    CONFIG_CLASSES = {
        "openAi": OpenAiBenchmarkConfig,
        "huggingface": HuggingFaceBenchmarkConfig,
        "vllm": VllmBenchmarkConfig,
    }

    @staticmethod
    def create_config(model_type: str, **kwargs) -> ModelConfig:
        return BenchmarkConfigFactory._create_config(
            model_type,
            BenchmarkConfigFactory.CONFIG_CLASSES,
            **kwargs,
        )
