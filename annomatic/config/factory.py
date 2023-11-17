from importlib import import_module

from annomatic.config.base import ModelConfig


class ConfigFactory:
    CONFIG_PATH = {
        "openai": "annomatic.llm.openai.config",
        "huggingface": "annomatic.llm.huggingface.config",
        "vllm": "annomatic.llm.vllm.config",
    }

    CONFIG_CLASSES = {
        "openai": "OpenAiConfig",
        "huggingface": "HuggingFaceConfig",
        "vllm": "VllmConfig",
    }

    @staticmethod
    def _create(model_type: str, config_classes, **kwargs):
        module_path = ConfigFactory.CONFIG_PATH.get(model_type.lower())
        if module_path is not None:
            module = import_module(module_path)
            config_class = getattr(
                module,
                config_classes.get(model_type.lower()),
            )
            return config_class(**kwargs)
        else:
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
            **kwargs,
        )


class BenchmarkConfigFactory(ConfigFactory):
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
            **kwargs,
        )
