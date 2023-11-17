import pytest

from annomatic.config import BenchmarkConfigFactory, ConfigFactory


def test_create_openai_config():
    config = ConfigFactory.create("openai", temperature=0.0)

    assert type(config).__name__ == "OpenAiConfig"


def test_create_huggingface_config():
    config = ConfigFactory.create("huggingface", temperature=0.0)
    assert type(config).__name__ == "HuggingFaceConfig"


def test_create_vllm_config():
    config = ConfigFactory.create("vllm", temperature=0.0)
    assert type(config).__name__ == "VllmConfig"


def test_create_huggingface_config_to_dict():
    config = ConfigFactory.create("huggingface", temperature=0.0)
    assert config.to_dict() == {"temperature": 0.0}


def test_create_unknown_model():
    with pytest.raises(
        ValueError,
        match="Unknown model: unknown_model. Available models:"
        " openai, huggingface, vllm",
    ):
        ConfigFactory.create("unknown_model")


def test_create_benchmark_openai_config():
    config = BenchmarkConfigFactory.create("openai")
    assert type(config).__name__ == "OpenAiBenchmarkConfig"


def test_create_benchmark_huggingface_config():
    config = BenchmarkConfigFactory.create("huggingface")
    assert type(config).__name__ == "HuggingFaceBenchmarkConfig"


def test_create_benchmark_vllm_config():
    config = BenchmarkConfigFactory.create("vllm")
    assert type(config).__name__ == "VllmBenchmarkConfig"
