import pytest

from annomatic.config.base import HuggingFaceConfig, OpenAiConfig, VllmConfig
from annomatic.config.benchmark import (
    HuggingFaceBenchmarkConfig,
    OpenAiBenchmarkConfig,
    VllmBenchmarkConfig,
)
from annomatic.config.factory import BenchmarkConfigFactory, ConfigFactory


def test_create_openai_config():
    config = ConfigFactory.create("openai", temperature=0.5)
    assert isinstance(config, OpenAiConfig)
    assert config.temperature == 0.5


def test_create_huggingface_config():
    config = ConfigFactory.create("huggingface", max_length=30)
    assert isinstance(config, HuggingFaceConfig)
    assert config.max_length == 30


def test_create_vllm_config():
    config = ConfigFactory.create("vllm", n=2)
    assert isinstance(config, VllmConfig)
    assert config.n == 2


def test_unknown_model_type():
    with pytest.raises(ValueError):
        ConfigFactory.create("unknown_model")


def test_create_openai_benchmark_config():
    config = BenchmarkConfigFactory.create("openai")
    assert isinstance(config, OpenAiBenchmarkConfig)
    assert config.temperature == 0.2


def test_create_huggingface_benchmark_config():
    config = BenchmarkConfigFactory.create("huggingface")
    assert isinstance(config, HuggingFaceBenchmarkConfig)
    assert config.temperature == 0.2
    assert config.do_sample


def test_create_vllm_benchmark_config():
    config = BenchmarkConfigFactory.create("vllm")
    assert isinstance(config, VllmBenchmarkConfig)
    assert config.temperature == 0.2


def test_unknown_model_type_benchmark():
    with pytest.raises(ValueError):
        BenchmarkConfigFactory.create("unknown_model")
