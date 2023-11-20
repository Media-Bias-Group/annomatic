from typing import Any, Dict, Optional

from annomatic.config.base import HuggingFaceConfig, OpenAiConfig, VllmConfig


class OpenAiBenchmarkConfig(OpenAiConfig):
    """
    OpenAiBenchmarkConfig is a class that holds the configuration for
    the OpenAi models that use for the benchmarking of the models.

    The temperature is set to 0.2.

    """

    def __init__(self, **kwargs):
        if kwargs.get("temperature") is not None:
            raise ValueError("Temperature should not be set for benchmarking!")

        super().__init__(temperature=0.2, **kwargs)


class HuggingFaceBenchmarkConfig(HuggingFaceConfig):
    """
    HuggingFaceBenchmarkConfig is a class that holds the configuration for
    the HuggingFace models that use for the benchmarking of the models.

    The temperature is set to 0.2 and do_sample to True.
    """

    def __init__(
        self,
        model_args: Optional[Dict[str, Any]] = None,
        tokenizer_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        if (
            kwargs.get("temperature") is not None
            or kwargs.get("do_sample") is not None
        ):
            raise ValueError(
                "Temperature should not be modified for benchmarking!",
            )
        super().__init__(
            temperature=0.2,
            do_sample=True,
            model_args=model_args,
            tokenizer_args=tokenizer_args,
            **kwargs,
        )


class VllmBenchmarkConfig(VllmConfig):
    """
    VllmBenchmarkConfig is a class that holds the configuration for
    the Vllm models that use for the benchmarking of the models.

    The temperature is set to 0.2 and do_sample to True.

    """

    def __init__(
        self,
        model_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        if kwargs.get("temperature") is not None:
            raise ValueError(
                "Temperature should not be modified " "for benchmarking!",
            )

        super().__init__(
            temperature=0.2,
            model_args=model_args,
            **kwargs,
        )
