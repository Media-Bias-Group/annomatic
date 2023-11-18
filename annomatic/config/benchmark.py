from typing import Any, Dict, Optional

from annomatic.config.base import HuggingFaceConfig, OpenAiConfig, VllmConfig


class OpenAiBenchmarkConfig(OpenAiConfig):
    """
    OpenAiBenchmarkConfig is a class that holds the configuration for
    the HuggingFace models that use for the benchmarking of the models.

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
        load_args: Optional[Dict[str, Any]] = None,
        token_args: Optional[Dict[str, Any]] = None,
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
            load_args=load_args,
            token_args=token_args,
            **kwargs,
        )


class VllmBenchmarkConfig(VllmConfig):
    """
    VllmBenchmarkConfig is a class that holds the configuration for
    the HuggingFace models that use for the benchmarking of the models.

    The temperature is set to 0.2 and do_sample to True.

    """

    def __init__(
        self,
        load_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        if kwargs.get("temperature") is not None:
            raise ValueError(
                "Temperature should not be modified " "for benchmarking!",
            )

        super().__init__(
            temperature=0.2,
            load_args=load_args,
            **kwargs,
        )
