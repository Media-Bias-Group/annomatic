from typing import Any, Dict, List, Optional

from annomatic.config.base import VllmConfig
from annomatic.llm.base import Model, ModelLoader


class VllmModelLoader(ModelLoader):
    """
    Abstract base class for Vllm annotators.

    Attributes:
        model_name (str): The name of the model.
        config (VllmConfig): The configuration of the model.
        system_prompt (Optional[str]): The system prompt.
        lib_args (Optional[Dict[str, Any]]): The library arguments.

    """

    DEFAULT_BATCH_SIZE = 5

    def __init__(
        self,
        model_name: str,
        config: Optional[VllmConfig] = None,
        model_args: Optional[Dict[str, Any]] = None,
        generation_args: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        batch_size: Optional[int] = DEFAULT_BATCH_SIZE,
        labels: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            config=config or VllmConfig(),
            system_prompt=system_prompt,
            lib_args={},
            batch_size=batch_size,
            labels=labels,
            **kwargs,
        )

        if hasattr(self.config, "model_args"):
            self.config.model_args = (
                getattr(
                    self.config,
                    "model_args",
                    {},
                )
                or {}
            )
            self.config.model_args.update(model_args or {})

        self.update_config_generation_args(generation_args)

    def load_model(self) -> Model:
        from annomatic.llm.vllm import VllmModel

        if not isinstance(self.config, VllmConfig):
            raise ValueError(
                "VLLM models require a VllmConfig object.",
            )
        return VllmModel(
            model_name=self.model_name,
            model_args=self.config.model_args,
            generation_args=self.config.to_dict(),
            system_prompt=self.system_prompt,
        )
