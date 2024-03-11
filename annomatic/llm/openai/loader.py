from abc import ABC
from typing import Any, Dict, List, Optional

from annomatic.config.base import OpenAiConfig
from annomatic.llm.base import Model, ModelLoader


class OpenAiModelLoader(ModelLoader, ABC):
    """
    Model loader for OpenAI models.

    This class is responsible for loading OpenAI models.

    Attributes:
        model_name (str): The name of the model.
        config (OpenAiConfig): The configuration of the model.
        system_prompt (Optional[str]): The system prompt.
        lib_args (Optional[Dict[str, Any]]): The library arguments.
    """

    DEFAULT_BATCH_SIZE = 1

    def __init__(
        self,
        model_name: str,
        config: Optional[OpenAiConfig] = None,
        model_args: Optional[Dict[str, Any]] = None,
        generation_args: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        batch_size: Optional[int] = DEFAULT_BATCH_SIZE,
        api_key: str = "",
        labels: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            config=config or OpenAiConfig(),
            system_prompt=system_prompt,
            lib_args={"api_key": api_key},
            batch_size=batch_size,
            labels=labels,
            **kwargs,
        )

        if hasattr(self.config, "model_args"):
            self.config.model_args = getattr(self.config, "model_args", {})
            self.config.model_args.update(model_args or {})

        self.update_config_generation_args(generation_args)

    def load_model(self) -> Model:
        from annomatic.llm.openai import OpenAiModel

        if not isinstance(self.config, OpenAiConfig):
            raise ValueError(
                "OpenAI models require a OpenAiConfig object.",
            )

        api_key = self.lib_args.get("api_key", None)
        return OpenAiModel(
            model_name=self.model_name,
            api_key=api_key,
            system_prompt=self.system_prompt,
            generation_args=self.config.to_dict(),
        )
