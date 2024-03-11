from abc import ABC
from typing import Any, Dict, List, Optional

from annomatic.config.base import HuggingFaceConfig
from annomatic.llm.base import Model, ModelLoader


class HuggingFaceModelLoader(ModelLoader, ABC):
    """
    Model loader for HuggingFace models.

    Attributes:
        model_name (str): The name of the model.
        config (HuggingFaceConfig): The configuration of the model.
        system_prompt (Optional[str]): The system prompt.
        lib_args (Optional[Dict[str, Any]]): The library arguments.
    """

    DEFAULT_BATCH_SIZE = 5

    def __init__(
        self,
        model_name: str,
        config: Optional[HuggingFaceConfig] = None,
        model_args: Optional[Dict[str, Any]] = None,
        tokenizer_args: Optional[Dict[str, Any]] = None,
        generation_args: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        batch_size: Optional[int] = DEFAULT_BATCH_SIZE,
        auto_model: str = "AutoModelForCausalLM",
        use_chat_template: bool = False,
        labels: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            config=config or HuggingFaceConfig(),
            system_prompt=system_prompt,
            lib_args={
                "auto_model": auto_model,
                "use_chat_template": use_chat_template,
            },
            batch_size=batch_size,
            labels=labels,
            **kwargs,
        )

        if hasattr(self.config, "model_args"):
            self.config.model_args = getattr(self.config, "model_args", {})
            self.config.model_args.update(model_args or {})

        if hasattr(self.config, "tokenizer_args"):
            self.config.tokenizer_args = (
                getattr(
                    self.config,
                    "tokenizer_args",
                    {},
                )
                or {}
            )
            self.config.tokenizer_args.update(tokenizer_args or {})

        self.update_config_generation_args(generation_args)

    def load_model(self) -> Model:
        if not isinstance(self.config, HuggingFaceConfig):
            raise ValueError(
                "Huggingface models require a HuggingfaceConfig object.",
            )

        model_args = self.config.model_args
        tokenizer_args = self.config.tokenizer_args
        generation_args = self.config.to_dict()
        auto_model = self.lib_args.get("auto_model", "AutoModelForCausalLM")
        use_chat_template = self.lib_args.get("use_chat_template", False)

        if auto_model == "AutoModelForCausalLM":
            from annomatic.llm.huggingface import HFAutoModelForCausalLM

            return HFAutoModelForCausalLM(
                model_name=self.model_name,
                model_args=model_args,
                tokenizer_args=tokenizer_args,
                generation_args=generation_args,
                system_prompt=self.system_prompt,
                use_chat_template=use_chat_template,
            )
        elif auto_model == "AutoModelForSeq2SeqLM":
            from annomatic.llm.huggingface import HFAutoModelForSeq2SeqLM

            return HFAutoModelForSeq2SeqLM(
                model_name=self.model_name,
                model_args=model_args,
                tokenizer_args=tokenizer_args,
                generation_args=generation_args,
                system_prompt=self.system_prompt,
                use_chat_template=use_chat_template,
            )
        else:
            raise ValueError(
                "auto_model must be either "
                "'AutoModelForCausalLM' or 'AutoModelForSeq2SeqLM')",
            )
