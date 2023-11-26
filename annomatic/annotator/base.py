from abc import ABC, abstractmethod
from typing import Optional

from annomatic.config.base import (
    HuggingFaceConfig,
    ModelConfig,
    OpenAiConfig,
    VllmConfig,
)
from annomatic.llm.base import Model


class BaseAnnotator(ABC):
    """
    Base class for annotator classes
    """

    @abstractmethod
    def annotate(self, **kwargs):
        """
        Annotates the input data and stores the annotated data.

        Args:
            kwargs: a dict containing the input variables for prompt templates
        """
        raise NotImplementedError()


class ModelLoadMixin(ABC):
    """
    Mixin for annotator to load a model from different libraries.
    """

    def __init__(self):
        self._model: Optional[Model] = None

    def _load_model(
        self,
        model_name: str,
        model_lib: str,
        config: ModelConfig,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Model:
        """
        Loads the model and store it in self.model.

        Returns:
            The loaded model.
        """

        if model_lib == "openai":
            from annomatic.llm.openai import OpenAiModel

            if not isinstance(config, OpenAiConfig):
                raise ValueError(
                    "OpenAI models require a OpenAiConfig object.",
                )

            api_key = kwargs.get("api_key", None)
            self._model = OpenAiModel(
                model_name=model_name,
                api_key=api_key,
                generation_args=config.to_dict(),
            )
            return self._model

        elif model_lib == "huggingface":
            if not isinstance(config, HuggingFaceConfig):
                raise ValueError(
                    "Huggingface models require a HuggingfaceConfig object.",
                )

            model_args = config.model_args
            tokenizer_args = config.tokenizer_args
            generation_args = config.to_dict()
            auto_model = kwargs.get("auto_model", "AutoModelForCausalLM")
            use_chat_template = kwargs.get("use_chat_template", False)

            if auto_model == "AutoModelForCausalLM":
                from annomatic.llm.huggingface import HFAutoModelForCausalLM

                self._model = HFAutoModelForCausalLM(
                    model_name=model_name,
                    model_args=model_args,
                    tokenizer_args=tokenizer_args,
                    generation_args=generation_args,
                    system_prompt=system_prompt,
                    use_chat_template=use_chat_template,
                )
                return self._model
            elif auto_model == "AutoModelForSeq2SeqLM":
                from annomatic.llm.huggingface import HFAutoModelForSeq2SeqLM

                self._model = HFAutoModelForSeq2SeqLM(
                    model_name=model_name,
                    model_args=model_args,
                    tokenizer_args=tokenizer_args,
                    generation_args=generation_args,
                    system_prompt=system_prompt,
                    use_chat_template=use_chat_template,
                )
                return self._model
            else:
                raise ValueError(
                    "auto_model must be either "
                    "'AutoModelForCausalLM' or 'AutoModelForSeq2SeqLM')",
                )
        elif model_lib == "vllm":
            from annomatic.llm.vllm import VllmModel

            if not isinstance(config, VllmConfig):
                raise ValueError(
                    "VLLM models require a VllmConfig object.",
                )
            _model = VllmModel(
                model_name=model_name,
                model_args=config.model_args,
                generation_args=config.to_dict(),
                system_prompt=system_prompt,
            )

            return _model
        else:
            raise ValueError(
                f"Model library {model_lib} not supported."
                f" Please choose from 'openai', 'huggingface', or 'vllm'.",
            )
