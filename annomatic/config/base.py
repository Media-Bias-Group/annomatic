from abc import ABC, abstractmethod
from typing import Any, Dict


class ModelConfig(ABC):
    """
    Base ModelConfig for LLMs
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initialize the model config.
        """
        self.kwargs = kwargs

    @staticmethod
    def get_default_values() -> Dict[str, Any]:
        """
        Return a dictionary of the default values for the model config.
        """
        raise NotImplementedError()

    def to_dict(self, exclude_kwargs: bool = False) -> Dict[str, Any]:
        """
        Convert the model config to a dictionary.

        Values that are different from the values set in the __init__ method
        are included in the dictionary, and the kwargs are flattened.

        Returns:
            dict: A dictionary representing the model configuration.
        """
        default_values = self.get_default_values()
        config_dict = {}

        for key, value in default_values.items():
            if getattr(self, f"{key}", None) != value:
                config_dict[key] = getattr(self, f"{key}")

        if not exclude_kwargs:
            config_dict.update(self.kwargs.items())

        return config_dict
