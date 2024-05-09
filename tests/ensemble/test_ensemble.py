import unittest
from typing import Any, Optional

import pandas as pd
from haystack.components.builders import PromptBuilder

from annomatic.annotator.base import BaseAnnotator
from annomatic.ensemble import AnnotatorEnsemble
from annomatic.io.base import DummyOutput


class MockAnnotator(BaseAnnotator):
    def __init__(self, name, prompt="old prompt"):
        super().__init__(model=name, prompt=prompt)

    def set_input(self, data: Any, data_variable: str):
        pass

    def annotate(
        self,
        data: Optional[Any] = None,
        return_df: bool = False,
        **kwargs,
    ):
        return pd.DataFrame({"text": [1, 2, 3]})


class TestAnnotatorEnsemble(unittest.TestCase):
    def test_from_annotators(self):
        annotator1 = MockAnnotator("mock1")
        annotator2 = MockAnnotator("mock2")

        ensemble = AnnotatorEnsemble.from_annotators(
            annotators=[annotator1, annotator2],
            output=DummyOutput(),
        )

        self.assertIsInstance(ensemble, AnnotatorEnsemble)
        self.assertEqual(len(ensemble.annotators), 2)

    def test_from_models(self):
        from haystack.components.generators import HuggingFaceLocalGenerator

        generator_1 = HuggingFaceLocalGenerator(
            model="google/flan-t5-base",
            task="text2text-generation",
            generation_kwargs={
                "max_new_tokens": 100,
                "temperature": 0.9,
            },
        )

        generator_2 = HuggingFaceLocalGenerator(
            model="mock_model",
            task="text2text-generation",
            generation_kwargs={
                "max_new_tokens": 100,
                "temperature": 0.9,
            },
        )

        ensemble = AnnotatorEnsemble.from_models(
            models=[generator_1, generator_2],
            output=DummyOutput(),
            prompt=PromptBuilder("test prompt"),
        )

        self.assertIsInstance(ensemble, AnnotatorEnsemble)
        self.assertEqual(len(ensemble.annotators), 2)
        self.assertEqual(
            ensemble.annotators[0]._prompt._template_string,
            "test prompt",
        )
        self.assertEqual(
            ensemble.annotators[1]._prompt._template_string,
            "test prompt",
        )
