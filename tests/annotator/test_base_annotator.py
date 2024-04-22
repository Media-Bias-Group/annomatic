import unittest
from unittest.mock import MagicMock

import pandas as pd
import pytest
from haystack.components.builders import PromptBuilder

from annomatic.annotator import FileAnnotator
from annomatic.annotator.annotation import DefaultAnnotation

mock_result = {
    "replies": ["NOT BIASED"],
    "meta": [
        {
            "model": "gpt-3.5-turbo-0125",
            "index": 0,
            "finish_reason": "stop",
            "usage": {
                "completion_tokens": 4,
                "prompt_tokens": 38,
                "total_tokens": 42,
            },
        },
    ],
}


class BaseAnnotatorTests(unittest.TestCase):
    def setUp(self):
        # Mock the model's predict method
        self.mock_model_predict = MagicMock(
            return_value=mock_result,
        )

        self.mock_model = MagicMock()
        self.mock_model.run = self.mock_model_predict

    def test_set_data_csv(self):
        annotator = FileAnnotator(
            out_path="./tests/data/output.csv",
            labels=["BIASED", "NOT BIASED"],
            out_format="csv",
            model=self.mock_model,
            annotation_process=DefaultAnnotation(),
        )
        annotator.set_data(
            data="./tests/data/input.csv",
            data_variable="input",
        )
        assert isinstance(annotator.data, pd.DataFrame)

    def test_set_data_prompt_matching(self):
        annotator = FileAnnotator(
            out_path="./tests/data/output.csv",
            labels=["BIASED", "NOT BIASED"],
            out_format="csv",
            model=self.mock_model,
            annotation_process=DefaultAnnotation(),
        )

        template = (
            "Instruction: '{{input}}'"
            "\n\n"
            "Classify the sentence above as BIASED "
            "or NOT BIASED or as {{extra}}."
            "\n\n"
            "Output: "
        )
        annotator.set_prompt(prompt=template)
        df = pd.read_csv(
            "./tests/data/input.csv",
        )
        annotator.set_data(
            data=df,
            data_variable="input",
        )

        assert annotator.data_variable == "input"

    def test_set_prompt_str(self):
        annotator = FileAnnotator(
            out_path="./tests/data/output.csv",
            labels=["BIASED", "NOT BIASED"],
            out_format="csv",
            model=self.mock_model,
            annotation_process=DefaultAnnotation(),
        )

        template = (
            "Instruction: '{input}'"
            "\n\n"
            "Classify the sentence above as BIASED "
            "or NOT BIASED or as {extra}."
            "\n\n"
            "Output: "
        )
        annotator.set_prompt(prompt=template)
        assert isinstance(annotator._prompt, PromptBuilder)

    def test_set_prompt(self):
        annotator = FileAnnotator(
            out_path="./tests/data/output.csv",
            labels=["BIASED", "NOT BIASED"],
            out_format="csv",
            model=self.mock_model,
            annotation_process=DefaultAnnotation(),
        )

        template = (
            "Instruction: '{{input}}'"
            "\n\n"
            "Classify the sentence above as BIASED "
            "or NOT BIASED or as {{extra}}."
            "\n\n"
            "Output: "
        )
        prompt = PromptBuilder(template)
        annotator.set_prompt(prompt=prompt)
        assert isinstance(annotator._prompt, PromptBuilder)

    def test_soft_parse(self):
        df = pd.DataFrame(
            {
                "response": [
                    "This is a biased response.",
                    "A not biased response.",
                    "Another biased statement.",
                    "This sentence is not parseable.",
                    "This is a biased response or Not biased.",
                ],
            },
        )

        annotator = FileAnnotator(
            out_path="./tests/data/output.csv",
            labels=["BIASED", "NOT BIASED"],
            out_format="csv",
            model=self.mock_model,
            annotation_process=DefaultAnnotation(),
        )

        annotator.post_processor.labels = ["BIASED", "NOT BIASED"]
        annotator.post_processor.process(
            df,
        )

        assert df["label"].iloc[0] == "BIASED"
        assert df["label"].iloc[1] == "NOT BIASED"
        assert df["label"].iloc[2] == "BIASED"
        assert df["label"].iloc[3] == "?"
        assert df["label"].iloc[4] == "?"
