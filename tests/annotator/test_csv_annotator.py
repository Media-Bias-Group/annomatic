import os
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from annomatic.annotator import (
    FileAnnotator,
    HuggingFaceFileAnnotator,
    OpenAiFileAnnotator,
    VllmFileAnnotator,
)
from annomatic.annotator.annotation import DefaultAnnotation
from annomatic.llm import Response, ResponseList
from annomatic.prompt.prompt import Prompt


class OpenAiFileAnnotatorTests(unittest.TestCase):
    def test_OpenAiAnnotation_annotate(self):
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

        # Mock the model's predict method
        self.mock_model_predict = MagicMock(
            return_value=mock_result,
        )

        self.mock_model = MagicMock()
        self.mock_model.run = self.mock_model_predict

        # delete file if exists
        try:
            os.remove(
                "./tests/data/output.csv",
            )
        except OSError:
            pass

        annotator = FileAnnotator(
            out_path="./tests/data/output.csv",
            labels=["BIASED", "NOT BIASED"],
            out_format="csv",
            model=self.mock_model,
            annotation_process=DefaultAnnotation(),
        )
        data = pd.read_csv(
            "./tests/data/input.csv",
        )

        template = (
            "Instruction: '{input}'"
            "\n\n"
            "Classify the sentence above as PERSUASIVE TECHNIQUES "
            "or NO PERSUASIVE TECHNIQUES."
            "\n\n"
            "Output: "
        )
        prompt = Prompt(content=template)
        annotator.set_prompt(prompt=prompt)
        annotator.set_data(
            data=data,
            data_variable="input",
        )

        res = annotator.annotate(return_df=True)

        assert len(res) == data.shape[0]

        assert os.path.exists(
            "./tests/data/output.csv",
        )

        output = pd.read_csv(
            "./tests/data/output.csv",
        )
        assert output.shape[0] == data.shape[0]

    def test_openai_annotate_batch(self):
        mock_result = {
            "replies": [
                "NOT BIASED",
                "BIASED",
                "NOT BIASED",
                "BIASED",
                "NOT BIASED",
            ],
            "meta": [
                {
                    "model": "gpt-3.5-turbo-0125",
                    "index": 0,
                    "finish_reason": "stop",
                    "usage": {
                        "completion_tokens": 0,
                        "prompt_tokens": 0,
                        "total_tokens": 0,
                    },
                },
            ],
        }

        # Mock the model's predict method
        self.mock_model_predict = MagicMock(
            return_value=mock_result,
        )

        self.mock_model = MagicMock()
        self.mock_model.run = self.mock_model_predict

        # delete file if exists
        try:
            os.remove(
                "./tests/data/output.csv",
            )
        except OSError:
            pass

        data = pd.read_csv(
            "./tests/data/input.csv",
        )

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
            "or NO NOT BIASED."
            "\n\n"
            "Output: "
        )
        annotator.set_prompt(prompt=template)
        annotator.data_variable = "input"
        res = annotator.annotation_process._annotate_batch(
            model=self.mock_model,
            batch=data,
            prompt=Prompt(template),
            data_variable="input",
            return_df=True,
        )

        assert len(res) == 5
