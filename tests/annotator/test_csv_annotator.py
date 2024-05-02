import os
import unittest
from unittest.mock import MagicMock

import pandas as pd
from haystack.components.builders import PromptBuilder

from annomatic.annotator import FileAnnotator
from annomatic.annotator.annotation import DefaultAnnotation


class FileAnnotatorTests(unittest.TestCase):
    def test_FileAnnotator_annotate(self):
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
            output="./tests/data/output.csv",
            labels=["BIASED", "NOT BIASED"],
            model=self.mock_model,
            annotation_process=DefaultAnnotation(),
        )
        data = pd.read_csv(
            "./tests/data/input.csv",
        )

        template = (
            "Instruction: '{{input}}'"
            "\n\n"
            "Classify the sentence above as BIASED "
            "or NO BIASED."
            "\n\n"
            "Output: "
        )
        prompt = PromptBuilder(template)
        annotator.set_prompt(prompt=prompt)
        annotator.set_input(
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

    def test_FileAnnotator_annotate_batch(self):
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
            output="./tests/data/output.csv",
            labels=["BIASED", "NOT BIASED"],
            model=self.mock_model,
            annotation_process=DefaultAnnotation(),
        )

        template = (
            "Instruction: '{{input}}'"
            "\n\n"
            "Classify the sentence above as BIASED "
            "or NO BIASED."
            "\n\n"
            "Output: "
        )
        annotator.data_variable = "input"
        res = annotator.annotation_process._annotate_batch(
            model=self.mock_model,
            batch=data,
            prompt=PromptBuilder(template),
            data_variable="input",
            return_df=True,
        )

        assert len(res) == 5
