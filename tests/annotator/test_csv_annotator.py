import os
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from annomatic.annotator import (
    HuggingFaceFileAnnotator,
    OpenAiFileAnnotator,
    VllmFileAnnotator,
)
from annomatic.llm import Response, ResponseList
from annomatic.prompt.prompt import Prompt


class OpenAiFileAnnotatorTests(unittest.TestCase):
    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_model_predict = MagicMock(
            return_value=ResponseList.from_responses(
                [Response(answer="answer", data="data", query="query")] * 1,
            ),
        )
        self.mock_load_model = MagicMock(return_value=self.mock_model)
        self.patcher = patch(
            "annomatic.annotator.OpenAiFileAnnotator",
            return_value=self.mock_model,
        )
        self.patcher_predict = patch.object(
            OpenAiFileAnnotator,
            "_model_predict",
            side_effect=self.mock_model_predict,
        )
        self.patcher_load_model = patch.object(
            OpenAiFileAnnotator,
            "_load_model",
            side_effect=self.mock_load_model,
        )
        self.patcher.start()
        self.patcher_predict.start()
        self.patcher_load_model.start()

    def tearDown(self):
        self.patcher.stop()
        self.patcher_predict.stop()
        self.patcher_load_model.stop()

    def test_OpenAiAnnotation_annotate(self):
        # delete file if exists
        try:
            os.remove(
                "./tests/data/output.csv",
            )
        except OSError:
            pass

        annotator = OpenAiFileAnnotator(
            model_name="model",
            out_path="./tests/data/output.csv",
        )
        annotator.batch_size = 1
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

        self.mock_model._num_batches = annotator._num_batches

        annotator.annotate()

        assert os.path.exists(
            "./tests/data/output.csv",
        )

        output = pd.read_csv(
            "./tests/data/output.csv",
        )
        assert output.shape[0] == data.shape[0]

    def test_openai_annotate_batch(self):
        # delete file if exists
        try:
            os.remove(
                "./tests/data/output.csv",
            )
        except OSError:
            pass
        inp = pd.read_csv(
            "./tests/data/input.csv",
        )

        annotator = OpenAiFileAnnotator(
            model_name="model",
            out_path="./tests/data/output.csv",
        )

        template = (
            "Instruction: '{input}'"
            "\n\n"
            "Classify the sentence above as PERSUASIVE TECHNIQUES "
            "or NO PERSUASIVE TECHNIQUES."
            "\n\n"
            "Output: "
        )
        annotator.set_prompt(prompt=template)
        annotator.data_variable = "input"
        res = annotator._annotate_batch(inp)

        assert len(res) == 1


class HuggingFaceTests(unittest.TestCase):
    def setUp(self):
        # Create a mock and replace the original implementation
        self.mock_model = MagicMock()
        self.mock_model_predict = MagicMock(
            return_value=ResponseList.from_responses(
                [Response(answer="answer", data="data", query="query")] * 5,
            ),
        )
        self.mock_load_model = MagicMock(return_value=self.mock_model)
        self.patcher = patch(
            "annomatic.annotator.HuggingFaceFileAnnotator",
            return_value=self.mock_model,
        )
        self.patcher_predict = patch.object(
            HuggingFaceFileAnnotator,
            "_model_predict",
            side_effect=self.mock_model_predict,
        )
        self.patcher_load_model = patch.object(
            HuggingFaceFileAnnotator,
            "_load_model",
            side_effect=self.mock_load_model,
        )
        self.patcher.start()
        self.patcher_predict.start()
        self.patcher_load_model.start()

    def tearDown(self):
        self.patcher.stop()
        self.patcher_predict.stop()
        self.patcher_load_model.stop()

    def test_Huggingface_annotate(self):
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

        annotator = HuggingFaceFileAnnotator(
            model_name="model",
            out_path="./tests/data/output.csv",
            labels=["BIASED", "NON-BIASED"],
        )
        annotator.batch_size = 5

        template = (
            "Instruction: '{input}'"
            "\n\n"
            "Classify the sentence above as PERSUASIVE TECHNIQUES "
            "or NO PERSUASIVE TECHNIQUES."
            "\n\n"
            "Output: "
        )
        annotator.set_prompt(prompt=template)
        annotator.set_data(
            data=data,
            data_variable="input",
        )

        annotator.annotate()
        assert os.path.exists(
            "./tests/data/output.csv",
        )

        output = pd.read_csv(
            "./tests/data/output.csv",
        )
        assert output.shape[0] == data.shape[0]

    def test_huggingface_annotate_batch(self):
        # delete file if exists
        try:
            os.remove(
                "./tests/data/output.csv",
            )
        except OSError:
            pass

        inp = pd.read_csv(
            "./tests/data/input.csv",
        )

        annotator = HuggingFaceFileAnnotator(
            model_name="model",
            out_path="./tests/data/output.csv",
        )
        template = (
            "Instruction: '{input}'"
            "\n\n"
            "Classify the sentence above as PERSUASIVE TECHNIQUES "
            "or NO PERSUASIVE TECHNIQUES."
            "\n\n"
            "Output: "
        )
        annotator.set_prompt(prompt=template)
        annotator.data_variable = "input"
        res = annotator._annotate_batch(inp)

        assert len(res) == inp.shape[0]


class VllmFileAnnotatorTests(unittest.TestCase):
    def setUp(self):
        # Create a mock and replace the original implementation
        self.mock_model = MagicMock()
        self.mock_model_predict = MagicMock(
            return_value=ResponseList.from_responses(
                [Response(answer="answer", data="data", query="query")] * 5,
            ),
        )
        self.mock_load_model = MagicMock(return_value=self.mock_model)
        self.patcher = patch(
            "annomatic.annotator.VllmFileAnnotator",
            return_value=self.mock_model,
        )
        self.patcher_predict = patch.object(
            VllmFileAnnotator,
            "_model_predict",
            side_effect=self.mock_model_predict,
        )
        self.patcher_load_model = patch.object(
            VllmFileAnnotator,
            "_load_model",
            side_effect=self.mock_load_model,
        )
        self.patcher.start()
        self.patcher_predict.start()
        self.patcher_load_model.start()

    def tearDown(self):
        self.patcher.stop()
        self.patcher_predict.stop()
        self.patcher_load_model.stop()

    def test_vllm_annotate(self):
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

        annotator = VllmFileAnnotator(
            model_name="model",
            out_path="./tests/data/output.csv",
        )
        template = (
            "Instruction: '{input}'"
            "\n\n"
            "Classify the sentence above as PERSUASIVE TECHNIQUES "
            "or NO PERSUASIVE TECHNIQUES."
            "\n\n"
            "Output: "
        )
        annotator.set_prompt(prompt=template)
        annotator.annotate(data=data, in_col="input")
        assert os.path.exists(
            "./tests/data/output.csv",
        )

        output = pd.read_csv(
            "./tests/data/output.csv",
        )
        assert output.shape[0] == data.shape[0]

    def test_vllm_annotate_batch(self):
        # delete file if exists
        try:
            os.remove(
                "./tests/data/output.csv",
            )
        except OSError:
            pass
        inp = pd.read_csv(
            "./tests/data/input.csv",
        )

        annotator = VllmFileAnnotator(
            model_name="model",
            out_path="./tests/data/output.csv",
        )
        template = (
            "Instruction: '{input}'"
            "\n\n"
            "Classify the sentence above as PERSUASIVE TECHNIQUES "
            "or NO PERSUASIVE TECHNIQUES."
            "\n\n"
            "Output: "
        )
        annotator.set_prompt(prompt=template)
        annotator.data_variable = "input"
        res = annotator._annotate_batch(inp)

        assert len(res) == inp.shape[0]
