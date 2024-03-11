import os
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from annomatic.annotator import (
    HuggingFaceFileAnnotator,
    OpenAiFileAnnotator,
    VllmFileAnnotator,
)
from annomatic.annotator.annotation import DefaultAnnotation
from annomatic.llm import Response, ResponseList
from annomatic.prompt.prompt import Prompt


class OpenAiFileAnnotatorTests(unittest.TestCase):
    def setUp(self):
        # Mock the model's predict method
        self.mock_model_predict = MagicMock(
            return_value=ResponseList.from_responses(
                [Response(answer="answer", data="data", query="query")],
            ),
        )

        # Mock the model and assign the predict method
        self.mock_model = MagicMock()
        self.mock_model.predict = self.mock_model_predict

        self.mock_model_loader = MagicMock()
        self.mock_model_loader.load_model.return_value = self.mock_model

        # Patch the HuggingFaceModelLoader to return the mocked model loader
        self.patcher_model_loader = patch(
            "annomatic.llm.openai.loader.OpenAiModelLoader",
            return_value=self.mock_model_loader,
        )
        self.patcher_model_loader.start()

    def tearDown(self):
        self.patcher_model_loader.stop()

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
            model_loader=self.mock_model_loader,
            annotation_process=DefaultAnnotation(batch_size=1),
            labels=["PERSUASIVE TECHNIQUES", "NO PERSUASIVE TECHNIQUES"],
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
            model_loader=self.mock_model_loader,
            annotation_process=DefaultAnnotation(),
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
        res = annotator.annotation_process._annotate_batch(
            model=self.mock_model,
            batch=inp,
            prompt=Prompt(template),
            data_variable="input",
        )

        assert len(res) == 1


class HuggingFaceTests(unittest.TestCase):
    def setUp(self):
        # Mock the model's predict method
        self.mock_model_predict = MagicMock(
            return_value=ResponseList.from_responses(
                [Response(answer="answer", data="data", query="query")] * 5,
            ),
        )

        # Mock the model and assign the predict method
        self.mock_model = MagicMock()
        self.mock_model.predict = self.mock_model_predict

        self.mock_model_loader = MagicMock()
        self.mock_model_loader.load_model.return_value = self.mock_model

        # Patch the HuggingFaceModelLoader to return the mocked model loader
        self.patcher_model_loader = patch(
            "annomatic.llm.huggingface.loader.HuggingFaceModelLoader",
            return_value=self.mock_model_loader,
        )
        self.patcher_model_loader.start()

    def tearDown(self):
        self.patcher_model_loader.stop()

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
            labels=["PERSUASIVE TECHNIQUES", "NO PERSUASIVE TECHNIQUES"],
            model_loader=self.mock_model_loader,
            annotation_process=DefaultAnnotation(),
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
            model_loader=self.mock_model_loader,
            annotation_process=DefaultAnnotation(),
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
        res = annotator.annotation_process._annotate_batch(
            model=self.mock_model,
            batch=inp,
            prompt=Prompt(template),
            data_variable="input",
        )
        assert len(res) == inp.shape[0]


class VllmFileAnnotatorTests(unittest.TestCase):
    def setUp(self):
        # Mock the model's predict method
        self.mock_model_predict = MagicMock(
            return_value=ResponseList.from_responses(
                [Response(answer="answer", data="data", query="query")] * 5,
            ),
        )

        # Mock the model and assign the predict method
        self.mock_model = MagicMock()
        self.mock_model.predict = self.mock_model_predict

        self.mock_model_loader = MagicMock()
        self.mock_model_loader.load_model.return_value = self.mock_model

        # Patch the HuggingFaceModelLoader to return the mocked model loader
        self.patcher_model_loader = patch(
            "annomatic.llm.vllm.loader.VllmModelLoader",
            return_value=self.mock_model_loader,
        )
        self.patcher_model_loader.start()

    def tearDown(self):
        self.patcher_model_loader.stop()

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
            labels=["PERSUASIVE TECHNIQUES", "NO PERSUASIVE TECHNIQUES"],
            model_loader=self.mock_model_loader,
            annotation_process=DefaultAnnotation(),
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
            model_loader=self.mock_model_loader,
            annotation_process=DefaultAnnotation(),
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
        res = annotator.annotation_process._annotate_batch(
            model=self.mock_model,
            batch=inp,
            prompt=Prompt(template),
            data_variable="input",
        )
        assert len(res) == inp.shape[0]
