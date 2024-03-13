import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from annomatic.annotator import FileAnnotator
from annomatic.annotator.annotation import DefaultAnnotation
from annomatic.llm import Response, ResponseList
from annomatic.prompt import Prompt
from annomatic.retriever import DiversityRetriever

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


class YourTestClass(unittest.TestCase):
    def setUp(self):
        # Mock the model's predict method
        self.mock_model_predict = MagicMock(
            return_value=mock_result,
        )

        self.mock_model = MagicMock()
        self.mock_model.run = self.mock_model_predict

    def test_fill_prompt_without_examples(self):
        df = pd.DataFrame(
            {"text": ["This is a test sentence."]},
        )

        annotator = FileAnnotator(
            out_path="./tests/data/output.csv",
            labels=["BIASED", "NOT BIASED"],
            out_format="csv",
            model=self.mock_model,
            annotation_process=DefaultAnnotation(),
        )
        annotator.data_variable = "text"
        prompt = Prompt()
        prompt.add_part("Instruction: '{text}'")
        prompt.add_labels_part("Classify the sentence above as {label}.")
        prompt.add_part("Output: ")

        annotator.set_prompt(prompt)
        message = annotator.annotation_process.fill_prompt(
            batch=df,
            prompt=prompt,
            data_variable="text",
            label=["BIASED", "NOT BIASED"],
        )

        assert (
            message == "Instruction: 'This is a test sentence.'\n\n"
            "Classify the sentence above as BIASED "
            "or NOT BIASED.\n\nOutput: "
        )

    def test_fill_prompt_with_examples(self):
        annotator = FileAnnotator(
            out_path="./tests/data/output.csv",
            labels=["BIASED", "NOT BIASED"],
            out_format="csv",
            model=self.mock_model,
            annotation_process=DefaultAnnotation(),
        )

        df = pd.DataFrame(
            {"text": ["This is a test sentence."]},
        )

        df_examples = pd.DataFrame(
            {
                "text": [
                    "This is a examples.",
                    "This is a second examples.",
                ],
                "label": ["BIASED", "NOT BIASED"],
            },
        )
        annotator.data_variable = "text"

        prompt = Prompt()
        prompt.add_part("Instruction: '{text}'")
        prompt.add_labels_part("Classify the sentence above as {label}.")
        prompt.add_part("Output: ")

        annotator.set_prompt(prompt)
        annotator.set_context(df_examples)
        message = annotator.annotation_process.fill_prompt(
            batch=df,
            prompt=prompt,
            data_variable="text",
            label=["BIASED", "NOT BIASED"],
        )

        assert (
            message == "Instruction: 'This is a examples.'\n\n"
            "Classify the sentence above as BIASED or NOT BIASED."
            "\n\nOutput: BIASED\n\n"
            "Instruction: 'This is a second examples.'\n\n"
            "Classify the sentence above as BIASED or NOT BIASED."
            "\n\nOutput: NOT BIASED\n\n"
            "Instruction: 'This is a test sentence.'\n\n"
            "Classify the sentence above as BIASED or NOT BIASED.\n\nOutput: "
        )

    def test_fill_prompt_with_Retriever(self):
        annotator = FileAnnotator(
            out_path="./tests/data/output.csv",
            labels=["BIASED", "NOT BIASED"],
            out_format="csv",
            model=self.mock_model,
            annotation_process=DefaultAnnotation(),
        )

        df = pd.DataFrame(
            {"text": ["This is a test sentence."]},
        )

        df_examples = pd.DataFrame(
            {
                "text": ["This is a examples.", "This is a test example."],
                "label": ["BIASED", "N0T BIASED"],
            },
        )

        retriever = DiversityRetriever(
            k=1,
            pool=df_examples,
            text_variable="text",
            label_variable="label",
        )
        annotator.data_variable = "text"

        prompt = Prompt()
        prompt.add_part("Instruction: '{text}'")
        prompt.add_labels_part("Classify the sentence above as {label}.")
        prompt.add_part("Output: ")
        annotator.set_prompt(prompt)

        annotator.set_context(retriever)
        message = annotator.annotation_process.fill_prompt(
            batch=df,
            prompt=prompt,
            data_variable="text",
            label=["BIASED", "NOT BIASED"],
        )

        assert (
            message == "Instruction: 'This is a test example.'\n\n"
            "Classify the sentence above as BIASED or NOT BIASED."
            "\n\nOutput: N0T BIASED\n\n"
            "Instruction: 'This is a test sentence.'\n\n"
            "Classify the sentence above as BIASED or NOT BIASED."
            "\n\nOutput: "
        )

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
            "Instruction: '{input}'"
            "\n\n"
            "Classify the sentence above as PERSUASIVE TECHNIQUES "
            "or NO PERSUASIVE TECHNIQUES or as {extra}."
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

    def test_set_data_prompt_raise_value_error(self):
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
            "Classify the sentence above as PERSUASIVE TECHNIQUES "
            "or NO PERSUASIVE TECHNIQUES or as {extra}."
            "\n\n"
            "Output: "
        )
        annotator.set_prompt(prompt=template)
        df = pd.read_csv(
            "./tests/data/input.csv",
        )
        with pytest.raises(ValueError) as e_info:
            annotator.set_data(
                data=df,
                data_variable="?",
            )
            raise e_info

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
            "Classify the sentence above as PERSUASIVE TECHNIQUES "
            "or NO PERSUASIVE TECHNIQUES or as {extra}."
            "\n\n"
            "Output: "
        )
        annotator.set_prompt(prompt=template)
        assert isinstance(annotator._prompt, Prompt)

    def test_set_prompt_prompt(self):
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
            "Classify the sentence above as PERSUASIVE TECHNIQUES "
            "or NO PERSUASIVE TECHNIQUES or as {extra}."
            "\n\n"
            "Output: "
        )
        prompt = Prompt(content=template)
        annotator.set_prompt(prompt=prompt)
        assert isinstance(annotator._prompt, Prompt)

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

    def test_validate_labels_only_init(self):
        annotator = FileAnnotator(
            out_path="./tests/data/output.csv",
            labels=["BIASED", "NOT BIASED"],
            out_format="csv",
            model=self.mock_model,
            annotation_process=DefaultAnnotation(),
        )
        annotator._prompt = Prompt("This is a prompt")
        annotator._validate_labels()

        assert annotator._labels == ["BIASED", "NOT BIASED"]

    def test_validate_labels_only_prompt(self):
        annotator = FileAnnotator(
            out_path="./tests/data/output.csv",
            labels=["BIASED", "NOT BIASED"],
            out_format="csv",
            model=self.mock_model,
            annotation_process=DefaultAnnotation(),
        )

        prompt = Prompt()
        prompt.add_labels_part(
            content="This is a prompt {labels}",
            label_var="labels",
        )
        annotator._prompt = prompt
        annotator._validate_labels(labels=["BIASED", "NOT BIASED"])

        assert annotator._labels == ["BIASED", "NOT BIASED"]

    def test_validate_labels_matching(self):
        annotator = FileAnnotator(
            out_path="./tests/data/output.csv",
            labels=["BIASED", "NOT BIASED"],
            out_format="csv",
            model=self.mock_model,
            annotation_process=DefaultAnnotation(),
        )

        prompt = Prompt()
        prompt.add_labels_part(
            content="This is a prompt {label}",
            label_var="label",
        )
        annotator._prompt = prompt
        annotator._validate_labels(label=["BIASED", "NOT BIASED"])

        assert annotator._labels == ["BIASED", "NOT BIASED"]

    def test_validate_labels_not_matching(self):
        annotator = FileAnnotator(
            out_path="./tests/data/output.csv",
            labels=["BIASED", "NOT BIASED"],
            out_format="csv",
            model=self.mock_model,
            annotation_process=DefaultAnnotation(),
        )

        prompt = Prompt()
        prompt.add_labels_part(
            content="This is a prompt {label}",
            label_var="label",
        )
        annotator._prompt = prompt

        with pytest.raises(Exception):
            annotator._validate_labels(label=["BIASED", "Other Label"])
