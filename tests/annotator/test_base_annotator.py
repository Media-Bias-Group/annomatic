import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from annomatic.annotator import HuggingFaceFileAnnotator
from annomatic.prompt import Prompt
from annomatic.retriever import DiversityRetriever


class YourTestClass(unittest.TestCase):
    def test_fill_prompt_without_examples(self):
        df = pd.DataFrame(
            {"text": ["This is a test sentence."]},
        )

        with patch.object(
            HuggingFaceFileAnnotator,
            "_load_model",
            return_value=MagicMock(),
        ):
            annotator = HuggingFaceFileAnnotator(
                model_name="mock",
                out_path="./tests/data/output.csv",
                labels=["BIASED", "NOT BIASED"],
            )

            annotator.data_variable = "text"
            prompt = Prompt()
            prompt.add_part("Instruction: '{text}'")
            prompt.add_labels_part("Classify the sentence above as {label}.")
            prompt.add_part("Output: ")

            annotator.set_prompt(prompt)
            message = annotator.fill_prompt(
                batch=df,
                label=["BIASED", "NOT BIASED"],
            )

        assert message == [
            "Instruction: 'This is a test sentence.'\n\n"
            "Classify the sentence above as BIASED or NOT BIASED."
            "\n\nOutput: ",
        ]

    def test_fill_prompt_with_examples(self):
        with patch.object(
            HuggingFaceFileAnnotator,
            "_load_model",
            return_value=MagicMock(),
        ):
            annotator = HuggingFaceFileAnnotator(
                model_name="mock",
                out_path="./tests/data/output.csv",
                labels=["BIASED", "NOT BIASED"],
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
            annotator.context = df_examples
            prompt = Prompt()
            prompt.add_part("Instruction: '{text}'")
            prompt.add_labels_part("Classify the sentence above as {label}.")
            prompt.add_part("Output: ")

            annotator.set_prompt(prompt)
            message = annotator.fill_prompt(
                batch=df,
                label=["BIASED", "NOT BIASED"],
            )

        assert message == [
            "Instruction: 'This is a examples.'\n\n"
            "Classify the sentence above as BIASED or NOT BIASED."
            "\n\nOutput: BIASED\n\n"
            "Instruction: 'This is a second examples.'\n\n"
            "Classify the sentence above as BIASED or NOT BIASED."
            "\n\nOutput: NOT BIASED\n\n"
            "Instruction: 'This is a test sentence.'\n\n"
            "Classify the sentence above as BIASED or NOT BIASED."
            "\n\nOutput: ",
        ]

    def test_fill_prompt_with_Retriever(self):
        with patch.object(
            HuggingFaceFileAnnotator,
            "_load_model",
            return_value=MagicMock(),
        ):
            annotator = HuggingFaceFileAnnotator(
                model_name="mock",
                out_path="./tests/data/output.csv",
                labels=["BIASED", "NOT BIASED"],
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
            annotator.context = retriever
            prompt = Prompt()
            prompt.add_part("Instruction: '{text}'")
            prompt.add_labels_part("Classify the sentence above as {label}.")
            prompt.add_part("Output: ")

            annotator.set_prompt(prompt)
            message = annotator.fill_prompt(
                batch=df,
                label=["BIASED", "NOT BIASED"],
            )

            assert message == [
                "Instruction: 'This is a test example.'\n\n"
                "Classify the sentence above as BIASED or NOT BIASED."
                "\n\nOutput: N0T BIASED\n\n"
                "Instruction: 'This is a test sentence.'\n\n"
                "Classify the sentence above as BIASED or NOT BIASED."
                "\n\nOutput: ",
            ]

    def test_set_data_csv(self):
        with patch.object(
            HuggingFaceFileAnnotator,
            "_load_model",
            return_value=MagicMock(),
        ):
            annotator = HuggingFaceFileAnnotator(
                model_name="mock",
                out_path="./tests/data/output.csv",
            )
            annotator.set_data(
                data="./tests/data/input.csv",
                data_variable="input",
            )
            assert isinstance(annotator.data, pd.DataFrame)

    def test_set_data_prompt_matching(self):
        with patch.object(
            HuggingFaceFileAnnotator,
            "_load_model",
            return_value=MagicMock(),
        ):
            annotator = HuggingFaceFileAnnotator(
                model_name="mock",
                out_path="./tests/data/output.csv",
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
        with patch.object(
            HuggingFaceFileAnnotator,
            "_load_model",
            return_value=MagicMock(),
        ):
            annotator = HuggingFaceFileAnnotator(
                model_name="mock",
                out_path="./tests/data/output.csv",
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
        with patch.object(
            HuggingFaceFileAnnotator,
            "_load_model",
            return_value=MagicMock(),
        ):
            annotator = HuggingFaceFileAnnotator(
                model_name="mock",
                out_path="./tests/data/output.csv",
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
        with patch.object(
            HuggingFaceFileAnnotator,
            "_load_model",
            return_value=MagicMock(),
        ):
            annotator = HuggingFaceFileAnnotator(
                model_name="mock",
                out_path="./tests/data/output.csv",
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
        with patch.object(
            HuggingFaceFileAnnotator,
            "_load_model",
            return_value=MagicMock(),
        ):
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

            annotator = HuggingFaceFileAnnotator(
                model_name="mock",
                out_path="./tests/data/output.csv",
            )
            annotator._labels = ["BIASED", "NOT BIASED"]
            annotator.post_processor.process(
                df,
                "response",
                "label",
                annotator._labels,
            )

            assert df["label"].iloc[0] == "BIASED"
            assert df["label"].iloc[1] == "NOT BIASED"
            assert df["label"].iloc[2] == "BIASED"
            assert df["label"].iloc[3] == "?"
            assert df["label"].iloc[4] == "?"

    def test_validate_labels_only_init(self):
        with patch.object(
            HuggingFaceFileAnnotator,
            "_load_model",
            return_value=MagicMock(),
        ):
            annotator = HuggingFaceFileAnnotator(
                model_name="mock",
                out_path="./tests/data/output.csv",
                labels=["BIASED", "NON-BIASED"],
            )
            annotator._prompt = Prompt("This is a prompt")
            annotator._validate_labels()

        assert annotator._labels == ["BIASED", "NON-BIASED"]

    def test_validate_labels_only_prompt(self):
        with patch.object(
            HuggingFaceFileAnnotator,
            "_load_model",
            return_value=MagicMock(),
        ):
            annotator = HuggingFaceFileAnnotator(
                model_name="mock",
                out_path="./tests/data/output.csv",
            )

            prompt = Prompt()
            prompt.add_labels_part(
                content="This is a prompt {labels}",
                label_var="labels",
            )
            annotator._prompt = prompt
            annotator._validate_labels(labels=["BIASED", "NON-BIASED"])

        assert annotator._labels == ["BIASED", "NON-BIASED"]

    def test_validate_labels_matching(self):
        with patch.object(
            HuggingFaceFileAnnotator,
            "_load_model",
            return_value=MagicMock(),
        ):
            annotator = HuggingFaceFileAnnotator(
                model_name="mock",
                out_path="./tests/data/output.csv",
                labels=["BIASED", "NON-BIASED"],
            )

            prompt = Prompt()
            prompt.add_labels_part(
                content="This is a prompt {label}",
                label_var="label",
            )
            annotator._prompt = prompt
            annotator._validate_labels(label=["BIASED", "NON-BIASED"])

        assert annotator._labels == ["BIASED", "NON-BIASED"]

    def test_validate_labels_not_matching(self):
        with patch.object(
            HuggingFaceFileAnnotator,
            "_load_model",
            return_value=MagicMock(),
        ):
            annotator = HuggingFaceFileAnnotator(
                model_name="mock",
                out_path="./tests/data/output.csv",
                labels=["BIASED", "NON-BIASED"],
            )

            prompt = Prompt()
            prompt.add_labels_part(
                content="This is a prompt {label}",
                label_var="label",
            )
            annotator._prompt = prompt

        with pytest.raises(Exception):
            annotator._validate_labels(label=["BIASED", "Other Label"])
