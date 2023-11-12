import os

import pandas as pd
import pytest
from cfgv import Optional

from annomatic.annotator.csv_annotator import CsvAnnotator, VllmCsvAnnotator
from annomatic.llm import Response, ResponseList
from annomatic.prompt.prompt import Prompt
from tests.model.mock import (
    FakeHFAutoModelForCausalLM,
    FakeOpenAiModel,
    FakeVllmModel,
)


class FakeOpenAiCSVAnnotator(CsvAnnotator):
    """
    Fake Annotator class for OpenAI models that use CSV files
    as input and output.

    All the model calls are mocked.
    """

    def __init__(
        self,
        model_lib: str = " ",
        model_name: str = " ",
        config: Optional[dict] = None,
        batch_size: Optional[int] = 5,
        out_path: str = "",
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            model_lib=model_lib,
            config=config,
            out_path=out_path,
            batch_size=batch_size,
            **kwargs,
        )

    def _model_predict(self, batch, **kwargs):
        """
        Mocking the models batch prediction
        """

        return ResponseList.from_responses(
            [
                Response(
                    answer="answer",
                    data="data",
                    query="query",
                ),
            ]
            * len(batch),
        )

    def _load_model(self):
        """
        Mocking the model loading
        """
        self._model = FakeOpenAiModel(model_name=self.model_name)


class FakeHuggingFaceCsvAnnotator(CsvAnnotator):
    """
    Fake Annotator class for OpenAI models that use CSV files
    as input and output.

    All the model calls are mocked.
    """

    def __init__(
        self,
        model_lib: str,
        model_name: str = " ",
        config: Optional[dict] = None,
        batch_size: Optional[int] = 5,
        out_path: str = "",
        labels: Optional[list] = None,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            model_lib=model_lib,
            config=config,
            out_path=out_path,
            batch_size=batch_size,
            labels=labels,
            **kwargs,
        )

    def _model_predict(self, batch, **kwargs):
        """
        Mocking the models batch prediction
        """
        return ResponseList.from_responses(
            [
                Response(
                    answer="answer",
                    data="data",
                    query="query",
                ),
            ]
            * len(batch),
        )

    def _load_model(self):
        """
        Mocking the model loading
        """
        self._model = FakeHFAutoModelForCausalLM()


class FakeVllmCsvAnnotator(VllmCsvAnnotator):
    """
    Fake Annotator class for OpenAI models that use CSV files
    as input and output.

    All the model calls are mocked.
    """

    def __init__(
        self,
        model_lib: str,
        model_name: str = " ",
        config: Optional[dict] = None,
        batch_size: Optional[int] = 5,
        out_path: str = "",
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            config=config,
            out_path=out_path,
            batch_size=batch_size,
        )

    def _load_model(self):
        """
        Mocking the model loading
        """
        self._model = FakeVllmModel("test_model")


def test_set_data_csv():
    annotator = FakeOpenAiCSVAnnotator(model_name="model")
    annotator.set_data(data="tests/data/input.csv", in_col="input")
    assert isinstance(annotator._input, pd.DataFrame)


def test_set_data_df():
    annotator = FakeOpenAiCSVAnnotator(model_name="model")
    df = pd.read_csv("tests/data/input.csv")
    annotator.set_data(data=df, in_col="input")

    assert isinstance(annotator._input, pd.DataFrame)


def test_set_data_prompt_matching():
    annotator = FakeOpenAiCSVAnnotator(model_name="model")

    template = (
        "Instruction: '{input}'"
        "\n\n"
        "Classify the sentence above as PERSUASIVE TECHNIQUES "
        "or NO PERSUASIVE TECHNIQUES or as {extra}."
        "\n\n"
        "Output: "
    )
    annotator.set_prompt(prompt=template)
    df = pd.read_csv("./tests/data/input.csv")
    annotator.set_data(
        data=df,
        in_col="input",
    )

    assert annotator.in_col == "input"


def test_set_data_prompt_raise_value_error():
    annotator = FakeOpenAiCSVAnnotator(model_name="model")

    template = (
        "Instruction: '{input}'"
        "\n\n"
        "Classify the sentence above as PERSUASIVE TECHNIQUES "
        "or NO PERSUASIVE TECHNIQUES or as {extra}."
        "\n\n"
        "Output: "
    )
    annotator.set_prompt(prompt=template)
    df = pd.read_csv("./tests/data/input.csv")
    with pytest.raises(ValueError) as e_info:
        annotator.set_data(
            data=df,
            in_col="?",
        )
        raise e_info


def test_set_prompt_str():
    annotator = FakeOpenAiCSVAnnotator(model_name="model")

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


def test_set_prompt_prompt():
    annotator = FakeOpenAiCSVAnnotator(model_name="model")
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


def test_OpenAIAnnotation_annotate():
    # delete file if exists
    try:
        os.remove("./tests/data/output.csv")
    except OSError:
        pass

    annotator = FakeOpenAiCSVAnnotator(
        model_name="model",
        out_path="./tests/data/output.csv",
    )
    data = pd.read_csv("./tests/data/input.csv")

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
        in_col="input",
    )

    annotator.annotate()

    assert os.path.exists("./tests/data/output.csv")

    output = pd.read_csv("./tests/data/output.csv")
    assert output.shape[0] == data.shape[0]


def test_Huggingface_annotate():
    # delete file if exists
    try:
        os.remove("./tests/data/output.csv")
    except OSError:
        pass

    data = pd.read_csv("./tests/data/input.csv")

    annotator = FakeHuggingFaceCsvAnnotator(
        model_name="model",
        model_lib="hf",
        out_path="./tests/data/output.csv",
        labels=["BIASED", "NON-BIASED"],
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
    annotator.set_data(
        data=data,
        in_col="input",
    )

    annotator.annotate()
    assert os.path.exists("./tests/data/output.csv")

    output = pd.read_csv("./tests/data/output.csv")
    assert output.shape[0] == data.shape[0]


def test_vllm_annotate():
    # delete file if exists
    try:
        os.remove("./tests/data/output.csv")
    except OSError:
        pass
    data = pd.read_csv("./tests/data/input.csv")

    annotator = FakeVllmCsvAnnotator(
        model_name="model",
        model_lib="vllm",
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
    annotator._load_model()
    annotator.annotate(data=data, in_col="input")
    assert os.path.exists("./tests/data/output.csv")

    output = pd.read_csv("./tests/data/output.csv")
    assert output.shape[0] == data.shape[0]


def test_huggingface_annotate_batch():
    # delete file if exists
    try:
        os.remove("./tests/data/output.csv")
    except OSError:
        pass

    inp = pd.read_csv("./tests/data/input.csv")

    annotator = FakeHuggingFaceCsvAnnotator(
        model_name="model",
        model_lib="hf",
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
    annotator._load_model()
    annotator.in_col = "input"
    res = annotator._annotate_batch(inp)

    assert len(res) == inp.shape[0]


def test_openai_annotate_batch():
    # delete file if exists
    try:
        os.remove("./tests/data/output.csv")
    except OSError:
        pass
    inp = pd.read_csv("./tests/data/input.csv")

    annotator = FakeOpenAiCSVAnnotator(
        model_name="model",
        model_lib="hf",
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
    annotator._load_model()
    annotator.in_col = "input"
    res = annotator._annotate_batch(inp)

    assert len(res) == inp.shape[0]


def test_vllm_annotate_batch():
    # delete file if exists
    try:
        os.remove("./tests/data/output.csv")
    except OSError:
        pass
    inp = pd.read_csv("./tests/data/input.csv")

    annotator = FakeVllmCsvAnnotator(
        model_name="model",
        model_lib="vllm",
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
    annotator._load_model()
    annotator.in_col = "input"
    res = annotator._annotate_batch(inp)

    assert len(res) == inp.shape[0]


def test_parse_label_expect_biased():
    annotator = FakeOpenAiCSVAnnotator(
        out_path="./tests/data/output.csv",
    )
    annotator._labels = ["BIASED", "NON-BIASED"]
    res = annotator._parse_label("This text is biased since...", "?")

    assert res == "BIASED"


def test_parse_label_expect_non_biased():
    annotator = FakeOpenAiCSVAnnotator(
        out_path="./tests/data/output.csv",
    )
    annotator._labels = ["BIASED", "NON-BIASED"]
    # sort done in method
    annotator._labels.sort(
        key=lambda x: len(x),
        reverse=True,
    )

    res = annotator._parse_label("This text is non-biased since...", "?")

    assert res == "NON-BIASED"


def test_parse_label_expect_default():
    annotator = FakeOpenAiCSVAnnotator(
        out_path="./tests/data/output.csv",
    )
    annotator._labels = ["BIASED", "NON-BIASED"]
    res = annotator._parse_label("This text is not parseable", "?")

    assert res == "?"


def test_soft_parse():
    df = pd.DataFrame(
        {
            "response": [
                "This is a biased response.",
                "A non-biased response.",
                "Another biased statement.",
                "This sentence is not parseable.",
            ],
        },
    )

    annotator = FakeOpenAiCSVAnnotator(
        out_path="./tests/data/output.csv",
    )
    annotator._labels = ["BIASED", "NON-BIASED"]
    annotator._soft_parse(df, "response", "label")

    assert df["label"].iloc[0] == "BIASED"
    assert df["label"].iloc[1] == "NON-BIASED"
    assert df["label"].iloc[2] == "BIASED"
    assert df["label"].iloc[3] == "?"


def test_validate_labels_only_init():
    annotator = FakeOpenAiCSVAnnotator(
        out_path="./tests/data/output.csv",
        labels=["BIASED", "NON-BIASED"],
    )
    annotator._prompt = Prompt("This is a prompt")
    annotator._validate_labels()

    assert annotator._labels == ["BIASED", "NON-BIASED"]


def test_validate_labels_only_prompt():
    annotator = FakeOpenAiCSVAnnotator(
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


def test_validate_labels_matching():
    annotator = FakeOpenAiCSVAnnotator(
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


def test_validate_labels_not_matching():
    annotator = FakeOpenAiCSVAnnotator(
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
