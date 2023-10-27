import os

from cfgv import Optional

from annomatic.annotator.csv_annotator import CsvAnnotator, OpenAiCsvAnnotator
from annomatic.llm import Response, ResponseList
from annomatic.prompt.prompt import Prompt
from tests.model.mock import FakeHFAutoModelForCausalLM, FakeOpenAiModel


class FakeOpenAiCSVAnnotator(CsvAnnotator):
    """
    Fake Annotator class for OpenAI models that use CSV files
    as input and output.

    All the model calls are mocked.
    """

    def __init__(
        self,
        model_lib: str = "",
        model_name: str = " ",
        model_args: Optional[dict] = None,
        out_path: str = "",
        **kwargs,
    ):
        super().__init__(model_name, model_lib, model_args, out_path, **kwargs)

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
        self.model = FakeOpenAiModel(model_name=self.model_name)


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
        model_args: Optional[dict] = None,
        out_path: str = "",
        **kwargs,
    ):
        super().__init__(model_name, model_lib, model_args, out_path, **kwargs)

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
        self.model = FakeHFAutoModelForCausalLM()


def test_set_data_csv():
    import pandas as pd

    annotator = FakeOpenAiCSVAnnotator(model_name="model")
    annotator.set_data(data="tests/data/input.csv", in_col="input")
    assert isinstance(annotator._input, pd.DataFrame)


def test_set_data_df():
    import pandas as pd

    annotator = FakeOpenAiCSVAnnotator(model_name="model")
    df = pd.read_csv("tests/data/input.csv")
    annotator.set_data(data=df, in_col="input")

    assert isinstance(annotator._input, pd.DataFrame)


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


def test_OpenAIAnnotation_no_exception():
    # delete file if exists
    try:
        os.remove("tests/data/output.csv")
    except OSError:
        pass

    annotator = FakeOpenAiCSVAnnotator(
        model_name="model",
        out_path="tests/data/output.csv",
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
        data="tests/data/input.csv",
        in_col="input",
    )

    annotator.annotate()

    assert os.path.exists("tests/data/output.csv")


def test_Huggingface_no_exception():
    # delete file if exists
    try:
        os.remove("/home/sinix/dev/annomatic/tests/data/output.csv")
    except OSError:
        pass

    annotator = FakeHuggingFaceCsvAnnotator(
        model_name="model",
        model_lib="hf",
        out_path="/home/sinix/dev/annomatic/tests/data/output.csv",
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
        data="/home/sinix/dev/annomatic/tests/data/input.csv",
        in_col="input",
    )

    annotator.annotate()
    assert os.path.exists("/home/sinix/dev/annomatic/tests/data/output.csv")
