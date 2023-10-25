from cfgv import Optional

from annomatic.annotator.csv_annotator import CsvAnnotator, OpenAiCsvAnnotator
from annomatic.prompt.prompt import Prompt
from tests.model.mock import FakeOpenAiModel


class FakeCSVAnnotator(CsvAnnotator):
    """
    Fake Annotator class for OpenAI models that use CSV files
    as input and output.

    All the model calls are mocked.
    """

    def __init__(
        self,
        model_name: str,
        in_col: str = "input",
        model_lib: str = "huggingface",
        model_args: Optional[dict] = None,
        out_path: str = "",
    ):
        # exclude the super().__init__ call, due to model building
        # super().__init__(model_name=model_name, model_lib="openai")
        self._in_col = in_col
        self._prompt = None
        self._input = None
        self._model = FakeOpenAiModel(model_name=model_name)


def test_set_data_csv():
    import pandas as pd

    annotator = FakeCSVAnnotator(model_name="model")
    annotator.set_data(data="tests/data/input.csv", in_col="input")
    assert isinstance(annotator._input, pd.DataFrame)


def test_set_data_df():
    import pandas as pd

    annotator = FakeCSVAnnotator(model_name="model")
    df = pd.read_csv("tests/data/input.csv")
    annotator.set_data(data=df, in_col="input")

    assert isinstance(annotator._input, pd.DataFrame)


def test_set_prompt_str():
    annotator = FakeCSVAnnotator(model_name="model")

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
    annotator = FakeCSVAnnotator(model_name="model")
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
