from annomatic.annotator.csv_annotator import OpenAICsvAnnotator
from annomatic.prompt.prompt import Prompt
from tests.model.mock import FakeOpenAiModel


class FakeOpenAiCSVAnnotator(OpenAICsvAnnotator):
    """
    Fake Annotator class for OpenAI models that use CSV files
    as input and output.

    All the model calls are mocked.
    """

    def __init__(
        self,
        in_path: str,
        out_path: str,
        in_col: str = "input",
        model: str = "gpt-3.5-turbo",
    ):
        super().__init__(in_path, out_path, api_key="dummy_key")
        self._in_col = in_col
        self._prompt = None
        self._model = FakeOpenAiModel(model=model)


def test_csv_annotator():
    annotator = FakeOpenAiCSVAnnotator(
        in_path="../data/input.csv",
        out_path="../data/output.csv",
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

    annotator.annotate()


def test_csv_annotator_with_var():
    annotator = FakeOpenAiCSVAnnotator(
        in_path="../data/input.csv",
        out_path="../data/output.csv",
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

    annotator.annotate(extra="something")
