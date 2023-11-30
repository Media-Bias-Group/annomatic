import pandas as pd

from annomatic.prompt import Prompt
from tests.annotator.test_csv_annotator import FakeHuggingFaceCsvAnnotator


def test_fill_prompt_without_examples():
    df = pd.DataFrame(
        {"text": ["This is a test sentence."]},
    )

    annotator = FakeHuggingFaceCsvAnnotator(
        model_name="model",
        model_lib="hf",
        out_path="./tests/data/output.csv",
        labels=["BIASED", "NON-BIASED"],
    )

    annotator.data_variable = "text"
    prompt = Prompt()
    prompt.add_part("Instruction: '{text}'")
    prompt.add_labels_part("Classify the sentence above as {label}.")
    prompt.add_part("Output: ")

    annotator.set_prompt(prompt)
    message = annotator.fill_prompt(batch=df, label=["BIASED", "NOT BIASED"])

    assert message == [
        "Instruction: 'This is a test sentence.'\n\n"
        "Classify the sentence above as BIASED or NOT BIASED."
        "\n\nOutput: ",
    ]


def test_fill_prompt_with_examples():
    annotator = FakeHuggingFaceCsvAnnotator(
        model_name="model",
        model_lib="hf",
        out_path="./tests/data/output.csv",
        labels=["BIASED", "NON-BIASED"],
    )

    df = pd.DataFrame(
        {"text": ["This is a test sentence."]},
    )

    df_examples = pd.DataFrame(
        {
            "text": ["This is a examples."],
            "label": ["BIASED"],
        },
    )

    annotator.data_variable = "text"
    annotator.examples = df_examples
    prompt = Prompt()
    prompt.add_part("Instruction: '{text}'")
    prompt.add_labels_part("Classify the sentence above as {label}.")
    prompt.add_part("Output: ")

    annotator.set_prompt(prompt)
    message = annotator.fill_prompt(batch=df, label=["BIASED", "NOT BIASED"])

    assert message == [
        "Instruction: 'This is a examples.'\n\n"
        "Classify the sentence above as BIASED or NOT BIASED."
        "\n\nOutput: BIASED\n\n"
        "Instruction: 'This is a test sentence.'\n\n"
        "Classify the sentence above as BIASED or NOT BIASED."
        "\n\nOutput: ",
    ]
