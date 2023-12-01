import pandas as pd

from annomatic.prompt import Prompt
from annomatic.retriever import DiversityRetriever
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
        labels=["BIASED", "NOT BIASED"],
    )

    df = pd.DataFrame(
        {"text": ["This is a test sentence."]},
    )

    df_examples = pd.DataFrame(
        {
            "text": ["This is a examples.", "This is a second examples."],
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
    message = annotator.fill_prompt(batch=df, label=["BIASED", "NOT BIASED"])

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


def test_fill_prompt_with_Retriever():
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
    message = annotator.fill_prompt(batch=df, label=["BIASED", "NOT BIASED"])

    assert message == [
        "Instruction: 'This is a test example.'\n\n"
        "Classify the sentence above as BIASED or NOT BIASED."
        "\n\nOutput: N0T BIASED\n\n"
        "Instruction: 'This is a test sentence.'\n\n"
        "Classify the sentence above as BIASED or NOT BIASED."
        "\n\nOutput: ",
    ]
