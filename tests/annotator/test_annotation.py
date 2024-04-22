import datasets
import pandas as pd
import pytest
from haystack.components.builders import PromptBuilder

from annomatic.annotator.annotation import DefaultAnnotation

ZERO_SHOT_PROMPT = """Instruction: "{{text}}"
    Classify the sentence below as BIASED or NOT BIASED.
    Output: """

FEW_SHOT_PROMPT = """{% for key,value in documents.iterrows() %}Instruction: "{{value.text}}"
Classify the sentence below as BIASED or NOT BIASED.
Output: {{ value.label }}
{% endfor %}
Instruction: "{{text}}"
Classify the sentence below as BIASED or NOT BIASED.
Output: """


def test_prompt_fill_zero_shot():
    text = [
        "Recent studies suggest that the new technology is revolutionary.",
        "Critics argue that the government's policies are misguided and harmful.",
    ]
    df = pd.DataFrame({"text": text})

    expected0 = """Instruction: "Recent studies suggest that the new technology is revolutionary."
    Classify the sentence below as BIASED or NOT BIASED.
    Output: """

    expected1 = """Instruction: "Critics argue that the government's policies are misguided and harmful."
    Classify the sentence below as BIASED or NOT BIASED.
    Output: """

    prompt = PromptBuilder(ZERO_SHOT_PROMPT)

    result = DefaultAnnotation().fill_prompt(prompt, df, "text")

    assert result == [expected0, expected1]


def test_prompt_fill_few_shot():
    text = [
        "Recent studies suggest that the new technology is revolutionary.",
    ]
    df = pd.DataFrame({"text": text})
    data = {
        "text": [
            "Critics argue that the government's policies are misguided "
            "and harmful.",
        ],
        "label": [
            "NOT BIASED",
        ],
    }
    df_examples = pd.DataFrame(data, columns=["text", "label"])
    expected = """Instruction: "Critics argue that the government's policies are misguided and harmful."
Classify the sentence below as BIASED or NOT BIASED.
Output: NOT BIASED

Instruction: "Recent studies suggest that the new technology is revolutionary."
Classify the sentence below as BIASED or NOT BIASED.
Output: """

    prompt = PromptBuilder(FEW_SHOT_PROMPT)

    annotation = DefaultAnnotation()
    annotation.context = {"documents": df_examples}
    result = annotation.fill_prompt(prompt, df, "text")

    assert result == expected
