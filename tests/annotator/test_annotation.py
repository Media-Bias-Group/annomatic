import pandas as pd
import pytest
from haystack.components.builders import PromptBuilder

from annomatic.annotator.annotation import DefaultAnnotation

ZERO_SHOT_PROMPT = """Instruction: "{{text}}"
    Classify the sentence below as BIASED or NOT BIASED.
    Output: """

FEW_SHOT_PROMPT = """
    Given these documents, answer the question.\nDocuments:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}

    \nQuestion: {{query}}
    \nAnswer:
    """


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
