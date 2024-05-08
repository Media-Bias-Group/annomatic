import pandas as pd
from haystack.components.builders import PromptBuilder

from annomatic.annotator.annotation import DefaultAnnotation
from annomatic.retriever import SimilarityRetriever

ZERO_SHOT_PROMPT = (
    "Instruction: '{{text}}'\n"
    "Classify the sentence above as BIASED or NOT BIASED.\n"
    "Output: "
)

FEW_SHOT_PROMPT = (
    "{% for key,value in documents.iterrows() %}"
    "Instruction: '{{value.text}}'\n"
    "Classify the sentence above as BIASED or NOT BIASED.\n"
    "Output: {{ value.label }}\n\n"
    "{% endfor %}"
    "Instruction: '{{text}}'\n"
    "Classify the sentence above as BIASED or NOT BIASED.\n"
    "Output: "
)


def test_prompt_fill_zero_shot():
    text = [
        "Recent studies suggest that the new technology is revolutionary.",
        "Critics argue that the government's policies are misguided and harmful.",
    ]
    df = pd.DataFrame({"text": text})

    prompt = PromptBuilder(ZERO_SHOT_PROMPT)

    result = DefaultAnnotation().fill_prompt(prompt, df)

    assert result == [
        "Instruction: 'Recent studies suggest that the new technology is revolutionary.'\n"
        "Classify the sentence above as BIASED or NOT BIASED.\n"
        "Output: ",
        "Instruction: 'Critics argue that the government's policies are misguided and harmful.'\n"
        "Classify the sentence above as BIASED or NOT BIASED.\n"
        "Output: ",
    ]


def test_prompt_fill_few_shot_dataframe():
    df_data = pd.DataFrame(
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

    prompt = PromptBuilder(FEW_SHOT_PROMPT)

    annotation = DefaultAnnotation()
    annotation.context = {"documents": df_examples}
    result = annotation.fill_prompt(prompt, df_data)

    assert (
        result == "Instruction: 'This is a examples.'\n"
        "Classify the sentence above as BIASED or NOT BIASED.\n"
        "Output: BIASED\n\n"
        "Instruction: 'This is a second examples.'\n"
        "Classify the sentence above as BIASED or NOT BIASED.\n"
        "Output: NOT BIASED\n\n"
        "Instruction: 'This is a test sentence.'\n"
        "Classify the sentence above as BIASED or NOT BIASED.\n"
        "Output: "
    )


def test_prompt_fill_few_shot_retriever():
    df_data = pd.DataFrame(
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
    retriever = SimilarityRetriever(
        k=1,
        pool=df_examples,
        model_name="all-MiniLM-L6-v2",
        text_variable="text",
        label_variable="label",
    )
    prompt = PromptBuilder(FEW_SHOT_PROMPT)

    annotation = DefaultAnnotation()
    annotation.context = {"documents": retriever}
    result = annotation.fill_prompt(prompt, df_data)

    assert (
        result == "Instruction: 'This is a examples.'\n"
        "Classify the sentence above as BIASED or NOT BIASED.\n"
        "Output: BIASED\n\n"
        "Instruction: 'This is a test sentence.'\n"
        "Classify the sentence above as BIASED or NOT BIASED.\n"
        "Output: "
    )
