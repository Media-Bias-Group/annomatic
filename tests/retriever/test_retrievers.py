import pandas as pd

from annomatic.retriever import DiversityRetriever, SimilarityRetriever


def test_similarity_retriever_identity():
    test_sentences = [
        "I like to eat bananas",
        "I like to eat apples",
        "I like to eat oranges",
    ]

    df = pd.DataFrame({"text": test_sentences, "label": [1, 2, 3]})

    retriever = SimilarityRetriever(
        model_name="all-MiniLM-L6-v2",
        k=1,
        seed=42,
        pool=df,
    )
    result = retriever.select(query="I like to eat apples")

    assert result.iloc[0]["text"] == "I like to eat apples"


def test_similarity_retriever():
    test_sentences = [
        "I like to eat bananas",
        "This sentence is very different from the others",
        "I like to eat apples",
    ]
    # Create a DataFrame with 'text' column
    df = pd.DataFrame({"text": test_sentences, "label": [1, 2, 3]})

    retriever = SimilarityRetriever(
        model_name="all-MiniLM-L6-v2",
        k=2,
        seed=42,
        pool=df,
    )
    result = retriever.select(query="I like to eat banana")

    assert result.iloc[0]["text"] == "I like to eat bananas"
    assert result.iloc[1]["text"] == "I like to eat apples"


def test_diversity_retriever():
    test_sentences = [
        "I like to eat bananas",
        "This sentence is very different from the others",
        "I like to eat apples",
    ]
    # Create a DataFrame with 'text' column
    df = pd.DataFrame({"text": test_sentences, "label": [1, 2, 3]})

    retriever = DiversityRetriever(
        model_name="all-MiniLM-L6-v2",
        k=2,
        pool=df,
        seed=42,
    )
    result = retriever.select()

    assert (
        result.iloc[0]["text"]
        == "This sentence is very different from the others"
    )
    assert result.iloc[1]["text"] == "I like to eat apples"

    result = retriever.select()
    assert (
        result.iloc[0]["text"]
        == "This sentence is very different from the others"
    )
    assert result.iloc[1]["text"] == "I like to eat apples"
