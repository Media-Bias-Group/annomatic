from annomatic.retriever import DiversityRetriever, SimilarityRetriever


def test_similarity_retriever_identity():
    test_sentences = [
        "I like to eat bananas",
        "I like to eat apples",
        "I like to eat oranges",
    ]

    retriever = SimilarityRetriever(k=1, seed=42)
    result = retriever.select(test_sentences, query="I like to eat apples")

    assert result[0] == "I like to eat apples"


def test_similarity_retriever():
    test_sentences = [
        "I like to eat bananas",
        "This sentence is very different from the others",
        "I like to eat apples",
    ]

    retriever = SimilarityRetriever(k=2, seed=42)
    result = retriever.select(test_sentences, query="I like to eat banana")

    assert result[0] == "I like to eat bananas"
    assert result[1] == "I like to eat apples"


def test_diversity_retriever():
    test_sentences = [
        "I like to eat bananas",
        "This sentence is very different from the others",
        "I like to eat apples",
    ]

    retriever = DiversityRetriever(k=2, seed=42)
    result = retriever.select(test_sentences)

    assert result[0] == "This sentence is very different from the others"
    assert result[1] == "I like to eat apples"
