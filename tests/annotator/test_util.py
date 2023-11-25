from annomatic.annotator.util import (
    find_all_occurrences_indices,
    find_contained_labels,
    find_label,
    find_labels_in_sentence,
)


def test_find_all_occurrences_indices():
    test_sentence = "'NOT BIASED' is better than 'BIASED'."
    result = find_all_occurrences_indices(test_sentence, "BIASED")
    assert result == [(5, 11), (29, 35)]


def test_find_labels_in_sentence():
    test_labels = ["BIASED", "NOT BIASED"]
    test_sentence = "'NOT BIASED' is better than 'BIASED'."

    result = find_labels_in_sentence(test_sentence, test_labels)
    assert result == [[(29, 35)], [(1, 11)]]
    assert test_labels[0] == test_sentence[result[0][0][0] : result[0][0][1]]
    assert test_labels[1] == test_sentence[result[1][0][0] : result[1][0][1]]

    test_labels = ["BIASED", "NOT BIASED"]
    test_sentence = "Sentence with no labels"
    result = find_labels_in_sentence(test_sentence, test_labels)
    assert result == [[], []]

    test_labels = []
    test_sentence = "Sentence with no labels"
    result = find_labels_in_sentence(test_sentence, test_labels)
    assert result == []


def test_find_contained_labels():
    test_labels = ["BIASED", "NOT BIASED"]

    result = find_contained_labels(test_labels)
    assert result == {"BIASED": ["NOT BIASED"], "NOT BIASED": []}


def test_find_label_1():
    test_sentence = (
        "'NOT BIASED'\n\nExplanation: This statement presents a "
        "neutral view on the environmental impact of renewable "
        "energy, avoiding any subjective language that "
        "might indicate bias."
    )
    test_labels = ["BIASED", "NOT BIASED"]
    result = find_label(test_sentence, test_labels)

    assert result == "NOT BIASED"


def test_find_label_no_labels():
    test_sentence = "No This statement contains no bias. NO BIAS"

    test_labels = ["BIASED", "NOT BIASED"]
    result = find_label(test_sentence, test_labels)

    assert result == "?"


def test_find_label_both_labels():
    test_sentence = "BIASED. No This statement contains not BIASED."

    test_labels = ["BIASED", "NOT BIASED"]
    result = find_label(test_sentence, test_labels)

    assert result == "?"
