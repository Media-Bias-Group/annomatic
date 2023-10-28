import pytest

from annomatic.prompt.base import FstringTemplater


def test_fstring_template_no_variable():
    template = FstringTemplater("This template is empty.")
    result = template.parse()
    assert result == "This template is empty."


def test_fstring_template():
    template = FstringTemplater("This template is a {purpose}.")
    result = template.parse(purpose="test")
    assert result == "This template is a test."


def test_fstring_template_multiple_variables():
    template = FstringTemplater(
        "Hello {person}. This is the {speaker} speaking."
        " We have some time till {answer} is known.",
    )
    result = template.parse(person="Traveler", speaker="me", answer=42)
    assert (
        result == "Hello Traveler. This is the me speaking."
        " We have some time till 42 is known."
    )


def test_fstring_template_missing_variables():
    template = FstringTemplater(
        "Hello {person}. This is the {speaker} speaking."
        " We have some time till {answer} is known.",
    )
    with pytest.raises(ValueError, match="Missing variable: answer"):
        template.parse(person="Traveler", speaker="me")


def test_fstring_get_variables():
    template = FstringTemplater(
        "Hello {person}. This is the {speaker} speaking."
        " We have some time till {answer} is known.",
    )
    result = template.get_variables()

    assert result == ["person", "speaker", "answer"]


if __name__ == "__main__":
    pytest.main()
