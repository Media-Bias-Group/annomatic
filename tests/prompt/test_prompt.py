import pytest

from annomatic.prompt.prompt import Prompt


def test_basic_prompt_single_input():
    result = (
        "Instruction: 'Since then, health care has turned out to be a very "
        "strong issue for Democrats, who campaigned on the issue aggressively "
        "during the 2018 midterms and enjoyed a net gain of 40 seats in the "
        "U.S. House of Representatives.'"
        "\n\n"
        "Classify the sentence above as PERSUASIVE TECHNIQUES "
        "or NO PERSUASIVE TECHNIQUES."
        "\n\n"
        "Output: "
    )

    template = (
        "Instruction: '{input}'"
        "\n\n"
        "Classify the sentence above as PERSUASIVE TECHNIQUES "
        "or NO PERSUASIVE TECHNIQUES."
        "\n\n"
        "Output: "
    )

    input = (
        "Since then, health care has turned out to be a very "
        "strong issue for Democrats, "
        "who campaigned on the issue aggressively during the "
        "2018 midterms and enjoyed a net "
        "gain of 40 seats in the U.S. House of Representatives."
    )

    prompt = Prompt(content=template)

    res = prompt.to_string(input=input)
    assert result == res


def test_basic_prompt_multiple_segments():
    result = (
        "Instruction: 'Since then, health care has turned out "
        "to be a very strong issue for Democrats, "
        "who campaigned on the issue aggressively during the "
        "2018 midterms and enjoyed a net"
        " gain of 40 seats in the U.S. House of Representatives.'"
        "\n\n"
        "Classify the sentence above as PERSUASIVE TECHNIQUES "
        "or NO PERSUASIVE TECHNIQUES."
        "\n\n"
        "Output: "
    )

    template_input = "Instruction: '{input}'"

    template_task = (
        "Classify the sentence above as PERSUASIVE TECHNIQUES "
        "or NO PERSUASIVE TECHNIQUES."
    )
    template_out = "Output: "

    input = (
        "Since then, health care has turned out to be a "
        "very strong issue for Democrats, who campaigned on the issue "
        "aggressively during the 2018 midterms and enjoyed "
        "a net gain of 40 seats in the U.S. House of Representatives."
    )

    prompt = Prompt()
    prompt.add_part(content=template_input)
    prompt.add_part(content=template_task)
    prompt.add_part(content=template_out)

    res = prompt.to_string(input=input)
    assert result == res


def test_basic_prompt_multiple_inputs():
    result = (
        "Instruction: 'Since then, health care has turned out "
        "to be a very strong issue for Democrats, "
        "who campaigned on the issue aggressively during the "
        "2018 midterms and enjoyed a net"
        " gain of 40 seats in the U.S. House of Representatives.'"
        "\n\n"
        "Classify the sentence above as PERSUASIVE TECHNIQUES "
        "or NO PERSUASIVE TECHNIQUES."
        "\n\n"
        "Output: "
    )

    template_input = "Instruction: '{input}'"

    template_task = "Classify the sentence above as {labels}."
    template_out = "{pred}: "

    input = (
        "Since then, health care has turned out to be a "
        "very strong issue for Democrats, who campaigned on the issue "
        "aggressively during the 2018 midterms and enjoyed "
        "a net gain of 40 seats in the U.S. House of Representatives."
    )

    prompt = Prompt()
    prompt.add_part(content=template_input)
    prompt.add_labels_part(content=template_task, label_var="labels")
    prompt.add_part(content=template_out)

    res = prompt.to_string(
        input=input,
        labels=[
            "PERSUASIVE TECHNIQUES",
            "NO PERSUASIVE TECHNIQUES",
        ],
        pred="Output",
    )
    assert result == res


def test_get_variable():
    template_input = "Instruction: '{input}'"

    template_out = "Instruction: '{output}'"

    prompt = Prompt()
    prompt.add_part(content=template_input)
    prompt.add_part(content=template_out)

    res = prompt.get_variables()
    assert res == ["input", "output"]


def test_str():
    exp_result = "Instruction: '{input}'\n\nOutput: '{output}'"

    template_input = "Instruction: '{input}'"
    template_out = "Output: '{output}'"

    prompt = Prompt()
    prompt.add_part(content=template_input)
    prompt.add_part(content=template_out)

    assert prompt.__str__() == exp_result
