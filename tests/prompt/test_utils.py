import pytest

from annomatic.prompt.utils import is_f_string_format


def test_is_f_string_format_expect_false():
    templ = "This should be a wrong f-String!"

    assert not is_f_string_format(templ)


def test_is_f_string_format_expect_false2():
    templ = "This should be a wrong} f-String!"

    assert not is_f_string_format(templ)


def test_is_f_string_format_expect_true():
    templ = "This should be a {wrong} f-String!"

    assert is_f_string_format(templ)


if __name__ == "__main__":
    pytest.main()
