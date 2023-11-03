import re
from typing import List


def check_template_format(template: str, format_type: str = "fString") -> bool:
    """
     Checks if the given template string is in a valid format.

        Args:
        template: The template string to be checked.
        format: The format to be checked. Defaults to "fString".

    Returns:
        bool: True if the template is the specified format; otherwise, False.
    """
    if template is None:
        return False

    if format_type == "fString":
        return is_f_string_format(template)
    else:
        return False


def is_f_string_format(template: str) -> bool:
    """
    Checks if the given template string is in f-string format.

    Args:
        template: The template string to be checked.

    Returns:
        bool: True if the template is in f-string format; otherwise, False.
    """
    try:
        _ = f"{template}"
        if "{" in template and "}" in template:
            return True
        else:
            return False
    except AssertionError:
        return False


def _template_variables(
    template: str,
    template_format: str,
) -> List[str]:
    """
    Extract variables from a template string.

    Args:
        template: A string containing a template.
        pattern: A regex pattern to be used for extracting variables.

    Returns:
        A list of variable names found in the template.
    """

    if template_format == "fString":
        pattern = r"\{(\w+)\}"
    else:
        raise NotImplementedError(f"Format {format} not implemented.")

    return re.findall(pattern, template)
