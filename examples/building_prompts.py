class PromptSegment:
    pass


class PromptPlainSegment(PromptSegment):
    """
    This is class represents a part of a Prompt as plaintext.
    """

    def __init__(self, message: str = ""):
        self.part = message


class PromptTemplateSegment(PromptSegment):
    """
    This is class represents a part of a Prompt witch has his own template.

    This may be used for templating e.g. examples

    """

    def __init__(self, template: str = ""):
        self.template = template
