def build_message(content: str, role: str = "user") -> dict[str, str]:
    """
    Build a structured message.

    This function creates the message in the format specified in the
    Chat Completions API provided by OpenAI.
    https://platform.openai.com/docs/guides/gpt/chat-completions-api

    The format is a dict containing a "role", usually "user" or "system",
    and the "content" which is the prompt message

    Arguments:
        content: content of the prompt.
        role : The role of the message sender/owner. default="user"

    Returns:
        dict: A dictionary representing a message
    """
    return {"role": role, "content": content}
