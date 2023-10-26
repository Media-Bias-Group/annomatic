import logging

from annomatic.llm.base import Response

LOGGER = logging.getLogger(__name__)


def build_message(content: str, role: str = "user"):
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


def _build_response(api_response: dict) -> Response:
    """
    Build the Response format from the OpenAI API.

    Uses

    Args:
        api_response: dict with the API response

    Returns:
        The formatted Response object
    """
    answer: str = ""
    data: dict = api_response

    if api_response["object"] == "chat.completion":
        answer = api_response["choices"][0]["message"]["content"]
    elif api_response["object"] == "text_completion":
        answer = api_response["choices"][0]["text"]
    else:
        LOGGER.warning("unknown OpenAI response format: ")

    return Response(answer=answer, data=data, query="")
