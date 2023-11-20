import logging

from annomatic.llm.base import Response, ResponseList

LOGGER = logging.getLogger(__name__)


def _build_response_List(api_response: dict) -> ResponseList:
    """
    Build the ResponseList from the OpenAI API.

    Uses

    Args:
        api_response: dict with the API response

    Returns:
        The formatted Response object
    """
    answer = ""
    data: dict = api_response

    if api_response["object"] == "chat.completion":
        answer = api_response["choices"][0]["message"]["content"]
    elif api_response.get("object") == "text_completion":
        answer = api_response["choices"][0]["text"]
    else:
        LOGGER.warning("unknown OpenAI response format: ")

    return ResponseList(answers=answer, data=data, queries=["TODO"])


def _build_response(message: str, api_response: dict) -> Response:
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

    return Response(answer=answer, data=data, query=message)
