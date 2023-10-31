import pytest

from tests.model.mock import FakeVllmModel


def test_vllm_predict():
    model = FakeVllmModel(model_name="test_model")
    result = model.predict(
        ["example message1", "example message2", "example message3"],
    )
    #
    assert len(result) == 3


def test_vllm_predict_model_None():
    model = FakeVllmModel(model_name="test_model")
    model.model = None

    with pytest.raises(Exception):
        model.predict(
            ["example message1", "example message2", "example message3"],
        )
