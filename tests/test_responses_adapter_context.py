import pytest
from openai.types.chat import ChatCompletionMessageParam
from openai.types.responses import Response
from openai.types.responses.response_input_param import ResponseInputParam
from pydantic import TypeAdapter

from SimpleLLMFunc.context.ir import (
    AssistantMessage,
    DeveloperMessage,
    Image,
    InputImagePart,
    OutputTextPart,
    ReasoningPart,
    Request,
    UserMessage,
)
from SimpleLLMFunc.interface.openai_responses_compatible import (
    _request_to_responses_kwargs,  # pyright: ignore[reportPrivateUsage]
    _response_to_completion,  # pyright: ignore[reportPrivateUsage]
)
from SimpleLLMFunc.interface.openai_compatible import (
    _request_to_create_kwargs,  # pyright: ignore[reportPrivateUsage]
)


def test_responses_adapter_preserves_output_text_parts() -> None:
    request = Request(
        model="test-model",
        messages=[
            UserMessage(content="question"),
            AssistantMessage(content=[OutputTextPart(text="answer")]),
        ],
    )

    kwargs = _request_to_responses_kwargs(request)

    assert kwargs["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "question"}],
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "input_text", "text": "answer"}],
        },
    ]


def test_responses_adapter_rejects_lossy_reasoning_replay() -> None:
    request = Request(
        model="test-model",
        messages=[
            UserMessage(content="question"),
            AssistantMessage(
                content=[
                    ReasoningPart(reasoning="reason", signature="provider-signature")
                ]
            ),
        ],
    )

    with pytest.raises(ValueError, match="cannot losslessly replay"):
        _request_to_responses_kwargs(request)


def test_responses_adapter_emits_sdk_valid_input_and_sampling_parameters() -> None:
    request = Request(
        model="test-model",
        messages=[
            DeveloperMessage(content="Follow project constraints."),
            UserMessage(
                content=[InputImagePart(image_url="https://example.test/image.png")]
            ),
        ],
        temperature=0.2,
        top_p=0.8,
    )

    kwargs = _request_to_responses_kwargs(request)

    TypeAdapter(ResponseInputParam).validate_python(kwargs["input"])
    assert kwargs["temperature"] == 0.2
    assert kwargs["top_p"] == 0.8
    assert kwargs["input"] == [
        {
            "type": "message",
            "role": "developer",
            "content": [
                {"type": "input_text", "text": "Follow project constraints."}
            ],
        },
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": "https://example.test/image.png",
                    "detail": "auto",
                }
            ],
        },
    ]


def test_multimodal_tool_projection_is_valid_for_both_openai_apis() -> None:
    data_url = Image.from_base64(b"image", media_type="image/png")
    request = Request(
        model="test-model",
        messages=[
            UserMessage(content="inspect"),
            AssistantMessage(content="I will inspect the screenshot."),
            UserMessage(content=[data_url]),
        ],
    )

    chat_kwargs = _request_to_create_kwargs(
        request,
        stream=False,
        stream_options=None,
    )
    messages = chat_kwargs["messages"]
    assert isinstance(messages, list)
    TypeAdapter(list[ChatCompletionMessageParam]).validate_python(messages)
    assert messages[-1] == {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,aW1hZ2U=",
                },
            }
        ],
    }

    responses_kwargs = _request_to_responses_kwargs(request)
    TypeAdapter(ResponseInputParam).validate_python(responses_kwargs["input"])
    assert responses_kwargs["input"][-1] == {
        "type": "message",
        "role": "user",
        "content": [
            {
                "type": "input_image",
                "image_url": "data:image/png;base64,aW1hZ2U=",
                "detail": "auto",
            }
        ],
    }


def test_responses_reasoning_round_trips_with_native_metadata() -> None:
    response = Response.model_validate(
        {
            "id": "response-1",
            "created_at": 1,
            "model": "gpt-5",
            "object": "response",
            "output": [
                {
                    "id": "reasoning-1",
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "reason"}],
                    "status": "completed",
                    "encrypted_content": "encrypted",
                },
                {
                    "id": "message-1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "answer", "annotations": []}
                    ],
                },
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "status": "completed",
        }
    )

    completion = _response_to_completion(response)
    content = completion.choices[0].message.content
    assert isinstance(content, list)
    assert isinstance(content[0], ReasoningPart)
    assert content[0].reasoning == "reason"
    assert content[0].signature is not None

    kwargs = _request_to_responses_kwargs(
        Request(
            model="gpt-5",
            messages=[
                UserMessage(content="question"),
                completion.choices[0].message,
            ],
        )
    )
    TypeAdapter(ResponseInputParam).validate_python(kwargs["input"])
    assert kwargs["input"][1]["type"] == "reasoning"
    assert kwargs["input"][1]["encrypted_content"] == "encrypted"


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("max_output_tokens", "length"),
        ("content_filter", "content_filter"),
    ],
)
def test_responses_incomplete_status_maps_to_chat_finish_reason(
    reason: str,
    expected: str,
) -> None:
    response = Response.model_validate(
        {
            "id": "response-incomplete",
            "created_at": 1,
            "model": "gpt-5",
            "object": "response",
            "output": [],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "status": "incomplete",
            "incomplete_details": {"reason": reason},
        }
    )

    completion = _response_to_completion(response)

    finish_reason = completion.choices[0].finish_reason
    assert finish_reason is not None
    assert finish_reason.value == expected
