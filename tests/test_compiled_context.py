import pytest

from SimpleLLMFunc.context.ir import Request, SystemMessage, UserMessage
from SimpleLLMFunc.loop import CompiledContext, ContextProvenance


def make_request(content: str = "hello") -> Request:
    return Request(
        model="test-model",
        messages=[
            SystemMessage(content="system"),
            UserMessage(content=content),
        ],
    )


def test_compiled_context_is_stable_and_returns_request_copies() -> None:
    context = CompiledContext.create(
        request=make_request(),
        source_revision=4,
        compiler_revision="compiler@1",
        provenance=(
            ContextProvenance(path="messages.0", event_ids=("system-1",)),
            ContextProvenance(path="messages.1", event_ids=("user-1",)),
        ),
    )

    request = context.request
    request.messages.append(UserMessage(content="mutation"))

    assert len(context.request.messages) == 2
    assert context.request_digest
    assert context.digest
    assert context.request.model == "test-model"

    equivalent = CompiledContext.create(
        request=make_request(),
        source_revision=4,
        compiler_revision="compiler@1",
        provenance=context.provenance,
    )
    assert equivalent.digest == context.digest


def test_derive_records_parent_and_changes_identity() -> None:
    context = CompiledContext.create(
        request=make_request(),
        source_revision=1,
        compiler_revision="compiler@1",
    )

    derived = context.derive(
        request=make_request().model_copy(update={"temperature": 0.2}),
        transformation="test.change_user_message",
    )

    assert derived.parent_digest == context.digest
    assert derived.transformation == "test.change_user_message"
    assert derived.digest != context.digest
    assert derived.request_digest != context.request_digest

    with pytest.raises(ValueError, match="cannot change messages"):
        context.derive(
            request=make_request("changed"),
            transformation="test.change_user_message",
        )


def test_context_rejects_duplicate_provenance_paths_and_invalid_source() -> None:
    provenance = ContextProvenance(path="messages.0", event_ids=("event-1",))

    try:
        CompiledContext.create(
            request=make_request(),
            source_revision=0,
            compiler_revision="compiler@1",
            provenance=(provenance, provenance),
        )
    except ValueError as exc:
        assert "duplicate provenance path" in str(exc)
    else:
        raise AssertionError("duplicate provenance path was accepted")

    try:
        CompiledContext.create(
            request=make_request(),
            source_revision=-1,
            compiler_revision="compiler@1",
        )
    except ValueError as exc:
        assert "source_revision" in str(exc)
    else:
        raise AssertionError("negative source revision was accepted")


def test_context_rejects_invalid_provenance_and_derivation_metadata() -> None:
    with pytest.raises(ValueError, match="duplicate event id"):
        ContextProvenance(path="messages.0", event_ids=("event-1", "event-1"))

    with pytest.raises(ValueError, match="must not be empty"):
        ContextProvenance(path="messages.0", event_ids=("",))

    with pytest.raises(ValueError, match="compiler_revision"):
        CompiledContext.create(
            request=make_request(),
            source_revision=0,
            compiler_revision="",
        )

    with pytest.raises(ValueError, match="provided together"):
        CompiledContext.create(
            request=make_request(),
            source_revision=0,
            compiler_revision="compiler@1",
            parent_digest="parent",
        )

    context = CompiledContext.create(
        request=make_request(),
        source_revision=0,
        compiler_revision="compiler@1",
    )
    with pytest.raises(ValueError, match="transformation"):
        context.derive(request=make_request(), transformation="")


def test_context_rejects_forged_serialized_evidence() -> None:
    context = CompiledContext.create(
        request=make_request(),
        source_revision=0,
        compiler_revision="compiler@1",
    )

    for field, value, message in (
        ("request_json", make_request("forged").model_dump_json(), "canonical JSON"),
        ("request_digest", "forged", "request_digest"),
        ("digest", "forged", "context digest"),
        ("parent_digest", "", "derivation metadata"),
    ):
        payload = context.model_dump(mode="python")
        payload[field] = value
        if field == "parent_digest":
            payload["transformation"] = "edited"
        with pytest.raises(ValueError, match=message):
            CompiledContext.model_validate(payload)

    restored = CompiledContext.model_validate_json(context.model_dump_json())
    assert restored == context

    invalid_payload = context.model_dump(mode="python")
    invalid_payload["request_json"] = "{}"
    with pytest.raises(ValueError, match="valid Request"):
        CompiledContext.model_validate(invalid_payload)

    duplicate = ContextProvenance(path="messages.0", event_ids=("event",))
    duplicate_payload = context.model_dump(mode="python")
    duplicate_payload["provenance"] = (duplicate, duplicate)
    with pytest.raises(ValueError, match="duplicate provenance path"):
        CompiledContext.model_validate(duplicate_payload)

    unpaired_payload = context.model_dump(mode="python")
    unpaired_payload["parent_digest"] = "parent"
    with pytest.raises(ValueError, match="provided together"):
        CompiledContext.model_validate(unpaired_payload)

    for path, message in (
        ("events.0", "target a Request message"),
        ("messages.nope", "target a Request message"),
        ("messages.9", "outside the Request"),
    ):
        invalid_path = context.model_dump(mode="python")
        invalid_path["provenance"] = (
            ContextProvenance(path=path, event_ids=("event",)),
        )
        with pytest.raises(ValueError, match=message):
            CompiledContext.model_validate(invalid_path)
