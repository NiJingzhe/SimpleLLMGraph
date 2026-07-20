import base64
from pathlib import Path

import pytest
from pydantic import ValidationError

from SimpleLLMFunc import Image
from SimpleLLMFunc.context.ir import ImageDetail, InputImageURL


def test_image_supports_http_and_base64_sources(tmp_path: Path) -> None:
    remote = Image.from_url(
        "https://example.test/image.png",
        detail=ImageDetail.HIGH,
    )
    assert remote.image_url == InputImageURL(
        url="https://example.test/image.png",
        detail=ImageDetail.HIGH,
    )

    inline = Image.from_base64(b"png", media_type="image/png")
    assert inline.image_url == "data:image/png;base64,cG5n"
    encoded = base64.b64encode(b"jpeg").decode("ascii")
    assert Image.from_base64(
        encoded,
        media_type="image/jpeg",
    ).image_url == f"data:image/jpeg;base64,{encoded}"

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"file")
    assert Image.from_path(image_path).image_url == "data:image/png;base64,ZmlsZQ=="


def test_image_rejects_invalid_sources(tmp_path: Path) -> None:
    for url in (
        "file:///tmp/image.png",
        "data:image/png,not-base64",
        "data:image/png;base64,not base64",
    ):
        with pytest.raises(ValidationError):
            Image.from_url(url)

    for media_type in ("text/plain", "image/", "image/png;utf8"):
        with pytest.raises(ValueError, match="image MIME type"):
            Image.from_base64(b"data", media_type=media_type)

    with pytest.raises(ValueError, match="valid base64"):
        Image.from_base64("not base64", media_type="image/png")

    unknown = tmp_path / "image.unknown"
    unknown.write_bytes(b"data")
    with pytest.raises(ValueError, match="determine image MIME type"):
        Image.from_path(unknown)

    explicit = Image.from_path(unknown, media_type="image/custom")
    assert explicit.image_url == "data:image/custom;base64,ZGF0YQ=="
