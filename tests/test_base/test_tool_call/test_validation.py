"""Tests for base.tool_call.validation module."""

from __future__ import annotations

from SimpleLLMFunc.base.tool_call.validation import (
    is_valid_tool_result,
    serialize_tool_output_for_langfuse,
)
from SimpleLLMFunc.type.multimodal import ImgPath, ImgUrl, Text


class TestIsValidToolResult:
    """Tests for is_valid_tool_result function."""

    def test_valid_string(self) -> None:
        """Test string result validation."""
        assert is_valid_tool_result("test") is True

    def test_valid_img_url(self, img_url: ImgUrl) -> None:
        """Test ImgUrl result validation."""
        assert is_valid_tool_result(img_url) is True

    def test_valid_img_path(self, img_path: ImgPath) -> None:
        """Test ImgPath result validation."""
        assert is_valid_tool_result(img_path) is True

    def test_valid_dict(self) -> None:
        """Test dict result validation."""
        assert is_valid_tool_result({"key": "value"}) is True

    def test_valid_list(self) -> None:
        """Test list result validation."""
        assert is_valid_tool_result([1, 2, 3]) is True

    def test_valid_tuple_with_image(self, img_url: ImgUrl) -> None:
        """Test tuple with image validation."""
        result = ("text", img_url)
        assert is_valid_tool_result(result) is True

    def test_valid_tuple_with_multiple_images(
        self,
        img_path: ImgPath,
        img_url: ImgUrl,
    ) -> None:
        """Test tuple with multiple images validation."""
        result = ("text", [img_url, img_path])
        assert is_valid_tool_result(result) is True

    def test_invalid_tuple(self) -> None:
        """Test invalid tuple validation."""
        result = ("text", "not_image")
        assert is_valid_tool_result(result) is False

    def test_invalid_type(self) -> None:
        """Test invalid type validation."""
        # Create a non-serializable object
        class NonSerializable:
            pass

        assert is_valid_tool_result(NonSerializable()) is False


class TestSerializeToolOutputForLangfuse:
    """Tests for serialize_tool_output_for_langfuse function."""

    def test_serialize_string(self) -> None:
        """Test serializing string."""
        result = serialize_tool_output_for_langfuse("test")
        assert result == "test"

    def test_serialize_img_url(self, img_url: ImgUrl) -> None:
        """Test serializing ImgUrl."""
        result = serialize_tool_output_for_langfuse(img_url)
        assert result["type"] == "image_url"
        assert result["url"] == img_url.url

    def test_serialize_img_path(self, img_path: ImgPath) -> None:
        """Test serializing ImgPath."""
        result = serialize_tool_output_for_langfuse(img_path)
        assert result["type"] == "image_path"
        assert "path" in result

    def test_serialize_text_object(self, text_content: Text) -> None:
        """Test serializing Text object."""
        result = serialize_tool_output_for_langfuse(text_content)
        assert isinstance(result, str)

    def test_serialize_tuple_with_image(self, img_url: ImgUrl) -> None:
        """Test serializing tuple with image."""
        result = serialize_tool_output_for_langfuse(("text", img_url))
        assert result["type"] == "text_with_image"
        assert result["text"] == "text"
        assert "image" in result

    def test_serialize_tuple_with_multiple_images(
        self,
        img_path: ImgPath,
        img_url: ImgUrl,
    ) -> None:
        """Test serializing tuple with multiple images."""
        result = serialize_tool_output_for_langfuse(("text", [img_url, img_path]))
        assert result["type"] == "text_with_images"
        assert result["text"] == "text"
        assert len(result["images"]) == 2

    def test_serialize_dict(self) -> None:
        """Test serializing dict."""
        data = {"key": "value", "number": 123}
        result = serialize_tool_output_for_langfuse(data)
        assert result == data

    def test_serialize_list(self) -> None:
        """Test serializing list."""
        data = [1, 2, 3]
        result = serialize_tool_output_for_langfuse(data)
        assert result == data

    def test_serialize_non_serializable(self) -> None:
        """Test serializing non-serializable object."""
        class NonSerializable:
            def __str__(self) -> str:
                return "non-serializable"

        obj = NonSerializable()
        result = serialize_tool_output_for_langfuse(obj)
        assert result == "non-serializable"
