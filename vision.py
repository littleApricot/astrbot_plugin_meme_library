"""Build bounded image previews and validate model JSON without schema coercion."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

from PIL import Image, ImageOps


def parse_json(text: str) -> dict:
    """Read a JSON object, allowing only an enclosing Markdown code fence.

    Args:
        text: Model completion text.

    Returns:
        Parsed JSON object.

    Raises:
        ValueError: The completion is not a JSON object.
    """
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("invalid_model_result")
    return value


def inspect_image(data: bytes, maximum: int) -> str:
    """Validate size and image format before keeping the original bytes.

    Args:
        data: Original image bytes.
        maximum: Maximum allowed file size in bytes.

    Returns:
        A safe extension determined by decoding the file.

    Raises:
        ValueError: The image is oversized or has an unsupported format.
    """
    if len(data) > maximum:
        raise ValueError("image_too_large")
    with Image.open(io.BytesIO(data)) as image:
        if image.width * image.height > 40_000_000:
            raise ValueError("image_dimensions_too_large")
        suffix = {"JPEG": ".jpg", "PNG": ".png", "GIF": ".gif", "WEBP": ".webp"}.get(
            image.format
        )
        if suffix is None:
            raise ValueError("unsupported_image_format")
        image.verify()
    return suffix


def previews(path: Path) -> list[str]:
    """Return real image frames as portable JPEG data URLs.

    Args:
        path: Original stored image.

    Returns:
        One static frame or up to four evenly spaced animation frames.
    """
    frames = []
    with Image.open(path) as image:
        count = getattr(image, "n_frames", 1)
        indices = (
            sorted({round(i * (count - 1) / 3) for i in range(4)}) if count > 1 else [0]
        )
        for index in indices:
            image.seek(index)
            frame = ImageOps.exif_transpose(image.convert("RGB"))
            frame.thumbnail((1280, 1280))
            buffer = io.BytesIO()
            frame.save(buffer, format="JPEG", quality=88)
            frames.append(
                "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
            )
    return frames
