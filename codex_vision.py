"""Recognize new library images through a locally authenticated Codex CLI."""

from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path
import shutil
import tempfile

from .vision import parse_json


class CodexRecognitionError(RuntimeError):
    """Expose a bounded error code without retaining CLI output or credentials."""


async def recognize_codex(
    frames: list[str],
    prompt: str,
    source_text: str,
    executable: str = "",
    timeout: float = 120,
) -> dict:
    """Classify image frames with Astra using the host user's ChatGPT login.

    Args:
        frames: JPEG data URLs produced by the existing image preview function.
        prompt: The library's editable recognition instructions.
        source_text: Nearby message used only as an image usage clue.
        executable: CLI name or absolute path; empty enables local discovery.
        timeout: Maximum duration in seconds, including CLI startup.

    Returns:
        Validated is_meme, name and description fields.

    Raises:
        CodexRecognitionError: The CLI is missing, fails or returns invalid fields.
        TimeoutError: Recognition exceeds the configured duration.
    """
    candidates = (
        [str(Path(executable).expanduser())]
        if executable.strip()
        else ["codex", str(Path.home() / ".local/bin/codex")]
    )
    binary = next(
        (path for candidate in candidates if (path := shutil.which(candidate))), None
    )
    if binary is None:
        raise CodexRecognitionError("codex_not_found")
    if not frames:
        raise CodexRecognitionError("codex_image_required")
    schema = {
        "type": "object",
        "properties": {
            "is_meme": {"type": "boolean"},
            "name": {"type": "string", "minLength": 1, "maxLength": 24},
            "description": {"type": "string", "minLength": 1, "maxLength": 120},
        },
        "required": ["is_meme", "name", "description"],
        "additionalProperties": False,
    }
    with tempfile.TemporaryDirectory(prefix="astrbot-meme-codex-") as directory:
        root = Path(directory)
        schema_path = root / "schema.json"
        schema_path.write_text(json.dumps(schema), encoding="utf-8")
        instructions = root / "instructions.txt"
        instructions.write_text(prompt, encoding="utf-8")
        output = root / "result.json"
        args = [
            binary,
            "exec",
            "-",
            "--model", "gpt-6-astra",
            "--ignore-user-config",
            "--ephemeral",
            "--skip-git-repo-check",
            "--sandbox", "read-only",
            "--color", "never",
            "--output-schema", str(schema_path),
            "--output-last-message", str(output),
            "-c", 'forced_login_method="chatgpt"',
            "-c", 'model_reasoning_effort="medium"',
            "-c", "model_instructions_file=" + json.dumps(str(instructions)),
            "-c", "project_doc_max_bytes=0",
            "-c", 'web_search="disabled"',
            "-c", "skills.bundled.enabled=false",
            "-c", "skills.include_instructions=false",
        ]
        for feature in (
            "shell_tool", "apps", "plugins", "remote_plugin", "multi_agent",
            "hooks", "memories", "goals", "view_image",
        ):
            args.extend(["--disable", feature])
        for index, frame in enumerate(frames):
            prefix = "data:image/jpeg;base64,"
            if not frame.startswith(prefix):
                raise CodexRecognitionError("codex_invalid_image")
            image = root / f"frame-{index}.jpg"
            image.write_bytes(base64.b64decode(frame[len(prefix):], validate=True))
            args.extend(["--image", str(image)])
        # Force account auth and keep inherited API credentials out of this child.
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in {"CODEX_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"}
        }
        process = await asyncio.create_subprocess_exec(
            *args,
            cwd=root,
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            await asyncio.wait_for(
                process.communicate(json.dumps(
                    {"图片附近的发言（仅作使用线索）": source_text}, ensure_ascii=False,
                ).encode("utf-8")),
                timeout=max(1, float(timeout)),
            )
        except BaseException:
            if process.returncode is None:
                process.kill()
            await process.wait()
            raise
        if process.returncode:
            raise CodexRecognitionError(f"codex_exit_{process.returncode}")
        if not output.is_file() or output.stat().st_size > 16_384:
            raise CodexRecognitionError("codex_invalid_result")
        value = parse_json(output.read_text(encoding="utf-8"))
        if (
            set(value) != {"is_meme", "name", "description"}
            or type(value["is_meme"]) is not bool
            or not isinstance(value["name"], str)
            or not 1 <= len(value["name"].strip()) <= 24
            or not isinstance(value["description"], str)
            or not 1 <= len(value["description"].strip()) <= 120
        ):
            raise CodexRecognitionError("codex_invalid_result")
        return value
