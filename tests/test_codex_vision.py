"""Check CLI process cleanup, credential boundaries and recognition routing."""

import asyncio
import base64
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from astrbot_plugin_meme_library.codex_vision import (
    CodexRecognitionError,
    recognize_codex,
)

from . import test_library as fixtures


class CodexProcessTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="codex-process-test-")
        self.root = Path(self.temp.name)
        self.binary = self.root / "test-cli"
        self.frames = [
            "data:image/jpeg;base64," + base64.b64encode(b"image-frame").decode()
        ] * 4

    def tearDown(self):
        self.temp.cleanup()

    def executable(self, body):
        self.binary.write_text("#!/usr/bin/env python3\n" + body)
        self.binary.chmod(0o700)
        return str(self.binary)

    async def test_cli_contract_uses_all_frames_and_preserves_shared_prompt(self):
        binary = self.executable('''import json, os, pathlib, sys
args = sys.argv[1:]
assert args[0:2] == ["exec", "-"]
assert args[args.index("--model") + 1] == "gpt-6-astra"
assert args[args.index("--sandbox") + 1] == "read-only"
assert "--ephemeral" in args and "--ignore-user-config" in args
assert args.count("--image") == 4
for index, arg in enumerate(args):
    if arg == "--image":
        assert pathlib.Path(args[index + 1]).read_bytes() == b"image-frame"
for key in ("CODEX_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
    assert key not in os.environ
for feature in ("shell_tool", "apps", "plugins", "multi_agent", "hooks"):
    assert any(args[i:i+2] == ["--disable", feature] for i in range(len(args)))
instruction = next(a.split("=", 1)[1] for a in args if a.startswith("model_instructions_file="))
assert pathlib.Path(json.loads(instruction)).read_text() == "shared custom instructions"
assert "nearby message" in next(iter(json.load(sys.stdin).values()))
schema = json.loads(pathlib.Path(args[args.index("--output-schema")+1]).read_text())
assert schema["additionalProperties"] is False
pathlib.Path(args[args.index("--output-last-message")+1]).write_text(json.dumps({
    "is_meme": True, "name": "角色笑", "description": "角色偷笑，用于调侃。"
}))
''')
        with patch.dict(os.environ, {
            "CODEX_API_KEY": "fixture-only",
            "OPENAI_API_KEY": "fixture-only",
            "OPENAI_BASE_URL": "https://invalid.example",
        }):
            result = await recognize_codex(
                self.frames, "shared custom instructions", "nearby message", binary
            )
        self.assertEqual(result["name"], "角色笑")

    async def test_timeout_and_cancellation_stop_the_real_child_and_remove_inputs(self):
        marker = self.root / "child.json"
        binary = self.executable(
            "import json, os, pathlib, time\n"
            f"pathlib.Path({str(marker)!r}).write_text(json.dumps({{"
            "'pid':os.getpid(),'cwd':str(pathlib.Path.cwd())}))\n"
            "time.sleep(20)\n"
        )
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                marker.unlink(missing_ok=True)
                task = asyncio.create_task(recognize_codex(
                    self.frames, "instructions", "", binary, 10 if cancel else 1
                ))
                for _ in range(100):
                    if marker.is_file():
                        break
                    await asyncio.sleep(0.01)
                self.assertTrue(marker.is_file())
                child = json.loads(marker.read_text())
                if cancel:
                    task.cancel()
                with self.assertRaises(asyncio.CancelledError if cancel else TimeoutError):
                    await task
                with self.assertRaises(ProcessLookupError):
                    os.kill(child["pid"], 0)
                self.assertFalse(Path(child["cwd"]).exists())

    async def test_missing_cli_invalid_output_and_failed_exit_are_bounded(self):
        with self.assertRaisesRegex(CodexRecognitionError, "codex_not_found"):
            await recognize_codex(self.frames, "", "", str(self.root / "missing"))
        invalid = json.dumps({"is_meme": "true", "name": "x", "description": "y"})
        invalid_script = (
            "import pathlib,sys\n"
            "path=pathlib.Path(sys.argv[sys.argv.index('--output-last-message')+1])\n"
            f"path.write_text({invalid!r})\n"
        )
        for body, expected in [
            ("import sys; sys.exit(7)\n", "codex_exit_7"),
            (invalid_script, "codex_invalid_result"),
        ]:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(CodexRecognitionError, expected):
                    await recognize_codex(self.frames, "", "", self.executable(body))


class CodexRoutingTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixtures.PluginTests.asyncSetUp
    asyncTearDown = fixtures.PluginTests.asyncTearDown
    ready = fixtures.PluginTests.ready

    async def test_only_collection_uses_codex_and_search_keeps_existing_provider(self):
        self.plugin.config.update({
            "recognition_backend": "codex",
            "recognition_prompt": "shared editable prompt",
        })
        row, _ = self.plugin.library.add(
            fixtures.image_bytes(), ".png", "group-a", "", "nearby message"
        )
        with patch("astrbot_plugin_meme_library.main.recognize_codex", new=AsyncMock(
            return_value={"is_meme": True, "name": "角色笑", "description": "角色偷笑。"}
        )) as recognize:
            self.plugin._start_worker()
            for _ in range(100):
                if self.plugin.library.get(row["id"])["status"] == "ready":
                    break
                await asyncio.sleep(0.01)
            self.assertEqual(self.plugin.library.get(row["id"])["status"], "ready")
            self.assertEqual(recognize.await_args.args[1:3], (
                "shared editable prompt", "nearby message"
            ))
            self.provider.text_chat.assert_not_awaited()
            self.provider.text_chat.return_value.completion_text = '{"matches":[]}'
            await self.plugin.search(fixtures.event(), "偷笑")
            self.provider.text_chat.assert_awaited_once()
            recognize.assert_awaited_once()

    async def test_missing_cli_is_visible_without_falling_back_to_another_model(self):
        self.plugin.config.update({
            "recognition_backend": "codex",
            "codex_path": str(Path(self.temp.name) / "missing-codex"),
        })
        row, _ = self.plugin.library.add(
            fixtures.image_bytes(), ".png", "group-a", "", ""
        )
        self.plugin._start_worker()
        for _ in range(100):
            if self.plugin.library.get(row["id"])["status"] == "failed":
                break
            await asyncio.sleep(0.01)
        result = self.plugin.library.get(row["id"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"], "codex_not_found")
        self.assertEqual(result["revision"], 0)
        self.provider.text_chat.assert_not_awaited()
