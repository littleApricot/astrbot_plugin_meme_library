"""Check collection filters with synthetic images and an isolated library."""

import asyncio
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from astrbot.api import AstrBotConfig
from astrbot.api.message_components import Image
from PIL import Image as PILImage

from astrbot_plugin_meme_library.main import MemeLibrary
from astrbot_plugin_meme_library.vision import ImageFiltered, inspect_image

from .test_library import event


def picture(size, format="PNG"):
    buffer = io.BytesIO()
    PILImage.new("RGB", size, "blue").save(buffer, format=format)
    return buffer.getvalue()


@pytest.mark.parametrize(
    "size,allowed",
    [
        ((1200, 400), True),
        ((400, 1200), True),
        ((1080, 1080), True),
        ((1201, 1000), False),
        ((1000, 1201), False),
        ((1200, 399), False),
        ((399, 1200), False),
    ],
)
def test_original_dimension_boundaries(size, allowed):
    data = picture(size)
    if allowed:
        assert inspect_image(
            data, 15 * 1024 * 1024, max_long_edge=1200, max_aspect_ratio=3
        ) == ".png"
    else:
        with pytest.raises(ImageFiltered):
            inspect_image(
                data, 15 * 1024 * 1024, max_long_edge=1200, max_aspect_ratio=3
            )


class CollectionPrefilterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.provider = SimpleNamespace(
            provider_config={"modalities": ["text", "image"]},
            text_chat=AsyncMock(
                return_value=SimpleNamespace(
                    completion_text=json.dumps(
                        {"is_meme": True, "name": "测试表情", "description": "测试用蓝色图片"}
                    )
                )
            ),
        )
        context = SimpleNamespace(
            register_web_api=MagicMock(),
            get_using_provider_async=AsyncMock(return_value=self.provider),
            get_provider_by_id=MagicMock(return_value=self.provider),
            get_config=lambda **kwargs: {"timezone": "Asia/Shanghai"},
        )
        self.location = patch(
            "astrbot_plugin_meme_library.main.StarTools.get_data_dir",
            return_value=self.root / "library",
        )
        self.location.start()
        self.plugin = MemeLibrary(context, {"enabled": True})
        self.codex = AsyncMock(side_effect=AssertionError("must not invoke Codex"))
        self.codex_patch = patch(
            "astrbot_plugin_meme_library.main.recognize_codex", self.codex
        )
        self.codex_patch.start()

    async def asyncTearDown(self):
        await self.plugin.terminate()
        self.codex_patch.stop()
        self.location.stop()
        self.temp.cleanup()

    def incoming(self, size):
        path = self.root / f"incoming-{size[0]}-{size[1]}.png"
        path.write_bytes(picture(size))
        return event(image=Image.fromFileSystem(path))

    async def drain(self):
        for _ in range(100):
            counts = self.plugin.library.counts()
            if not counts.get("queued") and not counts.get("processing"):
                return
            await asyncio.sleep(0.01)
        self.fail("synthetic collection queue did not finish")

    async def test_filtered_arrivals_never_enter_library_or_model_queue(self):
        for backend in ("provider", "codex"):
            self.plugin.config["recognition_backend"] = backend
            for size in ((1201, 1000), (1000, 1201), (900, 299), (299, 900)):
                incoming = self.incoming(size)
                original = copy.deepcopy(incoming.message_obj.message)
                await self.plugin.collect(incoming)
                self.assertEqual(incoming.message_obj.message, original)
                incoming.send.assert_not_awaited()
        self.assertEqual(self.plugin.library.counts(), {})
        self.assertEqual(list((self.root / "library" / "images").iterdir()), [])
        self.assertIsNone(self.plugin._worker)
        self.provider.text_chat.assert_not_awaited()
        self.codex.assert_not_awaited()

    async def test_boundary_arrivals_still_reach_recognition(self):
        for size in ((1200, 400), (400, 1200), (1080, 1080)):
            await self.plugin.collect(self.incoming(size))
            await self.drain()
        self.assertEqual(self.plugin.library.counts(), {"ready": 3})
        self.assertEqual(self.provider.text_chat.await_count, 3)

    async def test_saved_configuration_controls_both_filters(self):
        schema = json.loads((Path(__file__).parents[1] / "_conf_schema.json").read_text())
        path = self.root / "config.json"
        config = AstrBotConfig(str(path), schema=schema)
        self.assertEqual(config["max_image_long_edge"], 1200)
        self.assertEqual(config["max_image_aspect_ratio"], 3)
        config["max_image_long_edge"] = 1600
        config["max_image_aspect_ratio"] = 4
        config.save_config()
        self.plugin.config = AstrBotConfig(str(path), schema=schema)
        for size in ((1500, 1000), (800, 200)):
            await self.plugin.collect(self.incoming(size))
            await self.drain()
        self.assertEqual(self.plugin.library.counts(), {"ready": 2})
        self.assertEqual(self.provider.text_chat.await_count, 2)

    async def test_old_queue_filters_before_previews_and_keeps_processing(self):
        oversized, _ = self.plugin.library.add(
            picture((1500, 1000)), ".png", "group-a", "", ""
        )
        normal, _ = self.plugin.library.add(
            picture((300, 300)), ".png", "group-a", "", ""
        )
        self.plugin._start_worker()
        await self.drain()
        self.assertEqual(self.plugin.library.get(oversized["id"])["status"], "rejected")
        self.assertEqual(self.plugin.library.get(normal["id"])["status"], "ready")
        self.assertEqual(self.provider.text_chat.await_count, 1)

    async def test_old_codex_queue_is_filtered_before_backend_invocation(self):
        self.plugin.config["recognition_backend"] = "codex"
        row, _ = self.plugin.library.add(
            picture((900, 299)), ".png", "group-a", "", ""
        )
        with patch(
            "astrbot_plugin_meme_library.main.previews",
            side_effect=AssertionError("must filter before building previews"),
        ) as preview:
            self.plugin._start_worker()
            await self.drain()
        self.assertEqual(self.plugin.library.get(row["id"])["status"], "rejected")
        preview.assert_not_called()
        self.provider.text_chat.assert_not_awaited()
        self.codex.assert_not_awaited()

    async def test_download_retry_does_not_bypass_filter_or_store_image(self):
        incoming = self.incoming((1500, 1000))
        reference = incoming.get_messages()[-1].file
        row = self.plugin.library.download_error(reference, "group-a", "", "", "test")
        self.plugin.library.set_status(row["id"], "queued")
        self.plugin._start_worker()
        await self.drain()
        self.assertEqual(self.plugin.library.counts(), {"rejected": 1})
        self.assertEqual(list((self.root / "library" / "images").iterdir()), [])
        self.provider.text_chat.assert_not_awaited()
        self.codex.assert_not_awaited()
