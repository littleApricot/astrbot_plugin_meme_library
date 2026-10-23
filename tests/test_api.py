"""Exercise actual authenticated-bridge request adapters in isolation."""

import asyncio
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from astrbot.api import AstrBotConfig
from astrbot.api.web import PluginRequest, bind_request_context
from starlette.requests import Request

from astrbot_plugin_meme_library.main import RECOGNITION

from . import test_library as fixtures


class ApiTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixtures.PluginTests.asyncSetUp
    asyncTearDown = fixtures.PluginTests.asyncTearDown
    ready = fixtures.PluginTests.ready

    async def call_api(self, name, payload=None, query=""):
        data = json.dumps(payload or {}).encode()

        async def receive():
            return {"type": "http.request", "body": data, "more_body": False}

        req = Request(
            {
                "type": "http",
                "http_version": "1.1",
                "method": "POST" if payload is not None else "GET",
                "scheme": "http",
                "path": "/api/test",
                "query_string": query.encode(),
                "headers": [],
                "server": ("127.0.0.1", 6196),
            },
            receive,
        )
        with bind_request_context(
            PluginRequest(
                req, plugin_name="astrbot_plugin_meme_library", username="test"
            )
        ):
            response = await getattr(self.plugin, "api_" + name)()
        return response.status_code, json.loads(response.body)

    async def test_management_pagination_edit_conflict_history_and_restore(self):
        row = self.ready()
        status, data = await self.call_api(
            "list", query="status=ready&limit=1&offset=0"
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(data["records"]), 1)
        self.assertNotIn("origin", data["records"][0])
        self.assertNotIn("document", data["records"][0])
        self.assertEqual(data["records"][0]["name"], row["name"])
        original_path = data["records"][0]["local_path"]
        status, data = await self.call_api(
            "document",
            {
                "id": row["id"],
                "revision": 1,
                "name": "新名称",
                "description": "修改后的短描述",
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(data["revision"], 2)
        self.assertEqual(data["local_path"], original_path)
        status, _ = await self.call_api(
            "document",
            {"id": row["id"], "revision": 1, "name": "旧名称", "description": "stale"},
        )
        self.assertEqual(status, 409)
        status, data = await self.call_api("revisions", query="id=" + row["id"])
        self.assertEqual(len(data["revisions"]), 2)
        status, _ = await self.call_api(
            "action",
            {
                "id": row["id"],
                "action": "restore",
                "revision": 2,
                "restore_revision": 1,
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            self.plugin.library.get(row["id"])["description"], row["description"]
        )
        self.assertEqual(self.plugin.library.get(row["id"])["name"], row["name"])
        _, data = await self.call_api("image", query="id=" + row["id"])
        self.assertTrue(data["data_url"].startswith("data:image/jpeg;base64,"))

    async def test_unknown_image_does_not_resolve_arbitrary_paths(self):
        status, _ = await self.call_api("image", query="id=../../secret")
        self.assertEqual(status, 404)

    def configure_persistence(self):
        schema = json.loads(
            (Path(__file__).parents[1] / "_conf_schema.json").read_text()
        )
        path = Path(self.temp.name) / "config.json"
        self.plugin.config = AstrBotConfig(str(path), schema=schema)
        return path, schema

    async def test_prompt_page_and_plugin_config_share_persisted_value_used_by_worker(
        self,
    ):
        path, schema = self.configure_persistence()
        existing = self.ready()
        self.plugin.library.acknowledge("group-a", self.plugin.library.latest_seq())
        status, data = await self.call_api("recognition_prompt")
        self.assertEqual(status, 200)
        self.assertEqual(set(data), {"recognition_prompt", "default_prompt"})
        self.assertEqual(data["recognition_prompt"], RECOGNITION)
        custom = "用群里的梗起名，描述只写适用场景。\n返回 is_meme、name、description 的 JSON。"
        status, data = await self.call_api(
            "recognition_prompt", {"recognition_prompt": custom, "enabled": False}
        )
        self.assertEqual(status, 200)
        reloaded = AstrBotConfig(str(path), schema=schema)
        self.assertEqual(reloaded["recognition_prompt"], custom)
        self.assertTrue(reloaded["enabled"])
        self.assertEqual(self.plugin.library.unseen("group-a"), 0)
        self.assertEqual(self.plugin.library.get(existing["id"]), existing)
        row, _ = self.plugin.library.add(
            fixtures.image_bytes("purple"), ".png", "group-a", "", ""
        )
        self.plugin._start_worker()
        for _ in range(100):
            if self.plugin.library.get(row["id"])["status"] in {"ready", "failed"}:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(self.plugin.library.get(row["id"])["status"], "ready")
        self.assertEqual(
            self.provider.text_chat.await_args.kwargs["system_prompt"], custom
        )
        self.plugin.config.save_config(
            replace_config={"recognition_prompt": "配置页的修改"}
        )
        _, data = await self.call_api("recognition_prompt")
        self.assertEqual(data["recognition_prompt"], "配置页的修改")
        status, _ = await self.call_api(
            "recognition_prompt", {"recognition_prompt": ""}
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            AstrBotConfig(str(path), schema=schema)["recognition_prompt"], ""
        )
        self.assertEqual(self.plugin._recognition_prompt(), RECOGNITION)

    async def test_invalid_prompt_or_write_failure_keeps_saved_config(self):
        path, _ = self.configure_persistence()
        original = path.read_bytes()
        for body in ({}, [], {"recognition_prompt": 123}):
            status, _ = await self.call_api("recognition_prompt", body)
            self.assertEqual(status, 400)
            self.assertEqual(path.read_bytes(), original)
        with patch.object(AstrBotConfig, "_write_config_snapshot", side_effect=OSError):
            status, _ = await self.call_api(
                "recognition_prompt", {"recognition_prompt": "未成功保存的提示词"}
            )
        self.assertEqual(status, 500)
        self.assertEqual(self.plugin.config["recognition_prompt"], RECOGNITION)
        self.assertEqual(path.read_bytes(), original)
