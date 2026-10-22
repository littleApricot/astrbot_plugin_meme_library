"""Exercise actual authenticated-bridge request adapters in isolation."""

import json
import unittest

from astrbot.api.web import PluginRequest, bind_request_context
from starlette.requests import Request

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
                "method": "POST" if payload else "GET",
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
