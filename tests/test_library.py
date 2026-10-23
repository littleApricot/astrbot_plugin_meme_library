"""Verify collection, shared documents and actual AstrBot vision tool execution."""

import asyncio
import copy
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from astrbot.api import AstrBotConfig
from astrbot.api.message_components import Image, Plain
from astrbot.core.agent.hooks import BaseAgentRunHooks
from astrbot.core.agent.message import dump_messages_with_checkpoints
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
from astrbot.core.platform.message_type import MessageType
from astrbot.core.platform.platform_metadata import PlatformMetadata
from astrbot.core.provider.entities import LLMResponse, ProviderMeta, ProviderRequest
from astrbot.core.provider.provider import Provider
from PIL import Image as PILImage

from astrbot_plugin_meme_library.main import RECOGNITION, MemeLibrary
from astrbot_plugin_meme_library.store import Library
from astrbot_plugin_meme_library.vision import inspect_image, previews


def image_bytes(color="red"):
    buffer = io.BytesIO()
    PILImage.new("RGB", (30, 20), color).save(buffer, format="PNG")
    return buffer.getvalue()


def event(group="group-a", image=None):
    msg = AstrBotMessage()
    msg.type = MessageType.GROUP_MESSAGE
    msg.self_id = "bot"
    msg.session_id = msg.group_id = group
    msg.message_id = "test-" + group
    msg.sender = MessageMember(user_id="ordinary-member", nickname="Member")
    msg.message = [Plain("开心")] + ([image] if image else [])
    msg.message_str = "开心"
    result = AstrMessageEvent(
        "开心", msg, PlatformMetadata("qq_official", "qq_official", "test"), group
    )
    result.send = AsyncMock()
    return result


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Library(Path(self.temp.name))

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def ready(self):
        row, _ = self.store.add(image_bytes(), ".png", "group-a", "", "caption")
        self.store.set_status(row["id"], "processing")
        self.store.recognize(row["id"], "红色庆祝", "开心的红色图，表示庆祝", True)
        return self.store.get(row["id"])

    def test_shared_dedup_and_unified_revision_with_no_update_notice(self):
        row = self.ready()
        again, added = self.store.add(
            image_bytes(), ".png", "group-b", "", "other caption"
        )
        self.assertFalse(added)
        self.assertEqual(again["id"], row["id"])
        self.store.acknowledge("session", self.store.latest_seq())
        updated = self.store.edit(
            row["id"], "红色嘴硬", "群里也用来表达嘴硬", 1, "群友纠正", "model"
        )
        self.assertEqual(updated["revision"], 2)
        self.assertEqual(self.store.get(row["id"])["description"], "群里也用来表达嘴硬")
        self.assertEqual(len(self.store.revisions(row["id"])), 2)
        self.assertEqual(self.store.unseen("session"), 0)
        self.store.set_status(row["id"], "disabled")
        self.assertEqual(self.store.unseen("session"), 0)

    def test_conflicting_edits_preserve_newer_document(self):
        row = self.ready()
        self.store.edit(row["id"], "短名称", "new", 1, "fix", "model")
        with self.assertRaisesRegex(ValueError, "revision_conflict"):
            self.store.edit(row["id"], "短名称", "stale", 1, "fix", "model")
        self.assertEqual(self.store.get(row["id"])["description"], "new")

    def test_late_recognition_cannot_overwrite_manual_document(self):
        row, _ = self.store.add(image_bytes(), ".png", "group", "", "")
        self.store.set_status(row["id"], "processing")
        self.store.edit(row["id"], "短名称", "manual correction", 0, "fix", "dashboard")
        self.store.recognize(row["id"], "红色庆祝", "late vision output", True)
        row = self.store.get(row["id"])
        self.assertEqual(row["description"], "manual correction")
        self.assertEqual(row["status"], "failed")

    def test_multiframe_previews_and_size_validation(self):
        buffer = io.BytesIO()
        frames = [
            PILImage.new("RGB", (15, 15), color)
            for color in ("red", "green", "blue", "yellow")
        ]
        frames[0].save(
            buffer,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0,
        )
        data = buffer.getvalue()
        self.assertEqual(inspect_image(data, 100000), ".gif")
        row, _ = self.store.add(data, ".gif", "group", "", "")
        self.assertEqual(len(previews(self.store.path(row))), 4)
        with self.assertRaisesRegex(ValueError, "image_too_large"):
            inspect_image(data, 5)

    def test_legacy_database_migrates_without_losing_originals_or_notices(self):
        root = Path(self.temp.name) / "legacy"
        (root / "images").mkdir(parents=True)
        original = "旧图片的详细说明。" * 80
        image = image_bytes()
        (root / "images/old.gif").write_bytes(image)
        with sqlite3.connect(root / "library.sqlite3") as db:
            db.executescript("""
                CREATE TABLE memes(id TEXT PRIMARY KEY,digest TEXT UNIQUE NOT NULL,filename TEXT NOT NULL,status TEXT NOT NULL,document TEXT NOT NULL DEFAULT '',revision INTEGER NOT NULL DEFAULT 0,origin TEXT NOT NULL,provider_id TEXT NOT NULL,source_text TEXT NOT NULL,error TEXT NOT NULL DEFAULT '',added_seq INTEGER NOT NULL DEFAULT 0,created_at REAL NOT NULL,updated_at REAL NOT NULL);
                CREATE TABLE revisions(meme_id TEXT NOT NULL,revision INTEGER NOT NULL,document TEXT NOT NULL,reason TEXT NOT NULL,actor TEXT NOT NULL,created_at REAL NOT NULL,PRIMARY KEY(meme_id,revision));
                CREATE TABLE cursors(session TEXT PRIMARY KEY,seq INTEGER NOT NULL);
            """)
            db.execute(
                "INSERT INTO memes VALUES('old','digest','old.gif','ready',?,1,'group','','','',1,1,1)",
                (original,),
            )
            db.execute(
                "INSERT INTO revisions VALUES('old',1,?,'首次识别','vision',1)",
                (original,),
            )
            db.execute("INSERT INTO cursors VALUES('session',1)")
        migrated = Library(root)
        try:
            row = migrated.get("old")
            self.assertEqual(row["document"], original)
            self.assertLessEqual(len(row["name"]), 24)
            self.assertLessEqual(len(row["description"]), 120)
            self.assertEqual(migrated.path(row).read_bytes(), image)
            self.assertEqual(migrated.unseen("session"), 0)
            migrated.edit("old", "新名称", "简短且实用的描述。", 1, "精简", "dashboard")
            self.assertEqual(migrated.revisions("old")[1]["document"], original)
        finally:
            migrated.close()
        reopened = Library(root)
        try:
            self.assertEqual(reopened.get("old")["name"], "新名称")
            self.assertEqual(reopened.get("old")["revision"], 2)
            self.assertEqual(reopened.unseen("session"), 0)
        finally:
            reopened.close()

    def test_metadata_lengths_and_both_field_names_are_searchable(self):
        row = self.ready()
        for name, description in [("长" * 25, "短描述"), ("短名称", "长" * 121)]:
            with self.assertRaises(ValueError):
                self.store.edit(row["id"], name, description, 1, "test", "model")
        self.store.edit(
            row["id"], "捂嘴偷笑", "粉色桃子捂嘴，表示憋笑或害羞。", 1, "fix", "model"
        )
        self.assertEqual(self.store.records(query="捂嘴偷笑")[0]["id"], row["id"])
        self.assertEqual(self.store.records(query="害羞")[0]["id"], row["id"])


class PluginTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.provider = MagicMock(spec=Provider)
        self.provider.provider_config = {
            "id": "fake-vision",
            "modalities": ["text", "image"],
            "max_context_tokens": 128000,
        }
        self.provider.get_model.return_value = "fake-vision"
        self.provider.meta.return_value = ProviderMeta(
            "fake-vision", "fake-vision", "test", None
        )
        self.provider.text_chat = AsyncMock(
            return_value=LLMResponse(
                role="assistant",
                completion_text=json.dumps(
                    {
                        "is_meme": True,
                        "name": "开心庆祝",
                        "description": "红色图片，开心庆祝",
                    }
                ),
            )
        )
        self.context = SimpleNamespace(
            register_web_api=MagicMock(),
            get_using_provider_async=AsyncMock(return_value=self.provider),
            get_provider_by_id=MagicMock(return_value=self.provider),
        )
        self.patch = patch(
            "astrbot_plugin_meme_library.main.StarTools.get_data_dir",
            return_value=Path(self.temp.name),
        )
        self.patch.start()
        self.plugin = MemeLibrary(
            self.context, {"enabled": True, "search_batch_chars": 16000}
        )

    async def asyncTearDown(self):
        await self.plugin.terminate()
        self.patch.stop()
        self.temp.cleanup()

    def ready(self):
        row, _ = self.plugin.library.add(image_bytes(), ".png", "group-a", "", "")
        self.plugin.library.set_status(row["id"], "processing")
        self.plugin.library.recognize(row["id"], "开心庆祝", "红色图片，开心庆祝", True)
        return self.plugin.library.get(row["id"])

    async def test_first_image_is_recognized_and_other_groups_reuse_it(self):
        path = Path(self.temp.name) / "source.png"
        path.write_bytes(image_bytes())
        first = event(image=Image.fromFileSystem(path))
        await self.plugin.collect(first)
        for _ in range(100):
            if self.plugin.library.counts().get("ready"):
                break
            await asyncio.sleep(0.01)
        self.assertEqual(self.plugin.library.counts(), {"ready": 1})
        self.assertEqual(self.provider.text_chat.await_count, 1)
        await self.plugin.collect(event("group-b", Image.fromFileSystem(path)))
        self.assertEqual(self.provider.text_chat.await_count, 1)
        self.assertEqual(self.plugin.library.counts(), {"ready": 1})

    async def test_saved_recognition_prompt_reaches_worker_and_blank_uses_default(self):
        schema = json.loads(
            (Path(__file__).parents[1] / "_conf_schema.json").read_text()
        )
        config_path = Path(self.temp.name) / "plugin_config.json"
        config_path.write_text(json.dumps({"enabled": True, "search_results": 3}))
        config = AstrBotConfig(str(config_path), schema=schema)
        self.assertEqual(config["recognition_prompt"], RECOGNITION)
        self.assertEqual(config["search_results"], 3)
        custom = (
            "按群聊梗命名，描述只写用途。\n返回 JSON，包含 is_meme、name、description。"
        )
        for color, saved_prompt, expected in [
            ("blue", custom, custom),
            ("green", " \n ", RECOGNITION),
        ]:
            config["recognition_prompt"] = saved_prompt
            config.save_config()
            self.plugin.config = AstrBotConfig(str(config_path), schema=schema)
            self.assertEqual(self.plugin.config["recognition_prompt"], saved_prompt)
            row, _ = self.plugin.library.add(
                image_bytes(color), ".png", "group-a", "", "附近的聊天"
            )
            self.plugin._start_worker()
            for _ in range(100):
                if self.plugin.library.get(row["id"])["status"] != "queued":
                    break
                await asyncio.sleep(0.01)
            for _ in range(100):
                if self.plugin.library.get(row["id"])["status"] != "processing":
                    break
                await asyncio.sleep(0.01)
            self.assertEqual(self.plugin.library.get(row["id"])["status"], "ready")
            call = self.provider.text_chat.await_args.kwargs
            self.assertEqual(call["system_prompt"], expected)
            self.assertIn("附近的聊天", call["prompt"])
            self.assertTrue(call["image_urls"])

    async def test_failed_download_can_retry_and_deduplicates_real_image(self):
        path = Path(self.temp.name) / "retry.png"
        row = self.plugin.library.download_error(
            str(path), "group-a", "", "", "download_FileNotFoundError"
        )
        self.assertEqual(row["status"], "failed")
        path.write_bytes(image_bytes())
        self.plugin.library.set_status(row["id"], "queued")
        self.plugin._start_worker()
        for _ in range(100):
            if self.plugin.library.counts().get("ready"):
                break
            await asyncio.sleep(0.01)
        self.assertEqual(self.plugin.library.counts(), {"ready": 1, "retried": 1})
        self.assertEqual(self.provider.text_chat.await_count, 1)
        self.assertEqual(self.plugin.library.latest_seq(), 1)

    async def test_gif_bytes_keep_animation_through_collection_and_visual_recognition(
        self,
    ):
        buffer = io.BytesIO()
        frames = [
            PILImage.new("RGB", (20, 20), c) for c in ["red", "green", "blue", "yellow"]
        ]
        frames[0].save(
            buffer,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0,
        )
        original = buffer.getvalue()
        await self.plugin.collect(event(image=Image.fromBytes(original)))
        for _ in range(100):
            if self.plugin.library.counts().get("ready"):
                break
            await asyncio.sleep(0.01)
        row = self.plugin.library.records()[0]
        self.assertEqual(row["status"], "ready")
        self.assertTrue(row["filename"].endswith(".gif"))
        self.assertEqual(self.plugin.library.path(row).read_bytes(), original)
        self.assertEqual(
            len(self.provider.text_chat.await_args.kwargs["image_urls"]), 4
        )

    async def test_only_additions_notify_and_silence_does_not_consume_notice(self):
        row = self.ready()
        e = event()
        original = [{"role": "user", "content": "old history"}]
        req = ProviderRequest(prompt="hi", contexts=copy.deepcopy(original))
        await self.plugin.prepare(e, req)
        self.assertEqual(req.contexts, original)
        self.assertEqual(req.extra_user_content_parts[0].text, "图库有新增表情包。")
        self.assertEqual(self.plugin.library.unseen(e.unified_msg_origin), 1)
        await self.plugin.acknowledge(
            e, LLMResponse(role="assistant", completion_text="hi")
        )
        self.plugin.library.edit(
            row["id"], "新名称", "更新说明", 1, "correction", "model"
        )
        self.plugin.library.set_status(row["id"], "disabled")
        next_req = ProviderRequest(prompt="next")
        await self.plugin.prepare(event(), next_req)
        self.assertEqual(next_req.extra_user_content_parts, [])

    async def test_failed_recognition_is_visible_and_retryable(self):
        self.provider.text_chat.side_effect = TimeoutError()
        path = Path(self.temp.name) / "source.png"
        path.write_bytes(image_bytes())
        await self.plugin.collect(event(image=Image.fromFileSystem(path)))
        for _ in range(100):
            if self.plugin.library.counts().get("failed"):
                break
            await asyncio.sleep(0.01)
        row = self.plugin.library.records()[0]
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["error"], "TimeoutError")
        self.provider.text_chat.side_effect = None
        self.plugin.library.set_status(row["id"], "queued")
        self.plugin._wake.set()
        for _ in range(100):
            if self.plugin.library.counts().get("ready"):
                break
            await asyncio.sleep(0.01)
        self.assertEqual(self.plugin.library.counts(), {"ready": 1})

    async def test_preview_is_required_and_parallel_sends_are_deduplicated(self):
        row = self.ready()
        e = event()
        req = ProviderRequest()
        e.set_extra("provider_request", req)
        denied = json.loads(await self.plugin.send(e, row["id"], 1, ""))
        self.assertFalse(denied["sent"])
        await self.plugin.view(e, row["id"])
        denied = json.loads(await self.plugin.send(e, row["id"], 1, ""))
        self.assertFalse(denied["sent"])
        req.tool_calls_result = [SimpleNamespace()]

        async def slow_send(chain):
            await asyncio.sleep(0.02)

        e.send.side_effect = slow_send
        results = await asyncio.gather(
            self.plugin.send(e, row["id"], 1, "hello"),
            self.plugin.send(e, row["id"], 1, "hello"),
        )
        self.assertEqual(sum(json.loads(r)["sent"] for r in results), 1)
        self.assertEqual(e.send.await_count, 1)

    async def test_ordinary_member_can_correct_same_document_and_stale_selection_fails(
        self,
    ):
        row = self.ready()
        e = event()
        req = ProviderRequest()
        e.set_extra("provider_request", req)
        await self.plugin.view(e, row["id"])
        req.tool_calls_result = [SimpleNamespace()]
        result = json.loads(
            await self.plugin.edit(
                e, row["id"], "夸人厉害", "这张在群里是夸人厉害", 1, "普通群友明确纠正"
            )
        )
        self.assertTrue(result["ok"])
        self.assertEqual(
            self.plugin.library.get(row["id"])["description"], result["description"]
        )
        stale = json.loads(await self.plugin.send(e, row["id"], 1, ""))
        self.assertFalse(stale["sent"])
        e.send.assert_not_awaited()

    async def test_actual_runner_sees_image_then_sends_and_keeps_tool_history(self):
        row = self.ready()
        e = event()
        seen = []

        async def completion(**kwargs):
            seen.append(copy.deepcopy(kwargs["contexts"]))
            if len(seen) == 1:
                return LLMResponse(
                    role="assistant",
                    tools_call_name=["meme_view"],
                    tools_call_args=[{"meme_id": row["id"]}],
                    tools_call_ids=["view-1"],
                )
            if len(seen) == 2:
                self.assertIn("data:image/jpeg;base64,", json.dumps(kwargs["contexts"]))
                self.assertIn("local_path", json.dumps(kwargs["contexts"]))
                self.assertIn(
                    str(self.plugin.library.path(row).resolve()),
                    json.dumps(kwargs["contexts"]),
                )
                return LLMResponse(
                    role="assistant",
                    tools_call_name=["meme_send"],
                    tools_call_args=[
                        {
                            "meme_id": row["id"],
                            "expected_revision": 1,
                            "caption": "恭喜",
                        }
                    ],
                    tools_call_ids=["send-1"],
                )
            return LLMResponse(role="assistant", completion_text="已发送")

        self.provider.text_chat.side_effect = completion
        tools = ToolSet(
            [
                FunctionTool(
                    name="meme_view",
                    description="view",
                    parameters={
                        "type": "object",
                        "properties": {"meme_id": {"type": "string"}},
                        "required": ["meme_id"],
                    },
                    handler=self.plugin.view,
                ),
                FunctionTool(
                    name="meme_send",
                    description="send",
                    parameters={
                        "type": "object",
                        "properties": {
                            "meme_id": {"type": "string"},
                            "expected_revision": {"type": "number"},
                            "caption": {"type": "string"},
                        },
                        "required": ["meme_id", "expected_revision", "caption"],
                    },
                    handler=self.plugin.send,
                ),
            ]
        )
        req = ProviderRequest(
            prompt="庆祝一下",
            func_tool=tools,
            conversation=SimpleNamespace(cid="conv", token_usage=0),
        )
        e.set_extra("provider_request", req)
        runner = ToolLoopAgentRunner()
        await runner.reset(
            provider=self.provider,
            request=req,
            run_context=ContextWrapper(context=SimpleNamespace(event=e)),
            tool_executor=FunctionToolExecutor(),
            agent_hooks=BaseAgentRunHooks(),
        )
        async for _ in runner.step_until_done(5):
            pass
        e.send.assert_awaited_once()
        saved = dump_messages_with_checkpoints(runner.run_context.messages)
        self.assertIn(
            '"sent": true', json.dumps(saved, ensure_ascii=False).replace('\\"', '"')
        )
        self.assertTrue(
            any(
                m.get("role") == "tool" and m.get("tool_call_id") == "view-1"
                for m in saved
            )
        )
        self.assertIn("data:image/jpeg;base64,", json.dumps(saved))

    async def test_semantic_search_uses_updated_document_across_batches(self):
        row = self.ready()
        self.plugin.library.edit(
            row["id"], "夸人厉害", "实际上表示夸人厉害", 1, "纠正", "model"
        )
        self.provider.text_chat.return_value = LLMResponse(
            role="assistant",
            completion_text=json.dumps({"matches": [{"id": row["id"], "score": 0.95}]}),
        )
        result = json.loads(await self.plugin.search(event("group-b"), "夸人厉害"))
        self.assertEqual(result["candidates"][0]["revision"], 2)
        self.assertEqual(result["candidates"][0]["name"], "夸人厉害")
        self.assertNotIn("document", result["candidates"][0])
        path = Path(result["candidates"][0]["local_path"])
        self.assertTrue(path.is_absolute())
        self.assertEqual(path.read_bytes(), image_bytes())
        self.assertIn("夸人厉害", result["candidates"][0]["description"])
        prompt = self.provider.text_chat.await_args.kwargs["prompt"]
        self.assertNotIn("开心庆祝", prompt)

    async def test_view_exposes_original_gif_path_for_upload_instead_of_preview(self):
        buffer = io.BytesIO()
        PILImage.new("RGB", (20, 20), "blue").save(buffer, format="GIF")
        data = buffer.getvalue()
        row, _ = self.plugin.library.add(data, ".gif", "group", "", "")
        self.plugin.library.set_status(row["id"], "processing")
        self.plugin.library.recognize(
            row["id"], "蓝色动图", "蓝色背景，测试用图。", True
        )
        result = await self.plugin.view(event(), row["id"])
        metadata = json.loads(result.content[0].text)
        self.assertEqual(metadata["name"], "蓝色动图")
        self.assertTrue(metadata["local_path"].endswith(".gif"))
        self.assertEqual(Path(metadata["local_path"]).read_bytes(), data)
        self.assertEqual(result.content[1].mimeType, "image/jpeg")
