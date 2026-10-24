"""Verify collection, shared documents and actual AstrBot vision tool execution."""

import asyncio
import base64
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
from astrbot.core.agent.message import TextPart, dump_messages_with_checkpoints
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from astrbot.core.agent.tool import ToolSet
from astrbot.core.astr_agent_run_util import run_agent
from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor
from astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal import (
    InternalAgentSubStage,
)
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
from astrbot.core.platform.message_type import MessageType
from astrbot.core.platform.platform_metadata import PlatformMetadata
from astrbot.core.provider.entities import LLMResponse, ProviderMeta, ProviderRequest
from astrbot.core.provider.provider import Provider
from astrbot.core.provider.register import llm_tools
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
            for field in ("use_count", "last_used_at", "recommendation_count"):
                self.assertEqual(row[field], 0)
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
            "modalities": ["text", "image", "tool_use"],
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
        self.plugin = MemeLibrary(self.context, {"enabled": True})

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
        first.send.assert_not_awaited()
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

    async def test_collection_does_not_change_assembled_reply_request(self):
        original = [
            {"role": "user", "content": "群友之前的消息"},
            {"role": "assistant", "content": "机器人之前的文字回复"},
        ]
        image = "data:image/png;base64," + base64.b64encode(image_bytes()).decode()

        async def assembled():
            e = event()
            req = ProviderRequest(
                prompt="群友发来的新图片",
                contexts=copy.deepcopy(original),
                image_urls=[image],
                system_prompt="人格和记忆",
                extra_user_content_parts=[TextPart(text="已有群聊上下文")],
            )
            await self.plugin.prepare(e, req)
            self.assertEqual(req.contexts, original)
            self.assertEqual(req.image_urls, [image])
            self.assertEqual(
                req.extra_user_content_parts, [TextPart(text="已有群聊上下文")]
            )
            runner = ToolLoopAgentRunner()
            await runner.reset(
                provider=self.provider,
                request=req,
                run_context=ContextWrapper(context=SimpleNamespace(event=e)),
                tool_executor=FunctionToolExecutor(),
                agent_hooks=BaseAgentRunHooks(),
            )
            e.send.assert_not_awaited()
            return dump_messages_with_checkpoints(runner.run_context.messages)

        before = await assembled()
        row = self.ready()
        self.assertEqual(await assembled(), before)
        self.plugin.library.edit(
            row["id"], "新名称", "更新说明", 1, "correction", "model"
        )
        self.plugin.library.set_status(row["id"], "disabled")
        self.assertEqual(await assembled(), before)
        self.provider.text_chat.assert_not_awaited()

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
        denied = json.loads(await self.plugin.send(e, row["id"], 1))
        self.assertFalse(denied["sent"])
        await self.plugin.view(e, row["id"])
        denied = json.loads(await self.plugin.send(e, row["id"], 1))
        self.assertFalse(denied["sent"])
        req.tool_calls_result = [SimpleNamespace()]

        async def slow_send(chain):
            await asyncio.sleep(0.02)

        e.send.side_effect = slow_send
        results = await asyncio.gather(
            self.plugin.send(e, row["id"], 1),
            self.plugin.send(e, row["id"], 1),
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
        stale = json.loads(await self.plugin.send(e, row["id"], 1))
        self.assertFalse(stale["sent"])
        e.send.assert_not_awaited()

    async def _run_image_reply(self, *, fail_send=False):
        row = self.ready()
        e = event()
        seen = []
        e.set_extra("meme_library_notice_seq", self.plugin.library.latest_seq())
        if fail_send:
            e.send.side_effect = RuntimeError("simulated platform send failure")

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
                e.send.assert_not_awaited()
                preview = next(
                    message
                    for message in kwargs["contexts"]
                    if message.get("role") == "tool"
                    and message.get("tool_call_id") == "view-1"
                )
                metadata, _ = json.JSONDecoder().raw_decode(preview["content"])
                self.assertNotIn("instruction", metadata)
                self.assertNotIn("sent", metadata)
                self.assertIn("data:image/jpeg;base64,", json.dumps(kwargs["contexts"]))
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
                        }
                    ],
                    tools_call_ids=["send-1"],
                )
            return LLMResponse(
                role="assistant",
                completion_text="图片没发出去" if fail_send else "恭喜\n你真棒",
            )

        self.provider.text_chat.side_effect = completion
        registered_tools = []
        for name, handler in [
            ("meme_view", self.plugin.view),
            ("meme_send", self.plugin.send),
        ]:
            tool = copy.copy(llm_tools.get_func(name))
            self.assertIsNotNone(tool)
            tool.handler = handler
            registered_tools.append(tool)
        send_parameters = registered_tools[1].parameters["properties"]
        self.assertEqual(set(send_parameters), {"meme_id", "expected_revision"})
        req = ProviderRequest(
            prompt="庆祝一下",
            func_tool=ToolSet(registered_tools),
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
        replies = []
        async for reply in run_agent(runner, max_step=5, show_tool_use=False):
            replies.append(reply)
        self.assertTrue(runner.done())
        self.assertFalse(e.is_stopped())
        e.send.assert_awaited_once()
        chain = e.send.await_args.args[0].chain
        self.assertEqual(len(chain), 1)
        self.assertIsInstance(chain[0], Image)
        stage = SimpleNamespace(
            conv_manager=SimpleNamespace(update_conversation=AsyncMock())
        )
        await InternalAgentSubStage._save_to_history(
            stage,
            e,
            req,
            runner.get_final_llm_resp(),
            runner.run_context.messages,
            runner.stats,
        )
        stage.conv_manager.update_conversation.assert_awaited_once()
        saved = stage.conv_manager.update_conversation.await_args.kwargs["history"]
        self.assertTrue(
            any(
                m.get("role") == "tool" and m.get("tool_call_id") == "view-1"
                for m in saved
            )
        )
        self.assertIn("data:image/jpeg;base64,", json.dumps(saved))
        send_results = [
            message
            for message in saved
            if message.get("role") == "tool" and message.get("tool_call_id") == "send-1"
        ]
        self.assertEqual(len(send_results), 1)
        return runner, saved, send_results[0], replies

    async def test_actual_runner_sends_only_image_and_continues_normal_reply(self):
        runner, saved, result, replies = await self._run_image_reply()
        self.assertEqual(self.provider.text_chat.await_count, 3)
        self.assertEqual(runner.get_final_llm_resp().completion_text, "恭喜\n你真棒")
        self.assertEqual(
            [reply.get_plain_text() for reply in replies], ["恭喜\n你真棒"]
        )
        self.assertEqual(saved[-1]["role"], "assistant")
        self.assertEqual(
            saved[-1]["content"], [{"type": "text", "text": "恭喜\n你真棒"}]
        )
        self.assertTrue(json.loads(result["content"])["sent"])
        self.assertNotIn("caption", json.loads(result["content"]))
        self.assertEqual(
            self.plugin.library.unseen(
                runner.run_context.context.event.unified_msg_origin
            ),
            0,
        )

    async def test_actual_runner_failed_send_allows_error_response(self):
        runner, saved, result, replies = await self._run_image_reply(fail_send=True)
        self.assertEqual(self.provider.text_chat.await_count, 3)
        self.assertFalse(json.loads(result["content"])["sent"])
        self.assertEqual(json.loads(result["content"])["error"], "RuntimeError")
        self.assertEqual(
            self.plugin.library.unseen(
                runner.run_context.context.event.unified_msg_origin
            ),
            1,
        )
        self.assertEqual(
            [reply.get_plain_text() for reply in replies], ["图片没发出去"]
        )
        self.assertEqual(
            saved[-1]["content"], [{"type": "text", "text": "图片没发出去"}]
        )

    async def test_local_search_uses_updated_metadata_without_a_model(self):
        row = self.ready()
        self.plugin.library.edit(
            row["id"], "夸人厉害", "实际上表示夸人厉害", 1, "纠正", "model"
        )
        result = json.loads(await self.plugin.search(event("group-b"), "夸人 厉害"))
        self.assertEqual(result["candidates"][0]["revision"], 2)
        self.assertEqual(result["candidates"][0]["name"], "夸人厉害")
        self.assertNotIn("document", result["candidates"][0])
        path = Path(result["candidates"][0]["local_path"])
        self.assertTrue(path.is_absolute())
        self.assertEqual(path.read_bytes(), image_bytes())
        self.assertIn("夸人厉害", result["candidates"][0]["description"])
        self.provider.text_chat.assert_not_awaited()
        self.context.get_using_provider_async.assert_not_awaited()

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


class DecisionMetadataTests(unittest.IsolatedAsyncioTestCase):
    async def test_existing_metadata_is_read_without_recognition(self):
        from collections import OrderedDict

        with tempfile.TemporaryDirectory() as directory:
            library = Library(Path(directory))
            try:
                data = image_bytes()
                row, _ = library.add(data, ".png", "group", "", "")
                library.set_status(row["id"], "processing")
                library.recognize(row["id"], "企鹅拿捏", "被拿捏或求饶", True)
                plugin = object.__new__(MemeLibrary)
                plugin.config = {}
                plugin.library = library
                plugin._decision_images = OrderedDict(
                    [("https://isolated.invalid/image", row["id"])]
                )
                for reference in [
                    "https://isolated.invalid/image",
                    "data:image/png;base64," + base64.b64encode(data).decode(),
                    str(library.path(row)),
                ]:
                    result = await plugin.decision_image_description(reference)
                    self.assertIn("企鹅拿捏", result)
                    self.assertIn("被拿捏", result)
                self.assertEqual(
                    await plugin.decision_image_description(
                        "https://isolated.invalid/unknown"
                    ),
                    "",
                )
                self.assertEqual(
                    await plugin.decision_image_description("/etc/passwd"), ""
                )
            finally:
                library.close()
