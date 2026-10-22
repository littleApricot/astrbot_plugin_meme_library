"""Verify collection, shared documents and actual AstrBot vision tool execution."""

import asyncio
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

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

from astrbot_plugin_meme_library.main import MemeLibrary
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
        self.store.recognize(row["id"], "开心的红色图，表示庆祝", True)
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
            row["id"], "群里也用来表达嘴硬", 1, "群友纠正", "model"
        )
        self.assertEqual(updated["revision"], 2)
        self.assertEqual(self.store.get(row["id"])["document"], "群里也用来表达嘴硬")
        self.assertEqual(len(self.store.revisions(row["id"])), 2)
        self.assertEqual(self.store.unseen("session"), 0)
        self.store.set_status(row["id"], "disabled")
        self.assertEqual(self.store.unseen("session"), 0)

    def test_conflicting_edits_preserve_newer_document(self):
        row = self.ready()
        self.store.edit(row["id"], "new", 1, "fix", "model")
        with self.assertRaisesRegex(ValueError, "revision_conflict"):
            self.store.edit(row["id"], "stale", 1, "fix", "model")
        self.assertEqual(self.store.get(row["id"])["document"], "new")

    def test_late_recognition_cannot_overwrite_manual_document(self):
        row, _ = self.store.add(image_bytes(), ".png", "group", "", "")
        self.store.set_status(row["id"], "processing")
        self.store.edit(row["id"], "manual correction", 0, "fix", "dashboard")
        self.store.recognize(row["id"], "late vision output", True)
        row = self.store.get(row["id"])
        self.assertEqual(row["document"], "manual correction")
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
                    {"is_meme": True, "document": "红色图片，开心庆祝"}
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
        self.plugin.library.recognize(row["id"], "红色图片，开心庆祝", True)
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
        self.plugin.library.edit(row["id"], "更新说明", 1, "correction", "model")
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
                e, row["id"], "这张在群里是夸人厉害", 1, "普通群友明确纠正"
            )
        )
        self.assertTrue(result["ok"])
        self.assertEqual(
            self.plugin.library.get(row["id"])["document"], result["document"]
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
        self.plugin.library.edit(row["id"], "实际上表示夸人厉害", 1, "纠正", "model")
        self.provider.text_chat.return_value = LLMResponse(
            role="assistant",
            completion_text=json.dumps({"matches": [{"id": row["id"], "score": 0.95}]}),
        )
        result = json.loads(await self.plugin.search(event("group-b"), "夸人厉害"))
        self.assertEqual(result["candidates"][0]["revision"], 2)
        self.assertIn("夸人厉害", result["candidates"][0]["document"])
        prompt = self.provider.text_chat.await_args.kwargs["prompt"]
        self.assertNotIn("开心庆祝", prompt)
