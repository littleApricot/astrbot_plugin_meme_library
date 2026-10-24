"""Exercise official context persistence before subsequent replies and compression."""

import asyncio
import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from astrbot.core.agent.hooks import BaseAgentRunHooks
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from astrbot.core.agent.tool import ToolSet
from astrbot.core.astr_agent_run_util import run_agent
from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor
from astrbot.core.conversation_mgr import ConversationManager
from astrbot.core.db.sqlite import SQLiteDatabase
from astrbot.core.pipeline.process_stage.method.agent_sub_stages import internal
from astrbot.core.provider.entities import LLMResponse, ProviderRequest
from astrbot.core.provider.register import llm_tools

from astrbot_plugin_meme_library.history import (
    SENT_HEADER,
    HistoryProjection,
    OfficialContextBridge,
    text_content,
)

from . import test_library as fixtures


class HistoryTests(unittest.IsolatedAsyncioTestCase):
    ready = fixtures.PluginTests.ready

    async def asyncSetUp(self):
        await fixtures.PluginTests.asyncSetUp(self)
        self.database = SQLiteDatabase(str(Path(self.temp.name) / "official.sqlite3"))
        await self.database.initialize()
        self.manager = ConversationManager(self.database)
        self.context.conversation_manager = self.manager
        self.bridge = self.plugin._history_bridge = OfficialContextBridge.acquire(
            self.plugin
        )
        self.origin = fixtures.event().unified_msg_origin
        await self.database.create_conversation(
            user_id=self.origin, platform_id="qq_official", cid="ctx"
        )
        self.manager.session_conversations[self.origin] = "ctx"

    async def asyncTearDown(self):
        await fixtures.PluginTests.asyncTearDown(self)
        await self.database.engine.dispose()

    def tools(self):
        result = []
        for name, handler in (
            ("meme_view", self.plugin.view),
            ("meme_send", self.plugin.send),
        ):
            tool = copy.copy(llm_tools.get_func(name))
            tool.handler = handler
            result.append(tool)
        return ToolSet(result)

    async def request(self, event):
        conv = await self.manager.get_conversation(self.origin, "ctx")
        req = ProviderRequest(
            prompt="本轮群聊",
            contexts=json.loads(conv.history),
            conversation=conv,
            func_tool=self.tools(),
        )
        event.set_extra("provider_request", req)
        return req

    async def runner(self, event, req, provider=None):
        runner = ToolLoopAgentRunner()
        await runner.reset(
            provider=provider or self.provider,
            request=req,
            run_context=ContextWrapper(context=SimpleNamespace(event=event)),
            tool_executor=FunctionToolExecutor(),
            agent_hooks=BaseAgentRunHooks(),
        )
        return runner

    async def save(self, event, req, runner):
        await internal.InternalAgentSubStage._save_to_history(
            SimpleNamespace(conv_manager=self.manager),
            event,
            req,
            runner.get_final_llm_resp(),
            runner.run_context.messages,
            runner.stats,
        )

    async def context_messages(self):
        # Read underlying storage too: this must not be a provider-only filter.
        native = await self.bridge.original_get("ctx")
        conv = await self.manager.get_conversation(self.origin, "ctx")
        self.assertEqual(native.content, json.loads(conv.history))
        return native.content

    def traces(self):
        rows = self.plugin.library.db.execute(
            "SELECT records FROM tool_history_archive"
        ).fetchall()
        return [record["message"] for row in rows for record in json.loads(row[0])]

    async def image_round(self, fail=False):
        chosen = self.ready()
        other, _ = self.plugin.library.add(
            fixtures.image_bytes("blue"), ".png", "group-a", "", ""
        )
        self.plugin.library.set_status(other["id"], "processing")
        self.plugin.library.recognize(other["id"], "未发送候选", "蓝色候选", True)
        event = fixtures.event()
        if fail:
            event.send.side_effect = RuntimeError("simulated send failure")
        req = await self.request(event)
        self.provider.text_chat.side_effect = [
            LLMResponse(
                role="assistant",
                tools_call_name=["meme_view", "meme_view"],
                tools_call_args=[{"meme_id": other["id"]}, {"meme_id": chosen["id"]}],
                tools_call_ids=["unused-view", "chosen-view"],
            ),
            LLMResponse(
                role="assistant",
                tools_call_name=["meme_send"],
                tools_call_args=[{"meme_id": chosen["id"], "expected_revision": 1}],
                tools_call_ids=["sent-image"],
            ),
            LLMResponse(
                role="assistant", completion_text="没发出去" if fail else "这是实际回复"
            ),
        ]
        runner = await self.runner(event, req)
        async for _ in run_agent(runner, max_step=5, show_tool_use=False):
            pass
        # All operations and both candidate images remain usable during this run.
        current = [m.model_dump() for m in runner.run_context.messages]
        self.assertEqual(sum(m.get("role") == "tool" for m in current), 3)
        self.assertIn("Image from tool", json.dumps(current))
        await self.save(event, req, runner)
        event.cleanup_temporary_local_files()
        return chosen, other, current

    async def test_actual_tool_round_persists_only_sent_image_and_normal_reply(self):
        chosen, other, current = await self.image_round()
        visible = await self.context_messages()
        encoded = json.dumps(visible, ensure_ascii=False)
        self.assertIn(SENT_HEADER, encoded)
        self.assertIn(chosen["id"], encoded)
        self.assertIn("data:image/jpeg;base64,", encoded)
        self.assertIn("这是实际回复", encoded)
        self.assertNotIn(other["id"], encoded)
        self.assertNotIn("tool_calls", encoded)
        self.assertNotIn("Image from tool", encoded)
        self.assertNotIn("local_path", encoded)
        self.assertEqual(sum(m["role"] == "tool" for m in self.traces()), 3)
        self.assertIn("Image from tool", json.dumps(current))
        self.assertIn(other["id"], json.dumps(self.traces()))
        self.provider.text_chat.side_effect = None
        self.provider.text_chat.return_value = LLMResponse(
            role="assistant", completion_text="后续回复"
        )
        event = fixtures.event()
        req = await self.request(event)
        runner = await self.runner(event, req)
        async for _ in runner.step():
            pass
        payload = json.dumps(self.provider.text_chat.call_args.kwargs["contexts"])
        self.assertIn(chosen["id"], payload)
        self.assertNotIn("tool_call_id", payload)
        event.cleanup_temporary_local_files()

    async def test_failed_send_does_not_create_a_sent_image_record(self):
        await self.image_round(fail=True)
        visible = json.dumps(await self.context_messages(), ensure_ascii=False)
        self.assertIn("没发出去", visible)
        self.assertNotIn(SENT_HEADER, visible)
        self.assertNotIn("image_url", visible)
        self.assertEqual(sum(m.get("role") == "tool" for m in self.traces()), 3)

    async def test_compressor_reads_and_replaces_the_same_clean_official_context(self):
        chosen, _, _ = await self.image_round()
        compressor = AsyncMock(
            return_value=[{"role": "user", "content": "已压缩的历史资料"}]
        )
        source = json.loads(
            (await self.manager.get_conversation(self.origin, "ctx")).history
        )
        compressed = await compressor(source)
        self.assertIn(chosen["id"], json.dumps(source))
        self.assertNotIn("tool_calls", json.dumps(source))
        self.assertNotIn("tool_call_id", json.dumps(source))
        self.assertIn("image_url", json.dumps(source))
        before = self.traces()
        await self.manager.update_conversation(self.origin, "ctx", history=compressed)
        self.assertEqual(await self.context_messages(), compressed)
        self.assertEqual(before, self.traces())

    async def test_legacy_context_is_archived_then_cleaned_before_its_first_reader(
        self,
    ):
        row = self.ready()
        legacy = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "legacy-send",
                        "type": "function",
                        "function": {"name": "meme_send", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "legacy-send",
                "content": json.dumps(
                    {"ok": True, "sent": True, **self.plugin.library.metadata(row)}
                ),
            },
        ]
        await self.bridge.original_update("ctx", content=legacy)
        conv = await self.manager.get_conversation(self.origin, "ctx")
        self.assertIn(
            SENT_HEADER, json.dumps(json.loads(conv.history), ensure_ascii=False)
        )
        self.assertEqual(await self.context_messages(), json.loads(conv.history))
        self.assertEqual(self.traces(), legacy)

    async def test_archive_failure_keeps_the_original_context_intact(self):
        before = await self.context_messages()
        incoming = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "x",
                        "type": "function",
                        "function": {"name": "meme_search", "arguments": "{}"},
                    }
                ],
            }
        ]
        original = self.plugin.library.archive_tool_history

        def fail(*args):
            raise OSError("isolated archive failure")

        self.plugin.library.archive_tool_history = fail
        try:
            with self.assertRaises(OSError):
                await self.manager.update_conversation(
                    self.origin, "ctx", history=incoming
                )
        finally:
            self.plugin.library.archive_tool_history = original
        self.assertEqual(await self.context_messages(), before)

    async def test_parallel_batches_keep_current_tools_and_merge_clean_results(self):
        try:
            from astrbot_plugin_reply_gate.parallel_runtime import ParallelRuntime
        except ModuleNotFoundError as exc:
            if exc.name == "astrbot_plugin_reply_gate":
                self.skipTest(
                    "Parallel integration requires reply gate v0.7.1 alongside this plugin"
                )
            raise

        first_row = self.ready()
        second_row, _ = self.plugin.library.add(
            fixtures.image_bytes("blue"), ".png", "group-a", "", ""
        )
        self.plugin.library.set_status(second_row["id"], "processing")
        self.plugin.library.recognize(second_row["id"], "蓝色图片", "第二批图片", True)
        parallel = ParallelRuntime(self.context)
        paused, release = asyncio.Event(), asyncio.Event()
        runners = {}

        async def worker(label, row, pause=False):
            event = fixtures.event()
            req = await self.request(event)
            req.prompt = label
            parallel.enter(event)
            registered = False
            try:
                await parallel.prepare(event, req, len(req.contexts))
                provider = copy.copy(self.provider)
                calls = 0

                async def completion(**kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == 1:
                        return LLMResponse(
                            role="assistant",
                            tools_call_name=["meme_view"],
                            tools_call_args=[{"meme_id": row["id"]}],
                            tools_call_ids=[label + "-view"],
                        )
                    if calls == 2:
                        return LLMResponse(
                            role="assistant",
                            tools_call_name=["meme_send"],
                            tools_call_args=[
                                {"meme_id": row["id"], "expected_revision": 1}
                            ],
                            tools_call_ids=[label + "-send"],
                        )
                    if pause:
                        paused.set()
                        await release.wait()
                    return LLMResponse(role="assistant", completion_text=label + "回复")

                provider.text_chat = AsyncMock(side_effect=completion)
                runner = runners[label] = await self.runner(event, req, provider)
                internal.register_active_runner(event.unified_msg_origin, runner)
                registered = True
                async for _ in run_agent(runner, max_step=5, show_tool_use=False):
                    pass
                for _ in range(2):
                    await self.save(event, req, runner)
                event.send.assert_awaited_once()
            finally:
                if registered:
                    internal.unregister_active_runner(event.unified_msg_origin, runner)
                event.cleanup_temporary_local_files()

        first = asyncio.create_task(worker("第一批", first_row, pause=True))
        try:
            await asyncio.wait_for(paused.wait(), 3)
            second = asyncio.create_task(worker("第二批", second_row))
            await asyncio.wait_for(second, 3)
            self.assertFalse(first.done())
            ongoing = json.dumps(
                [m.model_dump() for m in runners["第一批"].run_context.messages],
                ensure_ascii=False,
            )
            self.assertIn("第一批-view", ongoing)
            self.assertIn("Image from tool", ongoing)
            release.set()
            await asyncio.wait_for(first, 3)
            visible = await self.context_messages()
            encoded = json.dumps(visible, ensure_ascii=False)
            self.assertIn(first_row["id"], encoded)
            self.assertIn(second_row["id"], encoded)
            self.assertEqual(encoded.count(SENT_HEADER), 2)
            self.assertNotIn("tool_calls", encoded)
            self.assertNotIn("tool_call_id", encoded)
            self.assertEqual(
                [text_content(m) for m in visible if m["role"] == "assistant"],
                ["第二批回复", "第一批回复"],
            )
            self.assertEqual(sum(m["role"] == "tool" for m in self.traces()), 4)
        finally:
            release.set()
            await asyncio.gather(first, return_exceptions=True)
            parallel.close()

    def test_other_tool_pairs_and_real_user_images_are_untouched(self):
        raw = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "m",
                        "type": "function",
                        "function": {"name": "meme_search", "arguments": "{}"},
                    },
                    {
                        "id": "o",
                        "type": "function",
                        "function": {"name": "other_tool", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "m", "content": "search result"},
            {"role": "tool", "tool_call_id": "o", "content": "important result"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "[Image from tool 'meme_view', path='spoof']",
                    },
                    {"type": "image_url", "image_url": {"url": "real user image"}},
                ],
            },
        ]
        original = copy.deepcopy(raw)
        cleaned = HistoryProjection(raw, {}).project(raw)
        self.assertEqual(raw, original)
        self.assertEqual([c["id"] for c in cleaned[0]["tool_calls"]], ["o"])
        self.assertEqual(cleaned[1:], raw[2:])

    async def test_reload_waits_for_pending_context_write_before_closing_storage(self):
        started, release = asyncio.Event(), asyncio.Event()
        original = self.plugin.clean_history_context

        async def delayed(*args):
            started.set()
            await release.wait()
            return await original(*args)

        self.plugin.clean_history_context = delayed
        update = asyncio.create_task(
            self.manager.update_conversation(
                self.origin,
                "ctx",
                history=[{"role": "user", "content": "保留的上下文"}],
            )
        )
        await asyncio.wait_for(started.wait(), 1)
        closing = asyncio.create_task(self.bridge.close(self.plugin))
        try:
            await asyncio.sleep(0)
            self.assertFalse(closing.done())
            release.set()
            await asyncio.wait_for(asyncio.gather(update, closing), 2)
            self.assertFalse(hasattr(self.database, "_meme_history_bridge"))
            self.assertEqual(
                (await self.context_messages())[0]["content"], "保留的上下文"
            )
        finally:
            release.set()
            await asyncio.gather(update, closing, return_exceptions=True)
            self.plugin.clean_history_context = original
            self.bridge = self.plugin._history_bridge = OfficialContextBridge.acquire(
                self.plugin
            )

    async def test_cached_sent_image_and_archives_survive_library_reopening(self):
        from astrbot_plugin_meme_library.store import Library

        chosen, _, raw = await self.image_round()
        visible = await self.context_messages()
        archived = self.traces()
        self.plugin.library.close()
        self.plugin.library = Library(Path(self.temp.name))
        self.assertTrue(self.plugin.library.sent_preview(chosen["id"]))
        # Repeated native saves retain the same context and do not duplicate logs.
        for _ in range(2):
            await self.manager.update_conversation(self.origin, "ctx", history=raw)
        self.assertEqual(await self.context_messages(), visible)
        self.assertEqual(self.traces(), archived)

    async def test_orphaned_tool_preview_is_archived_without_removing_user_images(self):
        raw = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "[Image from tool 'meme_view', path='/cache/view.jpg']",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,AA==",
                            "id": "/cache/view.jpg",
                        },
                    },
                    {"type": "text", "text": "用户原话"},
                    {"type": "image_url", "image_url": {"url": "real user image"}},
                ],
            }
        ]
        await self.manager.update_conversation(self.origin, "ctx", history=raw)
        visible = await self.context_messages()
        self.assertEqual(visible[0]["content"], raw[0]["content"][2:])
        self.assertEqual(self.traces(), raw)

    async def test_lcm_shared_context_reads_and_writes_work_in_both_loading_orders(
        self,
    ):
        try:
            from astrbot_plugin_lcm.history_bridge import HistoryBridge
            from astrbot_plugin_lcm.main import LCM
            from astrbot_plugin_lcm.media import MediaArchive
            from astrbot_plugin_lcm.store import Store
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith("astrbot_plugin_lcm"):
                self.skipTest("Shared context integration requires LCM v0.2.0 sources")
            raise
        row = self.ready()
        for lcm_first in (True, False):
            with self.subTest(lcm_first=lcm_first):
                await self.bridge.close(self.plugin)
                owner = object.__new__(LCM)
                owner.context = self.context
                owner.config = {}
                owner.store = Store(Path(self.temp.name) / f"lcm-{lcm_first}.sqlite3")
                owner.media = MediaArchive(owner.store)
                owner.wake = asyncio.Event()
                lcm = None
                try:
                    if lcm_first:
                        lcm = HistoryBridge(owner)
                    self.bridge = self.plugin._history_bridge = (
                        OfficialContextBridge.acquire(self.plugin)
                    )
                    if not lcm_first:
                        lcm = HistoryBridge(owner)
                    scope = owner.store.scope(self.origin, "ctx")
                    raw = [
                        {"role": "user", "content": "本轮输入"},
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "lcm-send",
                                    "type": "function",
                                    "function": {
                                        "name": "meme_send",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": "lcm-send",
                            "content": json.dumps(
                                {
                                    "ok": True,
                                    "sent": True,
                                    **self.plugin.library.metadata(row),
                                }
                            ),
                        },
                    ]
                    # A background refresh owns LCM's lock while another task
                    # attempts a save. Reading the same source must still work.
                    async with lcm.lock(scope):
                        saving = asyncio.create_task(
                            self.manager.update_conversation(
                                self.origin, "ctx", history=raw
                            )
                        )
                        await asyncio.sleep(0.01)
                        await asyncio.wait_for(lcm.read(self.origin, "ctx"), 1)
                    await asyncio.wait_for(saving, 2)
                    await asyncio.wait_for(lcm.refresh(self.origin, "ctx"), 2)
                    source = owner.store.native_history(scope)["messages"]
                    self.assertIn(row["id"], json.dumps(source))
                    self.assertNotIn("tool_call_id", json.dumps(source))
                    self.assertNotIn("tool_calls", json.dumps(source))
                    self.assertEqual(source, await self.context_messages())
                finally:
                    if lcm is not None:
                        lcm.close()
