"""Verify local ranking, real message usage and transient provider-only hints."""

import base64
import copy
import json
import sqlite3
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from astrbot.api.message_components import Image
from astrbot.core.agent.message import Message, TextPart, dump_messages_with_checkpoints
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from astrbot.core.agent.tool import ToolSet
from astrbot.core.astr_agent_hooks import MainAgentHooks
from astrbot.core.astr_agent_run_util import run_agent
from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor
from astrbot.core.pipeline.context_utils import call_event_hook
from astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal import (
    InternalAgentSubStage,
)
from astrbot.core.provider.entities import LLMResponse, ProviderRequest
from astrbot.core.provider.register import llm_tools
from astrbot.core.provider.sources.openai_source import ProviderOpenAIOfficial
from astrbot.core.star.star import star_map
from astrbot.core.star.star_handler import EventType, star_handlers_registry
from openai.types.chat import ChatCompletion

from astrbot_plugin_meme_library.main import RECOMMENDATION_HEADER, MemeLibrary
from astrbot_plugin_meme_library.store import Library

from . import test_library as fixtures

NOW = 1_790_000_000


class RankingTests(unittest.TestCase):
    setUp = fixtures.StoreTests.setUp
    tearDown = fixtures.StoreTests.tearDown

    def seed(self, count):
        rows = []
        for index in range(count):
            with patch(
                "astrbot_plugin_meme_library.store.time.time", return_value=NOW + index
            ):
                row, _ = self.store.add(
                    fixtures.image_bytes((index, 10, 20)), ".png", "group", "", ""
                )
                self.store.set_status(row["id"], "processing")
                self.store.recognize(
                    row["id"], f"表情{index}", f"测试表情{index}的用途", True
                )
                rows.append(self.store.get(row["id"]))
        return rows

    def test_four_ranked_and_one_random_from_all_remaining_usable_images(self):
        rows = self.seed(9)
        self.store.set_status(rows[6]["id"], "disabled")
        self.store.set_status(rows[7]["id"], "rejected")
        self.store.path(rows[8]).unlink()
        with (
            patch("astrbot_plugin_meme_library.store.time.time", return_value=NOW + 20),
            patch(
                "astrbot_plugin_meme_library.store.random.SystemRandom.choice",
                side_effect=lambda pool: pool[-1],
            ) as choose,
        ):
            chosen = self.store.recommend()
        self.assertEqual(
            [r["id"] for r in chosen[:4]], [r["id"] for r in rows[2:6][::-1]]
        )
        self.assertEqual(chosen[-1]["id"], rows[0]["id"])
        self.assertEqual(
            {r["id"] for r in choose.call_args.args[0]}, {r["id"] for r in rows[:2]}
        )
        self.assertEqual(len({r["id"] for r in chosen}), 5)
        selected = {r["id"] for r in chosen}
        for row in rows:
            current = self.store.get(row["id"])
            self.assertEqual(
                current["recommendation_count"], int(row["id"] in selected)
            )
            self.assertEqual(current["use_count"], 0)
            self.assertEqual(current["revision"], 1)

    def test_new_exposure_then_recent_use_then_frequency(self):
        rows = self.seed(5)
        with self.store.db:
            self.store.db.execute("UPDATE memes SET recommendation_count=4")
            self.store.db.execute(
                "UPDATE memes SET use_count=1000,last_used_at=? WHERE id=?",
                (NOW - 3 * 86400, rows[0]["id"]),
            )
            for row, age in ((rows[1], 60), (rows[2], 3600)):
                self.store.db.execute(
                    "UPDATE memes SET use_count=1,last_used_at=? WHERE id=?",
                    (NOW - age, row["id"]),
                )
            self.store.db.execute(
                "UPDATE memes SET use_count=30 WHERE id=?", (rows[3]["id"],)
            )
            self.store.db.execute(
                "UPDATE memes SET recommendation_count=0 WHERE id=?", (rows[4]["id"],)
            )
        with patch("astrbot_plugin_meme_library.store.time.time", return_value=NOW):
            selected = self.store.recommend()
        self.assertEqual(
            [r["id"] for r in selected[:4]], [rows[i]["id"] for i in (4, 1, 2, 0)]
        )

    def test_each_exposure_decays_new_priority_without_becoming_a_use(self):
        rows = self.seed(10)
        seq = self.store.latest_seq()
        self.store.acknowledge("session", seq)
        with patch(
            "astrbot_plugin_meme_library.store.time.time", return_value=NOW + 20
        ):
            first = self.store.recommend()
            second = self.store.recommend()
        self.assertTrue(
            {r["id"] for r in first}.isdisjoint({r["id"] for r in second[:4]})
        )
        self.assertEqual(self.store.latest_seq(), seq)
        self.assertEqual(self.store.unseen("session"), 0)
        for row in rows:
            self.assertEqual(self.store.get(row["id"])["use_count"], 0)
            self.assertEqual(self.store.get(row["id"])["name"], row["name"])
            self.assertEqual(len(self.store.revisions(row["id"])), 1)

    def test_small_library_and_statistics_survive_reopening(self):
        row = self.seed(1)[0]
        with patch("astrbot_plugin_meme_library.store.time.time", return_value=NOW + 1):
            self.store.record_use(row["id"])
            for _ in range(5):
                self.assertEqual(len(self.store.recommend()), 1)
        self.store.close()
        self.store = Library(self.store.root)
        current = self.store.get(row["id"])
        self.assertEqual(current["use_count"], 1)
        self.assertEqual(current["last_used_at"], NOW + 1)
        self.assertEqual(current["recommendation_count"], 5)
        self.assertEqual(
            self.store.path(current).read_bytes(), fixtures.image_bytes((0, 10, 20))
        )
        self.store.set_status(row["id"], "disabled")
        self.assertEqual(self.store.recommend(), [])


class RecommendationFlowTests(unittest.IsolatedAsyncioTestCase):
    ready = fixtures.PluginTests.ready

    async def asyncSetUp(self):
        await fixtures.PluginTests.asyncSetUp(self)
        self.old_handlers = star_handlers_registry._handlers
        self.metadata = star_map[MemeLibrary.__module__]
        self.old_name = self.metadata.name
        self.metadata.name = "astrbot_plugin_meme_library"
        bound = []
        for handler in self.old_handlers:
            if handler.handler_module_path == MemeLibrary.__module__:
                handler = copy.copy(handler)
                handler.handler = getattr(self.plugin, handler.handler_name)
                bound.append(handler)
        star_handlers_registry._handlers = bound

    async def asyncTearDown(self):
        star_handlers_registry._handlers = self.old_handlers
        self.metadata.name = self.old_name
        await fixtures.PluginTests.asyncTearDown(self)

    async def run_reply(self, incoming, request):
        await call_event_hook(incoming, EventType.OnLLMRequestEvent, request)
        self.assertNotIn(RECOMMENDATION_HEADER, request.system_prompt)
        original = copy.deepcopy(
            (request.system_prompt, request.contexts, request.extra_user_content_parts)
        )
        incoming.set_extra("provider_request", request)
        runner = ToolLoopAgentRunner()
        await runner.reset(
            provider=self.provider,
            request=request,
            run_context=ContextWrapper(context=SimpleNamespace(event=incoming)),
            tool_executor=FunctionToolExecutor(),
            agent_hooks=MainAgentHooks(),
        )
        async for _ in run_agent(runner, max_step=5, show_tool_use=False):
            pass
        self.assertTrue(runner.done())
        self.assertEqual(
            (request.system_prompt, request.contexts, request.extra_user_content_parts),
            original,
        )
        manager = SimpleNamespace(update_conversation=AsyncMock())
        await InternalAgentSubStage._save_to_history(
            SimpleNamespace(conv_manager=manager),
            incoming,
            request,
            runner.get_final_llm_resp(),
            runner.run_context.messages,
            runner.stats,
        )
        manager.update_conversation.assert_awaited_once()
        saved = manager.update_conversation.await_args.kwargs["history"]
        self.assertNotIn(RECOMMENDATION_HEADER, json.dumps(saved, ensure_ascii=False))
        self.assertNotIn(
            RECOMMENDATION_HEADER,
            json.dumps(
                dump_messages_with_checkpoints(runner.run_context.messages),
                ensure_ascii=False,
            ),
        )
        return saved

    async def test_real_hooks_show_optional_recommendations_only_to_current_reply(self):
        row = self.ready()
        self.provider.text_chat.return_value = LLMResponse(
            role="assistant", completion_text="正常聊天"
        )
        history = [
            {"role": "user", "content": "OLD" * 3000},
            {"role": "assistant", "content": "完整旧回复"},
        ]
        for decision in (True, None):
            incoming = fixtures.event()
            if decision is not None:
                incoming.set_extra("reply_gate_decision", {"reply": decision})
            request = ProviderRequest(
                prompt="继续聊",
                contexts=copy.deepcopy(history),
                system_prompt="人格与记忆",
                conversation=SimpleNamespace(cid="conv", token_usage=0),
            )
            history = await self.run_reply(incoming, request)
            call = self.provider.text_chat.await_args.kwargs
            payload = json.dumps(
                {
                    key: call[key]
                    for key in ("contexts", "system_prompt", "prompt")
                    if key in call
                },
                ensure_ascii=False,
            )
            self.assertEqual(payload.count(RECOMMENDATION_HEADER), 1)
            self.assertIn("图库推荐只是可选候选", payload)
            self.assertIn("图库推荐只是可选候选", request.system_prompt)
            self.assertIn("meme_search", payload)
            self.assertIn("OLD" * 3000, payload)
            self.assertIn("人格与记忆", payload)
            self.assertNotIn(
                RECOMMENDATION_HEADER,
                str([m for m in call["contexts"] if m["role"] == "system"]),
            )
            self.assertEqual(call["contexts"][-1]["role"], "user")
            self.assertIn(RECOMMENDATION_HEADER, str(call["contexts"][-1]["content"]))
            self.assertNotIn(RECOMMENDATION_HEADER, str(call["contexts"][:-1]))
            incoming.send.assert_not_awaited()
        current = self.plugin.library.get(row["id"])
        self.assertEqual(current["recommendation_count"], 2)
        self.assertEqual(current["use_count"], 0)

    async def test_real_tools_remain_in_history_while_recommendation_does_not(self):
        row = self.ready()
        incoming = fixtures.event()
        incoming.set_extra("reply_gate_decision", {"reply": True})
        seen = []

        async def completion(**kwargs):
            seen.append(
                copy.deepcopy(
                    {
                        key: kwargs[key]
                        for key in ("contexts", "system_prompt", "prompt")
                        if key in kwargs
                    }
                )
            )
            self.assertIn(
                RECOMMENDATION_HEADER, json.dumps(seen[-1], ensure_ascii=False)
            )
            if len(seen) == 1:
                self.assertIn(row["id"], json.dumps(seen[-1]))
                return LLMResponse(
                    role="assistant",
                    tools_call_name=["meme_view"],
                    tools_call_args=[{"meme_id": row["id"]}],
                    tools_call_ids=["view-1"],
                )
            if len(seen) == 2:
                self.assertEqual(self.plugin.library.get(row["id"])["use_count"], 0)
                self.assertIn("data:image/jpeg;base64,", json.dumps(seen[-1]))
                return LLMResponse(
                    role="assistant",
                    tools_call_name=["meme_send"],
                    tools_call_args=[{"meme_id": row["id"], "expected_revision": 1}],
                    tools_call_ids=["send-1"],
                )
            return LLMResponse(role="assistant", completion_text="接着聊")

        self.provider.text_chat.side_effect = completion
        tools = []
        for name, handler in (
            ("meme_view", self.plugin.view),
            ("meme_send", self.plugin.send),
        ):
            tool = copy.copy(llm_tools.get_func(name))
            tool.handler = handler
            tools.append(tool)
        request = ProviderRequest(
            prompt="聊点有趣的",
            system_prompt="人格",
            func_tool=ToolSet(tools),
            conversation=SimpleNamespace(cid="conv", token_usage=0),
        )
        saved = await self.run_reply(incoming, request)
        self.assertEqual(len(seen), 3)
        self.assertEqual(self.plugin.library.get(row["id"])["recommendation_count"], 1)
        self.assertEqual(self.plugin.library.get(row["id"])["use_count"], 1)
        self.assertEqual(
            {m["tool_call_id"] for m in saved if m["role"] == "tool"},
            {"view-1", "send-1"},
        )
        incoming.send.assert_awaited_once()

    async def test_real_openai_adapter_preserves_full_prefix_when_hints_change(self):
        row = self.ready()
        previous = self.provider
        provider = ProviderOpenAIOfficial(
            {
                "id": "adapter-test",
                "type": "openai_chat_completion",
                "model": "adapter-test",
                "key": ["fixture-only-key"],
                "api_base": "https://example.invalid/v1",
                "max_context_tokens": 128000,
            },
            {},
        )
        provider.client.chat.completions.create = AsyncMock(
            return_value=ChatCompletion(
                id="fixture",
                created=0,
                model="adapter-test",
                object="chat.completion",
                choices=[
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "正常回复"},
                    }
                ],
            )
        )
        self.provider = provider
        try:
            history = [
                {"role": "user", "content": "完整旧历史" * 10000},
                {"role": "assistant", "content": "原来的完整回复"},
            ]
            image = (
                "data:image/png;base64,"
                + base64.b64encode(fixtures.image_bytes()).decode()
            )
            calls = []
            for description in ("第一次的可选推荐", "另一次不同的推荐"):
                request = ProviderRequest(
                    prompt="当前用户消息",
                    contexts=copy.deepcopy(history),
                    system_prompt="固定人格与记忆",
                    image_urls=[image],
                    extra_user_content_parts=[
                        TextPart(text="<Quoted Message>引用原文</Quoted Message>")
                    ],
                    conversation=SimpleNamespace(cid="conv", token_usage=0),
                )
                with patch.object(
                    self.plugin.library,
                    "recommend",
                    return_value=[dict(row, description=description)],
                ):
                    saved = await self.run_reply(fixtures.event(), request)
                messages = copy.deepcopy(
                    provider.client.chat.completions.create.await_args.kwargs[
                        "messages"
                    ]
                )
                calls.append(messages)
                serialized = json.dumps(messages, ensure_ascii=False)
                self.assertEqual(serialized.count(RECOMMENDATION_HEADER), 1)
                self.assertIn(row["id"], serialized)
                self.assertIn(description, serialized)
                self.assertIn("完整旧历史" * 10000, serialized)
                self.assertNotIn("_no_save", serialized)
                self.assertEqual(
                    [m["role"] for m in messages],
                    ["system", "user", "assistant", "user"],
                )
                self.assertNotIn(RECOMMENDATION_HEADER, str(messages[:-1]))
                self.assertIn(RECOMMENDATION_HEADER, str(messages[-1]))
                current_parts = messages[-1]["content"]
                self.assertIn("当前用户消息", str(current_parts))
                self.assertIn("引用原文", str(current_parts))
                self.assertTrue(
                    any(p.get("type") == "image_url" for p in current_parts)
                )
                self.assertEqual(request.contexts, history)
                self.assertEqual(request.image_urls, [image])
                self.assertNotIn(
                    RECOMMENDATION_HEADER, json.dumps(saved, ensure_ascii=False)
                )
            self.assertEqual(provider.client.chat.completions.create.await_count, 2)
            self.assertEqual(calls[0][:-1], calls[1][:-1])
            self.assertEqual(calls[0][-1]["content"][:-1], calls[1][-1]["content"][:-1])
            self.assertNotEqual(
                calls[0][-1]["content"][-1], calls[1][-1]["content"][-1]
            )
        finally:
            self.provider = previous
            await provider.client.close()

    async def test_rejected_disabled_or_empty_turns_do_not_consume_exposure(self):
        row = self.ready()
        for mode in ("rejected", "stopped", "disabled", "empty"):
            with self.subTest(mode=mode):
                incoming = fixtures.event()
                incoming.set_extra("reply_gate_decision", {"reply": mode != "rejected"})
                if mode == "stopped":
                    incoming.stop_event()
                self.plugin.config["enabled"] = mode != "disabled"
                if mode == "empty":
                    self.plugin.library.set_status(row["id"], "disabled")
                context = SimpleNamespace(
                    messages=[Message(role="system", content="PERSONA")]
                )
                await self.plugin.notify_after_decision(incoming, context)
                self.assertNotIn(RECOMMENDATION_HEADER, str(context.messages))
                self.assertEqual(
                    self.plugin.library.get(row["id"])["recommendation_count"], 0
                )
        self.provider.text_chat.assert_not_awaited()

    async def test_repeated_hook_reuses_one_hint_and_one_exposure(self):
        row = self.ready()
        incoming = fixtures.event()
        context = SimpleNamespace(messages=[Message(role="user", content="hi")])
        await self.plugin.notify_after_decision(incoming, context)
        await self.plugin.notify_after_decision(incoming, context)
        text = str(context.messages)
        self.assertEqual(text.count(RECOMMENDATION_HEADER), 1)
        self.assertEqual(self.plugin.library.get(row["id"])["recommendation_count"], 1)
        self.provider.text_chat.assert_not_awaited()

    async def test_no_user_message_does_not_change_system_or_recommend(self):
        row = self.ready()
        incoming = fixtures.event()
        context = SimpleNamespace(
            messages=[Message(role="system", content="人格与记忆")]
        )
        before = copy.deepcopy(context.messages)
        await self.plugin.notify_after_decision(incoming, context)
        self.assertEqual(context.messages, before)
        self.assertEqual(self.plugin.library.get(row["id"])["recommendation_count"], 0)
        self.provider.text_chat.assert_not_awaited()

    async def test_received_images_update_usage_once_across_groups_not_on_recognition(
        self,
    ):
        row = self.ready()
        source = self.plugin.library.path(row)
        first = fixtures.event(image=Image.fromFileSystem(source))
        await self.plugin.collect(first)
        await self.plugin.collect(first)
        await self.plugin.collect(
            fixtures.event("group-b", Image.fromFileSystem(source))
        )
        own = fixtures.event(image=Image.fromFileSystem(source))
        own.message_obj.sender.user_id = own.get_self_id()
        await self.plugin.collect(own)
        current = self.plugin.library.get(row["id"])
        self.assertEqual(current["use_count"], 2)
        self.assertGreater(current["last_used_at"], 0)
        self.assertEqual(current["recommendation_count"], 0)
        self.provider.text_chat.assert_not_awaited()

    async def test_failed_send_does_not_count_and_stats_failure_does_not_hide_success(
        self,
    ):
        row = self.ready()
        incoming = fixtures.event()
        request = ProviderRequest()
        incoming.set_extra("provider_request", request)
        await self.plugin.view(incoming, row["id"])
        request.tool_calls_result = [SimpleNamespace()]
        incoming.send.side_effect = RuntimeError("simulated send failure")
        result = json.loads(await self.plugin.send(incoming, row["id"], 1))
        self.assertFalse(result["sent"])
        self.assertEqual(self.plugin.library.get(row["id"])["use_count"], 0)
        second = fixtures.event("group-b")
        request = ProviderRequest()
        second.set_extra("provider_request", request)
        await self.plugin.view(second, row["id"])
        request.tool_calls_result = [SimpleNamespace()]
        with patch.object(
            self.plugin.library,
            "record_use",
            side_effect=sqlite3.OperationalError("test-only"),
        ):
            result = json.loads(await self.plugin.send(second, row["id"], 1))
        self.assertTrue(result["sent"])
        second.send.assert_awaited_once()

    async def test_statistics_error_does_not_block_a_normal_reply(self):
        self.ready()
        self.provider.text_chat.return_value = LLMResponse(
            role="assistant", completion_text="正常回复"
        )
        with patch.object(
            self.plugin.library,
            "recommend",
            side_effect=sqlite3.OperationalError("test-only"),
        ):
            saved = await self.run_reply(
                fixtures.event(),
                ProviderRequest(
                    prompt="hi", conversation=SimpleNamespace(cid="conv", token_usage=0)
                ),
            )
        self.assertEqual(saved[-1]["role"], "assistant")
        self.provider.text_chat.assert_awaited_once()
