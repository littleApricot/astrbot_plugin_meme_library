"""Check delayed inventory notices at the admitted reply boundary."""

import copy
import unittest
from types import SimpleNamespace

from astrbot.core.agent.hooks import BaseAgentRunHooks
from astrbot.core.agent.message import dump_messages_with_checkpoints
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor
from astrbot.core.provider.entities import LLMResponse, ProviderRequest

from astrbot_plugin_meme_library.main import COLLECTION_NOTICE

from . import test_library as fixtures


class NoticeTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixtures.PluginTests.asyncSetUp
    asyncTearDown = fixtures.PluginTests.asyncTearDown
    ready = fixtures.PluginTests.ready

    async def reply_context(self, incoming, decision):
        if decision is not None:
            incoming.set_extra("reply_gate_decision", {"reply": decision})
        request = ProviderRequest(
            prompt="用户当前的消息",
            system_prompt="人格和记忆",
            contexts=[{"role": "user", "content": "之前的聊天"}],
        )
        await self.plugin.prepare(incoming, request)
        decision_input = copy.deepcopy(request.system_prompt)
        runner = ToolLoopAgentRunner()
        await runner.reset(
            provider=self.provider,
            request=request,
            run_context=ContextWrapper(context=SimpleNamespace(event=incoming)),
            tool_executor=FunctionToolExecutor(),
            agent_hooks=BaseAgentRunHooks(),
        )
        await self.plugin.notify_after_decision(incoming, runner.run_context)
        self.assertNotIn(COLLECTION_NOTICE, decision_input)
        self.assertEqual(request.system_prompt, decision_input)
        self.assertEqual(request.contexts, [{"role": "user", "content": "之前的聊天"}])
        self.assertEqual(request.extra_user_content_parts, [])
        incoming.send.assert_not_awaited()
        return dump_messages_with_checkpoints(runner.run_context.messages)

    def has_notice(self, messages):
        matching = [
            message
            for message in messages
            if COLLECTION_NOTICE in str(message["content"])
        ]
        if matching:
            self.assertEqual(len(matching), 1)
            self.assertEqual(matching[0]["role"], "system")
        return bool(matching)

    async def test_same_turn_and_rejected_turn_defer_until_later_admitted_reply(self):
        image_turn = fixtures.event()
        await self.plugin.collect(image_turn)
        self.ready()
        self.assertFalse(self.has_notice(await self.reply_context(image_turn, True)))
        rejected = fixtures.event()
        await self.plugin.collect(rejected)
        self.assertFalse(self.has_notice(await self.reply_context(rejected, False)))
        self.assertEqual(self.plugin.library.unseen(rejected.unified_msg_origin), 1)
        admitted = fixtures.event()
        await self.plugin.collect(admitted)
        self.assertTrue(self.has_notice(await self.reply_context(admitted, True)))
        await self.plugin.acknowledge_notice(
            admitted, LLMResponse(role="err", completion_text="failed")
        )
        self.assertEqual(self.plugin.library.unseen(admitted.unified_msg_origin), 1)
        retry = fixtures.event()
        await self.plugin.collect(retry)
        self.assertTrue(self.has_notice(await self.reply_context(retry, True)))
        await self.plugin.acknowledge_notice(
            retry, LLMResponse(role="assistant", completion_text="正常回复")
        )
        after = fixtures.event()
        await self.plugin.collect(after)
        self.assertFalse(self.has_notice(await self.reply_context(after, True)))

    async def test_new_arrival_during_reply_is_not_acknowledged_by_older_notice(self):
        row = self.ready()
        incoming = fixtures.event()
        await self.plugin.collect(incoming)
        newer, _ = self.plugin.library.add(
            fixtures.image_bytes("blue"), ".png", "group", "", ""
        )
        self.plugin.library.set_status(newer["id"], "processing")
        self.plugin.library.recognize(
            newer["id"], "蓝色图片", "蓝色背景的测试图片。", True
        )
        self.assertTrue(self.has_notice(await self.reply_context(incoming, True)))
        await self.plugin.acknowledge_notice(
            incoming, LLMResponse(role="assistant", completion_text="正常回复")
        )
        self.assertEqual(self.plugin.library.unseen(incoming.unified_msg_origin), 1)
        following = fixtures.event()
        await self.plugin.collect(following)
        self.assertTrue(self.has_notice(await self.reply_context(following, True)))
        await self.plugin.acknowledge_notice(
            following, LLMResponse(role="assistant", completion_text="正常回复")
        )
        self.plugin.library.edit(
            row["id"], "修订名称", "修订用途。", 1, "纠正", "model"
        )
        final = fixtures.event()
        await self.plugin.collect(final)
        self.assertFalse(self.has_notice(await self.reply_context(final, True)))

    async def test_missing_gate_decision_never_adds_notice(self):
        self.ready()
        incoming = fixtures.event()
        await self.plugin.collect(incoming)
        self.assertFalse(self.has_notice(await self.reply_context(incoming, None)))
        self.assertEqual(self.plugin.library.unseen(incoming.unified_msg_origin), 1)
