"""Exercise model-driven retirement through the real tool and history pipeline."""

import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from astrbot.core.agent.hooks import BaseAgentRunHooks
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from astrbot.core.agent.tool import ToolSet
from astrbot.core.astr_agent_run_util import run_agent
from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor
from astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal import (
    InternalAgentSubStage,
)
from astrbot.core.provider.entities import LLMResponse, ProviderRequest
from astrbot.core.provider.register import llm_tools

from astrbot_plugin_meme_library.store import Library

from . import test_api as api_fixtures
from . import test_library as fixtures


class DisableTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixtures.PluginTests.asyncSetUp
    asyncTearDown = fixtures.PluginTests.asyncTearDown
    ready = fixtures.PluginTests.ready

    async def test_actual_runner_views_disables_and_retains_normal_tool_history(self):
        row = self.ready()
        incoming = fixtures.event()
        calls = []

        async def completion(**kwargs):
            calls.append(copy.deepcopy(kwargs["contexts"]))
            if len(calls) == 1:
                return LLMResponse(
                    role="assistant",
                    tools_call_name=["meme_view"],
                    tools_call_args=[{"meme_id": row["id"]}],
                    tools_call_ids=["view-1"],
                )
            if len(calls) == 2:
                self.assertIn("data:image/jpeg;base64,", json.dumps(calls[-1]))
                return LLMResponse(
                    role="assistant",
                    tools_call_name=["meme_disable"],
                    tools_call_args=[
                        {
                            "meme_id": row["id"],
                            "expected_revision": 1,
                            "reason": "群友提醒这张误收录，核对图片后采纳",
                        }
                    ],
                    tools_call_ids=["disable-1"],
                )
            result = next(
                message
                for message in calls[-1]
                if message.get("role") == "tool"
                and message.get("tool_call_id") == "disable-1"
            )
            self.assertEqual(json.loads(result["content"])["status"], "disabled")
            return LLMResponse(role="assistant", completion_text="这张停用了。")

        self.provider.text_chat.side_effect = completion
        tools = []
        for name, handler in (
            ("meme_view", self.plugin.view),
            ("meme_disable", self.plugin.disable),
        ):
            tool = copy.copy(llm_tools.get_func(name))
            self.assertIsNotNone(tool)
            tool.handler = handler
            tools.append(tool)
        self.assertEqual(
            set(tools[-1].parameters["properties"]),
            {"meme_id", "expected_revision", "reason"},
        )
        req = ProviderRequest(
            prompt="这张误收录了，删掉它",
            func_tool=ToolSet(tools),
            conversation=SimpleNamespace(cid="conv", token_usage=0),
        )
        incoming.set_extra("provider_request", req)
        await self.plugin.prepare(incoming, req)
        self.assertIn("meme_disable", req.system_prompt)
        runner = ToolLoopAgentRunner()
        await runner.reset(
            provider=self.provider,
            request=req,
            run_context=ContextWrapper(context=SimpleNamespace(event=incoming)),
            tool_executor=FunctionToolExecutor(),
            agent_hooks=BaseAgentRunHooks(),
        )
        replies = []
        async for result in run_agent(runner, max_step=5, show_tool_use=False):
            replies.append(result.get_plain_text())
        self.assertEqual(replies, ["这张停用了。"])
        self.assertEqual(len(calls), 3)
        self.assertEqual(self.plugin.library.get(row["id"])["status"], "disabled")
        incoming.send.assert_not_awaited()
        manager = SimpleNamespace(update_conversation=AsyncMock())
        await InternalAgentSubStage._save_to_history(
            SimpleNamespace(conv_manager=manager),
            incoming,
            req,
            runner.get_final_llm_resp(),
            runner.run_context.messages,
            runner.stats,
        )
        saved = manager.update_conversation.await_args.kwargs["history"]
        disabled = [
            m
            for m in saved
            if m.get("role") == "tool" and m.get("tool_call_id") == "disable-1"
        ]
        self.assertEqual(len(disabled), 1)
        self.assertEqual(json.loads(disabled[0]["content"])["revision"], 2)
        self.assertEqual(saved[-1]["role"], "assistant")

    async def test_disable_requires_current_view_in_an_earlier_model_step(self):
        row = self.ready()
        incoming = fixtures.event()
        req = ProviderRequest()
        incoming.set_extra("provider_request", req)
        for viewed in (False, True):
            if viewed:
                await self.plugin.view(incoming, row["id"])
            result = json.loads(
                await self.plugin.disable(incoming, row["id"], 1, "误收")
            )
            self.assertEqual(result["error"], "read_current_metadata_first")
            self.assertEqual(self.plugin.library.get(row["id"])["status"], "ready")
        req.tool_calls_result = [SimpleNamespace()]
        self.assertTrue(
            json.loads(await self.plugin.disable(incoming, row["id"], 1, "误收"))["ok"]
        )

    async def test_stale_view_cannot_disable_a_newer_correction(self):
        row = self.ready()
        incoming = fixtures.event()
        req = ProviderRequest()
        incoming.set_extra("provider_request", req)
        await self.plugin.view(incoming, row["id"])
        req.tool_calls_result = [SimpleNamespace()]
        self.plugin.library.edit(
            row["id"], "新名称", "已核对修正用途", 1, "纠正", "dashboard"
        )
        result = json.loads(await self.plugin.disable(incoming, row["id"], 1, "旧判断"))
        self.assertEqual(result["error"], "revision_conflict")
        current = self.plugin.library.get(row["id"])
        self.assertEqual(current["status"], "ready")
        self.assertEqual(current["description"], "已核对修正用途")

    async def test_retirement_persists_excludes_all_groups_and_preserves_original(self):
        row = self.ready()
        seq = self.plugin.library.latest_seq()
        incoming = fixtures.event()
        req = ProviderRequest()
        incoming.set_extra("provider_request", req)
        await self.plugin.view(incoming, row["id"])
        req.tool_calls_result = [SimpleNamespace()]
        self.plugin.library.acknowledge(incoming.unified_msg_origin, seq)
        result = json.loads(
            await self.plugin.disable(incoming, row["id"], 1, "普通群友纠正")
        )
        self.assertTrue(result["ok"])
        current = self.plugin.library.get(row["id"])
        self.assertEqual(current["revision"], 2)
        for field in ("filename", "digest", "name", "description", "added_seq"):
            self.assertEqual(current[field], row[field])
        self.assertEqual(
            self.plugin.library.path(current).read_bytes(), fixtures.image_bytes()
        )
        self.assertEqual(self.plugin.library.latest_seq(), seq)
        self.assertEqual(self.plugin.library.unseen(incoming.unified_msg_origin), 0)
        audit = self.plugin.library.revisions(row["id"])[0]
        self.assertEqual(audit["reason"], "停用：普通群友纠正")
        self.assertEqual(audit["actor"], "model:" + incoming.unified_msg_origin)
        again, added = self.plugin.library.add(
            fixtures.image_bytes(), ".png", "group-b", "", ""
        )
        self.assertFalse(added)
        self.assertEqual(again["status"], "disabled")
        stale = json.loads(await self.plugin.send(incoming, row["id"], 1))
        self.assertFalse(stale["sent"])
        for group in ("group-a", "group-b"):
            found = json.loads(await self.plugin.search(fixtures.event(group), "开心"))
            self.assertEqual(found["candidates"], [])
        incoming.send.assert_not_awaited()
        self.plugin.library.close()
        self.plugin.library = Library(self.plugin.library.root)
        self.assertEqual(self.plugin.library.get(row["id"])["status"], "disabled")
        self.assertEqual(self.plugin.library.revisions(row["id"])[0], audit)

    async def test_inflight_search_omits_a_just_disabled_candidate(self):
        row = self.ready()
        incoming = fixtures.event()
        req = ProviderRequest()
        incoming.set_extra("provider_request", req)
        await self.plugin.view(incoming, row["id"])
        req.tool_calls_result = [SimpleNamespace()]

        async def completion(**kwargs):
            self.assertIn(row["id"], kwargs["prompt"])
            result = json.loads(
                await self.plugin.disable(incoming, row["id"], 1, "误收")
            )
            self.assertTrue(result["ok"])
            return LLMResponse(
                role="assistant",
                completion_text=json.dumps(
                    {"matches": [{"id": row["id"], "score": 0.99}]}
                ),
            )

        self.provider.text_chat.side_effect = completion
        result = json.loads(await self.plugin.search(fixtures.event("group-b"), "开心"))
        self.assertEqual(result["candidates"], [])

    async def test_invalid_input_disabled_plugin_and_repeat_leave_catalog_intact(self):
        row = self.ready()
        incoming = fixtures.event()
        req = ProviderRequest()
        incoming.set_extra("provider_request", req)
        await self.plugin.view(incoming, row["id"])
        req.tool_calls_result = [SimpleNamespace()]
        for revision, reason, error in (
            (1, "   ", "reason_required"),
            (True, "误收", "invalid_revision"),
            (1.0, "误收", "invalid_revision"),
        ):
            with self.subTest(revision=revision, reason=reason):
                result = json.loads(
                    await self.plugin.disable(incoming, row["id"], revision, reason)
                )
                self.assertEqual(result["error"], error)
                self.assertEqual(self.plugin.library.get(row["id"]), row)
        self.plugin.config["enabled"] = False
        result = json.loads(await self.plugin.disable(incoming, row["id"], 1, "误收"))
        self.assertEqual(result["error"], "disabled")
        self.plugin.config["enabled"] = True
        await self.plugin.disable(incoming, row["id"], 1, "误收")
        await self.plugin.view(incoming, row["id"])
        req.tool_calls_result.append(SimpleNamespace())
        repeat = json.loads(await self.plugin.disable(incoming, row["id"], 2, "已停用"))
        self.assertTrue(repeat["ok"])
        self.assertEqual(len(self.plugin.library.revisions(row["id"])), 2)
        missing = await self.plugin.view(incoming, "not-an-image")
        self.assertTrue(missing.isError)
        result = json.loads(
            await self.plugin.disable(incoming, "not-an-image", 1, "误收")
        )
        self.assertFalse(result["ok"])

    async def test_dashboard_can_reenable_a_model_disabled_image(self):
        row = self.ready()
        incoming = fixtures.event()
        req = ProviderRequest()
        incoming.set_extra("provider_request", req)
        await self.plugin.view(incoming, row["id"])
        req.tool_calls_result = [SimpleNamespace()]
        await self.plugin.disable(incoming, row["id"], 1, "误收")
        status, result = await api_fixtures.ApiTests.call_api(
            self, "action", {"id": row["id"], "action": "enable"}
        )
        self.assertEqual(status, 200)
        self.assertTrue(result["ok"])
        self.assertEqual(self.plugin.library.get(row["id"])["status"], "ready")
        self.assertEqual(self.plugin.library.get(row["id"])["revision"], 2)
