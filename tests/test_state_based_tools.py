"""离线验证状态筛选、模型工具绑定、工具循环与执行权限边界。"""

import contextlib
import io
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.types import Command
from pydantic import Field

import langchain_state_based_tools as demo


class OfflineModel(BaseChatModel):
    replies: list[AIMessage] = Field(default_factory=list)
    seen: list[list[BaseMessage]] = Field(default_factory=list)
    offered: list[list[str]] = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "offline-state-based-tools-model"

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=tools, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        self.offered.append([t.name for t in kwargs.get("tools", [])])
        if not self.replies:
            raise AssertionError("离线模型回复耗尽")
        return ChatResult(generations=[ChatGeneration(message=self.replies.pop(0))])


def tool_call(name, call_id="search-call"):
    return AIMessage(content="", tool_calls=[{
        "name": name, "args": {"query": "北京的天气"},
        "id": call_id, "type": "tool_call",
    }])


class StateBasedToolsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sync_block = patch("httpx.Client.send", side_effect=AssertionError("禁止联网"))
        cls.async_block = patch("httpx.AsyncClient.send", side_effect=AssertionError("禁止联网"))
        cls.sync_block.start()
        cls.async_block.start()

    @classmethod
    def tearDownClass(cls):
        cls.async_block.stop()
        cls.sync_block.stop()

    def invoke(self, agent, auth=False, *, omit_auth=False):
        state = {"messages": [{"role": "user", "content": "北京的天气如何"}]}
        if not omit_auth:
            state["auth"] = auth
        return agent.invoke(state)

    def test_false_exposes_only_public_and_does_not_execute_tools_by_filtering(self):
        model = OfflineModel(replies=[AIMessage(content="直接回答")])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = self.invoke(demo.build_agent(model), False)
        self.assertEqual(model.offered, [["public_search"]])
        self.assertEqual(output.getvalue(), "")
        self.assertFalse(result["auth"])
        self.assertEqual([m.type for m in result["messages"]], ["human", "ai"])

    def test_true_exposes_only_private_not_public_or_advanced(self):
        model = OfflineModel(replies=[AIMessage(content="直接回答")])
        result = self.invoke(demo.build_agent(model), True)
        self.assertEqual(model.offered, [["private_search"]])
        self.assertTrue(result["auth"])

    def test_missing_auth_uses_false_fallback_at_runtime(self):
        model = OfflineModel(replies=[AIMessage(content="直接回答")])
        self.invoke(demo.build_agent(model), omit_auth=True)
        self.assertEqual(model.offered, [["public_search"]])

    def test_public_tool_loop_preserves_pair_and_filters_both_model_calls(self):
        model = OfflineModel(replies=[tool_call("public_search"), AIMessage(content="公开结果已收到")])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = self.invoke(demo.build_agent(model), False)
        self.assertEqual(model.offered, [["public_search"], ["public_search"]])
        self.assertEqual([m.type for m in result["messages"]], ["human", "ai", "tool", "ai"])
        message = result["messages"][2]
        self.assertEqual(message.name, "public_search")
        self.assertEqual(message.tool_call_id, "search-call")
        self.assertIn("[公开搜索结果]", message.content)
        self.assertEqual(output.getvalue().count("[公开搜索结果]"), 1)

    def test_private_tool_loop_reads_auth_from_extended_state(self):
        model = OfflineModel(replies=[tool_call("private_search"), AIMessage(content="私有演示结果已收到")])
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.invoke(demo.build_agent(model), True)
        self.assertEqual(model.offered, [["private_search"], ["private_search"]])
        self.assertEqual(result["messages"][2].name, "private_search")
        self.assertIn("[私有搜索结果]", result["messages"][2].content)
        self.assertTrue(result["auth"])

    def test_same_agent_can_switch_auth_without_permanently_pruning_registry(self):
        model = OfflineModel(replies=[AIMessage(content="第一次"), AIMessage(content="第二次")])
        agent = demo.build_agent(model)
        self.invoke(agent, False)
        self.invoke(agent, True)
        self.assertEqual(model.offered, [["public_search"], ["private_search"]])

    def test_override_creates_new_request_and_preserves_original_tool_list(self):
        before = []
        after = []

        @wrap_model_call
        def capture_before(request, handler):
            before.append(request)
            return handler(request)

        @wrap_model_call
        def capture_after(request, handler):
            after.append(request)
            return handler(request)

        model = OfflineModel(replies=[AIMessage(content="完成")])
        agent = create_agent(model=model, tools=[demo.public_search, demo.private_search, demo.advanced_search],
            middleware=[capture_before, demo.state_based_tools, capture_after])
        self.invoke(agent, False)
        self.assertIsNot(before[0], after[0])
        self.assertEqual([t.name for t in before[0].tools], ["public_search", "private_search", "advanced_search"])
        self.assertEqual([t.name for t in after[0].tools], ["public_search"])
        self.assertEqual(after[0].state["auth"], False)

    def test_no_prefix_match_produces_empty_model_tool_list(self):
        model = OfflineModel(replies=[AIMessage(content="没有适合的工具")])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.invoke(demo.build_agent(model, tools=[demo.advanced_search]), True)
        self.assertEqual(model.offered, [[]])
        self.assertEqual(output.getvalue(), "")

    def test_filter_is_not_execution_authorization_for_fabricated_tool_call(self):
        # 故意让模型替身请求未暴露但已注册的工具，检查执行端是否另有权限校验。
        model = OfflineModel(replies=[tool_call("private_search"), AIMessage(content="边界测试结束")])
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.invoke(demo.build_agent(model), False)
        self.assertEqual(model.offered, [["public_search"], ["public_search"]])
        self.assertFalse(result["auth"])
        self.assertEqual(result["messages"][2].name, "private_search")
        self.assertIn("[私有搜索结果]", result["messages"][2].content)

    def test_visible_tool_exception_is_not_swallowed_by_model_wrapper(self):
        original = RuntimeError("演示搜索服务失败")

        @tool("public_search")
        def broken_search(query: str) -> str:
            """抛出未处理错误的演示工具。"""
            raise original

        model = OfflineModel(replies=[tool_call("public_search")])
        with self.assertRaises(RuntimeError) as caught:
            self.invoke(demo.build_agent(model, tools=[broken_search]), False)
        self.assertIs(caught.exception, original)
        self.assertEqual(model.offered, [["public_search"]])

    def test_state_change_between_model_calls_recomputes_visible_tools(self):
        @tool("public_search")
        def transition_search(query: str, runtime: ToolRuntime) -> Command:
            """仅在测试中模拟上游流程更新状态，不执行真实身份认证。"""
            return Command(update={"auth": True, "messages": [ToolMessage(
                content="测试状态已更新", tool_call_id=runtime.tool_call_id,
            )]})

        model = OfflineModel(replies=[tool_call("public_search", "first"),
            tool_call("private_search", "second"), AIMessage(content="结束")])
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.invoke(demo.build_agent(model, tools=[transition_search, demo.private_search]), False)
        self.assertEqual(model.offered, [["public_search"], ["private_search"], ["private_search"]])
        self.assertTrue(result["auth"])
        self.assertEqual([m.tool_call_id for m in result["messages"] if isinstance(m, ToolMessage)], ["first", "second"])

    def test_model_factory_explicitly_selects_openrouter(self):
        model = OfflineModel()
        with patch.dict(os.environ, {
            "OPENROUTER_API_KEY": "offline-placeholder",
            "DYNAMIC_TOOLS_MODEL": "openai/gpt-5-mini",
        }), patch.object(demo, "init_chat_model", return_value=model) as factory:
            demo.build_agent()
        self.assertEqual(factory.call_args.kwargs, {
            "model": "openai/gpt-5-mini", "model_provider": "openrouter",
        })


if __name__ == "__main__":
    unittest.main()
