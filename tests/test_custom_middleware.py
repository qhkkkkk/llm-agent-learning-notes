"""用离线模型驱动真实 Agent，检查钩子时序、重试次数与工具返回协议。"""

import contextlib
import io
import os
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import ToolException
from langgraph.types import Command
from pydantic import Field

import langchain_custom_middleware as demo


class OfflineModel(BaseChatModel):
    script: list[Any] = Field(default_factory=list)
    seen: list[list[BaseMessage]] = Field(default_factory=list)
    events: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "offline-custom-middleware-model"

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=tools, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        self.events.append("model")
        if not self.script:
            raise AssertionError("离线模型脚本耗尽")
        next_item = self.script.pop(0)
        if isinstance(next_item, Exception):
            raise next_item
        return ChatResult(generations=[ChatGeneration(message=next_item)])


def weather_call():
    return AIMessage(content="", tool_calls=[{
        "name": "get_weather_for_location", "args": {"city": "北京"},
        "id": "weather-call", "type": "tool_call",
    }])


class OrderMiddleware(AgentMiddleware):
    def __init__(self, label, events):
        self.label = label
        self.events = events

    @property
    def name(self):
        return self.label

    def before_agent(self, state, runtime):
        self.events.append(self.label + ".before_agent")

    def before_model(self, state, runtime):
        self.events.append(self.label + ".before_model")

    def after_model(self, state, runtime):
        self.events.append(self.label + ".after_model")

    def after_agent(self, state, runtime):
        self.events.append(self.label + ".after_agent")

    def wrap_model_call(self, request, handler):
        self.events.append(self.label + ".wrap.enter")
        result = handler(request)
        self.events.append(self.label + ".wrap.exit")
        return result


class CustomMiddlewareTests(unittest.TestCase):
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

    def invoke(self, agent, query="北京的天气如何"):
        return agent.invoke({"messages": [{"role": "user", "content": query}]})

    def test_direct_answer_has_one_model_cycle(self):
        model = OfflineModel(script=[AIMessage(content="你好")])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = self.invoke(demo.build_agent(model, tools=[]), "你好")
        self.assertEqual(output.getvalue().splitlines(), [
            "即将执行Agent", "即将执行LLM", "【wrap model】最新消息：你好",
            "【wrap model】模型调用完成", "LLM执行完成", "Agent执行完成",
        ])
        self.assertEqual(result["messages"][-1].content, "你好")
        self.assertEqual(len(model.seen), 1)

    def test_weather_tool_cycle_exact_trace_and_message_pair(self):
        model = OfflineModel(script=[weather_call(), AIMessage(content="北京的演示天气是晴天。")])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = self.invoke(demo.build_agent(model))
        self.assertEqual(output.getvalue().splitlines(), [
            "即将执行Agent", "即将执行LLM", "【wrap model】最新消息：北京的天气如何",
            "【wrap model】模型调用完成", "LLM执行完成",
            "【wrap tool】执行工具：get_weather_for_location",
            "【wrap tool】参数：{'city': '北京'}", "【wrap tool】工具调用完成",
            "即将执行LLM", "【wrap model】最新消息：在北京总是阳光明媚！",
            "【wrap model】模型调用完成", "LLM执行完成", "Agent执行完成",
        ])
        self.assertEqual(len(model.seen), 2)
        self.assertEqual([m.type for m in result["messages"]], ["human", "ai", "tool", "ai"])
        tool_message = result["messages"][2]
        self.assertEqual(tool_message.tool_call_id, "weather-call")
        self.assertEqual(tool_message.content, "在北京总是阳光明媚！")

    def test_two_failures_then_success_are_three_attempts_one_hook_cycle(self):
        model = OfflineModel(script=[TimeoutError("first"), ConnectionError("second"), AIMessage(content="成功")])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = self.invoke(demo.build_agent(model, tools=[]))
        lines = output.getvalue().splitlines()
        self.assertEqual(len(model.seen), 3)
        self.assertEqual(lines.count("即将执行LLM"), 1)
        self.assertEqual(lines.count("LLM执行完成"), 1)
        self.assertEqual(lines.count("Agent执行完成"), 1)
        self.assertEqual(sum("准备第" in line for line in lines), 2)
        self.assertEqual(result["messages"][-1].content, "成功")
        self.assertEqual([[m.content for m in batch] for batch in model.seen],
                         [[m.content for m in model.seen[0]]] * 3)

    def test_exhaustion_reraises_original_exception_without_after_hooks(self):
        original = TimeoutError("仍然超时")
        model = OfflineModel(script=[TimeoutError("1"), TimeoutError("2"), original])
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(TimeoutError) as caught:
            self.invoke(demo.build_agent(model, tools=[]))
        self.assertIs(caught.exception, original)
        self.assertEqual(len(model.seen), 3)
        self.assertNotIn("LLM执行完成", output.getvalue())
        self.assertNotIn("Agent执行完成", output.getvalue())

    def test_broad_exception_demo_also_retries_permanent_errors(self):
        model = OfflineModel(script=[ValueError("配置错误")] * 3)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            self.invoke(demo.build_agent(model, tools=[]))
        self.assertEqual(len(model.seen), 3)

    def test_second_model_retry_does_not_repeat_finished_tool(self):
        executed = []

        @tool("get_weather_for_location")
        def counted_weather(city: str) -> str:
            """返回计数工具的演示天气。"""
            executed.append(city)
            return "测试天气"

        model = OfflineModel(script=[weather_call(), TimeoutError("回答阶段超时"), AIMessage(content="完成")])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.invoke(demo.build_agent(model, tools=[counted_weather]))
        self.assertEqual(executed, ["北京"])
        self.assertEqual(len(model.seen), 3)
        self.assertEqual(output.getvalue().splitlines().count("即将执行LLM"), 2)
        self.assertEqual(output.getvalue().splitlines().count("LLM执行完成"), 2)

    def test_tool_exception_is_logged_and_reraised_once(self):
        executed = []
        original = RuntimeError("工具服务不可用")

        @tool("get_weather_for_location")
        def broken_weather(city: str) -> str:
            """始终抛错的测试工具。"""
            executed.append(city)
            raise original

        model = OfflineModel(script=[weather_call()])
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(RuntimeError) as caught:
            self.invoke(demo.build_agent(model, tools=[broken_weather]))
        self.assertIs(caught.exception, original)
        self.assertEqual(executed, ["北京"])
        self.assertEqual(len(model.seen), 1)
        self.assertIn("【wrap tool】工具调用出现错误：工具服务不可用", output.getvalue())
        self.assertNotIn("Agent执行完成", output.getvalue())

    def test_error_tool_message_is_returned_without_python_exception(self):
        @tool("get_weather_for_location")
        def unavailable_weather(city: str) -> str:
            """用已配置的 ToolException 转换为错误消息。"""
            raise ToolException("没有天气数据")

        unavailable_weather.handle_tool_error = True
        model = OfflineModel(script=[weather_call(), AIMessage(content="暂时无法查询")])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = self.invoke(demo.build_agent(model, tools=[unavailable_weather]))
        error_message = result["messages"][2]
        self.assertIsInstance(error_message, ToolMessage)
        self.assertEqual(error_message.status, "error")
        self.assertEqual(error_message.tool_call_id, "weather-call")
        self.assertIn("工具调用完成", output.getvalue())
        self.assertNotIn("工具调用出现错误", output.getvalue())

    def test_tool_command_state_update_is_preserved(self):
        @tool("get_weather_for_location")
        def command_weather(city: str, runtime: ToolRuntime) -> Command:
            """通过 Command 把天气工具消息写回状态。"""
            return Command(update={"messages": [ToolMessage(
                content=f"Command 天气：{city}", tool_call_id=runtime.tool_call_id,
            )]})

        model = OfflineModel(script=[weather_call(), AIMessage(content="完成")])
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.invoke(demo.build_agent(model, tools=[command_weather]))
        self.assertEqual(result["messages"][2].content, "Command 天气：北京")
        self.assertEqual(model.seen[1][-1].tool_call_id, "weather-call")

    def test_multiple_middleware_before_forward_after_reverse_wrap_nested(self):
        model = OfflineModel(script=[AIMessage(content="完成")])
        events = model.events
        agent = create_agent(model=model, tools=[], middleware=[
            OrderMiddleware("A", events), OrderMiddleware("B", events),
        ])
        self.invoke(agent)
        self.assertEqual(events, [
            "A.before_agent", "B.before_agent", "A.before_model", "B.before_model",
            "A.wrap.enter", "B.wrap.enter", "model", "B.wrap.exit", "A.wrap.exit",
            "B.after_model", "A.after_model", "B.after_agent", "A.after_agent",
        ])

    def test_openrouter_factory_does_not_infer_provider_from_model_prefix(self):
        model = OfflineModel(script=[])
        with patch.dict(os.environ, {
            "OPENROUTER_API_KEY": "offline-placeholder",
            "CUSTOM_MIDDLEWARE_MODEL": "openai/gpt-5-mini",
        }), patch.object(demo, "init_chat_model", return_value=model) as factory:
            demo.build_agent()
        self.assertEqual(factory.call_args.kwargs, {
            "model": "openai/gpt-5-mini", "model_provider": "openrouter",
        })

    def test_without_checkpointer_separate_invocations_do_not_share_messages(self):
        model = OfflineModel(script=[AIMessage(content="第一轮"), AIMessage(content="第二轮")])
        agent = demo.build_agent(model, tools=[])
        with contextlib.redirect_stdout(io.StringIO()):
            first = self.invoke(agent, "你好")
            second = self.invoke(agent, "再见")
        self.assertEqual(len(first["messages"]), 2)
        self.assertEqual(len(second["messages"]), 2)
        self.assertEqual(second["messages"][0].content, "再见")
        self.assertNotIn("你好", [m.content for m in model.seen[1]])


if __name__ == "__main__":
    unittest.main()
