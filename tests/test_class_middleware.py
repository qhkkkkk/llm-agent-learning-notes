"""离线驱动真实 Agent，验证类式钩子顺序、嵌套重试和工具边界。"""

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

from langchain.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

import langchain_class_middleware as demo


class OfflineModel(BaseChatModel):
    script: list[Any] = Field(default_factory=list)
    seen: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "offline-class-middleware-model"

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=tools, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        if not self.script:
            raise AssertionError("离线模型脚本耗尽")
        result = self.script.pop(0)
        if isinstance(result, Exception):
            raise result
        return ChatResult(generations=[ChatGeneration(message=result)])


def weather_call():
    return AIMessage(content="", tool_calls=[{
        "name": "get_weather_for_location", "args": {"city": "北京"},
        "id": "class-weather-call", "type": "tool_call",
    }])


class ClassMiddlewareTests(unittest.TestCase):
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

    def invoke(self, agent):
        return agent.invoke({"messages": [{"role": "user", "content": "北京的天气如何"}]})

    def count_entries(self, output, label):
        return sum(line.startswith(f"【{label}】【wrap model】最新消息：")
                   for line in output.getvalue().splitlines())

    def test_default_order_two_outside_one_inside(self):
        model = OfflineModel(script=[AIMessage(content="完成")])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = self.invoke(demo.build_agent(model, tools=[]))
        self.assertEqual(output.getvalue().splitlines(), [
            "【2】即将执行Agent", "【1】即将执行Agent",
            "【2】【wrap model】最新消息：北京的天气如何",
            "【1】【wrap model】最新消息：北京的天气如何",
            "【1】【wrap model】模型调用完成", "【2】【wrap model】模型调用完成",
            "【1】Agent执行完成", "【2】Agent执行完成",
        ])
        self.assertEqual(len(model.seen), 1)
        self.assertEqual(result["messages"][-1].content, "完成")

    def test_reversing_list_reverses_nesting_not_class_labels(self):
        model = OfflineModel(script=[AIMessage(content="完成")])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.invoke(demo.build_agent(model, tools=[], middleware=[
                demo.LoggingMiddleware(), demo.Logging2Middleware(),
            ]))
        self.assertEqual(output.getvalue().splitlines(), [
            "【1】即将执行Agent", "【2】即将执行Agent",
            "【1】【wrap model】最新消息：北京的天气如何",
            "【2】【wrap model】最新消息：北京的天气如何",
            "【2】【wrap model】模型调用完成", "【1】【wrap model】模型调用完成",
            "【2】Agent执行完成", "【1】Agent执行完成",
        ])
        self.assertEqual(len(model.seen), 1)

    def test_weather_tool_reenters_wrappers_not_agent_hooks(self):
        model = OfflineModel(script=[weather_call(), AIMessage(content="北京的演示天气是晴天。")])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = self.invoke(demo.build_agent(model))
        self.assertEqual(output.getvalue().splitlines(), [
            "【2】即将执行Agent", "【1】即将执行Agent",
            "【2】【wrap model】最新消息：北京的天气如何",
            "【1】【wrap model】最新消息：北京的天气如何",
            "【1】【wrap model】模型调用完成", "【2】【wrap model】模型调用完成",
            "【2】【wrap model】最新消息：在北京总是阳光明媚！",
            "【1】【wrap model】最新消息：在北京总是阳光明媚！",
            "【1】【wrap model】模型调用完成", "【2】【wrap model】模型调用完成",
            "【1】Agent执行完成", "【2】Agent执行完成",
        ])
        self.assertEqual(len(model.seen), 2)
        self.assertEqual([m.type for m in result["messages"]], ["human", "ai", "tool", "ai"])
        self.assertEqual(result["messages"][2].tool_call_id, "class-weather-call")
        self.assertEqual(result["messages"][2].content, "在北京总是阳光明媚！")

    def test_inner_recovers_before_outer_needs_retry(self):
        model = OfflineModel(script=[TimeoutError("1"), TimeoutError("2"), AIMessage(content="成功")])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.invoke(demo.build_agent(model, tools=[]))
        self.assertEqual(len(model.seen), 3)
        self.assertEqual(self.count_entries(output, 2), 1)
        self.assertEqual(self.count_entries(output, 1), 3)
        self.assertNotIn("【2】【wrap model】第1次尝试失败", output.getvalue())
        self.assertEqual(output.getvalue().count("【1】即将执行Agent"), 1)

    def test_outer_retries_full_inner_chain_after_three_failures(self):
        model = OfflineModel(script=[TimeoutError("1"), TimeoutError("2"), TimeoutError("3"), AIMessage(content="第4次成功")])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = self.invoke(demo.build_agent(model, tools=[]))
        self.assertEqual(len(model.seen), 4)
        self.assertEqual(self.count_entries(output, 2), 2)
        self.assertEqual(self.count_entries(output, 1), 4)
        self.assertIn("【2】【wrap model】第1次尝试失败，准备第2/3次尝试", output.getvalue())
        self.assertEqual(output.getvalue().count("【2】即将执行Agent"), 1)
        self.assertEqual(output.getvalue().count("【2】Agent执行完成"), 1)
        self.assertEqual(result["messages"][-1].content, "第4次成功")

    def test_persistent_failure_makes_nine_attempts_and_preserves_exception(self):
        original = TimeoutError("始终超时")
        model = OfflineModel(script=[TimeoutError(str(n)) for n in range(8)] + [original])
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(TimeoutError) as caught:
            self.invoke(demo.build_agent(model, tools=[]))
        self.assertIs(caught.exception, original)
        self.assertEqual(len(model.seen), 9)
        self.assertEqual(self.count_entries(output, 2), 3)
        self.assertEqual(self.count_entries(output, 1), 9)
        self.assertNotIn("Agent执行完成", output.getvalue())
        self.assertNotIn("模型调用完成", output.getvalue())

    def test_ninth_attempt_can_succeed_and_return_through_both_wrappers(self):
        model = OfflineModel(script=[TimeoutError(str(n)) for n in range(8)] + [AIMessage(content="第9次成功")])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = self.invoke(demo.build_agent(model, tools=[]))
        self.assertEqual(len(model.seen), 9)
        self.assertEqual(self.count_entries(output, 2), 3)
        self.assertEqual(self.count_entries(output, 1), 9)
        self.assertEqual(output.getvalue().count("【1】【wrap model】模型调用完成"), 1)
        self.assertEqual(output.getvalue().count("【2】【wrap model】模型调用完成"), 1)
        self.assertEqual(result["messages"][-1].content, "第9次成功")

    def test_one_registered_retry_layer_has_only_three_attempts(self):
        model = OfflineModel(script=[TimeoutError("永久失败")] * 3)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(TimeoutError):
            self.invoke(demo.build_agent(model, tools=[], middleware=[demo.Logging2Middleware()]))
        self.assertEqual(len(model.seen), 3)

    def test_retrying_second_model_chain_does_not_repeat_finished_tool(self):
        calls = []

        @tool("get_weather_for_location")
        def counted_weather(city: str) -> str:
            """返回演示天气并记录执行次数。"""
            calls.append(city)
            return "天气演示结果"

        model = OfflineModel(script=[weather_call(), *[TimeoutError("回答超时")] * 3, AIMessage(content="完成")])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.invoke(demo.build_agent(model, tools=[counted_weather]))
        self.assertEqual(calls, ["北京"])
        self.assertEqual(len(model.seen), 5)
        self.assertEqual(self.count_entries(output, 2), 3)
        self.assertEqual(self.count_entries(output, 1), 5)

    def test_tool_exception_is_outside_model_wrapper_retry_scope(self):
        calls = []
        original = RuntimeError("工具失败")

        @tool("get_weather_for_location")
        def broken_weather(city: str) -> str:
            """抛出未处理异常的演示工具。"""
            calls.append(city)
            raise original

        model = OfflineModel(script=[weather_call()])
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(RuntimeError) as caught:
            self.invoke(demo.build_agent(model, tools=[broken_weather]))
        self.assertIs(caught.exception, original)
        self.assertEqual(len(model.seen), 1)
        self.assertEqual(calls, ["北京"])
        self.assertEqual(self.count_entries(output, 2), 1)
        self.assertNotIn("准备第", output.getvalue())
        self.assertNotIn("Agent执行完成", output.getvalue())

    def test_broad_capture_also_multiplies_permanent_errors(self):
        model = OfflineModel(script=[ValueError("参数错误")] * 9)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            self.invoke(demo.build_agent(model, tools=[]))
        self.assertEqual(len(model.seen), 9)

    def test_openrouter_factory_uses_correct_provider_keyword(self):
        model = OfflineModel(script=[])
        with patch.dict(os.environ, {
            "OPENROUTER_API_KEY": "offline-placeholder",
            "CLASS_MIDDLEWARE_MODEL": "openai/gpt-5-mini",
        }), patch.object(demo, "init_chat_model", return_value=model) as factory:
            demo.build_agent()
        self.assertEqual(factory.call_args.kwargs, {
            "model": "openai/gpt-5-mini", "model_provider": "openrouter",
        })


if __name__ == "__main__":
    unittest.main()
