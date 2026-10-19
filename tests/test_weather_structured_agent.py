"""离线运行真实 Agent 图，不访问模型 API，也不读取真实密钥。"""

import contextlib
import io
import os
import runpy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ["LANGGRAPH_STRICT_MSGPACK"] = "true"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.store.memory import InMemoryStore
from pydantic import Field

import langchain_weather_structured_agent as demo


def tool_call(name, args, call_id):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def final_answer(text="晴天也晴心", weather=None, call_id="answer"):
    return tool_call(
        "ResponseFormat",
        {"punny_response": text, "weather_conditions": weather},
        call_id,
    )


class ScriptedModel(BaseChatModel):
    """按预先给定的消息模拟模型决策；实际工具、校验和检查点由框架执行。"""

    replies: list[AIMessage] = Field(default_factory=list)
    seen_messages: list[list] = Field(default_factory=list)
    seen_options: list[dict] = Field(default_factory=list)
    bindings: list[dict] = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "offline-weather-study-model"

    def bind_tools(self, tools, **kwargs):
        self.bindings.append(
            {"tools": [convert_to_openai_tool(t) for t in tools], **kwargs}
        )
        return self.bind(tools=tools, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen_messages.append(list(messages))
        self.seen_options.append(kwargs)
        if not self.replies:
            raise AssertionError("离线脚本已用完模型回复；图没有按预期结束。")
        return ChatResult(generations=[ChatGeneration(message=self.replies.pop(0))])


class WeatherAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 即使将来代码误用真实 HTTP 客户端，测试也会立即失败。
        cls.sync_block = patch("httpx.Client.send", side_effect=AssertionError("禁止联网"))
        cls.async_block = patch("httpx.AsyncClient.send", side_effect=AssertionError("禁止联网"))
        cls.sync_block.start()
        cls.async_block.start()

    @classmethod
    def tearDownClass(cls):
        cls.async_block.stop()
        cls.sync_block.stop()

    def invoke(self, agent, text="天气怎么样？", user="2", thread="11"):
        with contextlib.redirect_stdout(io.StringIO()):
            return agent.invoke(
                {"messages": [{"role": "user", "content": text}]},
                config={"configurable": {"thread_id": thread}, "recursion_limit": 20},
                context=demo.Context(user_id=user),
            )

    def test_two_turn_tools_context_store_and_checkpoint(self):
        model = ScriptedModel(
            replies=[
                tool_call("get_user_location", {}, "location"),
                tool_call("get_weather_for_location", {"city": "上海"}, "weather"),
                final_answer(weather="在上海总是阳光明媚！"),
                final_answer(text="不客气", call_id="thanks"),
            ]
        )
        store = InMemoryStore()
        agent = demo.build_agent(model, store=store)
        first = self.invoke(agent)
        second = self.invoke(agent, "谢谢！")
        self.assertIsInstance(first["structured_response"], demo.ResponseFormat)
        self.assertEqual(first["structured_response"].weather_conditions, "在上海总是阳光明媚！")
        self.assertEqual(second["structured_response"].punny_response, "不客气")
        self.assertEqual(store.get(("users",), "2").value, {"name": "name_2"})
        outputs = [m.content for m in first["messages"] if isinstance(m, ToolMessage)]
        self.assertIn("上海", outputs)
        self.assertIn("在上海总是阳光明媚！", outputs)
        human_texts = [m.content for m in model.seen_messages[-1] if isinstance(m, HumanMessage)]
        self.assertEqual(human_texts, ["天气怎么样？", "谢谢！"])
        snapshot = agent.get_state({"configurable": {"thread_id": "11"}})
        self.assertIsInstance(snapshot.values["structured_response"], demo.ResponseFormat)
        self.assertEqual(snapshot.next, ())

    def test_user_one_gets_beijing(self):
        model = ScriptedModel(replies=[tool_call("get_user_location", {}, "loc"), final_answer()])
        result = self.invoke(demo.build_agent(model), user="1")
        outputs = [m.content for m in result["messages"] if isinstance(m, ToolMessage)]
        self.assertIn("北京", outputs)

    def test_new_thread_has_no_previous_chat_but_shared_store_survives(self):
        model = ScriptedModel(
            replies=[tool_call("get_user_location", {}, "loc"), final_answer(), final_answer(call_id="new")]
        )
        store = InMemoryStore()
        agent = demo.build_agent(model, store=store)
        self.invoke(agent, "旧线程的问题", thread="old")
        self.invoke(agent, "新线程的问题", thread="new")
        human_texts = [m.content for m in model.seen_messages[-1] if isinstance(m, HumanMessage)]
        self.assertEqual(human_texts, ["新线程的问题"])
        self.assertEqual(store.get(("users",), "2").value, {"name": "name_2"})

    def test_user_records_use_distinct_keys(self):
        model = ScriptedModel(
            replies=[
                tool_call("get_user_location", {}, "loc1"), final_answer(call_id="a1"),
                tool_call("get_user_location", {}, "loc2"), final_answer(call_id="a2"),
            ]
        )
        store = InMemoryStore()
        agent = demo.build_agent(model, store=store)
        self.invoke(agent, user="1", thread="one")
        self.invoke(agent, user="2", thread="two")
        self.assertEqual(store.get(("users",), "1").value["name"], "name_1")
        self.assertEqual(store.get(("users",), "2").value["name"], "name_2")

    def test_runtime_is_hidden_and_response_schema_is_a_tool(self):
        model = ScriptedModel(replies=[final_answer()])
        self.invoke(demo.build_agent(model))
        bound = model.bindings[-1]
        schemas = {t["function"]["name"]: t["function"] for t in bound["tools"]}
        self.assertEqual(schemas["get_user_location"]["parameters"]["properties"], {})
        self.assertIn("city", schemas["get_weather_for_location"]["parameters"]["properties"])
        self.assertIn("ResponseFormat", schemas)
        self.assertNotIn("response_format", bound)
        # LangChain 使用 any 表达必须选一个工具；各提供商适配器再转换名称。
        self.assertEqual(bound["tool_choice"], "any")

    def test_invalid_structured_output_is_reported_and_retried(self):
        model = ScriptedModel(
            replies=[
                tool_call("ResponseFormat", {"weather_conditions": None}, "missing-required"),
                final_answer(text="补齐必填字段", call_id="valid"),
            ]
        )
        result = self.invoke(demo.build_agent(model))
        self.assertEqual(result["structured_response"].punny_response, "补齐必填字段")
        errors = [m.content for m in model.seen_messages[-1] if isinstance(m, ToolMessage)]
        self.assertTrue(any("punny_response" in str(message) for message in errors))

    def test_optional_field_may_be_omitted(self):
        model = ScriptedModel(
            replies=[tool_call("ResponseFormat", {"punny_response": "不客气"}, "optional")]
        )
        result = self.invoke(demo.build_agent(model))
        self.assertIsNone(result["structured_response"].weather_conditions)

    def test_registered_dataclass_round_trip_in_strict_mode(self):
        value = demo.ResponseFormat("晴天", None)
        serde = JsonPlusSerializer(
            allowed_msgpack_modules=[(demo.ResponseFormat.__module__, demo.ResponseFormat.__name__)]
        )
        restored = serde.loads_typed(serde.dumps_typed(value))
        self.assertEqual(restored, value)
        self.assertIsInstance(restored, demo.ResponseFormat)

    def test_script_entry_point_handles_main_module_class(self):
        model = ScriptedModel(
            replies=[
                tool_call("get_user_location", {}, "entry-location"),
                tool_call("get_weather_for_location", {"city": "上海"}, "entry-weather"),
                final_answer(text="第一轮", call_id="entry-answer"),
                final_answer(text="第二轮", call_id="entry-thanks"),
            ]
        )
        output = io.StringIO()
        with (
            patch("langchain.chat_models.init_chat_model", return_value=model),
            patch("dotenv.load_dotenv", return_value=False),
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "offline-placeholder", "WEATHER_USER_ID": "2"}),
            contextlib.redirect_stdout(output),
            self.assertNoLogs("langgraph.checkpoint.serde.jsonplus", level="WARNING"),
        ):
            runpy.run_path(str(Path(demo.__file__)), run_name="__main__")
        self.assertIn("user_name:name_2", output.getvalue())
        self.assertIn("punny_response='第一轮'", output.getvalue())
        self.assertIn("punny_response='第二轮'", output.getvalue())

    def test_unregistered_dataclass_is_blocked_by_strict_mode(self):
        serde = JsonPlusSerializer(allowed_msgpack_modules=[])
        value = demo.ResponseFormat("晴天", None)
        with self.assertLogs("langgraph.checkpoint.serde.jsonplus", level="WARNING") as captured:
            restored = serde.loads_typed(serde.dumps_typed(value))
        # 当前版本拒绝构造这个类，但保留字段字典；不能假定一定抛异常。
        self.assertIsInstance(restored, dict)
        self.assertNotIsInstance(restored, demo.ResponseFormat)
        self.assertEqual(restored["punny_response"], value.punny_response)
        self.assertTrue(any("Blocked deserialization" in line for line in captured.output))


if __name__ == "__main__":
    unittest.main(verbosity=2)
