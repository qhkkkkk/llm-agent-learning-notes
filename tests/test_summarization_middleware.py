"""离线验证真实摘要中间件、工具消息边界与两轮检查点恢复。"""

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

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

import langchain_summarization_middleware as demo


def weather_call(city, call_id):
    return AIMessage(
        content="",
        tool_calls=[{
            "name": "get_weather_for_location", "args": {"city": city},
            "id": call_id, "type": "tool_call",
        }],
    )


class OfflineModel(BaseChatModel):
    """主模型按脚本申请工具；摘要模型返回固定的测试摘要。"""

    replies: list[AIMessage] | None = None
    seen: list[list[BaseMessage]] = Field(default_factory=list)
    summary_text: str = "较早对话讨论北京的演示天气，继续处理保留的最近消息。"

    @property
    def _llm_type(self):
        return "offline-summarization-study-model"

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=tools, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        if self.replies is None:
            reply = AIMessage(content=self.summary_text)
        elif self.replies:
            reply = self.replies.pop(0)
        else:
            raise AssertionError("主模型的离线回复已经用完，图未按预期结束。")
        return ChatResult(generations=[ChatGeneration(message=reply)])


class SummarizationTests(unittest.TestCase):
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

    def make_session(self, **kwargs):
        chat = OfflineModel(replies=[
            weather_call("北京", "beijing"), AIMessage(content="北京的演示天气是晴天。"),
            weather_call("太原", "taiyuan"), AIMessage(content="太原的演示天气是晴天。"),
        ])
        summary = OfflineModel()
        agent = demo.build_agent(chat, summary, **kwargs)
        return chat, summary, agent

    def invoke(self, agent, prompt, thread="11"):
        return agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config={"configurable": {"thread_id": thread}, "recursion_limit": 30},
        )

    def test_both_model_factories_explicitly_select_openrouter(self):
        chat = OfflineModel(replies=[])
        summary = OfflineModel()
        with (
            patch.dict(os.environ, {
                "OPENROUTER_API_KEY": "offline-placeholder",
                "SUMMARY_CHAT_MODEL": "openai/gpt-5-mini",
                "SUMMARY_MODEL": "openai/gpt-4o-mini",
            }),
            patch.object(demo, "init_chat_model", side_effect=[chat, summary]) as factory,
        ):
            demo.build_agent()
        self.assertEqual(factory.call_count, 2)
        self.assertEqual(factory.call_args_list[0].kwargs,
                         {"model": "openai/gpt-5-mini", "model_provider": "openrouter"})
        self.assertEqual(factory.call_args_list[1].kwargs,
                         {"model": "openai/gpt-4o-mini", "model_provider": "openrouter"})

    def test_two_turn_summary_and_second_city_tool(self):
        chat, summary, agent = self.make_session(token_counter=lambda _: 100)
        first = self.invoke(agent, "北京的天气怎么样？")
        self.assertEqual(summary.seen, [])
        self.assertIn("北京", first["messages"][-1].content)
        second = self.invoke(agent, "太原呢？")
        self.assertGreaterEqual(len(summary.seen), 1)
        self.assertIn("太原", second["messages"][-1].content)
        summaries = [m for m in second["messages"]
                     if m.additional_kwargs.get("lc_source") == "summarization"]
        self.assertEqual(len(summaries), 1)
        self.assertIsInstance(summaries[0], HumanMessage)
        self.assertIn(summary.summary_text, summaries[0].content)
        tools = [m.content for m in second["messages"] if isinstance(m, ToolMessage)]
        self.assertIn("在太原总是阳光明媚！", tools)
        self.assertTrue(any(isinstance(m, HumanMessage) and m.content == "太原呢？"
                            for m in chat.seen[2]))

    def test_high_threshold_keeps_full_history(self):
        _, summary, agent = self.make_session(trigger=("tokens", 1000), token_counter=lambda _: 100)
        self.invoke(agent, "北京的天气怎么样？")
        result = self.invoke(agent, "太原呢？")
        self.assertEqual(summary.seen, [])
        texts = [m.content for m in result["messages"] if isinstance(m, HumanMessage)]
        self.assertEqual(texts, ["北京的天气怎么样？", "太原呢？"])

    def test_no_safe_old_history_means_no_summary_despite_low_threshold(self):
        chat, summary, agent = self.make_session(trigger=("tokens", 1), token_counter=lambda _: 100)
        result = self.invoke(agent, "北京的天气怎么样？")
        self.assertEqual(len(chat.seen), 2)
        self.assertEqual(summary.seen, [])
        self.assertEqual(len(result["messages"]), 4)

    def test_tool_calls_and_results_are_not_split(self):
        chat, _, agent = self.make_session(token_counter=lambda _: 100)
        self.invoke(agent, "北京的天气怎么样？")
        result = self.invoke(agent, "太原呢？")
        for messages in [*chat.seen, result["messages"]]:
            pending = set()
            for message in messages:
                if isinstance(message, AIMessage):
                    self.assertFalse(pending)
                    pending.update(call["id"] for call in message.tool_calls)
                elif isinstance(message, ToolMessage):
                    self.assertIn(message.tool_call_id, pending)
                    pending.remove(message.tool_call_id)
                else:
                    self.assertFalse(pending)
            self.assertFalse(pending)

    def test_latest_state_is_compressed_but_old_checkpoints_remain(self):
        _, _, agent = self.make_session(token_counter=lambda _: 100)
        self.invoke(agent, "北京的天气怎么样？")
        self.invoke(agent, "太原呢？")
        config = {"configurable": {"thread_id": "11"}}
        latest = agent.get_state(config)
        self.assertTrue(any(m.additional_kwargs.get("lc_source") == "summarization"
                            for m in latest.values["messages"]))
        self.assertFalse(any(m.content == "北京的天气怎么样？" for m in latest.values["messages"]))
        history = list(agent.get_state_history(config))
        self.assertTrue(any(any(m.content == "北京的天气怎么样？"
                                for m in snap.values.get("messages", [])) for snap in history))

    def test_or_trigger_accepts_message_threshold_when_token_threshold_is_not_met(self):
        _, summary, agent = self.make_session(
            trigger=[("tokens", 1000), ("messages", 5)], token_counter=lambda _: 0,
        )
        self.invoke(agent, "北京的天气怎么样？")
        self.invoke(agent, "太原呢？")
        self.assertGreaterEqual(len(summary.seen), 1)

    def test_new_thread_does_not_inherit_old_messages(self):
        chat, summary, agent = self.make_session(token_counter=lambda _: 100)
        self.invoke(agent, "北京的天气怎么样？", thread="old")
        self.invoke(agent, "太原呢？", thread="new")
        texts = [m.content for m in chat.seen[2] if isinstance(m, HumanMessage)]
        self.assertEqual(texts, ["太原呢？"])
        self.assertEqual(summary.seen, [])

    def test_message_display_omits_provider_metadata(self):
        output = io.StringIO()
        messages = [
            HumanMessage(content="摘要", additional_kwargs={"lc_source": "summarization"}),
            AIMessage(content="回答", additional_kwargs={"reasoning_details": [{"data": "opaque-payload"}]}),
        ]
        with contextlib.redirect_stdout(output):
            demo.print_messages(messages)
        self.assertIn("HumanMessage [摘要]", output.getvalue())
        self.assertIn("AIMessage", output.getvalue())
        self.assertNotIn("opaque-payload", output.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
