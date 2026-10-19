"""用离线模型和模拟审核结果验证流程，不测试真实审核模型的准确率。"""

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai.middleware import OpenAIModerationError
from openai import AuthenticationError
from openai.types.moderation import Moderation
from pydantic import Field

import langchain_openai_moderation as native
import langchain_openrouter_input_moderation as router


class OfflineModel(BaseChatModel):
    replies: list[AIMessage] = Field(default_factory=list)
    seen: list[list] = Field(default_factory=list)
    fail: bool = False

    @property
    def _llm_type(self):
        return "offline-content-review-model"

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=tools, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        if self.fail:
            raise RuntimeError("模拟审核服务不可用")
        if not self.replies:
            raise AssertionError("离线模型回复已用完")
        return ChatResult(generations=[ChatGeneration(message=self.replies.pop(0))])


def weather_call():
    return AIMessage(content="", tool_calls=[{
        "name": "get_weather_for_location", "args": {"city": "北京"},
        "id": "weather-call", "type": "tool_call",
    }])


def fake_result(flagged):
    """构造显式模拟的 SDK 数据对象；这些标记和分数不是模型测量值。"""
    categories = Moderation.model_fields["categories"].annotation.model_fields
    scores = Moderation.model_fields["category_scores"].annotation.model_fields
    return Moderation.model_validate({
        "flagged": flagged,
        "categories": {field.alias or name: bool(flagged and name == "illicit")
                       for name, field in categories.items()},
        "category_scores": {field.alias or name: float(flagged and name == "illicit")
                            for name, field in scores.items()},
        "category_applied_input_types": {field.alias or name: ["text"]
                                         for name, field in categories.items()},
    })


class FakeModerationClient:
    def __init__(self, blocked=(), fail=False):
        self.blocked = set(blocked)
        self.fail = fail
        self.calls = []
        self.moderations = SimpleNamespace(create=self.create)

    def create(self, *, model, input):
        self.calls.append({"model": model, "input": input})
        if self.fail:
            raise AuthenticationError(
                "offline invalid_api_key",
                response=httpx.Response(401, request=httpx.Request("POST", "https://api.openai.com/v1/moderations")),
                body={"error": {"code": "invalid_api_key"}},
            )
        return SimpleNamespace(results=[fake_result(input in self.blocked)])


class ContentModerationTests(unittest.TestCase):
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

    def invoke(self, agent, text="测试输入"):
        return agent.invoke({"messages": [HumanMessage(content=text, id="user-1")]})

    def models(self, decision="ALLOW"):
        return (OfflineModel(replies=[AIMessage(content="正常回答")]),
                OfflineModel(replies=[AIMessage(content=decision)]))

    def test_router_allow_preserves_original_input(self):
        chat, review = self.models()
        self.invoke(router.build_agent(chat, review))
        texts = [m.content for m in chat.seen[0] if isinstance(m, HumanMessage)]
        self.assertEqual(texts, ["测试输入"])

    def test_router_replace_overwrites_same_id_without_duplicate(self):
        chat, review = self.models("BLOCK")
        result = self.invoke(router.build_agent(chat, review))
        humans = [m for m in result["messages"] if isinstance(m, HumanMessage)]
        self.assertEqual(len(humans), 1)
        self.assertEqual(humans[0].id, "user-1")
        self.assertEqual(humans[0].content, router.VIOLATION_MESSAGE)
        self.assertFalse(any(m.content == "测试输入" for m in chat.seen[0]))
        self.assertEqual(len(chat.seen), 1)

    def test_router_end_skips_main_model(self):
        chat, review = self.models("BLOCK")
        result = self.invoke(router.build_agent(chat, review, exit_behavior="end"))
        self.assertEqual(chat.seen, [])
        self.assertEqual(result["messages"][-1].content, router.VIOLATION_MESSAGE)

    def test_router_error_raises_application_error(self):
        chat, review = self.models("BLOCK")
        with self.assertRaises(router.InputReviewError):
            self.invoke(router.build_agent(chat, review, exit_behavior="error"))
        self.assertEqual(chat.seen, [])

    def test_router_unknown_verdict_does_not_pass_original_input(self):
        for verdict in ["", "unexpected", "ALLOW with explanation"]:
            with self.subTest(verdict=verdict):
                chat, review = self.models(verdict)
                self.invoke(router.build_agent(chat, review))
                self.assertTrue(any(isinstance(m, HumanMessage) and m.content == router.VIOLATION_MESSAGE
                                    for m in chat.seen[0]))

    def test_router_review_service_error_stops_main_model(self):
        chat, _ = self.models()
        with self.assertRaisesRegex(RuntimeError, "模拟审核服务不可用"):
            self.invoke(router.build_agent(chat, OfflineModel(fail=True)))
        self.assertEqual(chat.seen, [])

    def test_router_tool_loop_only_reviews_input_once(self):
        chat = OfflineModel(replies=[weather_call(), AIMessage(content="北京演示天气")])
        review = OfflineModel(replies=[AIMessage(content="ALLOW")])
        result = self.invoke(router.build_agent(chat, review))
        self.assertEqual(len(chat.seen), 2)
        self.assertEqual(len(review.seen), 1)
        self.assertTrue(any(isinstance(m, ToolMessage) for m in result["messages"]))

    def test_router_rejects_non_string_content(self):
        chat, review = self.models()
        with self.assertRaisesRegex(ValueError, "只支持文本字符串"):
            router.build_agent(chat, review).invoke({"messages": [HumanMessage(content=[{"type": "text", "text": "输入"}])]})
        self.assertEqual(chat.seen, [])
        self.assertEqual(review.seen, [])

    def test_router_model_factories_both_select_openrouter(self):
        chat, review = self.models()
        with (
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "offline-placeholder"}),
            patch.object(router, "init_chat_model", side_effect=[chat, review]) as factory,
        ):
            router.build_agent()
        self.assertEqual(factory.call_count, 2)
        self.assertTrue(all(call.kwargs["model_provider"] == "openrouter" for call in factory.call_args_list))

    def test_native_end_skips_main_model(self):
        chat, _ = self.models()
        client = FakeModerationClient(blocked=["测试输入"])
        result = self.invoke(native.build_agent(chat, moderation_client=client, exit_behavior="end"))
        self.assertEqual(chat.seen, [])
        self.assertIn("illicit", result["messages"][-1].content)

    def test_native_error_raises_moderation_error(self):
        chat, _ = self.models()
        client = FakeModerationClient(blocked=["测试输入"])
        with self.assertRaises(OpenAIModerationError):
            self.invoke(native.build_agent(chat, moderation_client=client, exit_behavior="error"))
        self.assertEqual(chat.seen, [])

    def test_native_replace_preserves_message_id(self):
        chat, _ = self.models()
        client = FakeModerationClient(blocked=["测试输入"])
        result = self.invoke(native.build_agent(chat, moderation_client=client))
        humans = [m for m in result["messages"] if isinstance(m, HumanMessage)]
        self.assertEqual(len(humans), 1)
        self.assertEqual(humans[0].id, "user-1")
        self.assertIn("illicit", humans[0].content)
        self.assertEqual(len(chat.seen), 1)

    def test_native_input_only_does_not_review_output(self):
        chat, _ = self.models()
        client = FakeModerationClient()
        self.invoke(native.build_agent(chat, moderation_client=client))
        self.assertEqual([c["input"] for c in client.calls], ["测试输入"])

    def test_native_output_review_happens_after_main_model(self):
        chat, _ = self.models()
        client = FakeModerationClient(blocked=["正常回答"])
        result = self.invoke(native.build_agent(chat, moderation_client=client, check_output=True, exit_behavior="end"))
        self.assertEqual(len(chat.seen), 1)
        self.assertEqual([c["input"] for c in client.calls], ["测试输入", "正常回答"])
        self.assertIn("审核未通过", result["messages"][-1].content)

    def test_native_tool_result_review_stops_next_main_call(self):
        chat = OfflineModel(replies=[weather_call()])
        client = FakeModerationClient(blocked=["在北京总是阳光明媚！"])
        result = self.invoke(native.build_agent(
            chat, moderation_client=client, check_input=False,
            check_tool_results=True, exit_behavior="end",
        ))
        self.assertEqual(len(chat.seen), 1)
        self.assertIn("审核未通过", result["messages"][-1].content)

    def test_native_authentication_error_is_not_moderation_error(self):
        chat, _ = self.models()
        with self.assertRaises(AuthenticationError) as caught:
            self.invoke(native.build_agent(chat, moderation_client=FakeModerationClient(fail=True)))
        self.assertNotIsInstance(caught.exception, OpenAIModerationError)
        self.assertEqual(chat.seen, [])

    def test_native_client_uses_separate_official_key_and_endpoint(self):
        with (
            patch.dict(os.environ, {
                "OPENAI_MODERATION_API_KEY": "offline-native-placeholder",
                "OPENAI_BASE_URL": "https://example.invalid/compat",
            }),
            patch.object(native, "OpenAI") as factory,
        ):
            native.build_moderation_client()
        self.assertEqual(factory.call_args.kwargs, {
            "api_key": "offline-native-placeholder", "base_url": "https://api.openai.com/v1",
        })


if __name__ == "__main__":
    unittest.main(verbosity=2)
