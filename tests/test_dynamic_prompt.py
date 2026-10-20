"""离线验证角色提示词、Runtime Context、系统消息覆盖与工具循环。"""

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
from langchain.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

import langchain_dynamic_prompt as demo

NOVICE = "你是一位乐于助人的专家。用简单易懂的语言描述概念，避免使用专业术语。"
EXPERT = "你是一位乐于助人的专家。尽量提供详细的技术解答，需要使用专业术语。"
BASE = "你是一位乐于助人的专家。"


class OfflineModel(BaseChatModel):
    replies: list[AIMessage] = Field(default_factory=list)
    seen: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "offline-dynamic-prompt-model"

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=tools, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        if not self.replies:
            raise AssertionError("离线模型回复耗尽")
        return ChatResult(generations=[ChatGeneration(message=self.replies.pop(0))])


class DynamicPromptTests(unittest.TestCase):
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

    def make_agent(self, replies=None):
        model = OfflineModel(replies=[AIMessage(content="离线测试回答")] if replies is None else replies)
        return model, demo.build_agent(model)

    def invoke(self, agent, context):
        return agent.invoke({"messages": [{"role": "user", "content": "解释一下机器学习"}]}, context=context)

    def test_novice_context_injects_system_prompt_before_user_message(self):
        model, agent = self.make_agent()
        result = self.invoke(agent, {"user_role": "初学者"})
        self.assertEqual(len(model.seen), 1)
        self.assertIsInstance(model.seen[0][0], SystemMessage)
        self.assertEqual(model.seen[0][0].content, NOVICE)
        self.assertIsInstance(model.seen[0][1], HumanMessage)
        self.assertEqual(model.seen[0][1].content, "解释一下机器学习")
        self.assertEqual(result["messages"][-1].content, "离线测试回答")

    def test_expert_context_selects_technical_instruction(self):
        model, agent = self.make_agent()
        self.invoke(agent, {"user_role": "专家"})
        self.assertEqual(model.seen[0][0].content, EXPERT)

    def test_unknown_empty_and_none_role_values_return_base_prompt(self):
        for role in ["访客", "", None]:
            with self.subTest(role=role):
                model, agent = self.make_agent()
                self.invoke(agent, {"user_role": role})
                self.assertEqual(model.seen[0][0].content, BASE)

    def test_empty_context_uses_missing_key_fallback(self):
        model, agent = self.make_agent()
        self.invoke(agent, {})
        self.assertEqual(model.seen[0][0].content, NOVICE)

    def test_omitted_context_uses_none_guard(self):
        model, agent = self.make_agent()
        agent.invoke({"messages": [{"role": "user", "content": "解释一下机器学习"}]})
        self.assertEqual(model.seen[0][0].content, NOVICE)

    def test_same_agent_switches_roles_without_accumulating_system_prompts(self):
        model, agent = self.make_agent([AIMessage(content="第一轮"), AIMessage(content="第二轮")])
        self.invoke(agent, {"user_role": "专家"})
        self.invoke(agent, {"user_role": "初学者"})
        self.assertEqual([batch[0].content for batch in model.seen], [EXPERT, NOVICE])
        for batch in model.seen:
            self.assertEqual(sum(isinstance(m, SystemMessage) for m in batch), 1)

    def test_context_and_dynamic_system_prompt_are_not_added_to_state_messages(self):
        model, agent = self.make_agent()
        result = self.invoke(agent, {"user_role": "专家"})
        self.assertNotIn("user_role", result)
        self.assertEqual([m.type for m in result["messages"]], ["human", "ai"])
        self.assertFalse(any(isinstance(m, SystemMessage) for m in result["messages"]))
        self.assertIsInstance(model.seen[0][0], SystemMessage)

    def test_role_in_state_input_does_not_replace_runtime_context(self):
        model, agent = self.make_agent()
        agent.invoke({
            "messages": [{"role": "user", "content": "解释一下机器学习"}],
            "user_role": "专家",
        })
        self.assertEqual(model.seen[0][0].content, NOVICE)

    def test_dynamic_prompt_replaces_static_prompt_in_this_configuration(self):
        model = OfflineModel(replies=[AIMessage(content="完成")])
        agent = create_agent(model=model, system_prompt="静态系统提示词",
            context_schema=demo.Context, middleware=[demo.user_role_prompt])
        self.invoke(agent, {"user_role": "专家"})
        systems = [m.content for m in model.seen[0] if isinstance(m, SystemMessage)]
        self.assertEqual(systems, [EXPERT])

    def test_tool_cycle_receives_role_prompt_on_each_model_call(self):
        @tool
        def explain_term(term: str) -> str:
            """返回固定术语解释，用于离线工具循环。"""
            return f"术语示例：{term}"

        model = OfflineModel(replies=[AIMessage(content="", tool_calls=[{
            "name": "explain_term", "args": {"term": "机器学习"},
            "id": "prompt-tool", "type": "tool_call",
        }]), AIMessage(content="最终测试回答")])
        agent = create_agent(model=model, tools=[explain_term], context_schema=demo.Context,
            middleware=[demo.user_role_prompt])
        result = self.invoke(agent, {"user_role": "专家"})
        self.assertEqual(len(model.seen), 2)
        self.assertEqual([batch[0].content for batch in model.seen], [EXPERT, EXPERT])
        self.assertEqual(result["messages"][2].tool_call_id, "prompt-tool")

    def test_factory_explicitly_selects_openrouter(self):
        model = OfflineModel()
        with patch.dict(os.environ, {
            "OPENROUTER_API_KEY": "offline-placeholder",
            "DYNAMIC_PROMPT_MODEL": "openai/gpt-5-mini",
        }), patch.object(demo, "init_chat_model", return_value=model) as factory:
            demo.build_agent()
        self.assertEqual(factory.call_args.kwargs, {
            "model": "openai/gpt-5-mini", "model_provider": "openrouter",
        })

    def test_script_entry_passes_role_and_prints_only_final_text(self):
        model, agent = self.make_agent()
        output = io.StringIO()
        with patch.object(demo, "load_dotenv"), patch.object(demo, "build_agent", return_value=agent), \
                patch.dict(os.environ, {"DYNAMIC_PROMPT_ROLE": "专家", "DYNAMIC_PROMPT_QUERY": "解释一下机器学习"}), \
                contextlib.redirect_stdout(output):
            demo.main()
        self.assertEqual(model.seen[0][0].content, EXPERT)
        self.assertEqual(output.getvalue(), "离线测试回答\n")


if __name__ == "__main__":
    unittest.main()
