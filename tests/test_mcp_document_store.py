"""真实 stdio 资源读取、缓存工具和离线 Agent 集成；不请求模型 API。"""

import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from langchain_core.documents.base import Blob
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_mcp_adapters.resources import load_mcp_resources
from pydantic import Field

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "mcp_document_store"
spec = importlib.util.spec_from_file_location("mcp_document_store_client", EXAMPLE / "agent_client.py")
demo = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = demo
spec.loader.exec_module(demo)


class OfflineDocsModel(BaseChatModel):
    seen: list = Field(default_factory=list)
    offered: list = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "offline-docs-resource-model"

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=tools, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        self.offered.append([t.name for t in kwargs.get("tools", [])])
        results = [m for m in messages if isinstance(m, ToolMessage)]
        if not results:
            answer = AIMessage(content="", tool_calls=[{
                "name": "search_aq_docs", "args": {"query": "重置密码"},
                "id": "faq-call-1", "type": "tool_call",
            }])
        else:
            text = results[-1].content
            if not isinstance(text, str) or "忘记密码" not in text:
                raise AssertionError("离线模型没有收到预期文档")
            answer = AIMessage(content="请点击“忘记密码”链接。")
        return ChatResult(generations=[ChatGeneration(message=answer)])


def blob(text, mime="text/plain"):
    return Blob.from_data(text, mime_type=mime, metadata={"uri": demo.FAQ_URI})


class CachedDocumentTests(unittest.TestCase):
    def test_model_uses_openrouter(self):
        model = OfflineDocsModel()
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "offline-placeholder", "MCP_DOCS_MODEL": "openai/gpt-5-mini"}), patch.object(demo, "init_chat_model", return_value=model) as factory:
            self.assertIs(demo.build_model(), model)
        self.assertEqual(factory.call_args.kwargs, {
            "model": "openai/gpt-5-mini", "model_provider": "openrouter",
        })

    def test_missing_key_fails_before_model_factory(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}), patch.object(demo, "init_chat_model") as factory:
            with self.assertRaisesRegex(RuntimeError, "OPENROUTER_API_KEY"):
                demo.build_model()
            factory.assert_not_called()

    def test_stdio_uses_current_interpreter_and_sibling_script(self):
        config = demo.build_client().connections["DocumentStore"]
        self.assertEqual(config["transport"], "stdio")
        self.assertEqual(config["command"], sys.executable)
        self.assertEqual(Path(config["args"][0]), EXAMPLE / "docs_server.py")

    def test_query_does_not_filter_cached_content(self):
        tool = demo.make_doc_tool([blob("固定 FAQ")])
        self.assertEqual(tool.invoke({"query": "重置密码"}), "固定 FAQ")
        self.assertEqual(tool.invoke({"query": "完全无关的问题"}), "固定 FAQ")

    def test_empty_and_non_plain_text_have_fallback(self):
        for values in [[], [blob('{"a":1}', "application/json")]]:
            with self.subTest(values=values):
                self.assertEqual(demo.make_doc_tool(values).invoke({"query": "密码"}), "未找到相关文档")

    def test_multiple_blobs_join_with_blank_line(self):
        values = [blob("第一份"), blob("忽略", "application/json"), blob("第二份")]
        self.assertEqual(demo.make_doc_tool(values).invoke({"query": ""}), "第一份\n\n第二份")


class DocumentStoreIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # 本例只通过子进程管道通信，测试中的 HTTP 调用均视为错误。
        self.sync_guard = patch.object(httpx.Client, "send", side_effect=AssertionError("禁止外部 HTTP 请求"))
        self.async_guard = patch.object(httpx.AsyncClient, "send", side_effect=AssertionError("禁止外部 HTTP 请求"))
        self.sync_guard.start()
        self.async_guard.start()
        self.addCleanup(self.sync_guard.stop)
        self.addCleanup(self.async_guard.stop)

    async def test_lists_two_resources_and_no_registered_tools(self):
        async with demo.build_client().session("DocumentStore") as session:
            resources = await session.list_resources()
            self.assertEqual({str(r.uri) for r in resources.resources}, {
                "file:///help/guide.txt", demo.FAQ_URI,
            })
            self.assertEqual((await session.list_tools()).tools, [])
            blobs = await load_mcp_resources(session)
            self.assertEqual(len(blobs), 2)

    async def test_reads_faq_with_uri_mime_and_json_text(self):
        blobs = await demo.build_client().get_resources("DocumentStore", uris=[demo.FAQ_URI])
        self.assertEqual(len(blobs), 1)
        self.assertEqual(str(blobs[0].metadata["uri"]), demo.FAQ_URI)
        self.assertEqual(blobs[0].mimetype, "text/plain")
        self.assertEqual(json.loads(blobs[0].as_string()), {
            "q1": "如何重置密码？", "a1": "请点击“忘记密码”链接。",
        })

    async def test_reads_guide_separately(self):
        blobs = await demo.build_client().get_resources("DocumentStore", uris=["file:///help/guide.txt"])
        self.assertEqual(len(blobs), 1)
        self.assertIn("新建项目", blobs[0].as_string())
        self.assertNotIn("忘记密码", blobs[0].as_string())

    async def test_unknown_uri_is_an_error_not_empty_search(self):
        with self.assertRaises(Exception) as raised:
            await demo.build_client().get_resources("DocumentStore", uris=["file:///help/missing.txt"])
        def details(error):
            if isinstance(error, BaseExceptionGroup):
                return " ".join(details(e) for e in error.exceptions)
            return str(error) + (" " + details(error.__cause__) if error.__cause__ else "")
        self.assertIn("missing.txt", details(raised.exception))

    async def test_cached_blob_works_after_session_closed(self):
        async with demo.build_client().session("DocumentStore") as session:
            blobs = await load_mcp_resources(session, uris=[demo.FAQ_URI])
        # 工具只读取已在内存里的 Blob，不需要保持上述会话。
        text = await demo.make_doc_tool(blobs).ainvoke({"query": "密码"})
        self.assertIn("忘记密码", text)

    async def test_full_agent_preloads_once_and_receives_tool_message(self):
        client = demo.build_client()
        model = OfflineDocsModel()
        with patch.object(client, "get_resources", new=AsyncMock(wraps=client.get_resources)) as read, patch.object(demo, "build_client", return_value=client):
            result = await demo.run_demo(model)
        read.assert_awaited_once_with("DocumentStore", uris=[demo.FAQ_URI])
        messages = result["messages"]
        results = [m for m in messages if isinstance(m, ToolMessage)]
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].tool_call_id, "faq-call-1")
        self.assertEqual(results[0].name, "search_aq_docs")
        self.assertEqual(json.loads(results[0].content)["a1"], "请点击“忘记密码”链接。")
        self.assertEqual(messages[-1].content, "请点击“忘记密码”链接。")
        self.assertEqual(len(model.seen), 2)
        self.assertEqual(model.offered, [["search_aq_docs"], ["search_aq_docs"]])


if __name__ == "__main__":
    unittest.main()
