"""真实 stdio/本机 HTTP MCP 集成；模型用离线替身，禁止外部 HTTP 请求。"""

import importlib.util
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
from exceptiongroup import BaseExceptionGroup
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_mcp_adapters.tools import load_mcp_tools
from pydantic import Field

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "mcp_multi_server"
spec = importlib.util.spec_from_file_location("mcp_multi_server_client", EXAMPLE / "agent_client.py")
demo = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = demo
spec.loader.exec_module(demo)


class OfflineModel(BaseChatModel):
    script: list[Any] = Field(default_factory=list)
    seen: list[list[BaseMessage]] = Field(default_factory=list)
    offered: list[list[str]] = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "offline-mcp-integration-model"

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=tools, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        self.offered.append([t.name for t in kwargs.get("tools", [])])
        if not self.script:
            raise AssertionError("离线模型脚本耗尽")
        next_item = self.script.pop(0)
        if isinstance(next_item, Exception):
            raise next_item
        return ChatResult(generations=[ChatGeneration(message=next_item)])


def call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{
        "name": name, "args": args, "id": call_id, "type": "tool_call",
    }])


def text_content(value):
    if isinstance(value, ToolMessage):
        value = value.content
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(block.get("text", "") if isinstance(block, dict) else str(block)
                       for block in value)
    return json.dumps(value, ensure_ascii=False)


def exception_leaves(error):
    if isinstance(error, BaseExceptionGroup):
        return [leaf for child in error.exceptions for leaf in exception_leaves(child)]
    return [error]


class MCPModelConfigurationTests(unittest.TestCase):
    def test_explicit_openrouter_configuration(self):
        model = OfflineModel()
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "offline-placeholder", "MCP_AGENT_MODEL": "openai/gpt-5-mini"}), \
                patch.object(demo, "init_chat_model", return_value=model) as factory:
            self.assertIs(demo.build_model(), model)
        self.assertEqual(factory.call_args.kwargs, {
            "model": "openai/gpt-5-mini", "model_provider": "openrouter",
        })


class MCPIntegrationTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.env_patch = patch.dict(os.environ, {"NO_PROXY": "127.0.0.1,localhost"})
        cls.env_patch.start()
        cls.addClassCleanup(cls.env_patch.stop)
        sync_send = httpx.Client.send
        async_send = httpx.AsyncClient.send

        def local_sync(client, request, *args, **kwargs):
            if request.url.host not in {"127.0.0.1", "localhost"}:
                raise AssertionError("测试禁止外部 HTTP 请求")
            return sync_send(client, request, *args, **kwargs)

        async def local_async(client, request, *args, **kwargs):
            if request.url.host not in {"127.0.0.1", "localhost"}:
                raise AssertionError("测试禁止外部 HTTP 请求")
            return await async_send(client, request, *args, **kwargs)

        cls.sync_patch = patch.object(httpx.Client, "send", new=local_sync)
        cls.async_patch = patch.object(httpx.AsyncClient, "send", new=local_async)
        cls.sync_patch.start()
        cls.async_patch.start()
        cls.addClassCleanup(cls.sync_patch.stop)
        cls.addClassCleanup(cls.async_patch.stop)

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        cls.url = f"http://127.0.0.1:{port}/mcp"
        cls.temp = tempfile.TemporaryDirectory(prefix="mcp-notes-test-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.log_path = Path(cls.temp.name) / "weather.log"
        cls.log = cls.log_path.open("wb")
        cls.addClassCleanup(cls.log.close)
        env = {**os.environ, "MCP_WEATHER_PORT": str(port)}
        cls.process = subprocess.Popen(
            [sys.executable, str(EXAMPLE / "weather_server.py")],
            cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=cls.log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        cls.addClassCleanup(cls.stop_server)
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            if cls.process.poll() is not None:
                cls.log.flush()
                raise RuntimeError(cls.log_path.read_text(encoding="utf-8", errors="replace"))
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    return
            except OSError:
                time.sleep(0.1)
        raise RuntimeError("测试天气 MCP 服务未能启动")

    @classmethod
    def stop_server(cls):
        if cls.process.poll() is None:
            cls.process.terminate()
            try:
                cls.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cls.process.kill()
                cls.process.wait(timeout=5)

    def client(self):
        return demo.build_client(self.url)

    async def test_session_scopes_and_combined_tool_registry(self):
        client = self.client()
        async with client.session("math") as math, client.session("weather") as weather:
            math_tools = await load_mcp_tools(math)
            weather_tools = await load_mcp_tools(weather)
            self.assertEqual({t.name for t in math_tools}, {"add", "multiply"})
            self.assertEqual([t.name for t in weather_tools], ["get_weather_for_location"])
            self.assertEqual(len(math_tools + weather_tools), 3)
            self.assertEqual(client.connections["math"]["command"], sys.executable)
            self.assertTrue(Path(client.connections["math"]["args"][0]).is_file())

    async def test_stdio_arithmetic_and_zero_negative_cases(self):
        async with self.client().session("math") as session:
            tools = {t.name: t for t in await load_mcp_tools(session)}
            self.assertEqual(text_content(await tools["add"].ainvoke({"a": 3, "b": 4})).strip(), "7")
            self.assertEqual(text_content(await tools["multiply"].ainvoke({"a": 7, "b": 5})).strip(), "35")
            self.assertEqual(text_content(await tools["add"].ainvoke({"a": -3, "b": 4})).strip(), "1")
            self.assertEqual(text_content(await tools["multiply"].ainvoke({"a": 0, "b": 5})).strip(), "0")

    async def test_http_weather_tool_reuses_live_session(self):
        async with self.client().session("weather") as session:
            weather = (await load_mcp_tools(session))[0]
            self.assertEqual(text_content(await weather.ainvoke({"city": "西安"})), "在西安总是阳光明媚！")
            self.assertEqual(text_content(await weather.ainvoke({"city": "北京"})), "在北京总是阳光明媚！")

    async def test_complete_agent_uses_real_mcp_tools_with_offline_model(self):
        model = OfflineModel(script=[
            call("add", {"a": 3, "b": 4}, "add-1"),
            call("multiply", {"a": 7, "b": 5}, "multiply-1"),
            AIMessage(content="35"),
            call("get_weather_for_location", {"city": "西安"}, "weather-1"),
            AIMessage(content="西安的演示天气是阳光明媚。"),
        ])
        result = await demo.run_demo(model, weather_url=self.url)
        expected = {"add", "multiply", "get_weather_for_location"}
        self.assertEqual(set(result["tool_names"]), expected)
        self.assertEqual(len(model.seen), 5)
        self.assertTrue(all(set(names) == expected for names in model.offered))
        math_messages = result["math_result"]["messages"]
        tool_results = [m for m in math_messages if isinstance(m, ToolMessage)]
        self.assertEqual([m.tool_call_id for m in tool_results], ["add-1", "multiply-1"])
        self.assertEqual([text_content(m).strip() for m in tool_results], ["7", "35"])
        self.assertEqual(math_messages[-1].content, "35")
        weather_messages = result["weather_result"]["messages"]
        self.assertEqual(weather_messages[2].tool_call_id, "weather-1")
        self.assertEqual(text_content(weather_messages[2]), "在西安总是阳光明媚！")
        self.assertNotIn("(3 + 4)", str(weather_messages[0].content))

    async def test_get_tools_discovers_both_servers_and_calls_work(self):
        tools = {t.name: t for t in await self.client().get_tools()}
        self.assertEqual(set(tools), {"add", "multiply", "get_weather_for_location"})
        self.assertEqual(text_content(await tools["add"].ainvoke({"a": 3, "b": 4})).strip(), "7")
        self.assertEqual(text_content(await tools["get_weather_for_location"].ainvoke({"city": "西安"})), "在西安总是阳光明媚！")

    async def test_bound_tool_cannot_outlive_explicit_session(self):
        async with self.client().session("math") as session:
            add = next(t for t in await load_mcp_tools(session) if t.name == "add")
        with self.assertRaises(Exception):
            await add.ainvoke({"a": 3, "b": 4})

    async def test_missing_required_tool_argument_reports_error(self):
        async with self.client().session("math") as session:
            add = next(t for t in await load_mcp_tools(session) if t.name == "add")
            self.assertIn("b", add.args)
            result = await add.ainvoke({
                "name": "add", "args": {"a": 3},
                "id": "invalid-args", "type": "tool_call",
            })
            self.assertIsInstance(result, ToolMessage)
            self.assertEqual(result.status, "error")
            self.assertEqual(result.tool_call_id, "invalid-args")
            self.assertIn("b", text_content(result))

    async def test_nested_exception_group_preserves_model_failure(self):
        original = RuntimeError("离线注入的模型调用故障")
        model = OfflineModel(script=[original])
        with self.assertRaises(Exception) as caught:
            await demo.run_demo(model, weather_url=self.url)
        self.assertTrue(any(leaf is original for leaf in exception_leaves(caught.exception)))
        self.assertEqual(len(model.seen), 1)


if __name__ == "__main__":
    unittest.main()
