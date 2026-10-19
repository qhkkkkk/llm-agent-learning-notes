# 示例验证

## 结构化天气 Agent

```bash
python -m unittest discover -s tests -p test_weather_structured_agent.py -v
```

测试使用离线模型驱动真实的 Agent 图，不使用 API Key，并阻断 HTTP 客户端。

覆盖两轮对话、工具执行、Context 注入、线程隔离、共享 Store、结构化输出校验重试、可选字段、严格模式自定义类恢复与脚本入口，共 10 项。

这些测试不替代真实提供商的可用性、模型决策质量或天气事实准确性验证。

详见 [第 16 篇学习笔记](../notes/16-langchain-weather-structured-agent.md#validation)。
