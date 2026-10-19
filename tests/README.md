# 示例验证

## 结构化天气 Agent

```bash
python -m unittest discover -s tests -p test_weather_structured_agent.py -v
```

测试使用离线模型驱动真实的 Agent 图，不使用 API Key，并阻断 HTTP 客户端。

覆盖两轮对话、工具执行、Context 注入、线程隔离、共享 Store、结构化输出校验重试、可选字段、严格模式自定义类恢复与脚本入口，共 10 项。

这些测试不替代真实提供商的可用性、模型决策质量或天气事实准确性验证。

详见 [第 16 篇学习笔记](../notes/16-langchain-weather-structured-agent.md#validation)。

## 内置摘要中间件

```bash
python -m unittest discover -s tests -p test_summarization_middleware.py -v
```

使用独立的主模型和摘要模型替身驱动真实中间件，不使用真实密钥，并阻断 HTTP 客户端。

覆盖两个客户端的提供商配置、两轮天气工具、阈值与保留边界、工具消息配对、旧检查点、多条件 OR、新线程及摘要日志，共 9 项。

详见 [第 17 篇学习笔记](../notes/17-langchain-summarization-middleware.md#validation)。

## 运行全部验证

```bash
python -m unittest discover -s tests -v
```

当前共 19 项离线测试，不代替真实模型服务认证、网络连通性与摘要质量验证。
