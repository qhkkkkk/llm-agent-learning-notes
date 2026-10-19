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

## 内容审核中间件

```bash
python -m unittest discover -s tests -p test_content_moderation.py -v
```

使用离线聊天模型与显式模拟的官方审核结果，运行真实 Agent、自定义输入审核和 OpenAI 审核中间件。HTTP 客户端被阻断，不读取真实密钥发起请求。

覆盖三种未通过行为、消息 ID 替换、格式异常、服务异常、工具循环、输入范围、输出与工具结果审核，以及官方客户端配置，共 17 项。

模拟类别与分数不是实际审核结果；本组不评估分类准确率。

详见 [第 18 篇学习笔记](../notes/18-langchain-content-moderation.md#validation)。

## 自定义中间件

```bash
python -m unittest discover -s tests -p test_custom_middleware.py -v
```

使用可编排的离线聊天模型驱动真实 Agent、中间件与 ToolNode，阻断同步、异步 HTTP 客户端。

覆盖直接回答、天气工具循环、成功与耗尽重试、永久错误的宽泛捕获、模型重试不重复工具、工具异常传播、错误 ToolMessage、Command 透传、多个中间件顺序、OpenRouter 配置与无检查点的调用隔离，共 12 项。

不会读取真实 Key 发出请求，也不把离线日志写成真实模型执行记录。

详见 [第 19 篇学习笔记](../notes/19-langchain-custom-middleware.md#validation)。

## 运行全部验证

```bash
python -m unittest discover -s tests -v
```

当前共 48 项离线测试，不代替真实模型服务认证、网络连通性、摘要质量或内容审核准确性验证。
