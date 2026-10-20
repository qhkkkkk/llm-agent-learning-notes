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

## 类式中间件

```bash
python -m unittest discover -s tests -p test_class_middleware.py -v
```

用离线模型驱动真实 Agent 与两个类式中间件，阻断同步和异步 HTTP 客户端。

覆盖默认与交换后的嵌套顺序、天气工具循环、内层恢复、外层重进、九次耗尽与第九次成功、单层三次、已完成工具不重复、工具异常边界、永久错误放大及 OpenRouter 构造参数，共 12 项。

“底层尝试次数”按测试模型的实际执行计数，不等同于真实 HTTP 请求次数；未调用真实模型。

详见 [第 20 篇学习笔记](../notes/20-langchain-class-middleware.md#validation)。

## 状态驱动的工具筛选

```bash
python -m unittest discover -s tests -p test_state_based_tools.py -v
```

离线模型驱动真实 Agent、中间件、工具绑定及工具执行，记录每次模型实际拿到的工具名；同步和异步 HTTP 客户端被阻断。

覆盖 auth 两条分支及缺省、公开/私有工具循环、同一 Agent 切换状态、override 新请求与原清单、无匹配工具、执行权限边界、工具异常、循环中状态更新，以及 OpenRouter 构造参数，共 12 项。

权限边界测试故意返回未暴露但已注册的工具调用，用于确认示例没有实现执行端鉴权；工具仅返回固定文本，未访问真实私密数据。

详见 [第 21 篇学习笔记](../notes/21-langchain-state-based-tools.md#validation)。

## 动态提示词

```bash
python -m unittest discover -s tests -p test_dynamic_prompt.py -v
```

离线模型驱动真实 Agent 与 dynamic_prompt，捕获实际模型输入中的系统消息，阻断同步和异步 HTTP 客户端。

覆盖初学者、专家、未知角色、空上下文字典、缺失上下文、同一 Agent 切换角色、Context 与 State 分离、误放字段、静态提示词替换、补充工具循环、OpenRouter 参数及脚本最终输出，共 12 项。

这些测试验证提示词传递，不评价真实模型的解释质量或角色风格遵循程度。

详见 [第 22 篇学习笔记](../notes/22-langchain-dynamic-prompt.md#validation)。

## 运行全部验证

```bash
python -m unittest discover -s tests -v
```

当前共 84 项离线测试，不代替真实模型服务认证、网络连通性、摘要质量或内容审核准确性验证。
