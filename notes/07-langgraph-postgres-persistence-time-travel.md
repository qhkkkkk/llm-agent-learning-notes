# 07｜LangGraph PostgreSQL 持久化与时间旅行：Checkpoint、Replay 与 Fork

这份笔记在“模型判断 → 搜索工具 → 模型回答”的循环 Agent 上加入 PostgreSQL Checkpointer，学习如何让图在多次执行和程序重启后保留状态。读完后应能理解：**`thread_id` 如何形成会话记忆、检查点保存了什么、怎样查看状态历史，以及 Replay 与 Fork 两种时间旅行操作有什么区别。**

## 一句话理解

Checkpointer 会在图执行的每个关键步骤保存状态快照；以后只要使用相同的 `thread_id`，就能继续这段对话，也能从旧快照重新执行或修改状态后走出一条新分支。

## 最终流程

Agent 本身仍然是一个工具调用循环：

```mermaid
flowchart LR
    START((START)) --> L[llm_call<br/>模型判断]
    L -->|存在 tool_calls| T[tool_node<br/>执行搜索]
    T --> L
    L -->|没有 tool_calls| END((END))
```

PostgreSQL Checkpointer 位于执行流程之外，在每个 super-step 边界保存图状态：

```mermaid
flowchart TB
    A[thread_id] --> C1[Checkpoint 1<br/>用户消息已写入]
    C1 --> C2[Checkpoint 2<br/>模型请求工具]
    C2 --> C3[Checkpoint 3<br/>工具结果已写入]
    C3 --> C4[Checkpoint 4<br/>模型最终回答]
    C2 -. Replay .-> R[重新执行后续节点]
    C1 -. update_state .-> F[创建新分支]
```

## 1. 为什么 Agent 需要持久化

没有 Checkpointer 时，每次 `invoke()` 都像一次全新的执行。程序结束后，内存中的消息和节点状态也随之消失。

编译图时加入 Checkpointer：

```python
agent = builder.compile(checkpointer=checkpointer)
```

可以获得这些能力：

- 同一会话在多次调用之间保留上下文。
- 程序重启后从数据库恢复线程状态。
- 查看某个线程过去的状态快照。
- 从旧检查点重新执行后续节点。
- 修改旧状态并探索新的执行路径。
- 节点失败后从已经成功保存的位置恢复。

本例使用 `PostgresSaver`，适合演示持久化到外部数据库。只想快速实验时，也可以使用 `InMemorySaver`；但进程退出后，内存检查点就会丢失。

## 2. Checkpointer 与 Store 不是一回事

LangGraph 的持久化包含两个不同概念：

| 组件 | 保存什么 | 作用范围 | 常见用途 |
| --- | --- | --- | --- |
| Checkpointer | 图状态快照 | 单个 `thread_id` | 对话上下文、暂停恢复、时间旅行 |
| Store | 应用自定义数据 | 可以跨线程 | 用户偏好、长期事实、共享知识 |

本例只使用 Checkpointer，所以这里的“记忆”是线程范围内的短期记忆。它不会自动把一个线程学到的信息分享给另一个线程。

## 3. thread_id：一条会话的身份标识

调用带 Checkpointer 的图时，必须在配置中提供 `thread_id`：

```python
config = {
    "configurable": {
        "thread_id": "langgraph-postgres-demo"
    }
}
```

可以把 `thread_id` 理解为会话编号：

- 相同 `thread_id`：读取并继续同一条状态历史。
- 不同 `thread_id`：开始彼此隔离的新会话。

在实际应用中，它通常来自会话 ID、工单 ID 或 UUID。不要让多个无关用户共用同一个固定 ID，否则他们的消息会进入同一状态历史。

示例通过环境变量配置：

```env
THREAD_ID=langgraph-postgres-demo
```

## 4. Checkpoint 到底是什么

Checkpoint 是某个时间点的完整图状态快照。LangGraph 会在每个 super-step 边界保存它。

顺序图中的一次执行通常会产生多个快照，而不是只在最后保存一次。本例可能出现这样的历史：

```text
用户输入已进入状态，下一节点是 llm_call
    ↓
AIMessage 包含 tool_calls，下一节点是 tool_node
    ↓
ToolMessage 已进入状态，下一节点是 llm_call
    ↓
最终 AIMessage 已进入状态，下一节点为空
```

节点是否真的调用工具由模型决定，因此实际检查点数量并不固定。

## 5. PostgreSQL Checkpointer 的配置

导入：

```python
from langgraph.checkpoint.postgres import PostgresSaver
```

连接数据库并编译图：

```python
with PostgresSaver.from_conn_string(postgres_uri) as checkpointer:
    checkpointer.setup()
    agent = build_agent(checkpointer)
```

`setup()` 会创建 Checkpointer 所需的数据表并执行数据库迁移。第一次使用必须调用；示例每次启动都调用，以便自动检查尚未执行的迁移。

PostgreSQL 实现是独立安装包，因此依赖文件增加了：

```text
langgraph-checkpoint-postgres
psycopg[binary]
```

`psycopg[binary]` 适合本地学习和快速安装。生产部署应根据操作系统、部署方式和 Psycopg 官方建议选择合适的安装形式。

## 6. 不要把数据库密码写进源码

原始学习代码包含固定连接串：

```python
DB_URI = "postgresql://postgres:<password>@127.0.0.1:5432/postgres"
```

公开上传时，这种写法可能泄露真实账号与密码。完整示例改为从 `.env` 读取：

```python
postgres_uri = os.getenv("POSTGRES_URI")
```

`.env.example` 只保留占位值：

```env
POSTGRES_URI=postgresql://postgres:your_password@127.0.0.1:5432/postgres?sslmode=disable
```

真正的 `.env` 不应提交到公开仓库。线上环境还应使用独立的最小权限数据库账号，并根据部署环境正确配置 TLS。

## 7. 消息为什么改用 add_messages

状态定义如下：

```python
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    llm_calls: NotRequired[int]
```

`add_messages` 是专门处理消息列表的 Reducer：

- 新消息通常追加到列表末尾。
- 如果新消息与旧消息具有相同 ID，则替换旧消息。
- 输入的消息表示形式会被整理为 LangChain 消息对象。

普通的 `operator.add` 只会拼接两个列表，无法根据消息 ID 修改历史消息。本例 Fork 时要替换过去的用户问题，因此使用 `add_messages` 更合适。

`llm_calls` 没有 Reducer，节点返回新整数时会直接覆盖旧值：

```python
"llm_calls": state.get("llm_calls", 0) + 1
```

## 8. PostgreSQL 如何形成对话记忆

本轮输入只传入一条新消息：

```python
agent.invoke(
    {"messages": [input_message]},
    config=config,
)
```

如果数据库中已经存在相同 `thread_id` 的检查点，LangGraph 会先恢复旧状态，再通过 `add_messages` 把新消息合并进去。

因此下一次调用时，`llm_call` 读取的是完整会话：

```text
以前的 HumanMessage
以前的 AIMessage / ToolMessage
本次 HumanMessage
```

这就是线程级会话记忆。即使 Python 程序已经重启，只要 PostgreSQL 数据仍在、图的状态结构兼容，并使用相同 `thread_id`，历史仍可恢复。

## 9. get_state：查看最新状态

```python
latest_snapshot = agent.get_state(config)
```

返回值是 `StateSnapshot`，常用字段包括：

| 字段 | 含义 |
| --- | --- |
| `values` | 该检查点保存的状态字段 |
| `next` | 接下来要执行的节点；空元组表示本次已结束 |
| `config` | `thread_id`、`checkpoint_id` 等定位信息 |
| `metadata` | 步骤编号、来源、节点写入记录 |
| `created_at` | 创建时间 |
| `parent_config` | 父检查点配置 |
| `tasks` | 当前步骤对应的任务、错误或中断信息 |

如果只提供 `thread_id`，`get_state()` 返回该线程最新快照；同时提供 `checkpoint_id` 时，可以读取指定快照。

## 10. get_state_history：查看完整历史

```python
history = list(agent.get_state_history(config))
```

历史顺序是“最新在前”，不是最早在前。示例打印四个最重要的信息：

```text
step | source | next | checkpoint_id
```

- `step`：super-step 编号。
- `source`：快照来自输入、图循环还是手动更新。
- `next`：从这个快照继续时将执行哪些节点。
- `checkpoint_id`：数据库中唯一定位该快照的 ID。

调试时，`next` 往往比只看编号更直观。例如：

```python
snapshot.next == ("llm_call",)
```

表示当前状态已经保存，下一步将运行模型节点。

## 11. 为什么不能写死 checkpoint_id

原始代码通过固定字符串查找检查点：

```python
if checkpoint_id == "1f1bb0ad-...":
    ...
```

Checkpoint ID 是运行时生成的。换数据库、换线程或重新执行后，这个 ID 都会变化，其他人无法复现。

完整示例为本轮输入显式生成消息 ID：

```python
input_message = HumanMessage(
    content=QUESTION,
    id=str(uuid4()),
)
```

然后同时根据三个条件找快照：

```text
下一节点是 llm_call
最后一条消息是 HumanMessage
消息 ID 等于本次输入的 ID
```

这样找到的是“本次用户输入已经保存，但模型还没有处理它”的检查点，不依赖任何预先知道的 Checkpoint ID。

## 12. Replay：从旧快照重新执行

找到检查点后，可以把它自己的 `config` 传回 `invoke()`：

```python
replay_result = agent.invoke(
    None,
    config=input_checkpoint.config,
)
```

这里输入为 `None`，因为状态已经保存在检查点中。

Replay 的关键特征：

- 检查点之前的节点不会重新执行。
- 检查点之后的节点会再次执行。
- 模型调用、搜索请求和其他外部操作会真实发生。
- 即使提示词相同，非确定性模型或变化的外部数据也可能产生不同结果。
- 从 `next` 为空的最终检查点重放不会执行任何节点。

因此 Replay 不是简单读取缓存。它可能产生额外费用、触发限流，甚至重复有副作用的工具操作。重放包含发邮件、付款或数据库写入的工作流前，必须额外设计幂等和审批机制。

## 13. Fork：修改旧状态并走新分支

Fork 分为两步。

第一步，用 `update_state()` 从旧检查点创建一个更新后的检查点：

```python
replacement_message = HumanMessage(
    content=FORK_QUESTION,
    id=input_message.id,
)

fork_config = agent.update_state(
    input_checkpoint.config,
    {"messages": [replacement_message]},
)
```

替换消息使用了相同 ID，因此 `add_messages` 会替换原问题，而不是在后面再追加一条问题。

第二步，从新检查点继续：

```python
fork_result = agent.invoke(None, config=fork_config)
```

`update_state()` 不会回滚或覆盖原始执行历史。它会保留旧分支，并从指定检查点创建一条新分支。

## 14. Replay 与 Fork 对比

| 对比项 | Replay | Fork |
| --- | --- | --- |
| 是否先修改状态 | 否 | 是，调用 `update_state()` |
| 后续节点是否重跑 | 是 | 是 |
| 原历史是否保留 | 保留 | 保留 |
| 典型用途 | 复现问题、重试步骤 | 修改输入、尝试替代路径 |
| 是否再次调用模型和工具 | 会 | 会 |

可以简单记忆：

```text
Replay = 从过去再走一次
Fork   = 改变过去的状态，再走出一条新路
```

## 15. TIME_TRAVEL_MODE 的四种模式

为避免每次运行都意外重复模型和搜索调用，示例默认只查看历史：

```env
TIME_TRAVEL_MODE=history
```

支持的值：

| 模式 | 行为 |
| --- | --- |
| `history` | 执行当前问题，查看最新状态和历史 |
| `replay` | 再从输入检查点重放一次 |
| `fork` | 替换历史问题并运行新分支 |
| `all` | 依次演示 Replay 和 Fork |

建议先使用 `history` 理解快照，再分别运行 `replay` 和 `fork`。这样更容易观察数据库中的历史如何变化，也能控制模型调用成本。

## 16. 安装、配置与运行

安装依赖：

```bash
pip install -r requirements.txt
```

准备一个可连接的 PostgreSQL 数据库，然后复制环境变量模板：

```bash
# Windows PowerShell
Copy-Item .env.example .env

# macOS / Linux
cp .env.example .env
```

至少需要填写：

```env
OPENAI_API_KEY=你的模型服务密钥
TAVILY_API_KEY=你的 Tavily 密钥
POSTGRES_URI=postgresql://用户名:密码@主机:5432/数据库名
```

其余演示配置：

```env
THREAD_ID=langgraph-postgres-demo
PERSISTENCE_QUESTION=今天西安的天气如何？
FORK_QUESTION=我们之前聊过宠物相关的话题吗？
TIME_TRAVEL_MODE=history
LANGGRAPH_STRICT_MSGPACK=true
```

运行：

```bash
python langgraph_postgres_time_travel.py
```

程序会打印最终回答、累计模型调用次数、最新状态和检查点历史。选择 `replay`、`fork` 或 `all` 时，还会打印时间旅行产生的新结果。

## 17. 原始学习代码中的实用改进

完整示例保留了 PostgreSQL 持久化和时间旅行主题，并做了这些整理：

1. 删除源码中的固定数据库账号与密码，改用 `POSTGRES_URI`。
2. 增加 `langgraph-checkpoint-postgres` 和 Psycopg 依赖。
3. 使用 `add_messages` 替代普通列表拼接，为历史消息替换提供支持。
4. 为本轮输入分配 UUID，通过消息 ID 动态定位检查点。
5. 移除只能在原作者数据库中生效的固定 Checkpoint ID。
6. 把 Replay 和 Fork 分开，避免把状态分叉误称为普通重放。
7. 增加 `TIME_TRAVEL_MODE`，默认不重复发起外部调用。
8. 对缺少数据库连接串、非法模式和错误消息类型给出明确报错。
9. 工具结果显式转换为字符串，并为 `ToolMessage` 保存工具名称和调用 ID。
10. 使用 `if __name__ == "__main__"`，导入模块时不会连接数据库或执行 Agent。

## 18. 常见错误

### ModuleNotFoundError: langgraph.checkpoint.postgres

PostgreSQL Checkpointer 是独立包，需要安装：

```bash
pip install langgraph-checkpoint-postgres "psycopg[binary]"
```

### 缺少 POSTGRES_URI

复制 `.env.example` 为 `.env`，填写真实连接信息。不要把含密码的 `.env` 上传到 GitHub。

### 连接被拒绝

检查 PostgreSQL 是否正在运行、主机和端口是否正确、数据库是否存在，以及防火墙、用户权限和 TLS 参数是否符合服务器要求。

### 首次运行提示数据表不存在

确认连接数据库后调用了：

```python
checkpointer.setup()
```

数据库账号还需要创建和修改相关表的权限。

### 没有传 thread_id

带 Checkpointer 的图必须使用：

```python
{"configurable": {"thread_id": "..."}}
```

否则无法确定应保存或读取哪条线程历史。

### 换了 thread_id 后“失去记忆”

这是预期行为。不同 `thread_id` 的状态相互隔离；切回原 ID 才会读取原线程。

### 历史顺序看起来反了

`get_state_history()` 返回最新检查点在前。需要按时间正序展示时，可以对列表执行 `reversed(history)`。

### 从最终检查点 Replay 没有任何输出

最终检查点的 `next` 是空元组，说明没有待执行节点。从它重放是空操作。应选择 `next` 指向目标节点的更早检查点。

### Fork 后原来的历史消失了吗

没有。`update_state()` 创建新检查点和新分支，不会就地修改原始检查点。

### 替换问题却变成追加问题

确认状态使用 `add_messages`，并且新旧消息 ID 相同。若使用 `operator.add` 或不同 ID，新消息会被追加。

### Replay 重复调用了搜索工具

这是正常行为。Replay 会重新执行检查点之后的节点，包括模型调用、API 请求和工具调用。运行前应评估成本与副作用。

### 检查点数据持续增长

长会话会积累越来越多的消息与快照。生产环境需要设计保留策略、归档或删除机制，并监控数据库容量。

## 19. 复习问答

<details>
<summary>1. Checkpointer 保存的核心内容是什么？</summary>

它按线程保存图状态快照、节点写入和检查点之间的父子关系，使状态可以恢复、检查和继续执行。

</details>

<details>
<summary>2. thread_id 的作用是什么？</summary>

它标识一条独立线程。相同 ID 继续同一状态历史，不同 ID 创建彼此隔离的会话。

</details>

<details>
<summary>3. 为什么一次 invoke 会产生多个 Checkpoint？</summary>

LangGraph 会在每个 super-step 边界保存快照，而一次图执行通常经过多个节点或循环步骤。

</details>

<details>
<summary>4. get_state 与 get_state_history 有什么区别？</summary>

`get_state()` 读取最新或指定的一个快照；`get_state_history()` 返回线程的一组历史快照，默认最新在前。

</details>

<details>
<summary>5. Replay 会不会再次调用模型和工具？</summary>

会。检查点之后的节点会真实重跑，因此模型、API、工具和中断都可能再次触发。

</details>

<details>
<summary>6. update_state 会修改旧 Checkpoint 吗？</summary>

不会。它创建一个带更新值的新检查点，从过去状态分出新的执行路径，原历史仍被保留。

</details>

<details>
<summary>7. 为什么 Fork 中的新 HumanMessage 使用旧消息 ID？</summary>

`add_messages` 遇到相同 ID 时会替换旧消息。使用新 ID 则会把问题追加到对话末尾。

</details>

<details>
<summary>8. Checkpointer 与 Store 的差别是什么？</summary>

Checkpointer 保存单个线程的图状态快照；Store 保存应用定义的长期数据，并可供多个线程共享。

</details>

<details>
<summary>9. 为什么不应该写死 checkpoint_id？</summary>

它由每次运行动态生成，只对特定数据库和线程历史有效。可复现代码应按 `next`、消息 ID、步骤或元数据选择快照。

</details>

## 20. 可以继续练习

- 用相同 `thread_id` 连续运行两次，观察模型能否读取上一轮消息。
- 换一个 `thread_id`，验证新线程不会读取旧会话。
- 把 `TIME_TRAVEL_MODE` 依次改为 `replay`、`fork` 和 `all`，比较新增的历史快照。
- 使用 `metadata["source"] == "update"` 找出所有 Fork 产生的检查点。
- 为历史展示增加创建时间、消息数量和最后执行节点。
- 把同步 `PostgresSaver` 改成 `AsyncPostgresSaver`，配合 `ainvoke()` 使用。
- 给外部工具增加幂等键，避免 Replay 重复产生副作用。
- 配置检查点数据加密与定期清理策略。
- 再加入 LangGraph Store，区分线程记忆与跨线程长期记忆。

## 官方资料

- [LangGraph Persistence 概览](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangGraph Checkpointers：状态、历史与更新](https://docs.langchain.com/oss/python/langgraph/checkpointers)
- [LangGraph 时间旅行：Replay 与 Fork](https://docs.langchain.com/oss/python/langgraph/use-time-travel)
- [PostgresSaver API 参考](https://reference.langchain.com/python/langgraph/checkpoint/postgres/PostgresSaver)
- [add_messages API 参考](https://reference.langchain.com/python/langgraph/graph/message/add_messages)

## 完整源码

见 [`langgraph_postgres_time_travel.py`](../langgraph_postgres_time_travel.py)。
