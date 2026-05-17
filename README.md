# AstrBot 自动模型切换器

这个插件不使用聊天指令。它会像 judge 插件一样，在 AstrBot 发起 LLM 请求前改写请求的 provider/model；命中触发词后，插件会主动调用内部裁决工具，工具里再用 LLM 判断是否需要切换模型。

## 功能

- `switch_mode = AUTO`：默认聊天模型，命中触发词后插件主动调用裁决 LLM，由裁决结果决定是否切换。
- `switch_mode = HIGH`：所有允许路由的普通聊天自动切到高智商模型。
- `switch_mode = CHAT`：所有允许路由的普通聊天自动切到聊天模型。
- 当前会话处于高智商模型时，如果 `high_iq_idle_timeout_seconds` 秒内没有继续追问，下一条消息会先调用裁决 LLM；裁决认为不是继续追问时才切回聊天模型。
- 工具实际切换后，会优先调用 LLM 生成一句自然提示；失败时才使用 `high_iq_switch_reply` 或 `chat_switch_reply` 兜底。
- 支持单 provider/model，也支持列表和 `provider_id:model` 配对。
- 支持白名单、黑名单限制生效范围。

## 配置方式

高智商模型推荐填写：

```text
high_iq_routes = ["openai:gpt-4o"]
switch_mode = AUTO
high_iq_idle_timeout_seconds = 120
tool_trigger_keywords = ["高智商", "深度", "复杂", "切回", "普通聊天"]
```

聊天模型推荐填写：

```text
chat_routes = ["openai:gpt-4o-mini"]
default_mode = CHAT
```

也可以用列表一一对应：

```text
high_iq_provider_ids = ["openai", "anthropic"]
high_iq_models = ["gpt-4o", "claude-3-opus"]
```

插件会在 `on_llm_request` 中写入：

```python
req.provider_id = provider_id
req.model = model
```

并同步设置 `event.set_extra("selected_provider", provider_id)` / `selected_model`，贴近 judge 插件的路由方式。

## 裁决流程

插件保留了一个 LLM 工具函数：

```text
model_switcher_decide(target_mode, should_switch, reason)
```

实际自动流程是：

1. 用户消息命中 `tool_trigger_keywords`、`high_iq_trigger_keywords` 或 `chat_trigger_keywords` 任意一组关键词。
2. 插件立刻调用裁决 LLM。
3. 裁决 LLM 只输出 JSON：`should_switch`、`target_mode`、`reason`。
4. 如果裁决为切换，插件更新会话模型状态，并主动发送切换提示。
5. 当前轮请求构建前会读取新状态，尽量让本轮也使用新 provider/model。

切换提示默认由 LLM 生成，可用：

```text
enable_llm_switch_reply = true
switch_reply_provider_id = ""
switch_reply_model = ""
```

留空 provider 时，会优先复用 `judge_provider_id`，再退回当前会话 provider。

退出高智商模式也走同一套裁决：高智商空闲超时后，插件不会直接退出，而是把新消息交给裁决 LLM 判断是不是继续追问；如果不是追问或不再需要高智商上下文，才切回 `CHAT`。

裁决 LLM 需要基于语义判断：

- 明确需要强模型、深度分析、复杂推理：调用工具切到 `HIGH`。
- 明确要求切回普通聊天、轻量模式：调用工具切到 `CHAT`。
- 只是普通聊天或随口提到关键词：不调用工具。

这个裁决发生在主 Agent 构建前，比让主 LLM 自己决定调用工具更早，所以可以影响本轮 provider/model 选择。
