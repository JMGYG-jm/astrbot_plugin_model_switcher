from __future__ import annotations

import random
import json
import re
import time
from dataclasses import dataclass

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Plain
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register


@dataclass
class ModelRoute:
    provider_id: str
    model: str


@register(
    "astrbot_plugin_model_switcher",
    "JMGYG",
    "Route AstrBot LLM requests to configured high-IQ or chat models.",
    "1.2.0",
)
class ModelSwitcherPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.config = config or {}
        self._session_state: dict[str, dict[str, float | str]] = {}

    async def initialize(self):
        logger.info("[ModelSwitcher] loaded")

    @filter.on_waiting_llm_request(priority=110)
    async def before_llm_build(self, event: AstrMessageEvent):
        """Run before AstrBot builds the main agent, so provider selection can change."""
        await self._run_switch_judge_if_triggered(event)
        route = self._resolve_event_route(event)
        if not route:
            return
        event.set_extra("selected_provider", route.provider_id)
        if route.model:
            event.set_extra("selected_model", route.model)

    @filter.on_llm_request(priority=110)
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """Run before the LLM request is sent, matching judge-style request rewrite."""
        route = self._resolve_event_route(event)
        if not route:
            return

        req.provider_id = route.provider_id
        if route.model:
            req.model = route.model

        event.set_extra("selected_provider", route.provider_id)
        if route.model:
            event.set_extra("selected_model", route.model)

    @filter.llm_tool(name="model_switcher_decide")
    async def model_switcher_decide_tool(
        self,
        event: AstrMessageEvent,
        target_mode: str,
        should_switch: bool = True,
        reason: str = "",
    ):
        """Decide whether the current conversation should switch model mode.

        Call this tool when the user is clearly asking to enter or leave a
        high-IQ/deep-thinking mode, or when the current message is a programming,
        code, software project, webpage/frontend/backend, API, database, script,
        deployment, debugging, error-log, or plugin-development task that would
        benefit from a stronger model.

        For coding-related requests, judge by meaning, not by exact words. Phrases
        like "帮我写个网页", "这个程序报错了", "看看这段代码", "插件怎么改",
        "接口/数据库/前端/后端/脚本/项目有问题" often indicate HIGH mode.

        Switch back to CHAT when the user clearly returns to ordinary conversation
        or asks for normal/light chat. Do not call it for jokes, greetings, or a
        vague word mention with no real task.

        Args:
            target_mode(string): HIGH or CHAT.
            should_switch(bool): True if switching is needed, otherwise False.
            reason(string): Short reason for the decision.
        """
        if not self.config.get("enable_llm_tool", True):
            return "[TOOL_UNAVAILABLE] model switch tool is disabled."
        if not should_switch:
            return "[TOOL_SUCCESS] No model switch is needed. Continue normally."

        mode = str(target_mode or "").upper()
        if mode not in {"HIGH", "CHAT"}:
            return "[TOOL_ERROR] target_mode must be HIGH or CHAT."

        previous_mode = self._get_session_mode(event)
        self._set_session_mode(event, mode)
        if previous_mode != mode and self.config.get("reply_on_switch", True):
            reply = await self._build_switch_reply_with_llm(event, mode, reason)
            await event.send(event.chain_result([Plain(reply)]))

        return (
            f"[TOOL_SUCCESS] Switched conversation model mode to {mode}. "
            "Do not mention internal tool names or provider IDs to the user."
        )

    async def _run_switch_judge_if_triggered(self, event: AstrMessageEvent) -> None:
        if not self.config.get("enable_llm_judge_tool", True):
            return
        if event.get_extra("model_switcher_judged"):
            return
        event.set_extra("model_switcher_judged", True)

        trigger_reason = self._judge_trigger_reason(event)
        if not trigger_reason:
            return

        logger.info(
            "[ModelSwitcher] judge triggered: %s, current_mode=%s, message=%s",
            trigger_reason,
            self._get_session_mode(event),
            str(event.message_str or "")[:120],
        )
        decision = await self._judge_switch_with_llm(event)
        logger.info("[ModelSwitcher] judge decision: %s", decision)
        if not decision.get("should_switch"):
            logger.info("[ModelSwitcher] judge says no switch: %s", decision.get("reason", ""))
            return

        mode = str(decision.get("target_mode") or "").upper()
        if mode not in {"HIGH", "CHAT"}:
            logger.warning("[ModelSwitcher] judge returned invalid mode: %s", mode)
            return

        previous_mode = self._get_session_mode(event)
        self._set_session_mode(event, mode)
        logger.info(
            "[ModelSwitcher] mode changed by judge: %s -> %s, reason=%s",
            previous_mode,
            mode,
            decision.get("reason", ""),
        )
        if previous_mode != mode and self.config.get("reply_on_switch", True):
            reply = await self._build_switch_reply_with_llm(
                event,
                mode,
                str(decision.get("reason") or ""),
            )
            await event.send(
                event.chain_result([Plain(reply)])
            )

    async def _judge_switch_with_llm(self, event: AstrMessageEvent) -> dict:
        provider_id = str(self.config.get("judge_provider_id") or "").strip()
        if not provider_id:
            try:
                provider_id = await self.context.get_current_chat_provider_id(event.unified_msg_origin)
            except Exception as exc:
                logger.warning("[ModelSwitcher] cannot resolve judge provider: %s", exc)
                return {"should_switch": False, "target_mode": "", "reason": "no judge provider"}

        prompt = self._build_judge_prompt(event)
        kwargs = {"temperature": 0}
        judge_model = str(self.config.get("judge_model") or "").strip()
        if judge_model:
            kwargs["model"] = judge_model

        try:
            response = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt,
                system_prompt=(
                    "You are a strict routing classifier. Output only valid JSON. "
                    "Do not explain outside JSON."
                ),
                **kwargs,
            )
        except Exception as exc:
            logger.warning("[ModelSwitcher] judge LLM failed: %s", exc)
            return {"should_switch": False, "target_mode": "", "reason": "judge failed"}

        return self._parse_judge_response(response.completion_text or "")

    def _build_judge_prompt(self, event: AstrMessageEvent) -> str:
        current_mode = self._get_session_mode(event)
        high_words = ", ".join(self._as_list(self.config.get("high_iq_trigger_keywords", [])))
        chat_words = ", ".join(self._as_list(self.config.get("chat_trigger_keywords", [])))
        coding_signal = self._looks_like_programming_high_iq_request(str(event.message_str or ""))
        chat_signal = self._looks_like_chat_mode_request(str(event.message_str or ""))
        return (
            "Decide whether this AstrBot conversation should switch model mode.\n"
            "Return JSON only, with this schema:\n"
            '{"should_switch": true|false, "target_mode": "HIGH|CHAT", "reason": "short reason"}\n\n'
            f"Current mode: {current_mode}\n"
            f"High-IQ idle timeout reached: {self._high_iq_exit_judge_needed(event)}\n"
            f"Programming/code/web semantic signal: {coding_signal}\n"
            f"Normal-chat semantic signal: {chat_signal}\n"
            f"High-IQ intent hints: {high_words or 'deep analysis, complex reasoning'}\n"
            f"Chat intent hints: {chat_words or 'normal chat, switch back'}\n\n"
            "Rules:\n"
            "- Switch to HIGH only when the user clearly wants stronger/deeper reasoning, or the task obviously needs it.\n"
            "- Treat concrete programming, code, webpage/frontend/backend, API, database, script, debugging, error-log, deployment, or plugin-development work as HIGH-worthy unless it is only a casual mention.\n"
            "- Switch to CHAT only when the user clearly asks to return to normal/light chat.\n"
            "- If High-IQ idle timeout reached is true, switch to CHAT unless the new user message is clearly a follow-up that still needs high-IQ context.\n"
            "- If the message merely contains a trigger word without real switching intent, do not switch.\n"
            "- If the current mode already matches the target, should_switch must be false unless the user explicitly asks again.\n\n"
            f"User message:\n{event.message_str}"
        )

    def _parse_judge_response(self, text: str) -> dict:
        cleaned = str(text or "").strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.S)
            if not match:
                return {"should_switch": False, "target_mode": "", "reason": "invalid json"}
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return {"should_switch": False, "target_mode": "", "reason": "invalid json"}

        if not isinstance(data, dict):
            return {"should_switch": False, "target_mode": "", "reason": "invalid response"}

        target_mode = str(data.get("target_mode") or "").upper()
        return {
            "should_switch": bool(data.get("should_switch", False)) and target_mode in {"HIGH", "CHAT"},
            "target_mode": target_mode,
            "reason": str(data.get("reason") or "").strip(),
        }

    def _resolve_event_route(self, event: AstrMessageEvent) -> ModelRoute | None:
        if not self.config.get("enable", True):
            return None
        if not self._is_router_allowed(event):
            return None

        mode = self._determine_mode(event)

        route = self._select_route(mode)
        if not route.provider_id:
            if mode != "CHAT":
                logger.warning("[ModelSwitcher] no provider configured for mode=%s", mode)
            return None

        provider = self.context.get_provider_by_id(route.provider_id)
        if not provider:
            logger.warning("[ModelSwitcher] provider not found: %s", route.provider_id)
            return None

        logger.info(
            "[ModelSwitcher] route mode=%s provider=%s model=%s",
            mode,
            route.provider_id,
            route.model or "<provider-default>",
        )
        return route

    def _determine_mode(self, event: AstrMessageEvent) -> str:
        configured_mode = str(self.config.get("switch_mode", "AUTO") or "AUTO").upper()
        if configured_mode not in {"AUTO", "HIGH", "CHAT"}:
            configured_mode = "AUTO"

        umo = event.unified_msg_origin
        now = time.time()
        state = self._session_state.setdefault(umo, {})

        if configured_mode == "AUTO":
            mode = str(state.get("mode") or self.config.get("default_mode", "CHAT") or "CHAT").upper()
        else:
            mode = configured_mode

        if mode not in {"HIGH", "CHAT"}:
            mode = "CHAT"

        direct_timeout_enabled = not self.config.get("enable_llm_judge_tool", True)
        if mode == "HIGH" and direct_timeout_enabled and self._high_iq_idle_expired(state, now):
            state["mode"] = "CHAT"
            state.pop("last_high_ts", None)
            return "CHAT"

        if mode == "HIGH":
            state["mode"] = "HIGH"
            state["last_high_ts"] = now
        else:
            state["mode"] = "CHAT"
        return mode

    def _get_session_mode(self, event: AstrMessageEvent) -> str:
        state = self._session_state.setdefault(event.unified_msg_origin, {})
        return str(state.get("mode") or self.config.get("default_mode", "CHAT") or "CHAT").upper()

    def _set_session_mode(self, event: AstrMessageEvent, mode: str) -> None:
        state = self._session_state.setdefault(event.unified_msg_origin, {})
        mode = "HIGH" if str(mode).upper() == "HIGH" else "CHAT"
        state["mode"] = mode
        if mode == "HIGH":
            state["last_high_ts"] = time.time()
        else:
            state.pop("last_high_ts", None)

    def _should_trigger_judge_tool(self, event: AstrMessageEvent) -> bool:
        return bool(self._judge_trigger_reason(event))

    def _judge_trigger_reason(self, event: AstrMessageEvent) -> str:
        if not self.config.get("enable_llm_judge_tool", True):
            return ""
        if self._high_iq_exit_judge_needed(event):
            return "high_iq_idle_timeout"
        text = str(event.message_str or "")
        groups = {
            "tool_trigger_keywords": self._as_list(self.config.get("tool_trigger_keywords", [])),
            "high_iq_trigger_keywords": self._as_list(self.config.get("high_iq_trigger_keywords", [])),
            "chat_trigger_keywords": self._as_list(self.config.get("chat_trigger_keywords", [])),
        }
        text_folded = text.casefold()
        for group_name, keywords in groups.items():
            for keyword in keywords:
                if keyword.casefold() in text_folded:
                    return f"{group_name}:{keyword}"
        if self._looks_like_programming_high_iq_request(text):
            return "semantic_programming_request"
        if self._looks_like_chat_mode_request(text):
            return "semantic_chat_request"
        return ""

    def _looks_like_programming_high_iq_request(self, message: str) -> bool:
        """Detect coding/web/project requests so the judge LLM can decide routing."""
        text = str(message or "").strip()
        if not text:
            return False

        lower = text.casefold()
        compact = re.sub(r"\s+", "", lower)

        code_block_markers = ["```", "traceback", "stack trace", "syntaxerror", "typeerror", "attributeerror"]
        if any(marker in lower for marker in code_block_markers):
            return True

        file_or_command_patterns = [
            r"\b[a-z0-9_\-]+\.(py|js|ts|tsx|jsx|vue|html|css|json|yaml|yml|toml|md|java|go|rs|cpp|c|cs|php|sql|sh|ps1)\b",
            r"\b(npm|pnpm|yarn|pip|python|node|git|docker|kubectl|powershell|cmd)\s+[\w\-./:@]+",
            r"[a-z]:\\[^\s]+",
            r"/[\w./-]+/[\w.-]+",
        ]
        if any(re.search(pattern, lower, flags=re.I) for pattern in file_or_command_patterns):
            return True

        coding_terms = [
            "编程", "代码", "源码", "程序", "项目", "工程", "脚本", "插件", "模块",
            "函数", "类", "方法", "变量", "组件", "依赖", "库", "框架", "架构",
            "网页", "网站", "前端", "后端", "全栈", "页面", "样式", "布局",
            "html", "css", "javascript", "typescript", "python", "java", "golang",
            "react", "vue", "vite", "node", "npm", "api", "接口", "数据库", "sql",
            "服务端", "服务器", "部署", "构建", "打包", "报错", "异常", "bug",
            "日志", "测试", "单元测试", "正则", "json", "yaml", "git", "github",
            "astrbot", "llm_tool", "provider", "model", "路由",
        ]
        action_terms = [
            "写", "做", "做个", "开发", "实现", "加", "加上", "改", "修改", "修",
            "修复", "优化", "重构", "封装", "适配", "迁移", "排查", "调试", "分析",
            "解释", "检查", "看看", "看下", "帮我", "怎么", "为什么", "报错",
            "不生效", "没反应", "跑不起来", "安装失败", "触发不了", "打包",
        ]
        complexity_terms = [
            "复杂", "详细", "认真", "深度", "完整", "生产", "稳定", "兼容",
            "安全", "性能", "并发", "异步", "架构", "源码", "项目",
        ]
        casual_exclusions = [
            "代码是什么", "编程是什么", "网页是什么", "程序是什么", "随便聊",
            "不用写", "不要写", "开玩笑",
        ]

        has_coding = any(term in compact for term in coding_terms)
        has_action = any(term in compact for term in action_terms)
        has_complexity = any(term in compact for term in complexity_terms)
        if any(term in compact for term in casual_exclusions) and not has_action:
            return False

        if has_coding and (has_action or has_complexity):
            return True

        direct_phrases = [
            "帮我写个网页", "帮我写网页", "写个网页", "做个网页", "写个程序",
            "写段代码", "看看代码", "看下代码", "代码报错", "程序报错",
            "网页报错", "插件报错", "接口报错", "数据库报错", "项目跑不起来",
        ]
        return any(phrase in compact for phrase in direct_phrases)

    def _looks_like_chat_mode_request(self, message: str) -> bool:
        """Detect explicit requests to leave deep/coding mode and chat normally."""
        text = str(message or "").strip()
        if not text:
            return False
        compact = re.sub(r"\s+", "", text.casefold())
        chat_phrases = [
            "切回聊天", "切回正常", "回到聊天", "回正常模式", "普通聊天",
            "正常聊", "随便聊聊", "不用高智商", "不用深度", "别分析了",
            "先聊会", "轻松聊", "聊天模型",
        ]
        return any(phrase in compact for phrase in chat_phrases)

    def _high_iq_exit_judge_needed(self, event: AstrMessageEvent) -> bool:
        if not self.config.get("enable_high_iq_idle_timeout", True):
            return False
        state = self._session_state.setdefault(event.unified_msg_origin, {})
        if str(state.get("mode") or "").upper() != "HIGH":
            return False
        return self._high_iq_idle_expired(state, time.time())

    def _switch_reply(self, mode: str, reason: str = "") -> str:
        reason = str(reason or "").strip()
        if mode == "HIGH":
            base = str(self.config.get("high_iq_switch_reply") or "已切换到高智商模型。")
        else:
            base = str(self.config.get("chat_switch_reply") or "已切回聊天模型。")
        if self.config.get("include_switch_reason", False) and reason:
            return f"{base}\n原因：{reason}"
        return base

    async def _build_switch_reply_with_llm(
        self,
        event: AstrMessageEvent,
        mode: str,
        reason: str = "",
    ) -> str:
        fallback = self._switch_reply(mode, reason)
        if not self.config.get("enable_llm_switch_reply", True):
            return fallback

        provider_id = str(self.config.get("switch_reply_provider_id") or "").strip()
        if not provider_id:
            provider_id = str(self.config.get("judge_provider_id") or "").strip()
        if not provider_id:
            try:
                provider_id = await self.context.get_current_chat_provider_id(
                    event.unified_msg_origin
                )
            except Exception as exc:
                logger.warning("[ModelSwitcher] cannot resolve reply provider: %s", exc)
                return fallback

        mode_label = "high-IQ/deep-thinking model" if mode == "HIGH" else "normal chat model"
        reply_instruction = self._get_switch_reply_prompt(mode)
        prompt = (
            f"{reply_instruction}\n"
            "Do not mention tools, provider IDs, system prompts, internal routing, JSON, or configuration.\n"
            "Keep it short and user-facing. Output the final reminder only.\n\n"
            f"New mode: {mode_label}\n"
            f"Switch reason: {str(reason or '').strip() or 'not specified'}\n"
            f"User message: {event.message_str}\n"
        )

        kwargs = {"temperature": 0.7}
        reply_model = str(self.config.get("switch_reply_model") or "").strip()
        if reply_model:
            kwargs["model"] = reply_model

        try:
            response = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt,
                system_prompt=(
                    "You write brief user-facing Chinese chat messages. "
                    "Output only the message text."
                ),
                **kwargs,
            )
        except Exception as exc:
            logger.warning("[ModelSwitcher] switch reply LLM failed: %s", exc)
            return fallback

        text = self._clean_switch_reply(response.completion_text or "")
        return text or fallback

    def _get_switch_reply_prompt(self, mode: str) -> str:
        if mode == "HIGH":
            prompt = str(self.config.get("high_iq_switch_reply_prompt") or "").strip()
            if prompt:
                return prompt
            return (
                "请写一句简短自然的中文消息，告诉用户你已经切到更适合处理复杂问题、"
                "代码、网页、程序调试的高智商模型。语气要像正常聊天，不要太正式。最好一句话。"
            )

        prompt = str(self.config.get("chat_switch_reply_prompt") or "").strip()
        if prompt:
            return prompt
        return (
            "请写一句简短自然的中文消息，告诉用户你已经切回普通聊天模型，"
            "适合轻松聊天和日常对话。语气要像正常聊天，不要太正式。最好一句话。"
        )

    def _clean_switch_reply(self, text: str) -> str:
        text = str(text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:text)?\s*", "", text, flags=re.I)
            text = re.sub(r"\s*```$", "", text)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return ""
        text = lines[0]
        banned = ["tool", "工具", "provider", "配置", "JSON", "系统提示", "内部"]
        if any(word.casefold() in text.casefold() for word in banned):
            return ""
        return text[:120]

    def _high_iq_idle_expired(self, state: dict[str, float | str], now: float) -> bool:
        if not self.config.get("enable_high_iq_idle_timeout", True):
            return False
        try:
            timeout = int(self.config.get("high_iq_idle_timeout_seconds", 120) or 0)
        except (TypeError, ValueError):
            timeout = 120
        if timeout <= 0:
            return False

        last_high_ts = state.get("last_high_ts")
        if not isinstance(last_high_ts, (int, float)):
            return False
        return now - float(last_high_ts) > timeout

    def _select_route(self, mode: str) -> ModelRoute:
        if mode == "CHAT":
            routes = self._normalize_routes(
                self.config.get("chat_routes", []),
                self.config.get("chat_provider_ids", []),
                self.config.get("chat_models", []),
            )
            if not routes:
                routes = [
                    ModelRoute(
                        provider_id=str(self.config.get("chat_provider_id") or "").strip(),
                        model=str(self.config.get("chat_model") or "").strip(),
                    )
                ]
            return self._choose_route(
                routes,
                bool(self.config.get("enable_chat_polling", False)),
            )

        routes = self._normalize_routes(
            self.config.get("high_iq_routes", []),
            self.config.get("high_iq_provider_ids", []),
            self.config.get("high_iq_models", []),
        )
        if not routes:
            routes = [
                ModelRoute(
                    provider_id=str(self.config.get("high_iq_provider_id") or "").strip(),
                    model=str(self.config.get("high_iq_model") or "").strip(),
                )
            ]
        return self._choose_route(
            routes,
            bool(self.config.get("enable_high_iq_polling", True)),
        )

    def _normalize_routes(self, route_items, provider_ids, models) -> list[ModelRoute]:
        routes: list[ModelRoute] = []

        if isinstance(route_items, list):
            for item in route_items:
                route = self._parse_route_item(item)
                if route.provider_id:
                    routes.append(route)
            if routes:
                return routes

        if not isinstance(provider_ids, list):
            provider_ids = []
        if not isinstance(models, list):
            models = []

        for index, provider_id in enumerate(provider_ids):
            provider_id = str(provider_id or "").strip()
            if not provider_id:
                continue
            model = str(models[index] or "").strip() if index < len(models) else ""
            routes.append(ModelRoute(provider_id=provider_id, model=model))
        return routes

    def _parse_route_item(self, item) -> ModelRoute:
        if isinstance(item, dict):
            return ModelRoute(
                provider_id=str(
                    item.get("provider_id") or item.get("provider") or ""
                ).strip(),
                model=str(item.get("model") or "").strip(),
            )
        if isinstance(item, (list, tuple)):
            provider_id = str(item[0] if len(item) > 0 else "").strip()
            model = str(item[1] if len(item) > 1 else "").strip()
            return ModelRoute(provider_id=provider_id, model=model)

        text = str(item or "").strip()
        if ":" in text:
            provider_id, model = (part.strip() for part in text.split(":", 1))
            return ModelRoute(provider_id=provider_id, model=model)
        return ModelRoute(provider_id=text, model="")

    def _choose_route(self, routes: list[ModelRoute], enable_polling: bool) -> ModelRoute:
        routes = [route for route in routes if route.provider_id]
        if not routes:
            return ModelRoute(provider_id="", model="")
        if enable_polling and len(routes) > 1:
            return random.choice(routes)
        return routes[0]

    def _is_router_allowed(self, event: AstrMessageEvent) -> bool:
        whitelist = self._as_list(self.config.get("router_whitelist", []))
        blacklist = self._as_list(self.config.get("router_blacklist", []))
        keys = {
            event.unified_msg_origin,
            event.get_group_id(),
            event.get_sender_id(),
            event.get_session_id(),
        }
        keys = {key for key in keys if key}

        if whitelist and not keys.intersection(whitelist):
            return False
        if blacklist and keys.intersection(blacklist):
            return False
        return True

    def _as_list(self, value) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return []

    def _matches_any(self, text: str, value) -> bool:
        text = text.casefold()
        for keyword in self._as_list(value):
            if keyword.casefold() in text:
                return True
        return False
