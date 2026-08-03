"""Unified Chat Service for Piata AI agents (refactored from server.py).

Centralizes all chat runtime logic:
- Demo chat (rate-limited, no DB logging)
- Regular agent chat (full DB logging + tool execution)
- Native OpenAI tool-call handling + JSON-in-text fallback
- Conversation history management

Importable by server.py, api_server.py and the social_agent orchestrator
to avoid duplicated logic.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import aiohttp
from engine.provisioner import TOOL_SCHEMAS
from piata_tools.registry import get_registry

logger = logging.getLogger("piata-agency")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(name)s: %(levelname)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Default model alias map (mirrors server.py)
MODEL_ALIASES = {
    "mimo-v2.5-pro":       os.environ.get("NVIDIA_NIM_MODEL", "deepseek-ai/deepseek-v4-pro"),
    "mimo-v2":             os.environ.get("NVIDIA_NIM_MODEL", "deepseek-ai/deepseek-v4-pro"),
    "minimax-m3":          "mistralai/mistral-large",
    "minimax-m3:cloud":    "mistralai/mistral-large",
    "qwen3.5:9b":          "meta/llama-3.3-70b-instruct",
    "llama-3.3-70b":       "meta/llama-3.3-70b-instruct",
    "deepseek-v4-pro":     "deepseek-ai/deepseek-v4-pro",
    "deepseek-v4-flash":   "deepseek-ai/deepseek-v4-flash",
    "mistral-large":       "mistralai/mistral-large",
}


def _resolve_model(model_name: str) -> str:
    """Map old/friendly model names to real provider model IDs."""
    if not model_name:
        return os.environ.get("NVIDIA_NIM_MODEL", "deepseek-ai/deepseek-v4-pro")
    if model_name in MODEL_ALIASES:
        return MODEL_ALIASES[model_name]
    if "/" in model_name:
        return model_name
    return os.environ.get("NVIDIA_NIM_MODEL", "deepseek-ai/deepseek-v4-pro")


def parse_tools(tools_field) -> List[str]:
    """Normalize the agent.tools field into a list of tool names."""
    if not tools_field:
        return []
    if isinstance(tools_field, list):
        return [str(t).strip() for t in tools_field if t]
    s = str(tools_field).strip()
    if not s:
        return []
    if s.startswith("["):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(t).strip() for t in parsed if t]
        except Exception:
            return []
    return [t.strip() for t in s.split(",") if t.strip()]


def _tc_args_nonempty(tc: dict) -> bool:
    """True if this tool_call has a non-empty JSON arguments string."""
    args = (tc.get("function") or {}).get("arguments")
    if not isinstance(args, str) or not args.strip():
        return False
    try:
        parsed = json.loads(args)
    except (json.JSONDecodeError, ValueError):
        return False
    return bool(parsed) and parsed != {}


def _extract_tool_calls(content: str) -> List[Dict]:
    """Extract one or more JSON tool-call objects from LLM text.

    Uses JSONDecoder.raw_decode at each object boundary so nested braces
    are balanced correctly. Invalid/prose objects are skipped silently.
    """
    if not content:
        return []
    decoder = json.JSONDecoder()
    calls = []
    for index, char in enumerate(content):
        if char != "{":
            continue
        try:
            parsed, _end = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        tool_name = parsed.get("tool")
        if not isinstance(tool_name, str) or not tool_name.strip():
            continue
        tool_args = parsed.get(
            "args",
            parsed.get("arguments", parsed.get("params", parsed.get("parameters", {}))),
        )
        if not isinstance(tool_args, dict):
            continue
        calls.append(parsed)
    return calls


def _normalize_piata_args(tool_name: str, args: Dict) -> Dict:
    """Normalize common parameter name variations for piata tools."""
    if tool_name != "piata_post_ad":
        return args
    normalized = dict(args)
    if "location" in normalized and "city" not in normalized:
        normalized["city"] = normalized.pop("location")
    if "contact_info" in normalized and "phone" not in normalized:
        normalized["phone"] = normalized.pop("contact_info")
    if "categories" in normalized and isinstance(normalized["categories"], list):
        cats = normalized.pop("categories")
        if len(cats) >= 1 and "category" not in normalized:
            normalized["category"] = cats[0]
        if len(cats) >= 2 and "subcategory" not in normalized:
            normalized["subcategory"] = cats[1]
    if "price" in normalized and isinstance(normalized["price"], str):
        import re

        match = re.search(r"[\d,.]+", normalized["price"].replace(",", ""))
        if match:
            normalized["price"] = float(match.group())
    return normalized


def build_tool_descriptors(tools: List[str]):
    """Return (tool_desc_lines, openai_tools) for a set of tool names."""
    tool_desc: List[str] = []
    openai_tools: List[Dict] = []
    for t in tools:
        for schema in TOOL_SCHEMAS.get(t, []):
            params = schema.get("parameters", {}).get("properties", {})
            name = schema.get("name", "")
            desc = schema.get("description", "")
            param_str = (
                " | params: " + ", ".join(f"{k}:{v}" for k, v in params.items())
                if params else ""
            )
            tool_desc.append(f"- **{name}**: {desc}{param_str}")

            openai_properties: Dict[str, Any] = {}
            required = schema.get("parameters", {}).get("required", [])
            for pname, ptype_schema in params.items():
                if isinstance(ptype_schema, dict):
                    openai_properties[pname] = ptype_schema
                else:
                    openai_properties[pname] = {"type": "string", "description": str(ptype_schema)}
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": desc,
                        "parameters": {
                            "type": "object",
                            "properties": openai_properties,
                            **({"required": list(required)} if required else {}),
                        },
                    },
                }
            )
    return tool_desc, openai_tools


async def execute_tool_calls(tool_calls, messages, llm_content, agent,
                             proxy_url, api_key, model, session, agent_id,
                             openai_tools=None):
    """Execute tool calls (native OpenAI or JSON-in-text) and get final response.

    Returns a dict with role/content/tool_calls/agent_id/agent_name.
    """
    registry = get_registry()
    timeout_s = float(os.environ.get("PIATA_TOOL_TIMEOUT_SECONDS", "30"))
    results: List[Dict] = []

    for tc in tool_calls:
        if "id" not in tc:
            tc["id"] = f"call_{tc.get('index', 0)}"
        function_name = tc.get("function", {}).get("name", "")
        try:
            function_args = json.loads(tc["function"]["arguments"])
        except (json.JSONDecodeError, KeyError, TypeError):
            function_args = {}

        if not isinstance(function_name, str) or not isinstance(function_args, dict):
            results.append({"tool": str(function_name), "error": "invalid tool call shape"})
            continue

        function_args = _normalize_piata_args(function_name, function_args)
        try:
            result = await asyncio.wait_for(
                registry.call(function_name, **function_args),
                timeout=timeout_s,
            )
            results.append({"tool": function_name, "result": result})
        except asyncio.TimeoutError:
            results.append({"tool": function_name, "error": "tool execution timed out"})
        except Exception as e:
            results.append({"tool": function_name, "error": str(e)})

    # Build tool results message and append to conversation
    tool_msg_lines = []
    for r in results:
        r_json = json.dumps(r.get("result", r.get("error")), indent=2, default=str)
        tool_msg_lines.append(f"**{r['tool']}** result:\n```json\n{r_json}\n```")
    tool_msg = "\n\n".join(tool_msg_lines)

    messages.append({"role": "assistant", "content": llm_content})
    messages.append({"role": "user", "content": f"Tool results:\n{tool_msg}\n\nProvide your final answer."})

    # Second LLM call with tool results — preserve native tool spec if available
    req_body = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 2000,
    }
    if openai_tools:
        req_body["tools"] = openai_tools
        req_body["tool_choice"] = "auto"

    try:
        async with session.post(
            f"{proxy_url}/v1/chat/completions",
            json=req_body,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            data = await resp.json()
            if "error" in data:
                msg = data["error"].get("message", str(data["error"]))
                raise RuntimeError(msg)
            choice = (data.get("choices") or [{}])[0]
            msg_obj = choice.get("message", {}) if isinstance(choice, dict) else {}
            return {
                "role": "assistant",
                "content": msg_obj.get("content", "") or "",
                "tool_calls": results,
                "agent_id": agent.get("agent_id", agent_id),
                "agent_name": agent.get("name", ""),
            }
    except RuntimeError as e:
        logger.error(f"Tool-followup LLM call failed: {e}")
        return {
            "role": "assistant",
            "content": "Error generating final response after tool calls. See logs.",
            "tool_calls": results,
            "agent_id": agent.get("agent_id", agent_id),
            "agent_name": agent.get("name", ""),
        }


class ChatService:
    """Encapsulates all Piata AI chat runtime logic.

    Example:
        from chat_service import ChatService
        svc = ChatService(provisioner)
        reply = await svc.run_agent_chat(agent_id, \"What's the weather?\")
    """

    def __init__(self, provisioner=None, registry=None):
        self.provisioner = provisioner
        self.registry = registry or get_registry()
        self._demo_usage: Dict[str, Dict] = {}
        # Connection pool — reuse TCP connections for long-running server
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()
        # Per-agent conversation memory (sliding window)
        # { agent_id: {"messages": [...], "last_access": timestamp} }
        self._conversations: Dict[str, Dict] = {}
        self._conv_lock = asyncio.Lock()
        self._max_history = int(os.environ.get("AGENT_HISTORY_SIZE", "20"))
        self._history_ttl = int(os.environ.get("AGENT_HISTORY_TTL_SECONDS", "3600"))  # 1 hour
        self._gc_task: Optional[asyncio.Task] = None

    async def _get_conversation(self, agent_id: str) -> List[Dict]:
        """Get sliding-window conversation history for an agent."""
        async with self._conv_lock:
            conv = self._conversations.get(agent_id)
            if conv is None:
                return []
            now = time.time()
            if now - conv["last_access"] > self._history_ttl:
                del self._conversations[agent_id]
                return []
            conv["last_access"] = now
            return list(conv["messages"])

    async def _append_conversation(self, agent_id: str, message: Dict):
        """Append a message to the conversation and trim old entries."""
        async with self._conv_lock:
            conv = self._conversations.get(agent_id)
            if conv is None:
                self._conversations[agent_id] = conv = {"messages": [], "last_access": time.time()}
            conv["messages"].append(message)
            conv["last_access"] = time.time()
            # Keep only the last N messages
            if len(conv["messages"]) > self._max_memory_messages():
                conv["messages"] = conv["messages"][-self._max_memory_messages():]

    def _max_memory_messages(self):
        """Calculate total messages to keep (user + assistant pairs)."""
        return max(4, self._max_history * 2)

    async def gc_conversations(self):
        """Clean up stale conversations (call periodically from server)."""
        async with self._conv_lock:
            now = time.time()
            stale = [aid for aid, cv in self._conversations.items()
                     if now - cv["last_access"] > self._history_ttl]
            for aid in stale:
                del self._conversations[aid]
            if stale:
                logger.info(f"chat_service GC: removed {len(stale)} expired conversations")

    async def _ensure_gc_loop(self):
        """Start the conversation GC loop on first use (idempotent).

        Runs every 60s so stale conversations don't accumulate in RAM.
        """
        if self._gc_task is not None and not self._gc_task.done():
            return

        async def _loop():
            while True:
                try:
                    await self.gc_conversations()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("conversation GC error")
                await asyncio.sleep(60)

        self._gc_task = asyncio.ensure_future(_loop())

    async def close(self):
        """Cancel the background GC loop. Call from the server shutdown path."""
        if self._gc_task is not None and not self._gc_task.done():
            self._gc_task.cancel()
            try:
                await self._gc_task
            except (asyncio.CancelledError, Exception):
                pass
            self._gc_task = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create a long-lived aiohttp session (connection pooling)."""
        if self._session is None or self._session.closed:
            async with self._session_lock:
                if self._session is None or self._session.closed:
                    connector = aiohttp.TCPConnector(
                        limit=50,                   # Max concurrent connections
                        limit_per_host=20,          # Per-host limit
                        ttl_dns_cache=300,           # DNS cache TTL
                        keepalive_timeout=60,        # TCP keepalive
                        force_close=False,           # Reuse connections
                    )
                    timeout = aiohttp.ClientTimeout(total=120, connect=10, sock_read=90)
                    self._session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=timeout,
                    )
                    logger.info("chat_service: created new HTTP session pool")
        return self._session

    # --- public API ------------------------------------------------------
    async def run_agent_chat(self, agent_id: str, message: str,
                             history: Optional[List[Dict]] = None,
                             agent: Optional[Dict] = None) -> Dict:
        """Run a chat turn for a provisioned agent (full DB logging)."""
        if agent is None:
            agent = self.provisioner.get_agent(agent_id) if self.provisioner else None
        if not agent:
            return {"error": "Agent not found", "agent_id": agent_id}
        if agent.get("status") != "active":
            return {"error": f"Agent is {agent['status']}", "agent_id": agent_id}

        return await self._run_chat_impl(agent, message, history or [], agent_id, log=True)

    async def run_demo_chat(self, agent: Dict, message: str,
                            history: Optional[List[Dict]] = None) -> Dict:
        """Run a demo chat turn (rate-limited, no DB logging).

        Accepts the demo agent dict directly (not looked up via provisioner).
        """
        agent_id = agent.get("agent_id", "demo")

        # Rate limit: 5 messages per agent per hour
        now = time.time()
        usage = self._demo_usage.setdefault(agent_id, {"count": 0, "reset_at": now + 3600})
        if now > usage["reset_at"]:
            usage["count"] = 0
            usage["reset_at"] = now + 3600
        if usage["count"] >= 5:
            reset_mins = int((usage["reset_at"] - now) / 60) + 1
            return {
                "error": (
                    f"Demo limit reached (5 messages/hour). Deploy your own agent "
                    f"at piata-ai.ro/agency for unlimited access."
                ),
                "retry_in_minutes": reset_mins,
                "agent_id": agent_id,
            }
        usage["count"] += 1

        return await self._run_chat_impl(agent, message, history or [], agent_id, log=False)

    # --- core implementation --------------------------------------------
    async def _run_chat_impl(self, agent: Dict, message: str, history: List[Dict],
                             agent_id: str, log: bool) -> Dict:
        tools = parse_tools(agent.get("tools", ""))
        # Scope the persistent memory tool to this agent so agents don't
        # read/write each other's memories (set_agent_id feeds the module
        # global in piata_tools.tools.memory used by memory_store/retrieve).
        try:
            from piata_tools.tools.memory import set_agent_id
            set_agent_id(agent_id)
        except Exception:
            pass
        if log:
            # check_plan_limits lives in billing.py (not on AgentProvisioner).
            # Lazy import avoids any module-import-order coupling.
            from billing import check_plan_limits
            ok, reason = check_plan_limits(agent_id, requested_tools=tools)
            if not ok:
                return {"error": reason, "agent_id": agent_id, "role": "error"}

        tool_desc, openai_tools = build_tool_descriptors(tools)
        system_prompt = self._build_system_prompt(agent, tool_desc, openai_tools)

        # In-process conversation memory: when no explicit history is passed,
        # continue the agent's previous turns from the sliding-window cache so
        # consecutive dashboard chats stay in context.
        await self._ensure_gc_loop()
        memory = await self._get_conversation(agent_id)
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            # Caller supplied full context (e.g. DB history) — use it as-is.
            messages.extend(history[-self._max_history:])
        elif memory:
            messages.extend(memory[-self._max_history:])
        messages.append({"role": "user", "content": message})
        await self._append_conversation(agent_id, {"role": "user", "content": message})

        if log and self.provisioner:
            self.provisioner.log_chat(agent_id, "user", message)

        proxy_url = os.environ.get("LLM_PROXY_URL", "http://127.0.0.1:5007")
        api_key = os.environ.get("OPENAI_API_KEY", "***")
        model = _resolve_model(agent.get("model", ""))

        # Use shared session pool for connection reuse on long runs
        session = await self._get_session()

        try:
            req_body = {
                "model": model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2000,
            }
            if openai_tools:
                req_body["tools"] = openai_tools
                req_body["tool_choice"] = "auto"

            llm_resp = await self._llm_chat_completion(session, proxy_url, api_key, req_body)
            content = llm_resp["content"]
            tool_calls = llm_resp["tool_calls"]

            has_valid_tool_calls = (
                bool(tool_calls) and all(_tc_args_nonempty(tc) for tc in tool_calls)
            )

            if has_valid_tool_calls:
                result = await execute_tool_calls(
                    tool_calls, messages, content, agent,
                    proxy_url, api_key, model, session, agent_id, openai_tools,
                )
                await self._append_conversation(
                    agent_id, {"role": "assistant", "content": result["content"]}
                )
                if log and self.provisioner:
                    used_tools = json.dumps([tc["function"]["name"] for tc in tool_calls])
                    self.provisioner.log_chat(agent_id, "assistant", result["content"], used_tools)
                return result

            # Fallback: JSON-in-text tool extraction
            extracted_tool_calls = _extract_tool_calls(content)
            if extracted_tool_calls:
                fake_tcs = []
                for i, call in enumerate(extracted_tool_calls):
                    args = call.get("args", {})
                    fake_tcs.append({
                        "id": f"text_{i}",
                        "function": {"name": call["tool"], "arguments": json.dumps(args)},
                    })
                result = await execute_tool_calls(
                    fake_tcs, messages, content, agent,
                    proxy_url, api_key, model, session, agent_id, openai_tools,
                )
                await self._append_conversation(
                    agent_id, {"role": "assistant", "content": result["content"]}
                )
                if log and self.provisioner:
                    self.provisioner.log_chat(
                        agent_id, "assistant", result["content"],
                        json.dumps([c["tool"] for c in extracted_tool_calls])
                    )
                return result

            await self._append_conversation(agent_id, {"role": "assistant", "content": content})
            if log and self.provisioner:
                self.provisioner.log_chat(agent_id, "assistant", content)
            return {
                "role": "assistant",
                "content": content,
                "agent_id": agent_id,
                "agent_name": agent.get("name", ""),
            }

        except Exception as e:
            logger.error(f"Agent chat error: {e}")
            safe_msg = "I encountered an error processing that request. See server logs."
            if log and self.provisioner:
                self.provisioner.log_chat(agent_id, "assistant", safe_msg)
            return {
                "role": "assistant",
                "content": safe_msg,
                "agent_id": agent_id,
                "agent_name": agent.get("name", ""),
            }

    # --- helpers --------------------------------------------------------
    def _build_system_prompt(self, agent: Dict, tool_desc: List[str], openai_tools: List[Dict]) -> str:
        if openai_tools:
            tools_guidance = (
                "You have access to tools. Use them when needed to fulfill requests.\n"
                "IMPORTANT: When you need to use a tool, output a JSON block in this EXACT format:\n"
                "```json\n{\"tool\": \"tool_name\", \"args\": {\"param1\": \"value1\"}}\n```\n"
                "Do NOT use native function calling. Use the JSON block format above."
            )
        else:
            tools_guidance = ""
        prompt = f"""{agent.get('system_prompt', '')}

## Available Tools
{chr(10).join(tool_desc)}

{tools_guidance}"""
        return prompt.rstrip() + "\n"

    async def _llm_chat_completion(self, session, proxy_url, api_key, body, *, timeout=60):
        """Post to the OpenAI-compatible LLM proxy and return the assistant message + raw response.

        Validates HTTP status, parses JSON safely, and surfaces a clear error instead
        of crashing on missing keys when the proxy returns 4xx/5xx or non-JSON.

        Returns: {"content": str, "tool_calls": list|None, "raw": dict}
        Raises: RuntimeError with a short, loggable message (no secrets) on failure.
        """
        try:
            async with session.post(
                f"{proxy_url}/v1/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    msg = text[:300]
                    try:
                        err = json.loads(text).get("error", {})
                        if isinstance(err, dict):
                            msg = err.get("message") or str(err)
                    except (json.JSONDecodeError, ValueError, AttributeError):
                        pass
                    raise RuntimeError(f"LLM proxy returned {resp.status}: {msg}")
                try:
                    data = json.loads(text)
                except (json.JSONDecodeError, ValueError) as e:
                    raise RuntimeError(f"LLM proxy returned non-JSON: {e}")
                if "error" in data:
                    err = data["error"]
                    msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                    raise RuntimeError(f"LLM proxy error: {msg}")
                choice = (data.get("choices") or [{}])[0]
                msg_obj = choice.get("message", {}) if isinstance(choice, dict) else {}
                return {
                    "content": msg_obj.get("content", "") or "",
                    "tool_calls": msg_obj.get("tool_calls", None),
                    "raw": data,
                }
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"LLM proxy call failed: {e}")


# Global instance (import in server.py / api_server.py and inject provisioner at startup)
chat_service = ChatService(None)  # type: ignore[arg-type]
