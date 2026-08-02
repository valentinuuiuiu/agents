"""MCP tool generator — the agency's Tool Registrar.

Spiritual successor to clean_rehoboam_project/mcp_creator.py (the original
was hand-built by Aryan/FannyMae in 2025). This version:

  * Uses an OpenAI-compatible LLM backend (no hardcoded provider keys). By
    default it points at Ollama's /v1/chat/completions endpoint (the same
    host the orchestrator uses for Thinker/Worker/Verifier), with a model
    fallback chain (minimax-m3:cloud → qwen2.5:3b) so a single 403 on a
    cloud-routed model doesn't stall tool creation. An explicit shared
    proxy can still be set via LLM_PROXY_URL="./v1/...".
  * Generates FastMCP-style tool modules (FastMCP 3.4+ is installed).
  * Sandboxes generated code with a restricted AST validator + restricted
    builtins at exec time.
  * Persists the registry to disk so generated tools survive restarts.
  * Hot-registers finished tools into the central `piata_tools.registry`
    so every agent can call them immediately as first-class tools.
  * Enforces a per-call asyncio timeout.

Role separation in PAI:
  * `TrinityOrchestrator` decides WHAT needs to be done and dispatches to
    agents. When it detects a tool gap, it REQUESTS a new tool from the
    Registrar — it does not write or validate code itself.
  * `MCPCreator` (this module) IS the Tool Registrar. Sole authority for
    the lifecycle of generated tools: create / validate / sandbox /
    register / list / delete. It owns who-can-call-what via the central
    `ToolRegistry`.
  * Agents call tools through the `ToolRegistry` only — they never see
    this module directly.

Only the TrinityOrchestrator (and admin tooling, server-side) may call
create_tool(); this is NOT exposed over HTTP to end users.
"""

from __future__ import annotations

import ast
import asyncio
import builtins as _builtins
import hashlib
import importlib.util
import json
import logging
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("piata.mcp_creator")
_REAL_IMPORT = _builtins.__import__

REGISTRY_DIR = Path(os.environ.get(
    "PIATA_GENERATED_TOOLS_DIR",
    "/root/piata-agency/generated_tools",
))
REGISTRY_FILE = REGISTRY_DIR / "registry.json"

# Dangerous calls that should never appear in agent-generated code.
_DANGEROUS_ATTRS = {
    "os.system", "os.popen", "os.exec", "os.execv", "os.execve",
    "os.spawn", "os.remove", "os.rmdir", "os.unlink",
    "shutil.rmtree", "shutil.move",
    "subprocess.run", "subprocess.Popen", "subprocess.call",
    "subprocess.check_output", "subprocess.check_call",
    "pty.spawn",
}
_DANGEROUS_BUILTINS = {
    "eval", "exec", "compile", "__import__", "open", "breakpoint",
    "exit", "quit",
}
# Bare import names that should be refused at the import-statement level.
_DANGEROUS_IMPORTS = {"subprocess", "pty", "ctypes", "multiprocessing"}


def _ollama_host() -> str:
    """Ollama base URL, overridable via OLLAMA_HOST (same env var the
    orchestrator already uses). Strips any trailing slash and any /v1 suffix
    so we can append `/v1/chat/completions` consistently below."""
    h = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    if h.endswith("/v1"):
        h = h[: -len("/v1")]
    return h


# Model fallback chain for code generation. Only the first reachable model
# is used per call. Order: cloud-routed (fast, matches orchestrator) → local
# (no cloud auth). The whole chain is overridable via LLM_PROXY_MODEL.
_DEFAULT_MODEL_CHAIN = [
    os.environ.get("NVIDIA_NIM_MODEL", ""),
    "minimax-m3:cloud",  # same model the orchestrator uses; cloud-routed, fast
    "qwen2.5:3b",         # local fallback, no cloud auth required
]


def _llm_proxy_env() -> tuple[str, str, list[str]]:
    """Resolve the agency LLM proxy connection from env (no hardcoded keys).

    Order of preference for the base URL:
      1. LLM_PROXY_URL (explicit shared proxy, e.g. http://127.0.0.1:5007)
      2. Ollama's OpenAI-compatible endpoint
         (OLLAMA_HOST=http://127.0.0.1:11434 → /v1/chat/completions)

    The agency's LLM proxy at 127.0.0.1:8080 is the admin dashboard, NOT
    a working OpenAI endpoint (see tests/test_orchestrator_e2e.py header),
    so we must not default to it. We default to Ollama instead, which the
    orchestrator already uses successfully for Thinker/Worker/Verifier.

    Returns (base_url, api_key, [model, fallback_model, ...]).
    """
    explicit = os.environ.get("LLM_PROXY_URL", "").strip().rstrip("/")
    if explicit:
        # Caller has a real shared proxy; honour it.
        base = explicit
        if not base.endswith("/v1"):
            base = base + "/v1"
    else:
        base = _ollama_host() + "/v1"
    api_key = os.environ.get("LLM_PROXY_KEY", os.environ.get("ADMIN_API_KEY", ""))

    # Model chain: honour LLM_PROXY_MODEL if set, otherwise fall through
    # the default chain (filtered to remove any empty entry).
    preset = os.environ.get("LLM_PROXY_MODEL", "").strip()
    if preset:
        models = [preset]
    else:
        models = [m for m in _DEFAULT_MODEL_CHAIN if m]
    return base, api_key, models


async def _call_llm(prompt: str, system: str, *, max_tokens: int = 1500,
                    temperature: float = 0.2, timeout: int = 90) -> str:
    """Call the agency's OpenAI-compatible LLM proxy to generate code.

    Tries each model in the configured fallback chain until one succeeds,
    so a single 403 / 404 / 502 on a cloud-routed model doesn't kill the
    whole tool-creation flow — we fall back to a local model instead.
    """
    import aiohttp
    base, api_key, models = _llm_proxy_env()
    url = f"{base}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    last_err: Optional[str] = None
    for model in models:
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=body, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    text = await resp.text()
                    if resp.status >= 400:
                        last_err = f"LLM proxy {model} {resp.status}: {text[:200]}"
                        logger.warning(f"mcp_creator: {last_err}; trying next model")
                        continue
                    data = json.loads(text)
                    if "error" in data:
                        last_err = f"LLM proxy {model} error: {data['error']}"
                        logger.warning(f"mcp_creator: {last_err}; trying next model")
                        continue
                    return data["choices"][0]["message"]["content"]
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_err = f"LLM proxy {model} network error: {e}"
            logger.warning(f"mcp_creator: {last_err}; trying next model")
            continue
    raise RuntimeError(f"All LLM models failed. Last error: {last_err}")


def _extract_code(llm_response: str) -> str:
    """Strip markdown fences from an LLM response."""
    if "```python" in llm_response:
        start = llm_response.find("```python") + len("```python")
        end = llm_response.find("```", start)
        return llm_response[start:end].strip()
    if "```" in llm_response:
        start = llm_response.find("```") + 3
        end = llm_response.find("```", start)
        return llm_response[start:end].strip()
    return llm_response.strip()


def _validate_code(code: str) -> tuple[bool, str]:
    """Static safety validator. Returns (ok, reason)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    for node in ast.walk(tree):
        # Refuse dangerous imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _DANGEROUS_IMPORTS:
                    return False, f"Import of '{alias.name}' is forbidden"
        if isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in _DANGEROUS_IMPORTS:
                return False, f"Import from '{node.module}' is forbidden"
        # Refuse dangerous attribute accesses (os.system, subprocess.run, ...)
        if isinstance(node, ast.Attribute):
            try:
                qual = ast.unparse(node)
            except Exception:
                qual = ""
            for danger in _DANGEROUS_ATTRS:
                if qual == danger or qual.startswith(danger + "."):
                    return False, f"Dangerous attribute: {qual}"
        # Refuse calls to dangerous builtins by name
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _DANGEROUS_BUILTINS:
                return False, f"Call to forbidden builtin: {func.id}"
    return True, "OK"


def _restricted_globals() -> dict:
    """Sandbox globals for executing generated tool code.
    Allows httpx + asyncio + stdlib pieces, but no eval/exec/open/__import__."""
    safe_builtins = {
        k: v for k, v in vars(_builtins).items()
        if k not in _DANGEROUS_BUILTINS
    }
    # Provide a guarded __import__ that blocks dangerous modules
    def _safe_import(name, *args, **kwargs):
        root = (name or "").split(".")[0]
        if root in _DANGEROUS_IMPORTS:
            raise ImportError(f"Import of {name} is forbidden in generated tools")
        return _REAL_IMPORT(name, *args, **kwargs)
    safe_builtins["__import__"] = _safe_import
    g: dict = {"__name__": "piata_generated_tool", "__builtins__": safe_builtins}
    # Pre-import safe utilities the LLM is allowed to use
    g.update({
        "asyncio": asyncio,
        "json": json,
        "hashlib": hashlib,
        "time": time,
        "os": _safe_os_module(),
        "Path": Path,
    })
    try:
        import httpx
        g["httpx"] = httpx
    except ImportError:
        pass
    return g


def _safe_os_module():
    """Return a copy of `os` with the dangerous surface stripped."""
    import os as _real_os
    class _SafeOs: pass
    safe = _SafeOs()
    # Whitelist the common, non-destructive attrs
    for attr in ("environ", "getenv", "path", "sep", "linesep", "name",
                 "cpu_count", "urandom", "cwd", "listdir", "walk",
                 "scandir", "stat", "exists", "isdir", "isfile",
                 "getsize", "makedirs", "mkdir", "curdir"):
        try:
            setattr(safe, attr, getattr(_real_os, attr))
        except Exception:
            pass
    return safe


class MCPCreator:
    """The agency's Tool Registrar.

    Sole authority for the lifecycle of generated tools. The
    `TrinityOrchestrator` requests tools from this class when it detects a
    capability gap; it never writes or validates tool code directly.
    """

    def __init__(self, registry=None):
        self.registry = registry
        REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        self._disk_registry: Dict[str, dict] = self._load_registry()

    # ── Persistence ──────────────────────────────────────────────
    def _load_registry(self) -> Dict[str, dict]:
        if REGISTRY_FILE.exists():
            try:
                return json.loads(REGISTRY_FILE.read_text())
            except json.JSONDecodeError:
                logger.warning("mcp_creator: registry.json corrupted, starting fresh")
        return {}

    def _save_registry(self):
        REGISTRY_FILE.write_text(json.dumps(self._disk_registry, indent=2))

    # ── Public API ───────────────────────────────────────────────
    async def create_tool(self, name: str, description: str,
                          parameters: Optional[Dict[str, str]] = None,
                          code: Optional[str] = None,
                          *, category: str = "generated",
                          timeout_seconds: int = 30) -> dict:
        """Generate (or register an explicit) tool and hot-register it.

        If `code` is None, an LLM is prompted to write the function body.
        """
        name = (name or "").strip().lower().replace(" ", "_")
        if not name or not name.replace("_", "").isalnum():
            return {"error": "Tool name must be alphanumeric/underscore"}

        if code is None:
            params = parameters or {}
            params_str = ", ".join(f"{n}: {t}" for n, t in params.items()) or "*args, **kwargs"
            prompt = (
                f"Generate a single Python async function named `{name}` that does the following:\n\n"
                f"{description}\n\n"
                f"Signature: `async def {name}({params_str}) -> dict:`\n\n"
                "Return a dict with at least a 'status' key ('ok'|'error') and a 'result' key.\n"
                "You may import inside the function. Use only stdlib + httpx for network.\n"
                "Output ONLY the function, no markdown fences, no prose."
            )
            system = ("You are a Python code generator for an agent tool factory. "
                      "Output ONLY valid Python code for a single async function. "
                      "No markdown fences, no explanations, no leading text.")
            try:
                raw = await _call_llm(prompt, system)
            except RuntimeError as e:
                return {"error": f"LLM call failed: {e}"}
            code = _extract_code(raw)

        valid, reason = _validate_code(code)
        if not valid:
            return {"error": f"Validation failed: {reason}", "raw_code": code}

        # Save source file + update disk registry
        source_path = REGISTRY_DIR / f"{name}.py"
        func_id = hashlib.sha256(f"{name}:{time.time()}".encode()).hexdigest()[:12]
        source_path.write_text(
            f"# Auto-generated by piata-agency MCPCreator\n"
            f"# func_id: {func_id}\n"
            f"# description: {description}\n"
            f"# created_at: {time.time()}\n\n"
            f"{code}\n"
        )
        meta = {
            "func_id": func_id,
            "name": name,
            "description": description,
            "params": parameters or {},
            "file": str(source_path),
            "category": category,
            "created_at": time.time(),
            "executions": 0,
        }
        self._disk_registry[func_id] = meta
        self._save_registry()

        # Hot-register into the central ToolRegistry. Do not leave a tool
        # advertised as created when compilation/registration failed: remove
        # the persisted source and metadata so the next attempt is clean.
        try:
            registered = await self._hot_register(meta, code, timeout_seconds)
        except Exception as e:
            logger.exception("mcp_creator: hot-registration raised for %s", name)
            registered = False
        if not registered:
            self._disk_registry.pop(func_id, None)
            try:
                source_path.unlink(missing_ok=True)
            except Exception:
                logger.warning("mcp_creator: failed to remove invalid tool source %s", source_path)
            self._save_registry()
            logger.warning("mcp_creator: tool %s rejected during hot-register", name)
            return {"error": "Tool failed hot-registration", "name": name,
                    "registered": False, "status": "rejected"}
        return {"func_id": func_id, "name": name, "file": str(source_path),
                "registered": True, "code": code, "status": "created"}

    async def _hot_register(self, meta: dict, code: str, timeout_seconds: int) -> bool:
        """Compile the generated code into a callable and register it."""
        try:
            ns = _restricted_globals()
            exec(compile(code, meta["file"], "exec"), ns)  # noqa: S102 — intentional, sandboxed
            handler = ns.get(meta["name"])
        except Exception as e:
            logger.error(f"mcp_creator: failed to compile {meta['name']}: {e}")
            return False
        if not callable(handler):
            logger.error(f"mcp_creator: {meta['name']} not callable after exec")
            return False

        # Wrap with a per-call timeout
        async def _safe_handler(**kwargs):
            try:
                return await asyncio.wait_for(
                    handler(**kwargs),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                return {"status": "error", "error": "tool execution timed out"}
        _safe_handler.__name__ = meta["name"]

        if self.registry is not None:
            # Convert {"param": "string"} -> {"type": "object", "properties": ...}
            params_obj: Dict[str, Any] = {"type": "object", "properties": {}}
            for p, t in (meta["params"] or {}).items():
                params_obj["properties"][p] = _map_param_type(t)
            previous = self.registry.tools.get(meta["name"])
            try:
                self.registry.register(
                    name=meta["name"],
                    description=meta["description"],
                    module="generated",
                    parameters=params_obj,
                    handler=_safe_handler,
                    is_async=True,
                    category=meta.get("category", "generated"),
                    emoji="🧬",
                    keywords=["generated", "mcp"],
                )
            except Exception:
                # Restore the previous definition (if any) so a failed
                # registration cannot leave a half-updated central registry.
                if previous is None:
                    self.registry.tools.pop(meta["name"], None)
                else:
                    self.registry.tools[meta["name"]] = previous
                raise
        return True

    def list_tools(self) -> List[dict]:
        return list(self._disk_registry.values())

    def delete_tool(self, name_or_id: str) -> dict:
        target = None
        for fid, meta in self._disk_registry.items():
            if meta["name"] == name_or_id or fid == name_or_id:
                target = fid
                break
        if not target:
            return {"error": f"Tool '{name_or_id}' not found"}
        meta = self._disk_registry.pop(target)
        # Remove source file
        try:
            Path(meta["file"]).unlink(missing_ok=True)
        except Exception:
            pass
        # Unregister from central registry
        if self.registry is not None:
            self.registry.tools.pop(meta["name"], None)
        self._save_registry()
        return {"deleted": meta["name"], "func_id": target}

    async def call_tool(self, name_or_id: str, arguments: dict) -> dict:
        """Invoke a generated tool directly (used by the orchestrator for trial)."""
        meta = None
        for fid, m in self._disk_registry.items():
            if m["name"] == name_or_id or fid == name_or_id:
                meta = m
                break
        if meta is None or self.registry is None or meta["name"] not in self.registry.tools:
            return {"error": f"Tool '{name_or_id}' not registered"}
        result = await asyncio.wait_for(
            self.registry.call(meta["name"], **arguments),
            timeout=60,
        )
        meta["executions"] = meta.get("executions", 0) + 1
        self._save_registry()
        return result


def _map_param_type(t: str) -> dict:
    """Map a loose type string to a JSON-schema property."""
    t = (t or "").lower().strip()
    if t in {"int", "integer", "long"}:
        return {"type": "integer"}
    if t in {"float", "double", "number", "decimal"}:
        return {"type": "number"}
    if t in {"bool", "boolean"}:
        return {"type": "boolean"}
    if t in {"list", "array"}:
        return {"type": "array", "items": {}}
    if t in {"dict", "object", "json"}:
        return {"type": "object"}
    # Default to string (covers "str", "string", unknown)
    return {"type": "string"}
