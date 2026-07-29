#!/usr/bin/env python3
"""
PraISON Trinity Orchestrator
=============================
Bazat pe arhitectura originală Trinity de Mervin Praison (PraisonAI).
Dynamic role dispatch: Thinker → Worker(s) → Verifier → Merge.

Nu hardcodezi echipe. Orchestratorul decide dinamic:
  - Ce roluri sunt necesare pentru task-ul ăsta
  - Ce agenți din pool sunt potriviți
  - Cum comunică între ei

Flow:
  1. Thinker: primește task-ul, îl descompune, decide roluri + agenți
  2. Worker(s): execută sarcinile în paralel
  3. Verifier: verifică output-ul, cere corecții dacă e cazul
  4. Merge: combină rezultatele într-un răspuns final

Usage:
  from engine.orchestrator_praison import TrinityOrchestrator
  orch = TrinityOrchestrator()
  result = await orch.run(task="analizează piața BTC", context={...})
"""

import asyncio
import json
import logging
import os
import time
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("praison-trinity")

# Default Ollama host — overridable via env so the Tailscale IP rotation
# (or running Ollama on localhost) doesn't require a code change.
_DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://100.75.36.5:11434")

# ── Pool de agenți disponibili ──
# Toți prin Ollama local pe VPS; host suprascriibil prin OLLAMA_HOST env var.
AGENT_POOL = {
    "minimax-m3": {
        "model": "minimax-m3:cloud",
        "strengths": ["multimodal", "reasoning", "coding", "analysis", "creative", "general"],
        "speed": "fast",
        "provider": "ollama",
        "ollama_host": _DEFAULT_OLLAMA_HOST,
    },
}

# ── Mapping task-type → roluri preferate ──
TASK_TYPE_MAP = {
    "coding":     {"roles": ["thinker", "coder", "verifier"],     "primary": "minimax-m3"},
    "analysis":   {"roles": ["thinker", "researcher", "verifier"], "primary": "minimax-m3"},
    "creative":   {"roles": ["thinker", "writer", "editor"],       "primary": "minimax-m3"},
    "math":       {"roles": ["thinker", "solver", "verifier"],     "primary": "minimax-m3"},
    "trading":    {"roles": ["thinker", "analyst", "verifier"],    "primary": "minimax-m3"},
    "data":       {"roles": ["thinker", "analyst", "verifier"],    "primary": "minimax-m3"},
    "general":    {"roles": ["thinker", "worker", "verifier"],     "primary": "minimax-m3"},
}


class TrinityOrchestrator:
    """Dynamic role dispatcher inspirat din Trinity (Mervin Praison)."""

    MEMORY_FILE = Path("/root/.hermes/trinity_memory.json")

    def __init__(self, registry=None):
        self.memory = self._load()
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        # The Tool Registrar — sole authority for generated tool lifecycle.
        # Lazy-imported here so the orchestrator stays the only caller of
        # MCPCreator.create_tool(); nothing else in the agency may register
        # generated tools.
        try:
            from engine.mcp_creator import MCPCreator
            self.registrar = MCPCreator(registry=registry)
        except Exception as e:
            logger.warning(f"TrinityOrchestrator: registrar unavailable: {e}")
            self.registrar = None

    async def request_tool(self, name: str, description: str,
                           parameters: Optional[dict] = None,
                           code: Optional[str] = None) -> dict:
        """Ask the Tool Registrar to create + hot-register a new tool.

        Called by `run()` when a capability gap is detected. This is the
        ONLY sanctioned entrypoint for generated tool creation in PAI.
        """
        if not self.registrar:
            return {"error": "Tool Registrar not initialized"}
        return await self.registrar.create_tool(
            name=name, description=description,
            parameters=parameters, code=code,
        )

    # ── Memorie Cross-Session ─────────────────────────────────────

    def _load(self) -> dict:
        if self.MEMORY_FILE.exists():
            try:
                data = json.loads(self.MEMORY_FILE.read_text())
                # Migrare: patterns era listă, acum e dict
                if isinstance(data.get("patterns"), list):
                    data["patterns"] = {}
                if "patterns" not in data:
                    data["patterns"] = {}
                if "session_count" not in data:
                    data["session_count"] = 0
                return data
            except (json.JSONDecodeError, OSError):
                pass
        return {
            "created": datetime.now(timezone.utc).isoformat(),
            "patterns": {},
            "session_count": 0,
        }

    def _save(self):
        self.MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.MEMORY_FILE.write_text(json.dumps(self.memory, indent=2, ensure_ascii=False))

    def _learn(self, task_type: str, success: bool, roles: list, agents: list, duration: float):
        """Învață din fiecare execuție."""
        pattern = self.memory["patterns"].setdefault(task_type, {
            "count": 0, "successes": 0, "failures": 0,
            "favorite_roles": {}, "favorite_agents": {},
            "avg_duration": 0.0, "success_rate": 1.0,
        })
        pattern["count"] += 1
        if success:
            pattern["successes"] += 1
        else:
            pattern["failures"] += 1
        pattern["success_rate"] = pattern["successes"] / max(pattern["count"], 1)
        pattern["avg_duration"] = (pattern["avg_duration"] * (pattern["count"] - 1) + duration) / pattern["count"]
        for r in roles:
            pattern["favorite_roles"][r] = pattern["favorite_roles"].get(r, 0) + 1
        for a in agents:
            pattern["favorite_agents"][a] = pattern["favorite_agents"].get(a, 0) + 1
        self.memory["session_count"] += 1
        self._save()

    # ── Ollama Call ───────────────────────────────────────────────

    def _ollama_chat(self, model: str, messages: list, host: str = _DEFAULT_OLLAMA_HOST,
                     temperature: float = 0.3, max_tokens: int = 500) -> str:
        """Trimite un mesaj la Ollama și returnează conținutul răspunsului."""
        payload = json.dumps({
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }).encode()

        import urllib.request
        req = urllib.request.Request(
            f"{host}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        try:
            resp = urllib.request.urlopen(req, timeout=60)
            data = json.loads(resp.read())
            return data.get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            raise

    # ── Task Type Detection ───────────────────────────────────────

    def _detect_task_type(self, task: str) -> str:
        """Detectează tipul de task pe baza cuvintelor cheie."""
        task_lower = task.lower()
        score = {t: 0 for t in TASK_TYPE_MAP}

        # Keyword scoring
        keywords = {
            "coding": ["implementează", "scrie cod", "api", "function", "class", "endpoint",
                       "refactor", "debug", "bug", "repository", "commit", "pull request",
                       "express", "react", "node", "python", "javascript", "typescript",
                       "docker", "deploy", "test", "unit test"],
            "analysis": ["analizează", "research", "cercetează", "studiu", "investighează",
                         "compară", "evaluează", "raport", "dashboard", "insights",
                         "revizuiește", "audit"],
            "creative": ["scrie", "creează", "design", "conținut", "articol", "post",
                         "email", "social media", "story", "poveste", "brand",
                         "logo", "visual", "imagine", "video"],
            "math": ["calculează", "matematică", "formulă", "ecuație", "statistică",
                     "probabilitate", "optimizare", "algoritm"],
            "trading": ["trading", "btc", "bitcoin", "crypto", "acțiuni", "stock",
                        "investiție", "portofoliu", "piață", "market", "semnal",
                        "buy", "sell", "long", "short"],
            "data": ["date", "dataset", "csv", "json", "sql", "query", "postgresql",
                     "mongodb", "analiză date", "chart", "grafic", "vizualizare",
                     "pipeline", "etl"],
        }

        for ttype, words in keywords.items():
            for w in words:
                if w in task_lower:
                    score[ttype] += 1

        # Winner
        best = max(score, key=score.get)
        if score[best] == 0:
            return "general"
        return best

    # ── Agent Selection ───────────────────────────────────────────

    def _select_agents(self, task_type: str) -> dict:
        """Alege rolurile și agenții pe baza task-type și a memoriei."""
        config = TASK_TYPE_MAP.get(task_type, TASK_TYPE_MAP["general"])
        roles = list(config["roles"])

        # Toți agenții sunt minimax-m3 (singurul model disponibil local)
        agents = ["minimax-m3"] * len(roles)

        return {
            "roles": roles,
            "agents": agents,
            "task_type": task_type,
            "config": config,
        }

    # ── Execuție ──────────────────────────────────────────────────

    async def _maybe_create_missing_tool(self, thinker_result: str,
                                         task_type: str, task: str) -> dict:
        """Inspect the Thinker's plan; if it names a capability the registry
        doesn't have, ask the Registrar to create it.

        The Thinker writes free-form Romanian/English plans like
        "I need a tool that fetches X from Y" — we extract a tool name +
        short description via an LLM call and forward to MCPCreator.

        Returns: {"created": [...], "skipped": [...]} or {"error": ...}
        """
        if not self.registrar:
            return {"error": "registrar not initialized"}

        # Ask the LLM to extract a structured list of missing-tool specs from
        # the Thinker's plan. Conservative: zero or one tool per turn to
        # avoid runaway tool-creation.
        prompt = (
            "From the following agent plan, identify AT MOST ONE missing tool "
            "that would be needed but is NOT already available. If no new tool "
            "is genuinely needed, return an empty list. Respond ONLY with JSON:\n"
            '{"tools": [{"name": "snake_case_name", "description": "one line", '
            '"parameters": {"p": "string"}}]}\n\n'
            f"Task type: {task_type}\n"
            f"Task: {task}\n"
            f"Thinker plan:\n{thinker_result[:1500]}\n"
        )
        try:
            raw = await self._ollama_chat_async(
                model=AGENT_POOL["minimax-m3"]["model"],
                messages=[{"role": "user", "content": prompt}],
                host=_DEFAULT_OLLAMA_HOST,
                temperature=0.0,
                max_tokens=400,
            )
            import re
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if not m:
                return {"created": [], "skipped": [], "reason": "no json in plan"}
            data = json.loads(m.group(0))
        except Exception as e:
            logger.warning(f"_maybe_create_missing_tool: LLM parse failed: {e}")
            return {"error": f"plan parse failed: {e}"}

        # What tools does the central registry already have? (cheap check)
        try:
            from piata_tools.registry import get_registry
            known = {t["name"] for t in get_registry().list_all()}
        except Exception:
            known = set()

        created, skipped = [], []
        for spec in (data.get("tools") or [])[:1]:  # hard cap 1
            name = (spec.get("name") or "").strip().lower()
            if not name or name in known:
                skipped.append(name or "<empty>")
                continue
            res = await self.registrar.create_tool(
                name=name,
                description=spec.get("description", name),
                parameters=spec.get("parameters") or {},
            )
            if res.get("status") == "created":
                created.append(name)
                # Remember in trinity memory so we don't ask again
                self._remember_tool_creation(name, task_type)
            else:
                skipped.append(f"{name}:{res.get('error','?')[:60]}")
        return {"created": created, "skipped": skipped}

    def _remember_tool_creation(self, name: str, task_type: str):
        """Track created tools in trinity memory so we learn from the gap."""
        try:
            created = self.memory.setdefault("created_tools", {})
            created[name] = {
                "task_type": task_type,
                "created_at": time.time(),
            }
            self._save()
        except Exception:
            pass

    async def _ollama_chat_async(self, model: str, messages: list,
                                  host: str = _DEFAULT_OLLAMA_HOST,
                                  temperature: float = 0.2,
                                  max_tokens: int = 400) -> str:
        """Async wrapper around the existing `_ollama_chat` (which is sync
        because it uses urllib). Runs the blocking call in a thread."""
        import urllib.request, urllib.error
        body = json.dumps({
            "model": model, "messages": messages,
            "stream": False, "options": {
                "temperature": temperature, "num_predict": max_tokens,
            },
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{host}/api/chat", data=body,
            headers={"Content-Type": "application/json"},
        )
        loop = asyncio.get_event_loop()

        def _do():
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        resp = await loop.run_in_executor(None, _do)
        return resp.get("message", {}).get("content", "")

    async def run(self, task: str, context: Optional[dict] = None) -> dict:
        """Rulează orchestratorul pe un task."""
        start = time.time()
        context = context or {}

        # 1. Detectează tipul
        task_type = self._detect_task_type(task)

        # 2. Alege roluri + agenți
        selection = self._select_agents(task_type)

        # 3. Thinker (dacă nu e test mode)
        thinker_result = None
        if not context.get("mode") == "test":
            try:
                agent = selection["agents"][0]
                agent_cfg = AGENT_POOL[agent]

                response = self._ollama_chat(
                    model=agent_cfg["model"],
                    messages=[
                        {"role": "system", "content": (
                            f"Tu ești role: THINKER. "
                            f"Task type: {task_type}. "
                            f"Task: {task}\n"
                            f"Planifică execuția. Ce trebuie făcut pas cu pas?"
                        )},
                        {"role": "user", "content": task}
                    ],
                    host=agent_cfg.get("ollama_host", _DEFAULT_OLLAMA_HOST),
                    temperature=0.3,
                    max_tokens=800,
                )
                thinker_result = response
            except Exception as e:
                logger.warning(f"Thinker call failed: {e}")

        # 3.5 Tool-Gap detection — if the Thinker's plan names a capability
        # we don't have a tool for, the orchestrator asks the Registrar to
        # create one. This is the ONLY sanctioned path for tool creation.
        tool_gap_result = None
        if thinker_result and self.registrar is not None and not context.get("mode") == "test":
            try:
                tool_gap_result = await self._maybe_create_missing_tool(
                    thinker_result, task_type, task
                )
            except Exception as e:
                logger.warning(f"Tool-gap detection failed: {e}")

        # 4. Worker
        worker_result = None
        if not context.get("mode") == "test":
            try:
                agent = selection["agents"][1] if len(selection["agents"]) > 1 else selection["agents"][0]
                agent_cfg = AGENT_POOL[agent]

                response = self._ollama_chat(
                    model=agent_cfg["model"],
                    messages=[
                        {"role": "system", "content": (
                            f"Tu ești role: WORKER. "
                            f"Task: {task}\n"
                            f"{'Thinker plan: ' + thinker_result[:500] if thinker_result else ''}"
                        )},
                        {"role": "user", "content": task}
                    ],
                    host=agent_cfg.get("ollama_host", _DEFAULT_OLLAMA_HOST),
                    temperature=0.4,
                    max_tokens=1000,
                )
                worker_result = response
            except Exception as e:
                logger.warning(f"Worker call failed: {e}")

        # 5. Verifier
        verifier_result = None
        if not context.get("mode") == "test":
            try:
                agent = selection["agents"][2] if len(selection["agents"]) > 1 else selection["agents"][0]
                agent_cfg = AGENT_POOL[agent]

                response = self._ollama_chat(
                    model=agent_cfg["model"],
                    messages=[
                        {"role": "system", "content": (
                            f"Tu ești role: VERIFIER. "
                            f"Verifică și îmbunătățește rezultatul Worker-ului.\n"
                            f"Worker output: {worker_result[:500] if worker_result else 'N/A'}"
                        )},
                        {"role": "user", "content": f"Task: {task}\nWorker: {worker_result[:500] if worker_result else 'N/A'}"}
                    ],
                    host=agent_cfg.get("ollama_host", _DEFAULT_OLLAMA_HOST),
                    temperature=0.2,
                    max_tokens=500,
                )
                verifier_result = response
            except Exception as e:
                logger.warning(f"Verifier call failed: {e}")

        duration = time.time() - start

        # 6. Learn
        success = worker_result is not None or context.get("mode") == "test"
        self._learn(task_type, success, selection["roles"], selection["agents"], duration)

        return {
            "status": "success" if success else "failed",
            "task": task,
            "task_type": task_type,
            "meta": {
                "roles_used": selection["roles"],
                "agents_used": selection["agents"],
                "duration": duration,
            },
            "results": {
                "thinker": thinker_result,
                "worker": worker_result,
                "verifier": verifier_result,
                "tool_gap": tool_gap_result,
            },
        }

    # ── Sugestie bazată pe memorie ────────────────────────────────

    def suggest(self, task: str) -> dict:
        """Sugerează configurația optimă pentru un task pe baza memoriei."""
        task_type = self._detect_task_type(task)
        pattern = self.memory["patterns"].get(task_type)
        if not pattern:
            return {"recommendation": "new", "task_type": task_type}

        # Favorite roles and agents
        best_roles = sorted(pattern["favorite_roles"], key=pattern["favorite_roles"].get, reverse=True)[:3]
        best_agents = sorted(pattern["favorite_agents"], key=pattern["favorite_agents"].get, reverse=True)[:3]

        return {
            "recommendation": "memory",
            "task_type": task_type,
            "success_rate": pattern["success_rate"],
            "avg_duration": pattern["avg_duration"],
            "suggested_roles": best_roles,
            "suggested_agents": best_agents,
            "sessions_trained": pattern["count"],
        }


# ── CLI Entry Point ───────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    task = " ".join(sys.argv[1:]) or "analizează piața crypto"

    async def main():
        orch = TrinityOrchestrator()
        result = await orch.run(task)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))

    asyncio.run(main())
