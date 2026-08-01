"""
title: Sacred Knowledge Graph
author: Vikarma
version: 1.0.0
description: |
  Query the Rudra Bhairava Sacred Knowledge Graph embedded in the piata-ai
  network: nodes (system/concept/agent), relations, kanban board, agents and
  pontaj (work log). Backed by sacred_api_server.py at api.piata-ai.ro/sacred/.
"""

import httpx
from pydantic import BaseModel, Field

SACRED_BASE = "http://127.0.0.1:9003"
TIMEOUT = 10.0


def _get(path: str, params: dict | None = None):
    try:
        r = httpx.get(f"{SACRED_BASE}{path}", params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


class QueryGraphIn(BaseModel):
    query: str = Field(..., description="Search term to match against node labels, ids, types and metadata.")
    limit: int = Field(10, description="Max number of nodes to return.")


def query_knowledge_graph(query: str, limit: int = 10) -> dict:
    """Search the sacred knowledge graph nodes by label/type/metadata keyword."""
    data = _get("/api/nodes")
    if "error" in data:
        return data
    q = query.lower()
    nodes = []
    for n in data.get("nodes", []):
        hay = " ".join(str(v).lower() for v in [
            n.get("id"), n.get("label"), n.get("type"),
            *(n.get("metadata") or {}).values(),
        ])
        if q in hay:
            nodes.append(n)
        if len(nodes) >= limit:
            break
    return {"query": query, "matches": len(nodes), "nodes": nodes}


class KanbanBoardIn(BaseModel):
    pass


def list_kanban_board() -> dict:
    """Return the sacred kanban board grouped by column."""
    return _get("/api/kanban/board")


def list_agents() -> dict:
    """Return registered agents in the sacred system."""
    return _get("/api/agents")


class PontajIn(BaseModel):
    limit: int = Field(20, description="Max pontaj log entries to return.")


def list_pontaj(limit: int = 20) -> dict:
    """Return recent pontaj (work log) entries."""
    data = _get("/api/pontaj")
    if isinstance(data, dict) and "error" in data:
        return data
    logs = data if isinstance(data, list) else data.get("logs", data.get("entries", []))
    return {"total": len(logs), "logs": logs[:limit]}


def sacred_health() -> dict:
    """Return health status of the sacred knowledge graph service."""
    return _get("/health")
