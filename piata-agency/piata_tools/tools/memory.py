"""
Memory tool — persistent agent memory with search.

Usage:
    from piata_tools import memory
    await memory.store("user_name", "Alex", namespace="conversations")
    val = await memory.retrieve("user_name", namespace="conversations")
    results = await memory.search("trading strategy")
"""

import json
import os
import sqlite3
import time
from ..registry import get_registry

DB_PATH = os.environ.get("PIATA_MEMORY_DB", "/tmp/piata_sandbox/agent_memory.db")


def _get_db():
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id TEXT NOT NULL,
            namespace TEXT DEFAULT 'default',
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            metadata TEXT DEFAULT '{}',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            UNIQUE(agent_id, namespace, key)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_mem_search
        ON memories(agent_id, namespace, key, value)
    """)
    conn.commit()
    return conn


# Per-request agent_id, set by the runtime
_current_agent_id = os.environ.get("PIATA_AGENT_ID", "default")


def set_agent_id(agent_id: str):
    global _current_agent_id
    _current_agent_id = agent_id


async def store(key: str = None, value: str = None, namespace: str = "default",
                metadata: dict = None, memory_key: str = None,
                memory_value: str = None) -> dict:
    """Store a memory entry.

    Accepts `memory_key`/`memory_value` aliases because the LLM frequently
    names the parameters that way even though the schema says key/value.
    """
    key = key or memory_key
    value = value if value is not None else memory_value
    if not key or value is None:
        return {"error": "missing required params (key and value)"}
    # Ensure value is string (LLM sometimes sends dicts)
    if isinstance(value, dict):
        value = json.dumps(value, ensure_ascii=False)
    else:
        value = str(value)
    conn = _get_db()
    now = time.time()
    try:
        conn.execute("""
            INSERT INTO memories (agent_id, namespace, key, value, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(agent_id, namespace, key)
            DO UPDATE SET value=?, metadata=?, updated_at=?
        """, (_current_agent_id, namespace, key, value,
              json.dumps(metadata or {}), now, now,
              value, json.dumps(metadata or {}), now))
        conn.commit()
        return {"stored": True, "key": key, "namespace": namespace}
    finally:
        conn.close()


async def retrieve(key: str = None, namespace: str = "default",
                   memory_key: str = None) -> dict:
    """Retrieve a memory entry by key."""
    key = key or memory_key
    if not key:
        return {"error": "missing required param (key)"}
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT value, metadata, created_at, updated_at FROM memories WHERE agent_id=? AND namespace=? AND key=?",
            (_current_agent_id, namespace, key)
        ).fetchone()
        if row:
            return {
                "key": key, "value": row[0],
                "metadata": json.loads(row[1]),
                "created_at": row[2], "updated_at": row[3],
            }
        return {"key": key, "found": False}
    finally:
        conn.close()


async def search(query: str, namespace: str = None, limit: int = 10) -> dict:
    """Search memories by key/value content."""
    conn = _get_db()
    try:
        if namespace:
            rows = conn.execute(
                "SELECT key, value, namespace FROM memories WHERE agent_id=? AND namespace=? AND (key LIKE ? OR value LIKE ?) LIMIT ?",
                (_current_agent_id, namespace, f"%{query}%", f"%{query}%", limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT key, value, namespace FROM memories WHERE agent_id=? AND (key LIKE ? OR value LIKE ?) LIMIT ?",
                (_current_agent_id, f"%{query}%", f"%{query}%", limit)
            ).fetchall()
        return {
            "query": query,
            "results": [{"key": r[0], "value": r[1], "namespace": r[2]} for r in rows],
        }
    finally:
        conn.close()


async def list_memories(namespace: str = "default", limit: int = 50) -> dict:
    """List all memories in a namespace."""
    conn = _get_db()
    try:
        rows = conn.execute(
            "SELECT key, value, updated_at FROM memories WHERE agent_id=? AND namespace=? ORDER BY updated_at DESC LIMIT ?",
            (_current_agent_id, namespace, limit)
        ).fetchall()
        return {
            "namespace": namespace,
            "memories": [{"key": r[0], "value": r[1][:200], "updated_at": r[2]} for r in rows],
        }
    finally:
        conn.close()


async def forget(key: str, namespace: str = "default") -> dict:
    """Delete a memory entry."""
    conn = _get_db()
    try:
        conn.execute(
            "DELETE FROM memories WHERE agent_id=? AND namespace=? AND key=?",
            (_current_agent_id, namespace, key)
        )
        conn.commit()
        return {"forgotten": True, "key": key}
    finally:
        conn.close()


# Register
_registry = get_registry()
for name, desc, handler, params in [
    ("memory_store", "Store a persistent memory entry", store,
     {"type": "object", "properties": {"key": {"type": "string"}, "value": {"type": "string"}, "namespace": {"type": "string", "default": "default"}}, "required": ["key", "value"]}),
    ("memory_retrieve", "Retrieve a memory entry by key", retrieve,
     {"type": "object", "properties": {"key": {"type": "string"}, "namespace": {"type": "string", "default": "default"}}, "required": ["key"]}),
    ("memory_search", "Search memories by content", search,
     {"type": "object", "properties": {"query": {"type": "string"}, "namespace": {"type": "string"}, "limit": {"type": "integer", "default": 10}}, "required": ["query"]}),
    ("memory_list", "List all memories in a namespace", list_memories,
     {"type": "object", "properties": {"namespace": {"type": "string", "default": "default"}, "limit": {"type": "integer", "default": 50}}}),
    ("memory_forget", "Delete a memory entry", forget,
     {"type": "object", "properties": {"key": {"type": "string"}, "namespace": {"type": "string", "default": "default"}}, "required": ["key"]}),
]:
    _registry.register(
        name=name, description=desc, module="piata_tools.tools.memory",
        parameters=params, handler=handler, category="memory", emoji="🧠",
    )
