#!/usr/bin/env python3
"""Install the Sacred Knowledge Graph tool into Open WebUI's webui.db."""
import json
import os
import sqlite3
import time

DB = "/root/.open-webui/data/webui.db"
SRC_FILE = "/root/sacred_graph_tool.py"

with open(SRC_FILE, "r") as f:
    TOOL_SOURCE = f.read()

SPECS = [
    {
        "name": "query_knowledge_graph",
        "description": "Search the sacred knowledge graph nodes by label/type/metadata keyword.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search term to match against node labels, ids, types and metadata."},
                "limit": {"type": "integer", "default": 10, "description": "Max number of nodes to return."}
            },
            "required": ["query"]
        }
    },
    {
        "name": "list_kanban_board",
        "description": "Return the sacred kanban board grouped by column.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "list_agents",
        "description": "Return registered agents in the sacred system.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "list_pontaj",
        "description": "Return recent pontaj (work log) entries.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 20, "description": "Max pontaj log entries to return."}
            }
        }
    },
    {
        "name": "sacred_health",
        "description": "Return health status of the sacred knowledge graph service.",
        "parameters": {"type": "object", "properties": {}}
    },
]

META = {
    "description": "Rudra Bhairava Sacred Knowledge Graph — nodes, kanban, agents, pontaj via api.piata-ai.ro/sacred/.",
    "manifest": {
        "title": "Sacred Knowledge Graph",
        "author": "Vikarma",
        "version": "1.0.0",
        "license": "MIT",
        "required_open_webui_version": "0.7.0",
        "description": "Query the sacred knowledge graph embedded in the piata-ai network."
    }
}


def install():
    if not os.path.exists(DB):
        print(f"❌ OpenWebUI database not found: {DB}")
        return False

    conn = sqlite3.connect(DB)
    now = int(time.time() * 1000)
    specs_json = json.dumps(SPECS, ensure_ascii=False)
    meta_json = json.dumps(META, ensure_ascii=False)

    conn.execute(
        """INSERT INTO tool (id, user_id, name, content, specs, meta, updated_at, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
               content = excluded.content,
               specs = excluded.specs,
               meta = excluded.meta,
               updated_at = excluded.updated_at""",
        (
            "sacred_graph_tool",
            None,
            "Sacred Knowledge Graph",
            TOOL_SOURCE,
            specs_json,
            meta_json,
            now,
            now,
        ),
    )
    conn.commit()

    row = conn.execute(
        "SELECT id, name, length(content) as len FROM tool WHERE id = 'sacred_graph_tool'"
    ).fetchone()
    conn.close()

    print(f"✅ Installed/Updated: {row[1]} (id={row[0]}, {row[2]} chars)")
    print(f"  {len(SPECS)} functions registered:")
    for s in SPECS:
        print(f"  - {s['name']}()")
    return True


if __name__ == "__main__":
    install()
