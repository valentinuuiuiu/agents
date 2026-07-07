#!/usr/bin/env python3
"""
Piata-AI Admin Panel — YOLO build
- Direct PostgreSQL access (nexus_storage)
- Auth via simple X-Admin-Key header
- Endpoints: /api/admin/login, /api/admin/stats, /api/admin/ads, /api/admin/users, /api/admin/chat
"""
import os
import json
import secrets
import hashlib
import hmac
from datetime import datetime, timezone
from pathlib import Path

import asyncpg
import httpx
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# ── Config ───────────────────────────────────────────────────────
ADMIN_USER = os.environ.get("ADMIN_USER", "ionut")
ADMIN_PASS = os.environ.get("ADMIN_PASS")
if not ADMIN_PASS:
    raise RuntimeError("ADMIN_PASS environment variable must be set in production (never hardcoded)")
PG_CONN = os.environ.get("PG_CONN", "postgresql://postgres:postgres@localhost:5432/nexus_storage")
LLM_PROXY = os.environ.get("LLM_PROXY", "http://127.0.0.1:8080")
ADMIN_DIR = Path("/var/www/frontends/nexus-lab/public/admin")

app = FastAPI(title="Piata-AI Admin")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── DB ───────────────────────────────────────────────────────────
db_pool: asyncpg.Pool | None = None

@app.on_event("startup")
async def startup():
    global db_pool
    db_pool = await asyncpg.create_pool(PG_CONN, min_size=1, max_size=5, command_timeout=30)

@app.on_event("shutdown")
async def shutdown():
    if db_pool:
        await db_pool.close()

def token_for(user: str) -> str:
    raw = f"{user}:{ADMIN_PASS}"
    return hashlib.sha256(raw.encode()).hexdigest()[:48]

def verify_token(token: str) -> bool:
    expected = token_for(ADMIN_USER)
    return hmac.compare_digest(token, expected)

# ── Models ───────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    user: str
    password: str

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []

# ── Auth ─────────────────────────────────────────────────────────
@app.post("/api/admin/login")
async def login(req: LoginRequest):
    if req.user == ADMIN_USER and req.password == ADMIN_PASS:
        return {"ok": True, "token": token_for(req.user), "user": req.user}
    raise HTTPException(401, "Invalid credentials")

# ── Stats ────────────────────────────────────────────────────────
@app.get("/api/admin/stats")
async def stats(x_admin_key: str = Header(...)):
    if not verify_token(x_admin_key):
        raise HTTPException(401, "Bad token")
    async with db_pool.acquire() as c:
        ads = await c.fetchval("SELECT count(*) FROM ads")
        ads_published = await c.fetchval("SELECT count(*) FROM ads WHERE status='published'")
        ads_pending = await c.fetchval("SELECT count(*) FROM ads WHERE status='pending'")
        clients = await c.fetchval("SELECT count(*) FROM clients")
        leads = await c.fetchval("SELECT count(*) FROM leads")
        leads_pending = await c.fetchval("SELECT count(*) FROM leads WHERE status='pending'")
        # recent ads
        recent = await c.fetch("""
            SELECT id, title, price, category, status, contact_phone, created_at,
                   COALESCE(array_length(image_urls, 1), 0) AS num_imgs
            FROM ads ORDER BY created_at DESC LIMIT 10
        """)
    return {
        "ok": True,
        "ads": ads, "ads_published": ads_published, "ads_pending": ads_pending,
        "clients": clients, "leads": leads, "leads_pending": leads_pending,
        "recent_ads": [dict(r) for r in recent]
    }

# ── List ads (paginated) ─────────────────────────────────────────
@app.get("/api/admin/ads")
async def list_ads(
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
    x_admin_key: str = Header(...)
):
    if not verify_token(x_admin_key):
        raise HTTPException(401, "Bad token")
    sql_data = """
        SELECT id, title, description, price, category, sub_category,
               location, phone, image_urls, status, contact_name, contact_phone,
               source_url, source_platform, created_at, confirmed_at,
               COALESCE(array_length(image_urls, 1), 0) AS num_imgs
        FROM ads
        WHERE ($3::text IS NULL OR status = $3::text)
        ORDER BY created_at DESC
        LIMIT $1::int OFFSET $2::int
    """
    sql_count = """
        SELECT count(*) FROM ads
        WHERE ($1::text IS NULL OR status = $1::text)
    """
    async with db_pool.acquire() as c:
        rows = await c.fetch(sql_data, int(limit), int(offset), status or None)
        if status:
            total = await c.fetchval(sql_count, str(status))
        else:
            total = await c.fetchval("SELECT count(*) FROM ads")
    return {
        "ok": True, "total": total, "limit": limit, "offset": offset,
        "ads": [dict(r) for r in rows]
    }

# ── Approve / Reject ad ─────────────────────────────────────────
class AdAction(BaseModel):
    action: str  # "approve" | "reject" | "delete"

@app.post("/api/admin/ads/{ad_id}/action")
async def ad_action(ad_id: str, body: AdAction, x_admin_key: str = Header(...)):
    if not verify_token(x_admin_key):
        raise HTTPException(401, "Bad token")
    if body.action == "approve":
        sql = "UPDATE ads SET status='published', confirmed_at=NOW() WHERE id=$1 RETURNING id, status"
    elif body.action == "reject":
        sql = "UPDATE ads SET status='archived' WHERE id=$1 RETURNING id, status"
    elif body.action == "delete":
        sql = "DELETE FROM ads WHERE id=$1 RETURNING id"
    else:
        raise HTTPException(400, "Unknown action")
    async with db_pool.acquire() as c:
        row = await c.fetchrow(sql, ad_id)
    if not row:
        raise HTTPException(404, "Ad not found")
    return {"ok": True, "ad": dict(row)}

# ── Clients & Leads ─────────────────────────────────────────────
@app.get("/api/admin/clients")
async def list_clients(x_admin_key: str = Header(...)):
    if not verify_token(x_admin_key):
        raise HTTPException(401, "Bad token")
    async with db_pool.acquire() as c:
        rows = await c.fetch("SELECT * FROM clients ORDER BY created_at DESC LIMIT 100")
    return {"ok": True, "clients": [dict(r) for r in rows]}

@app.get("/api/admin/leads")
async def list_leads(x_admin_key: str = Header(...)):
    if not verify_token(x_admin_key):
        raise HTTPException(401, "Bad token")
    async with db_pool.acquire() as c:
        rows = await c.fetch("SELECT * FROM leads ORDER BY created_at DESC LIMIT 100")
    return {"ok": True, "leads": [dict(r) for r in rows]}

# ── Chat with admin agent (queries DB) ──────────────────────────
SYSTEM_PROMPT = """You are the Piata-AI Admin Agent. You have read-only access to the marketplace database.
The database has these tables:
- ads: id, title, description, price, category, sub_category, location (jsonb), phone, image_urls (text[]), status (draft/pending/published/archived), contact_name, contact_phone, source_url, source_platform, created_at, confirmed_at
- clients: id, lead_id, nume, telefon, email, created_at
- leads: id, nume, telefon, email, link_olx, descriere, status (pending/approved/rejected/expired), token, sursa, confirmed_at, created_at, updated_at

Answer in Romanian, be concise, and help with:
- Finding ads by status, category, or city
- Identifying patterns in pending ads
- Suggesting moderation actions
- Explaining DB structure

If asked for SQL queries, format them in code blocks."""

@app.post("/api/admin/chat")
async def admin_chat(req: ChatRequest, x_admin_key: str = Header(...)):
    if not verify_token(x_admin_key):
        raise HTTPException(401, "Bad token")
    # Build messages
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in req.history[-10:]:
        msgs.append(h)
    msgs.append({"role": "user", "content": req.message})
    # Call LLM proxy (OpenAI-compatible)
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                f"{LLM_PROXY}/v1/chat/completions",
                json={"model": "gpt-4o-mini", "messages": msgs, "max_tokens": 800, "temperature": 0.4}
            )
            data = r.json()
            answer = data.get("choices", [{}])[0].get("message", {}).get("content", "Nu am primit răspuns de la LLM.")
    except Exception as e:
        answer = f"Eroare LLM proxy: {e}. Verifică conexiunea la {LLM_PROXY}."
    return {"ok": True, "answer": answer}

# ── Serve admin HTML ─────────────────────────────────────────────
if ADMIN_DIR.exists():
    @app.get("/", include_in_schema=False)
    @app.get("/admin", include_in_schema=False)
    @app.get("/admin/", include_in_schema=False)
    async def admin_index():
        return FileResponse(ADMIN_DIR / "index.html")

    @app.get("/admin/{path:path}", include_in_schema=False)
    async def admin_static(path: str):
        f = ADMIN_DIR / path
        if f.is_file():
            return FileResponse(f)
        return FileResponse(ADMIN_DIR / "index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 5011)))
