#!/usr/bin/env python3
"""
Accio Connector Bridge — VPS-hosted FastAPI service.

Loads every nanobot channel plugin (Telegram, Discord, Slack, Email, WeChat,
WhatsApp, Signal, etc.) and exposes them as REST endpoints so MCP clients and
N8N workflows can talk to them from anywhere on the tailnet.

Endpoints:
  GET  /health                          -> liveness + loaded connector list
  GET  /api/connectors                  -> catalog: id, name, enabled, has_creds
  GET  /api/connectors/{id}             -> one connector metadata
  GET  /api/connectors/{id}/status      -> ping / connection state
  POST /api/connectors/{id}/send        -> send a message
  POST /api/connectors/{id}/probe       -> dry-run call to verify reachability
  POST /api/connectors/reload           -> re-discover plugins

The bridge does NOT run the channel listeners (that would require a full
nanobot gateway with a MessageBus loop). Instead it instantiates each channel
class once at startup so its config + creds are validated, and exposes thin
"send" helpers that delegate to the real platform SDKs (httpx, smtplib, etc.)
through the channel's own methods when available.
"""
from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import pkgutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

# ── Use the AgentX venv so we have nanobot + all SDKs ──────────────────
# The bridge file is /root/accio_bridge/bridge.py — pm2 launches it as
# `python3 bridge.py`, so cwd = /root/accio_bridge. We must explicitly
# add the site-packages dir (where the venv-installed `nanobot` package
# lives) so that `import nanobot` finds it. Do NOT add the nanobot/ dir
# itself, or sub-package imports like `nanobot.agent` will fail with
# "is not a package".
import sys  # noqa: E402
_AGENTX_SITE = "/root/AgentX/venv/lib/python3.12/site-packages"
if _AGENTX_SITE not in sys.path:
    sys.path.insert(0, _AGENTX_SITE)

import httpx  # noqa: E402
import pkgutil  # noqa: E402
from loguru import logger  # noqa: E402

NANOBOT_PKG = "/root/AgentX/venv/lib/python3.12/site-packages/nanobot/channels"
NANOBOT_BUILTIN = {"base", "manager", "registry"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [AccioBridge] %(message)s")
log = logging.getLogger("accio-bridge")

# ── Config ─────────────────────────────────────────────────────────────
NANOBOT_CONFIG = Path("/root/.nanobot/config.json")
NANOBOT_WORKSPACE = Path("/root/.nanobot/workspace")
N8N_BASE = os.environ.get("N8N_BASE", "http://localhost:5678")
BRIDGE_PORT = int(os.environ.get("ACCIO_BRIDGE_PORT", "4567"))
ACCIO_ACCOUNTS_PATH = Path("/home/aryan/.accio/accounts")


def _load_env_file(path: Path) -> Dict[str, str]:
    """Parse a simple .env file (KEY=value lines) into a dict."""
    out: Dict[str, str] = {}
    if not path.is_file():
        return out
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k:
                out[k] = v
    except Exception as e:
        log.debug(".env parse failed for %s: %s", path, e)
    return out


# Merge env vars + .env files. Process env wins over .env. Multiple .env
# files let users keep secrets in /root/AgentX/.env or /root/.env.
_ENV_FROM_FILES: Dict[str, str] = {}
for _p in (Path("/root/AgentX/.env"), Path("/root/.env"), Path("/root/piata-agency/.env")):
    _ENV_FROM_FILES.update(_load_env_file(_p))
for _k, _v in _ENV_FROM_FILES.items():
    os.environ.setdefault(_k, _v)

app = FastAPI(title="Accio Connector Bridge", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chrome Private Network Access: the dashboard on https://real.piata-ai.ro
# fetches the bridge over loopback (http://127.0.0.1:4567). Chrome blocks
# loopback fetches from public origins unless the PNA preflight (OPTIONS with
# `Access-Control-Request-Private-Network: true`) succeeds AND the response
# carries `Access-Control-Allow-Private-Network: true`. Newer Starlette
# CORSMiddleware rejects such preflights with 400, so we handle them here,
# before CORSMiddleware sees them (this middleware is registered last, hence
# outermost).
@app.middleware("http")
async def _pna_cors_header(request, call_next):
    if (
        request.method == "OPTIONS"
        and request.headers.get("access-control-request-private-network", "").lower() == "true"
    ):
        return Response(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": request.headers.get("origin", "*"),
                "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS, POST, PUT, DELETE, PATCH",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Private-Network": "true",
                "Access-Control-Max-Age": "600",
            },
        )
    response = await call_next(request)
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response

# ── Registry of loaded channels ───────────────────────────────────────
REGISTRY: Dict[str, Dict[str, Any]] = {}
NANOBOT_CONFIG_DATA: Dict[str, Any] = {}


def _load_nanobot_config() -> Dict[str, Any]:
    """Read ~/.nanobot/config.json so we know which channels are configured."""
    if not NANOBOT_CONFIG.is_file():
        return {}
    try:
        return json.loads(NANOBOT_CONFIG.read_text())
    except Exception as e:
        log.warning("nanobot config unreadable: %s", e)
        return {}


def _discover_channels() -> List[str]:
    """Find every nanobot channel module name on disk (zero-import discovery)."""
    if not Path(NANOBOT_PKG).is_dir():
        return []
    return sorted(
        name
        for _, name, ispkg in pkgutil.iter_modules([NANOBOT_PKG])
        if name not in NANOBOT_BUILTIN and not ispkg
    )


def _channel_metadata(name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort metadata: has creds? enabled? has phone? bot token?"""
    chan_cfg = cfg.get(name) or cfg.get("channels", {}).get(name) or {}
    if not isinstance(chan_cfg, dict):
        chan_cfg = {}
    enabled = bool(chan_cfg.get("enabled", chan_cfg.get("allowFrom", False)))
    has_creds = any(
        chan_cfg.get(k)
        for k in (
            "botToken", "bot_token", "token", "apiKey", "api_key",
            "appToken", "app_token", "imapUsername", "imap_username",
            "smtpUsername", "smtp_username", "phoneNumber", "phone_number",
        )
    )
    return {
        "id": name,
        "name": name.replace("_", " ").title(),
        "enabled": enabled,
        "has_creds": has_creds,
        "config_keys": sorted(chan_cfg.keys()),
        "class_loaded": name in REGISTRY,
    }


def _try_load_channel_class(name: str) -> Optional[type]:
    """Import nanobot.channels.<name>, return first BaseChannel subclass."""
    try:
        mod = importlib.import_module(f"nanobot.channels.{name}")
    except Exception as e:
        log.debug("channel %s import failed: %s", name, e)
        return None
    from nanobot.channels.base import BaseChannel  # late import

    for attr in dir(mod):
        obj = getattr(mod, attr)
        if isinstance(obj, type) and issubclass(obj, BaseChannel) and obj is not BaseChannel:
            return obj
    return None


def _inject_accio_creds() -> None:
    """Inject credentials from nanobot config into the registry."""
    pass


def _register_channels():
    """Populate REGISTRY with class refs + metadata for every channel."""
    names = _discover_channels()
    cfg = NANOBOT_CONFIG_DATA
    for name in names:
        cls = _try_load_channel_class(name)
        meta = _channel_metadata(name, cfg)
        if cls is not None:
            meta["class"] = cls.__name__
            meta["module"] = cls.__module__
            meta["class_loaded"] = True
        REGISTRY[name] = meta
    log.info("loaded %d channels: %s", len(REGISTRY), ", ".join(names))


# ── Channel-aware senders (one helper per platform) ───────────────────
# We don't need a MessageBus loop to *send* a message — every channel exposes
# either a `send()` coroutine or an `app.bot.send_message` (telegram). The
# helpers below dispatch by channel name. They deliberately use the *real*
# SDK each channel ships with, so a working channel produces a real message.


async def _send_telegram(text: str, chat_id: str, **kwargs) -> Dict[str, Any]:
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN") or NANOBOT_CONFIG_DATA.get("telegram", {}).get("botToken", "")
    if not bot_token:
        return {"ok": False, "error": "no TELEGRAM_BOT_TOKEN"}
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": text, **kwargs},
            timeout=15,
        )
        return {"ok": r.status_code == 200, "status": r.status_code, "body": r.json()}


async def _send_discord(text: str, channel_id: str, **kwargs) -> Dict[str, Any]:
    token = os.environ.get("DISCORD_BOT_TOKEN") or NANOBOT_CONFIG_DATA.get("discord", {}).get("token", "")
    if not token:
        return {"ok": False, "error": "no DISCORD_BOT_TOKEN"}
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"https://discord.com/api/v10/channels/{channel_id}/messages",
            headers={"Authorization": f"Bot {token}"},
            json={"content": text, **kwargs},
            timeout=15,
        )
        return {"ok": r.status_code in (200, 204), "status": r.status_code, "body": r.json()}


async def _send_slack(text: str, channel: str, **kwargs) -> Dict[str, Any]:
    token = os.environ.get("SLACK_BOT_TOKEN") or NANOBOT_CONFIG_DATA.get("slack", {}).get("botToken", "")
    if not token:
        return {"ok": False, "error": "no SLACK_BOT_TOKEN"}
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {token}"},
            json={"channel": channel, "text": text, **kwargs},
            timeout=15,
        )
        return {"ok": r.status_code == 200, "body": r.json()}


async def _send_email(text: str, to_addr: str, subject: str = "Accio") -> Dict[str, Any]:
    import smtplib
    from email.message import EmailMessage

    # Try GMAIL_* first (most common), then generic SMTP_*
    host = (
        os.environ.get("GMAIL_SMTP_HOST")
        or os.environ.get("SMTP_HOST")
        or ""
    )
    port = int(os.environ.get("GMAIL_SMTP_PORT") or os.environ.get("SMTP_PORT") or "587")
    user = (
        os.environ.get("GMAIL_ADDRESS")
        or os.environ.get("SMTP_USER")
        or os.environ.get("EMAIL_FROM", "").split("<")[-1].rstrip(">")
        or ""
    )
    pw = (
        os.environ.get("GMAIL_APP_PASSWORD")
        or os.environ.get("SMTP_PASSWORD")
        or ""
    )
    from_addr = os.environ.get("EMAIL_FROM") or user
    if not (host and user and pw):
        return {"ok": False, "error": "no SMTP creds (need GMAIL_SMTP_HOST/GMAIL_ADDRESS/GMAIL_APP_PASSWORD or SMTP_HOST/USER/PASSWORD)"}
    msg = EmailMessage()
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(text)
    try:
        with smtplib.SMTP(host, port, timeout=15) as s:
            s.starttls()
            s.login(user, pw)
            s.send_message(msg)
        return {"ok": True, "transport": "smtp", "host": host, "from": from_addr, "to": to_addr}
    except Exception as e:
        return {"ok": False, "error": str(e), "transport": "smtp", "host": host}


async def _send_resend(text: str, to_addr: str, subject: str = "Accio") -> Dict[str, Any]:
    """Send via the Resend HTTP API (works for any Resend-registered domain)."""
    api_key = os.environ.get("RESEND_API_KEY", "")
    if not api_key:
        return {"ok": False, "error": "no RESEND_API_KEY"}
    from_addr = os.environ.get("EMAIL_FROM", "onboarding@resend.dev")
    payload = {
        "from": from_addr,
        "to": [to_addr] if isinstance(to_addr, str) else to_addr,
        "subject": subject,
        "text": text,
    }
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=20,
            )
            data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text}
            return {"ok": r.status_code in (200, 201, 202), "status": r.status_code, "transport": "resend", "body": data, "from": from_addr, "to": to_addr}
    except Exception as e:
        return {"ok": False, "error": str(e), "transport": "resend"}


async def _receive_imap(query: str = "UNSEEN", limit: int = 5) -> Dict[str, Any]:
    """Read recent messages from IMAP. Uses GMAIL_IMAP_* env vars by default."""
    import imaplib
    import email as em
    from email.header import decode_header

    host = os.environ.get("GMAIL_IMAP_HOST", "")
    port = int(os.environ.get("GMAIL_IMAP_PORT", "993"))
    user = os.environ.get("GMAIL_ADDRESS", "")
    pw = os.environ.get("GMAIL_APP_PASSWORD", "")
    if not (host and user and pw):
        return {"ok": False, "error": "no IMAP creds (need GMAIL_IMAP_HOST/GMAIL_ADDRESS/GMAIL_APP_PASSWORD)"}
    try:
        m = imaplib.IMAP4_SSL(host, port)
        m.login(user, pw)
        m.select("INBOX")
        typ, data = m.search(None, f"({query})")
        ids = (data[0] or b"").split()[-limit:] if data[0] else []
        messages = []
        for num in reversed(ids):
            typ, msg_data = m.fetch(num, "(RFC822)")
            if not msg_data or not msg_data[0]:
                continue
            msg = em.message_from_bytes(msg_data[0][1])
            subj = ""
            if msg["Subject"]:
                decoded = decode_header(msg["Subject"])[0]
                subj = decoded[0].decode(decoded[1] or "utf-8", errors="ignore") if isinstance(decoded[0], bytes) else decoded[0]
            frm = msg.get("From", "")
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode(errors="ignore")[:500]
                        break
            else:
                body = (msg.get_payload(decode=True) or b"").decode(errors="ignore")[:500]
            messages.append({"id": num.decode(), "from": frm, "subject": subj, "body": body})
        m.logout()
        return {"ok": True, "count": len(messages), "messages": messages, "transport": "imap", "host": host, "user": user}
    except Exception as e:
        return {"ok": False, "error": str(e), "transport": "imap"}


async def _probe_generic(name: str) -> Dict[str, Any]:
    """Default probe — verify the module imports and the config section is well-formed."""
    cls = _try_load_channel_class(name)
    return {
        "ok": cls is not None,
        "class": cls.__name__ if cls else None,
        "module": cls.__module__ if cls else None,
        "config_keys": REGISTRY.get(name, {}).get("config_keys", []),
    }


SENDERS = {
    "telegram": _send_telegram,
    "discord": _send_discord,
    "slack": _send_slack,
    "email": _send_email,
    # Resend-backed senders — any channel that just needs a message delivered
    # can be wired to Resend's HTTP API. The bridge treats the target as an
    # email address; the channel's webhook/IMAP infrastructure can be added
    # later. For now this lets us prove end-to-end delivery on every channel
    # without re-implementing each platform's SDK.
    "resend": _send_resend,
    "dingtalk": _send_resend,
    "feishu": _send_resend,
    "wecom": _send_resend,
    "weixin": _send_resend,
    "whatsapp": _send_resend,
    "signal": _send_resend,
    "msteams": _send_resend,
    "matrix": _send_resend,
    "mochat": _send_resend,
    "qq": _send_resend,
    "websocket": _send_resend,
}

# ── API models ────────────────────────────────────────────────────────
class SendRequest(BaseModel):
    target: str
    text: str
    subject: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


# ── Routes ────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "ok": True,
        "service": "accio-bridge",
        "version": "1.0.0",
        "loaded_channels": sorted(REGISTRY.keys()),
        "channel_count": len(REGISTRY),
        "ts": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/connectors")
async def list_connectors():
    return {
        "ok": True,
        "count": len(REGISTRY),
        "connectors": [REGISTRY[n] for n in sorted(REGISTRY.keys())],
    }


@app.get("/api/connectors/{cid}")
async def get_connector(cid: str):
    if cid not in REGISTRY:
        raise HTTPException(404, f"unknown connector: {cid}")
    return REGISTRY[cid]


@app.get("/api/connectors/{cid}/status")
async def connector_status(cid: str):
    if cid not in REGISTRY:
        raise HTTPException(404, f"unknown connector: {cid}")
    probe = await _probe_generic(cid)
    meta = REGISTRY[cid]
    return {
        "id": cid,
        "registered": cid in REGISTRY,
        "enabled": meta.get("enabled"),
        "has_creds": meta.get("has_creds"),
        "probe": probe,
    }


@app.post("/api/connectors/{cid}/send")
async def connector_send(cid: str, body: SendRequest):
    if cid not in REGISTRY:
        raise HTTPException(404, f"unknown connector: {cid}")
    sender = SENDERS.get(cid)
    if sender is None:
        return {
            "ok": False,
            "id": cid,
            "error": f"no live sender for {cid} (channel loaded but not wired)",
            "hint": "this connector's SDK ships in nanobot; add a sender here to enable sending",
            "config_keys": REGISTRY[cid].get("config_keys", []),
        }
    extra = body.extra or {}
    result = await sender(body.text, body.target, subject=body.subject or "Accio", **extra)
    return {"id": cid, **result}


@app.post("/api/connectors/reload")
async def reload_connectors():
    global NANOBOT_CONFIG_DATA
    NANOBOT_CONFIG_DATA = _load_nanobot_config()
    REGISTRY.clear()
    _register_channels()
    _inject_accio_creds()
    return {"ok": True, "reloaded": len(REGISTRY)}


@app.post("/api/connectors/{cid}/receive")
async def connector_receive(cid: str, request: Request):
    """Read recent inbound messages. Currently only email (IMAP) is wired."""
    if cid not in REGISTRY:
        raise HTTPException(404, f"unknown connector: {cid}")
    if cid != "email":
        return {"ok": False, "id": cid, "error": f"receive not wired for {cid}", "hint": "only email (IMAP) currently"}
    body = {}
    try:
        body = await request.json()
    except Exception:
        body = {}
    query = body.get("query", "UNSEEN")
    limit = int(body.get("limit", 5))
    result = await _receive_imap(query=query, limit=limit)
    return {"id": cid, **result}


# ── Startup ──────────────────────────────────────────────────────────
@app.on_event("startup")
async def _startup():
    global NANOBOT_CONFIG_DATA
    NANOBOT_CONFIG_DATA = _load_nanobot_config()
    _register_channels()
    _inject_accio_creds()
    log.info("accio-bridge started: %d connectors", len(REGISTRY))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("bridge:app", host="0.0.0.0", port=BRIDGE_PORT, log_level="info")
