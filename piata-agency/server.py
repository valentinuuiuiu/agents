"""
Piata Agency — Main Server

Endpoints:
  POST /api/agency/agents          — Provision a new agent (returns webhook URL)
  GET  /api/agency/agents          — List all agents
  GET  /api/agency/agents/{id}     — Get agent details
  POST /api/agency/agents/{id}/chat — Chat with agent (API key auth)
  POST /webhook/{id}/{token}       — Webhook endpoint (external integrations)
  GET  /chat/{id}                  — Chatbox UI (browser)
  GET  /api/agency/catalog         — Available tools and plans
  GET  /api/agency/health          — Health check
"""

import asyncio
import hashlib
import json
import logging
import os
import secrets
import sys
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import httpx
import jwt
from eth_account import Account
from eth_account.messages import encode_defunct
from fastapi import FastAPI, HTTPException, Request, Depends, Header, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import psycopg2
from psycopg2.extras import execute_values

# Firebase Auth verification
try:
    import firebase_admin
    from firebase_admin import auth as firebase_auth, credentials
    FIREBASE_ADMIN_AVAILABLE = True
except ImportError:
    FIREBASE_ADMIN_AVAILABLE = False

# Initialize Firebase Admin
_firebase_app = None
_firebase_api_key = os.environ.get("FIREBASE_API_KEY", "")

async def _verify_firebase_id_token_rest(id_token: str) -> dict:
    """Verify a Firebase ID token via the public identitytoolkit REST endpoint.

    Used as a fallback when no service-account credentials are available — no
    `firebase-admin` Admin SDK credentials (service-account JSON or ADC) are
    required, only the web API key (`FIREBASE_API_KEY`).
    """
    if not _firebase_api_key:
        raise RuntimeError("FIREBASE_API_KEY not configured")
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:lookup?key={_firebase_api_key}"
    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0)) as client:
        resp = await client.post(url, json={"idToken": id_token})
        if resp.status_code != 200:
            raise_http = {
                "status": resp.status_code,
                "detail": (resp.json().get("error", {}).get("message", "Invalid token")
                           if resp.headers.get("content-type", "").startswith("application/json")
                           else resp.text[:200]),
            }
            raise ValueError(json.dumps(raise_http))
        data = resp.json()
    users = data.get("users") or []
    if not users:
        raise ValueError("No user for token")
    return users[0]


def _init_firebase_admin():
    """Initialize the Firebase Admin SDK with a service account, if available.

    Returns None when no service-account credentials are configured — callers
    fall back to the REST-based token verifier (`_verify_firebase_id_token_rest`).
    """
    global _firebase_app
    if _firebase_app is not None:
        return _firebase_app
    if not FIREBASE_ADMIN_AVAILABLE:
        return None
    try:
        # Use service account if available, otherwise leave it uninitialized
        # so the REST verifier is used instead.
        service_account_path = os.environ.get("FIREBASE_SERVICE_ACCOUNT_PATH")
        if service_account_path and os.path.exists(service_account_path):
            cred = credentials.Certificate(service_account_path)
        else:
            logger.info(
                "Firebase Admin: no service-account credentials configured; "
                "using REST-based ID token verification."
            )
            return None
        _firebase_app = firebase_admin.initialize_app(cred, {
            'projectId': os.environ.get("FIREBASE_PROJECT_ID")
        })
        return _firebase_app
    except Exception as e:
        logger.warning(f"Firebase Admin init failed: {e}")
        return None

# NOTE: _init_firebase_admin() is called lazily from /api/auth/firebase so
# that `logger` is guaranteed to be defined (it is created further below).

from web3 import Web3

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))
# Allow importing qwen-bots helpers (news-spike dashboard, etc.)
sys.path.insert(0, "/root/qwen-bots")

from engine.provisioner import AgentProvisioner, TOOL_DESCRIPTIONS, TOOL_SCHEMAS, check_secret
import piata_tools  # Triggers tool registration
from news_spike_dashboard import _DASHBOARD_TEMPLATE as _NEWS_SPIKE_TEMPLATE, get_history as _news_spike_history, _fmt_ts as _news_fmt_ts
from piata_tools.registry import get_registry
from piata_tools.tools.mcp_client import call_tool as mcp_call_tool
from chat_service import ChatService
from billing import (
    check_plan_limits, increment_usage, get_usage_summary, PLANS as BILLING_PLANS,
    create_checkout_record, get_checkout_by_agent, update_checkout_status,
)
from payments import (
    create_checkout_session,
    verify_webhook_signature,
    handle_webhook_event as payments_handle_webhook_event,
    create_customer_portal_session,
)
try:
    from payments import cancel_subscription
except Exception:
    def cancel_subscription(*args, **kwargs):
        raise HTTPException(status_code=503, detail="Subscription cancellation unavailable")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("piata-agency")

# Agent runtime cache — stores loaded agent instances
_agent_instances: dict = {}

provisioner: AgentProvisioner = None
chat_service: ChatService = None

ADMIN_API_KEYS = {k.strip() for k in os.environ.get("ADMIN_API_KEYS", "").split(",") if k.strip()}
# 2026-08-02: removed the silent `secrets.token_urlsafe(32)` fallback —
# generating a new secret per process invalidates every active JWT/dashboard
# session on every deploy. JWT_SECRET must now be set explicitly via env so
# multiple replicas / restarts share one signing key.
JWT_SECRET = os.environ.get("JWT_SECRET")
if not JWT_SECRET:
    raise RuntimeError(
        "JWT_SECRET must be set in the environment. The auto-generated "
        "fallback (per-process random secret) was removed 2026-08-02 "
        "because it silently logged every active user out on each restart."
    )
JWT_ALGORITHM = "HS256"
# Nonce TTL (seconds) for the SIWE challenge. A non-zero TTL here is just
# hygiene — the canonical nonce expires via `pop()` on /verify — but it
# defends against the case where /verify never gets called and the dict
# accumulates garbage from abandoned login flows.
_METAMASK_NONCE_TTL_SECONDS = 10 * 60
_metamask_nonces: dict[str, tuple[str, float]] = {}  # address_lower -> (nonce, exp)
_metamask_nonces: dict[str, tuple[str, float]] = {}  # address_lower -> (nonce, exp)
TARS_URL = os.environ.get("TARS_URL", "http://" + "100.91.238.121:8889")

# ── Model alias map ──────────────────────────────────
# Old / friendly model names get remapped to the actual
# NVIDIA NIM model IDs before the chat-call is sent.
# Set NVIDIA_NIM_MODEL in env to override the default.
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
    # If it already contains a slash (e.g. "meta/llama-3.3-70b-instruct") assume it's valid.
    if "/" in model_name:
        return model_name
    # Unknown — fall back to default rather than sending a bad ID.
    return os.environ.get("NVIDIA_NIM_MODEL", "deepseek-ai/deepseek-v4-pro")


async def _llm_chat_completion(
    session: "aiohttp.ClientSession",
    proxy_url: str,
    api_key: str,
    body: dict,
    *,
    timeout: int = 60,
) -> dict:
    """Post to the OpenAI-compatible LLM proxy and return the assistant message
    + raw response.

    Validates HTTP status, parses JSON safely, and surfaces a clear error instead
    of crashing on `KeyError` when the proxy returns 4xx/5xx or non-JSON.

    Returns: {"content": str, "tool_calls": list|None, "raw": dict}
    Raises: RuntimeError with a short, loggable message (no secrets) on failure.
    """
    import aiohttp as _aiohttp
    try:
        async with session.post(
            f"{proxy_url}/v1/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=_aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            text = await resp.text()
            if resp.status >= 400:
                # Try to extract a clean error message from the proxy response
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

# Dashboard login credentials. Set DASHBOARD_EMAIL and DASHBOARD_PASSWORD env vars.
DASHBOARD_EMAIL = os.environ.get("DASHBOARD_EMAIL", "admin@piata-ai.ro")
DASHBOARD_PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "")
# Dashboard token TTL (seconds) — matches the lifetime of the API-key JWT we mint.
_DASHBOARD_TOKEN_TTL_SECONDS = 7 * 24 * 3600
# token -> expiry epoch (float). Bounded: expired entries are swept on every
# create/verify, so the dict cannot grow without limit.
_dashboard_tokens: dict[str, float] = {}


def _sweep_dashboard_tokens() -> None:
    """Drop expired dashboard tokens. Runs on every touch of the dict."""
    if not _dashboard_tokens:
        return
    now = time.time()
    expired = [t for t, exp in _dashboard_tokens.items() if exp <= now]
    for t in expired:
        _dashboard_tokens.pop(t, None)


def _create_dashboard_token() -> str:
    _sweep_dashboard_tokens()
    token = secrets.token_urlsafe(32)
    _dashboard_tokens[token] = time.time() + _DASHBOARD_TOKEN_TTL_SECONDS
    return token


def _verify_dashboard_token(token: Optional[str]) -> bool:
    if not token:
        return False
    exp = _dashboard_tokens.get(token)
    if exp is None:
        return False
    if exp <= time.time():
        # Expired — drop on the way through so the next call doesn't have to.
        _dashboard_tokens.pop(token, None)
        return False
    return True


def _mask_secret(value: str, prefix="pia_") -> str:
    if not value:
        return ""
    if len(value) <= 12:
        return "***"
    return f"{prefix}...{value[-6:]}"


def _public_agent(agent: dict) -> dict:
    public = dict(agent)
    token = public.pop("webhook_token", None)
    if token and "agent_id" in public:
        public["webhook_url"] = f"/webhook/{public['agent_id']}/{token}"
    api_key = public.pop("api_key", None)
    if api_key:
        # New agents store api_key as 'sha256$<salt>$<hex>' — show the
        # fingerprint so admins can verify they have the right key without
        # re-storing any recoverable secret.
        if api_key.startswith("sha256$"):
            try:
                public["api_key_masked"] = f"sha256…{api_key[-6:]}"
            except Exception:
                public["api_key_masked"] = "**hashed**"
        else:
            # Legacy plaintext — fall back to masking the prefix+last 6.
            public["api_key_masked"] = _mask_secret(api_key)
    return public


def _public_provision_result(result: dict) -> dict:
    public = dict(result)
    api_key = public.pop("api_key", None)
    if api_key:
        public["api_key"] = _mask_secret(api_key)
        public["api_key_masked"] = public["api_key"]
    return public


class MetaMaskNonceRequest(BaseModel):
    address: str


class MetaMaskVerifyRequest(BaseModel):
    address: str
    signature: str
    nonce: str


def _admin_required(authorization: Optional[str] = Header(default=None), x_api_key: Optional[str] = Header(default=None)) -> dict:
    if not ADMIN_API_KEYS:
        return {"admin": False}
    token = authorization[7:].strip() if authorization and authorization.lower().startswith("bearer ") else authorization
    if token in ADMIN_API_KEYS or x_api_key in ADMIN_API_KEYS:
        return {"admin": True}
    raise HTTPException(status_code=401, detail="Admin API key required")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global provisioner, chat_service
    provisioner = AgentProvisioner()
    # Initialize the tool registry
    registry = get_registry()
    mcp_config = os.path.join(os.path.dirname(__file__), "piata_tools", "mcp_servers.yaml")
    registry.load_mcp_config(mcp_config)
    # Initialize chat service
    chat_service = ChatService(provisioner, registry)
    logger.info(f"Piata Agency started — {len(registry.tools)} tools registered, "
                f"{len(registry.mcp_servers)} MCP servers")
    yield
    logger.info("Piata Agency shutting down")


app = FastAPI(
    title="Piata Agency",
    description="AI Agent-as-a-Service — provision custom agents with MCP tools",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — 2026-08-02: replaced the wildcard `["*"]` with an env-driven allow-
# list. Public storefront routes can stay permissive via the default set;
# private/dashboard endpoints are protected by the JWT/admin gate PLUS an
# origin filter so a leaked JWT cannot be replayed from an arbitrary domain.
_DEFAULT_CORS = ",".join([
    "https://piata-ai.ro",
    "https://real.piata-ai.ro",
    "https://app.piata-ai.ro",
    "https://admin.piata-ai.ro",
    "http://localhost:8080",
    "http://localhost:4200",
    "http://localhost:5173",
    "http://localhost:3000",
])
_cors_origins = [
    o.strip() for o in os.environ.get("PIATA_CORS_ORIGINS", _DEFAULT_CORS).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Request/Response Models ===

class ProvisionRequest(BaseModel):
    name: str = Field(..., description="Agent name")
    purpose: str = Field(..., description="What the agent does")
    tools: List[str] = Field(default=["web", "memory", "code"],
                            description="Tools to give the agent")
    model: str = Field(default="mimo-v2.5-pro", description="LLM model to use")
    system_prompt: Optional[str] = Field(default=None, description="Custom system prompt")
    owner_email: Optional[str] = Field(default=None)
    owner_name: Optional[str] = Field(default=None)
    mcp_servers: Optional[List[str]] = Field(default=None, description="MCP server names to connect")


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    history: Optional[List[dict]] = Field(default=None, description="Conversation history")


class WebhookRequest(BaseModel):
    message: Optional[str] = None
    text: Optional[str] = None
    body: Optional[str] = None
    sender: Optional[str] = None
    metadata: Optional[dict] = None


# === Auth ===

async def verify_api_key(authorization: Optional[str] = Header(None)) -> str:
    """Extract and verify API key from Authorization header."""
    if not authorization:
        raise HTTPException(401, "Missing Authorization header")
    key = authorization.replace("Bearer ", "").strip()
    if not key.startswith("pia_"):
        raise HTTPException(401, "Invalid API key format")
    return key


# AGENCY_BEARER_TOKEN: required for /api/agency/ingest and chat_api when caller is not 127.0.0.1
# ponytail: global bearer — local peers bypass for the admin dashboard
AGENCY_BEARER_TOKEN = os.environ.get("AGENCY_BEARER_TOKEN", "").strip()


async def get_current_user(request: Request, authorization: Optional[str] = Header(default=None), x_api_key: Optional[str] = Header(default=None), x_dashboard_token: Optional[str] = Header(default=None)) -> dict:
    """Auth gate for sensitive endpoints.
    Accepts (in priority order):
      1. Localhost peer (127.0.0.1) — admin dashboard
      2. AGENCY_BEARER_TOKEN via Authorization: Bearer <token>
      3. Admin API key (ADMIN_API_KEYS)
      4. Dashboard login token via X-Dashboard-Token header
    """
    client_host = request.client.host if request.client else ""
    if client_host in ("127.0.0.1", "::1", "localhost"):
        return {"auth": "local", "client": client_host}

    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if not token and x_api_key:
        token = x_api_key.strip()

    if AGENCY_BEARER_TOKEN and token and secrets.compare_digest(token, AGENCY_BEARER_TOKEN):
        return {"auth": "bearer", "client": client_host}
    if ADMIN_API_KEYS and token in ADMIN_API_KEYS:
        return {"auth": "admin", "client": client_host}
    if _verify_dashboard_token(x_dashboard_token):
        return {"auth": "dashboard", "client": client_host}

    raise HTTPException(status_code=401, detail="Authentication required")


# === Public Endpoints ===


@app.post("/api/agency/ingest")
async def ingest_knowledge(request: Request, _user: dict = Depends(get_current_user)):
    """
    Knowledge Siphon: Ingests data, generates embeddings via proxy, and stores in pgvector.
    Auth: AGENCY_BEARER_TOKEN, ADMIN_API_KEY, or 127.0.0.1 peer.
    """
    try:
        data = await request.json()
        title = data.get("title", "Unknown")
        text = data.get("text", "")
        metadata = data.get("metadata", {})
        source = metadata.get("source", "unknown")

        if not text:
            return JSONResponse(status_code=400, content={"error": "Missing text content"})

        async with httpx.AsyncClient(timeout=30) as client:
            emb_resp = await client.post(
                "http://localhost:5007/v1/embeddings",
                json={"model": "nvidia/embed-qa-4", "input": text}
            )
            if emb_resp.status_code != 200:
                return JSONResponse(status_code=502, content={"error": "Embedding proxy failure"})
            emb_data = emb_resp.json()
            if not emb_data.get("data") or len(emb_data["data"]) == 0:
                return JSONResponse(status_code=502, content={"error": "Embedding provider returned no data"})
            if not emb_data.get("data") or len(emb_data["data"]) == 0:
                return JSONResponse(status_code=502, content={"error": "Embedding provider returned no data"})
            vector = emb_data["data"][0]["embedding"]

        import psycopg2
        conn = psycopg2.connect("dbname=nexus_storage user=postgres")
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO ads (title, content, embedding, metadata) VALUES (%s, %s, %s, %s)",
                (title, text, vector, json.dumps(metadata))
            )
        conn.commit()
        conn.close()

        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                "http://localhost:5002/api/hive-mind/ingest",
                json={"title": title, "text": text, "source": source}
            )

        return {"status": "success", "agent": "Siphon", "vector_dim": len(vector)}
    except Exception as e:
        import logging
        logging.getLogger("piata-agency").error("Siphon error: %s", e)
        return JSONResponse(status_code=500, content={"error": str(e)})

class LoginRequest(BaseModel):
    email: str
    password: str


@app.post("/api/auth/login")
async def dashboard_login(req: LoginRequest):
    """Dashboard login. Returns a session token stored in browser localStorage."""
    if not DASHBOARD_PASSWORD:
        raise HTTPException(503, "Dashboard password not configured")
    if req.email.lower().strip() != DASHBOARD_EMAIL.lower().strip():
        raise HTTPException(401, "Invalid email or password")
    # Allow either plain password or bcrypt hash comparison if password starts with $
    if DASHBOARD_PASSWORD.startswith("$"):
        import bcrypt
        if not bcrypt.checkpw(req.password.encode(), DASHBOARD_PASSWORD.encode()):
            raise HTTPException(401, "Invalid email or password")
    else:
        if not secrets.compare_digest(req.password, DASHBOARD_PASSWORD):
            raise HTTPException(401, "Invalid email or password")
    return {"token": _create_dashboard_token()}


@app.get("/api/auth/me")
async def dashboard_me(x_dashboard_token: Optional[str] = Header(default=None)):
    """Verify dashboard session is still valid."""
    if _verify_dashboard_token(x_dashboard_token):
        return {"ok": True, "auth": "dashboard"}
    raise HTTPException(401, "Invalid or expired token")


@app.get("/api/agency/health")
async def health():
    registry = get_registry()
    return {
        "status": "ok",
        "tools": len(registry.tools),
        "mcp_servers": len(registry.mcp_servers),
        "agents": len(provisioner.list_agents()),
    }


@app.get("/api/agency/tools/health")
async def tools_health():
    """Health check for all registered tools and MCP servers."""
    registry = get_registry()
    tools_status = {}
    for name, tool in registry.tools.items():
        tools_status[name] = {
            "module": tool.module,
            "category": tool.category,
            "mcp_server": tool.mcp_server,
            "has_handler": tool.handler is not None,
            "requires_env": tool.requires_env,
            "env_ok": all(os.environ.get(env) for env in tool.requires_env),
        }
    
    mcp_status = {}
    for name, cfg in registry.mcp_servers.items():
        mcp_status[name] = {
            "transport": cfg.get("transport", "http"),
            "url": cfg.get("url", ""),
            "description": cfg.get("description", ""),
        }
    
    return {
        "status": "ok",
        "tools_count": len(tools_status),
        "tools": tools_status,
        "mcp_servers_count": len(mcp_status),
        "mcp_servers": mcp_status,
    }


@app.get("/api/auth/config")
async def auth_config():
    firebase_configured = bool(os.environ.get("FIREBASE_PROJECT_ID") and os.environ.get("FIREBASE_API_KEY"))
    google_oauth_configured = bool(
        (os.environ.get("GOOGLE_CLIENT_ID") and os.environ.get("GOOGLE_CLIENT_SECRET"))
        or firebase_configured  # Firebase Google Auth is configured
    )
    return {
        "admin_api_key_configured": bool(ADMIN_API_KEYS),
        "metamask": True,
        "google_oauth_configured": google_oauth_configured,
        "firebase_configured": firebase_configured,
        "stripe": True,
        "metamask_payment_ready": True,
    }


@app.post("/api/auth/metamask/nonce")
async def metamask_nonce(req: MetaMaskNonceRequest):
    address = Web3.to_checksum_address(req.address)
    now = time.time()
    # Sweep any expired nonces for this address before issuing a new one.
    key = address.lower()
    prev = _metamask_nonces.get(key)
    if prev and prev[1] <= now:
        _metamask_nonces.pop(key, None)
    nonce = secrets.token_urlsafe(18)
    _metamask_nonces[key] = (nonce, now + _METAMASK_NONCE_TTL_SECONDS)
    return {"address": address, "nonce": nonce, "message": f"Sign in to Piata-AI\nNonce: {nonce}"}


@app.post("/api/auth/metamask/verify")
async def metamask_verify(req: MetaMaskVerifyRequest):
    address = Web3.to_checksum_address(req.address).lower()
    entry = _metamask_nonces.pop(address, None)
    if not entry:
        raise HTTPException(status_code=401, detail="Invalid or expired nonce")
    stored_nonce, exp = entry
    if exp <= time.time() or stored_nonce != req.nonce:
        raise HTTPException(status_code=401, detail="Invalid or expired nonce")
    try:
        recovered = Account.recover_message(encode_defunct(text=f"Sign in to Piata-AI\nNonce: {req.nonce}"), signature=req.signature).lower()
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid signature")
    if recovered != address:
        raise HTTPException(status_code=401, detail="Signature does not match address")
    token = jwt.encode({
        "sub": address,
        "iat": int(time.time()),
        "exp": int(time.time()) + 7 * 86400,
    }, JWT_SECRET, algorithm=JWT_ALGORITHM)
    result = {
        "access_token": token,
        "token_type": "Bearer",
        "address": Web3.to_checksum_address(address),
        "expires_in": 7 * 86400,
    }
    # Link the wallet to a real Firebase user (best-effort). The returned
    # email/password let the frontend sign in to Firebase via
    # signInWithEmailAndPassword on every login.
    fb = await _upsert_wallet_firebase_user(address)
    if fb:
        email, uid, custom_token = fb
        result["email"] = email
        result["uid"] = uid
        result["custom_token"] = custom_token
        result["firebase_linked"] = True
    else:
        result["firebase_linked"] = False
    return result


# ── Wallet → Firebase linking ──────────────────────────────────────────────
# When a wallet signs in via MetaMask, we also create a REAL Firebase Auth
# user for it so the same identity works across the whole piata stack
# (Firestore, Google-authed endpoints, piata backend). The email is a
# deterministic per-address email; the account password is derived from a
# server secret only to satisfy create_user (email/password sign-in is
# DISABLED in this project, so the password is NOT the auth mechanism).
# Authentication uses a Firebase custom token minted here — the standard
# wallet-auth pattern — which works regardless of provider settings.


def _wallet_firebase_email(address: str) -> str:
    return f"{address.lower()}@wallet.piata-ai"


def _wallet_firebase_password(address: str) -> str:
    # Only used to create the Firebase user (a password is required there).
    # It is NOT an auth mechanism — login happens via custom token.
    secret = os.environ.get("WALLET_AUTH_SECRET", "") or JWT_SECRET
    return hashlib.sha256(f"{secret}:{address.lower()}".encode()).hexdigest()[:24]


def _upsert_wallet_firebase_user_sync(address: str):
    """Sync body of the wallet→Firebase upsert (runs in a thread)."""
    try:
        app = _init_firebase_admin()
        if app is None:
            logger.info("metamask: Firebase linking skipped (no Admin SDK)")
            return None
        email = _wallet_firebase_email(address)
        password = _wallet_firebase_password(address)
        uid = f"wallet:{address.lower()}"
        try:
            user = firebase_auth.get_user_by_email(email)
            # uid drift guard: if the account exists under another uid, keep its uid
            uid = user.uid
        except Exception:
            try:
                user = firebase_auth.create_user(
                    email=email,
                    password=password,
                    uid=uid,
                    display_name=f"Wallet {address[:6]}…{address[-4:]}",
                )
            except Exception as e:
                if "already exists" not in str(e).lower():
                    raise
                # Race: another request just created the user — re-fetch it.
                user = firebase_auth.get_user_by_email(email)
                uid = user.uid
        custom_token = firebase_auth.create_custom_token(uid)
        return (email, uid, custom_token)
    except Exception as e:
        logger.warning(f"metamask: Firebase user linking failed: {e}")
        return None


async def _upsert_wallet_firebase_user(address: str):
    """Create (or fetch) a real Firebase Auth user for a wallet address and
    mint a Firebase custom token for it (runs off the event loop).

    Returns (email, uid, custom_token) so the client can sign in to Firebase
    via signInWithCustomToken (works even when Email/Password sign-in is
    disabled in the Firebase console), or None when Firebase is unavailable —
    the piata JWT still works, linking is best-effort.
    """
    return await asyncio.to_thread(_upsert_wallet_firebase_user_sync, address)


class FirebaseAuthRequest(BaseModel):
    id_token: str


@app.post("/api/auth/firebase")
async def firebase_auth_endpoint(req: FirebaseAuthRequest):
    """Verify Firebase ID token and return dashboard session token.

    Uses the Firebase Admin SDK when a service account is configured, otherwise
    falls back to the public identitytoolkit REST verifier (only requires
    FIREBASE_API_KEY).
    """
    pure_now = False  # pure REST path
    try:
        app = _init_firebase_admin()
        if app is not None and FIREBASE_ADMIN_AVAILABLE:
            decoded_token = firebase_auth.verify_id_token(req.id_token)
            email = (decoded_token.get('email') or '').lower()
            email_verified = decoded_token.get('email_verified', False)
            name = decoded_token.get('name', '')
            picture = decoded_token.get('picture', '')
            uid = decoded_token.get('uid', '')
            pure_now = True
        else:
            user = await _verify_firebase_id_token_rest(req.id_token)
            email = (user.get('email') or '').lower()
            email_verified = user.get('emailVerified', False)
            name = user.get('displayName', '')
            picture = user.get('photoUrl', '')
            uid = user.get('localId', '')

        if not email:
            raise HTTPException(401, "No email in token")
        if not email_verified:
            raise HTTPException(401, "Email not verified")

        dashboard_token = _create_dashboard_token()
        return {
            "token": dashboard_token,
            "email": email,
            "name": name,
            "picture": picture,
            "uid": uid,
        }
    except HTTPException:
        raise
    except Exception as e:
        msg = str(e)
        try:
            parsed = json.loads(msg)
            if parsed.get("status") in (400, 401, 403):
                raise HTTPException(parsed["status"], f"Invalid token: {parsed.get('detail', msg)}")
        except (json.JSONDecodeError, TypeError):
            pass
        raise HTTPException(401, f"Authentication failed: {msg}")


@app.get("/api/mcp/hub")
async def mcp_hub():
    return {
        "hub": "piata-mcp-hub",
        "servers": [
            {"name": "openwebui-local-gateway", "url": os.environ.get("OPENWEBUI_URL"), "status": "configured", "tools": ["models", "chat-completions"]},
            {"name": "agent-tars", "url": TARS_URL, "status": "configured", "tools": ["lead-search", "scrape"]},
            {"name": "babyagi", "url": "http://127.0.0.1:5012", "status": "configured", "tools": ["convene", "settle", "leaderboard"]},
            {"name": "bhairava-council", "url": "http://127.0.0.1:5013", "status": "configured", "tools": ["bhairavas", "artifact", "consensus"]},
            {"name": "agt-wallet", "url": "http://127.0.0.1:5009", "status": "configured", "tools": ["wallet", "trade", "send-eth"]},
            {"name": "rehoboam", "url": "http://127.0.0.1:5002", "status": "configured", "tools": ["agent-tools"]},
            {"name": "n8n", "url": os.environ.get("N8N_URL", "http://127.0.0.1:5678"), "status": "configured", "tools": ["workflows"]},
            {"name": "stripe", "status": "configured", "tools": ["checkout", "portal", "webhook"]},
            {"name": "metamask", "status": "configured", "tools": ["signin", "verify-payment"]},
        ],
    }


@app.get("/api/tools/community")
async def community_tools():
    hub = await mcp_hub()
    return {"community_tools": hub["servers"]}


@app.get("/api/tars/health")
async def tars_health():
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=3.0)) as client:
        r = await client.get(f"{TARS_URL}/health")
        return {"ok": r.status_code == 200, "status": r.status_code, "body": r.text[:500]}


@app.post("/api/tars/search")
async def tars_search(request: Request):
    body = await request.json()
    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=5.0)) as client:
        r = await client.post(f"{TARS_URL}/search", json=body)
        return JSONResponse(r.json(), status_code=r.status_code)


@app.get("/api/agency/catalog")
async def catalog():
    """Show available tools, plans, and MCP servers."""
    registry = get_registry()
    return {
        "tools": {
            name: {
                "description": desc,
                "schemas": TOOL_SCHEMAS.get(name, []),
            }
            for name, desc in TOOL_DESCRIPTIONS.items()
        },
        "mcp_servers": [
            {"name": name, **cfg}
            for name, cfg in registry.mcp_servers.items()
        ],
        "plans": [
            {"name": "Agent", "price": 100, "currency": "RON",
             "price_usd": 20,
             "includes": "1 agent la alegere, 1000 requesturi/lună, toate tool-urile, web + memory + code + data"},
            {"name": "Pro", "price": 400, "currency": "RON",
             "price_usd": 80,
             "includes": "10 agenți, 10000 requesturi/lună, toate tool-urile, MCP custom, API access"},
            {"name": "Enterprise", "price": 1000, "currency": "RON",
             "price_usd": 200,
             "includes": "Agenți nelimitați, requesturi nelimitate, suport dedicat, model custom, white label"},
        ],
    }


# === Agent Templates (pre-built for one-click deploy) ===

AGENT_TEMPLATES = [
    {
        "id": "crypto-analyst",
        "name": "Crypto Analyst",
        "description": "Real-time crypto market analysis, price tracking, and trading signals for BTC, ETH, SOL and top altcoins.",
        "icon": "📊",
        "category": "Finance",
        "tools": ["web", "memory", "data", "code", "shell"],
        "model": "mimo-v2.5-pro",
        "system_prompt": """You are a professional Crypto Analyst with deep expertise in cryptocurrency markets, blockchain technology, and DeFi. You analyze real-time market data, identify trends, provide trading insights, and explain complex crypto concepts clearly. Always cite your data sources. Be honest about market uncertainty and never give financial advice — only analysis.""",
        "mcp_servers": ["chainlink", "etherscan"],
        "pricing": "Agent",
    },
    {
        "id": "shopify-support",
        "name": "Shopify Support Agent",
        "description": "Customer support automation for Shopify stores — handles inquiries, order tracking, returns, and product questions 24/7.",
        "icon": "🛍️",
        "category": "E-commerce",
        "tools": ["web", "memory", "data", "whatsapp", "dispatch"],
        "model": "openai/gpt-4o-mini",
        "system_prompt": """You are a friendly and efficient Shopify Customer Support Agent. You help customers with order status, returns, product questions, and general inquiries. Be empathetic, concise, and solution-oriented. Always confirm order details before taking action. Escalate complex issues with clear notes.""",
        "mcp_servers": ["accio"],
        "pricing": "Agent",
    },
    {
        "id": "lead-generator",
        "name": "Lead Generator",
        "description": "Automated lead research, enrichment, and outreach across web, social media, and WhatsApp — find and qualify prospects.",
        "icon": "🎯",
        "category": "Sales",
        "tools": ["web", "memory", "browser", "data", "whatsapp", "dispatch", "social"],
        "model": "mimo-v2.5-pro",
        "system_prompt": """You are a B2B Lead Generation specialist. You research companies, find decision-makers, enrich contact data, and craft personalized outreach messages. You use web scraping, social media, and WhatsApp/email dispatch tools. Always respect data privacy and anti-spam laws. Be professional and data-driven.""",
        "mcp_servers": ["accio", "accio_connectors"],
        "pricing": "Pro",
    },
    {
        "id": "content-marketer",
        "name": "Content Marketer",
        "description": "AI-powered content creation — blog posts, social media, email campaigns, SEO optimization, and analytics.",
        "icon": "✍️",
        "category": "Marketing",
        "tools": ["web", "memory", "data", "social", "vision"],
        "model": "openai/gpt-4o",
        "system_prompt": """You are a creative Content Marketing strategist. You write compelling blog posts, social media content, email campaigns, and ad copy. You optimize for SEO, analyze content performance, and adapt to brand voice. You can create visual content briefs using vision tools. Always deliver polished, publication-ready content.""",
        "mcp_servers": ["supabase"],
        "pricing": "Pro",
    },
    {
        "id": "data-analyst",
        "name": "Data Analyst",
        "description": "Query databases, analyze CSV/JSON data, generate charts and reports — your personal BI assistant.",
        "icon": "📈",
        "category": "Analytics",
        "tools": ["data", "code", "file", "web", "memory"],
        "model": "mimo-v2.5-pro",
        "system_prompt": """You are a skilled Data Analyst. You query databases, analyze CSV/JSON files, generate charts and visualizations, and produce clear analytical reports. You write clean Python/SQL code for data manipulation. Always explain your methodology and highlight key insights. Be precise with numbers.""",
        "mcp_servers": ["supabase"],
        "pricing": "Agent",
    },
    {
        "id": "devops-assistant",
        "name": "DevOps Assistant",
        "description": "Server monitoring, deployment automation, log analysis, and infrastructure management — your AI SRE.",
        "icon": "🖥️",
        "category": "DevOps",
        "tools": ["shell", "code", "file", "web", "memory"],
        "model": "mimo-v2.5-pro",
        "system_prompt": """You are a senior DevOps/SRE engineer. You monitor servers, analyze logs, write deployment scripts, manage Docker containers, and troubleshoot infrastructure issues. You write safe, idempotent shell scripts. Always warn before destructive operations. Be concise and technical.""",
        "mcp_servers": [],
        "pricing": "Pro",
    },
    {
        "id": "social-media-manager",
        "name": "Social Media Manager",
        "description": "Schedule posts, analyze engagement, reply to comments across X/Twitter, LinkedIn, Instagram — all from one agent.",
        "icon": "📱",
        "category": "Social",
        "tools": ["social", "web", "memory", "data", "vision"],
        "model": "openai/gpt-4o-mini",
        "system_prompt": """You are a Social Media Manager handling X/Twitter, LinkedIn, and Instagram. You craft platform-optimized posts, analyze engagement metrics, schedule content, and respond to comments. You maintain brand voice consistency. You generate image briefs for visual content. Always be authentic and engaging.""",
        "mcp_servers": ["accio"],
        "pricing": "Agent",
    },
    {
        "id": "market-researcher",
        "name": "Market Researcher",
        "description": "Competitive analysis, market trends, consumer insights, and industry reports.",
        "icon": "🔍",
        "category": "Business",
        "tools": ["web", "browser", "data", "memory", "file"],
        "model": "openai/gpt-4o",
        "system_prompt": "You are a senior market researcher. Conduct competitive analysis, identify trends, gather insights, and produce structured research reports. Be objective, thorough, and insightful.",
        "mcp_servers": [],
        "pricing": "Pro",
    },
    {
        "id": "whatsapp-business",
        "name": "WhatsApp Business Agent",
        "description": "Automate WhatsApp customer communication - promotions, support, reminders.",
        "icon": "💬",
        "category": "Communication",
        "tools": ["whatsapp", "web", "memory", "data", "dispatch"],
        "model": "openai/gpt-4o-mini",
        "system_prompt": "You are a WhatsApp Business agent for piata-ai.ro. Send promotions, welcome contacts, handle inquiries. Be professional, concise, GDPR-compliant. Respond in Romanian by default.",
        "mcp_servers": [],
        "pricing": "Agent",
    },
    {
        "id": "yolo-vision",
        "name": "Vision Inspector (YOLOv8)",
        "description": "Object detection and image analysis - classify products, detect defects, analyze marketplace images.",
        "icon": "👁️",
        "category": "AI & Vision",
        "tools": ["yolo", "vision", "web", "file", "memory"],
        "model": "openai/gpt-4o",
        "system_prompt": "You are a computer vision specialist using YOLOv8. Detect and classify objects, identify defects, analyze marketplace photos. Report confidence scores. Be precise and technical.",
        "mcp_servers": [],
        "pricing": "Pro",
    },
    {
        "id": "marketplace-scout",
        "name": "Marketplace Scout",
        "description": "Monitor marketplace listings, analyze competitor pricing, identify gaps and opportunities.",
        "icon": "🔎",
        "category": "Marketplace",
        "tools": ["web", "browser", "data", "memory", "file"],
        "model": "mimo-v2.5-pro",
        "system_prompt": "You are a Marketplace Scout. Monitor listings, analyze competitor pricing, identify market gaps, and report actionable insights with numbers. Be direct and data-driven.",
        "mcp_servers": ["accio", "accio_connectors"],
        "pricing": "Agent",
    },
    {
        "id": "revenue-tracker",
        "name": "Revenue Tracker",
        "description": "Track revenue, costs, P&L and generate financial dashboards for the agency.",
        "icon": "💰",
        "category": "Finance",
        "tools": ["data", "code", "memory", "web"],
        "model": "mimo-v2.5-pro",
        "system_prompt": "You are a Revenue Tracker. Track revenue, costs, P&L and generate financial dashboards. Be precise with numbers and never round figures without stating it.",
        "mcp_servers": ["supabase"],
        "pricing": "Agent",
    },
    {
        "id": "sentry-monitor",
        "name": "Sentry Monitor",
        "description": "Monitor VPS services, disk usage, CPU and PM2 health with military precision.",
        "icon": "🛡️",
        "category": "DevOps",
        "tools": ["shell", "code", "web", "memory"],
        "model": "mimo-v2.5-pro",
        "system_prompt": "You are a Sentry Monitor. Monitor VPS services, disk usage, CPU and PM2 health. Report status precisely, escalate incidents immediately, and never declare all-clear without verification.",
        "mcp_servers": [],
        "pricing": "Pro",
    },
    {
        "id": "video-director",
        "name": "Video Director",
        "description": "Generate scripts and orchestrate short-form video content for TikTok, Reels and Shorts.",
        "icon": "🎬",
        "category": "Creative",
        "tools": ["web", "memory", "code", "vision"],
        "model": "openai/gpt-4o",
        "system_prompt": "You are a Video Director. Generate scripts and orchestrate short-form video content for TikTok, Reels and Shorts. Be creative, visual, and concise.",
        "mcp_servers": [],
        "pricing": "Pro",
    },
]


@app.get("/api/agency/templates")
async def list_templates(category: Optional[str] = None):
    """Return pre-built agent templates for one-click provisioning."""
    templates = AGENT_TEMPLATES
    if category:
        templates = [t for t in templates if t["category"].lower() == category.lower()]
    return {
        "templates": templates,
        "count": len(templates),
        "categories": sorted(set(t["category"] for t in AGENT_TEMPLATES)),
    }


@app.post("/api/agency/templates/{template_id}/provision")
async def provision_from_template(template_id: str,
                                   owner_email: Optional[str] = None,
                                   owner_name: Optional[str] = None,
                                   _user: dict = Depends(get_current_user)):
    """Provision a new agent from a pre-built template."""
    template = next((t for t in AGENT_TEMPLATES if t["id"] == template_id), None)
    if not template:
        raise HTTPException(404, f"Template '{template_id}' not found")
    try:
        result = provisioner.provision(
            name=template["name"],
            purpose=template["description"],
            tools=template["tools"],
            model=template["model"],
            system_prompt=template["system_prompt"],
            owner_email=owner_email,
            owner_name=owner_name,
            mcp_servers=template["mcp_servers"],
        )
        logger.info(f"Provisioned agent from template {template_id}: {result['agent_id']}")
        return _public_provision_result(result)
    except Exception as e:
        logger.error(f"Template provisioning failed: {e}")
        raise HTTPException(500, f"Provisioning failed: {e}")




@app.post("/api/agency/templates/{template_id}/deploy")
async def deploy_template_compat(template_id: str, request: Request, _user: dict = Depends(get_current_user)):
    """Compatibility alias for /provision endpoint. Accepts optional owner_email/owner_name in JSON body."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    return await provision_from_template(template_id, body.get("owner_email"), body.get("owner_name"))
# === Demo Chat (try before you buy) ===

DEMO_AGENT = {
    "agent_id": "agt_demo_showcase",
    "name": "Piata AI Demo Agent",
    "purpose": "Showcase the power of AI agents — try tools live before you deploy your own.",
    "tools": ["web", "memory", "data", "code", "shell"],
    "model": "mimo-v2.5-pro",
    "system_prompt": """You are the Piata AI Demo Agent — a showcase of what AI agents can do on the piata-ai.ro marketplace.

You have access to real tools: web search, memory storage, data analysis, code execution, and shell commands.

Your job is to impress visitors. When they ask a question, use the most relevant tool to give them a concrete, useful answer. Show them the power of tool-augmented AI.

Examples:
- If asked about crypto prices, use web_search to get real-time data
- If asked to analyze something, use code_execute or data_analyze_csv
- If asked to remember something, use memory_store

Be friendly, enthusiastic, and SHOW your tool usage. Mention that they can deploy their own custom agent with 40+ tools and 7 MCP servers at piata-ai.ro/agency.

Keep responses concise but impressive. Limit to 5 messages per visitor.""",
    "webhook_token": "demo_public_token",
    "api_key": "demo_public_key",
    "status": "active",
}

# Track demo usage per IP (rate limit: 5 messages per IP per hour)
_demo_usage: dict = {}


def _cleanup_demo_usage():
    """Purge expired demo usage entries to prevent memory leak."""
    now = time.time()
    expired = [ip for ip, u in _demo_usage.items() if now > u["reset_at"]]
    for ip in expired:
        del _demo_usage[ip]


@app.post("/api/agency/demo/chat")
async def demo_chat(req: ChatRequest, request: Request):
    """Demo chat endpoint — try an AI agent without signing up. Rate-limited."""
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()

    # Periodic cleanup of expired entries (every ~100 requests)
    if len(_demo_usage) > 100:
        _cleanup_demo_usage()

    # Rate limit: 5 messages per IP per hour
    if client_ip not in _demo_usage:
        _demo_usage[client_ip] = {"count": 0, "reset_at": now + 3600}

    usage = _demo_usage[client_ip]
    if now > usage["reset_at"]:
        usage["count"] = 0
        usage["reset_at"] = now + 3600

    if usage["count"] >= 5:
        reset_mins = int((usage["reset_at"] - now) / 60) + 1
        return {
            "error": f"Demo limit reached (5 messages/hour). Deploy your own agent at piata-ai.ro/agency for unlimited access.",
            "retry_in_minutes": reset_mins,
        }

    usage["count"] += 1
    remaining = 5 - usage["count"]

    # Use chat_service for demo chat (rate-limited, no DB logging)
    response = await chat_service.run_demo_chat(DEMO_AGENT, req.message, req.history)
    response["demo_remaining"] = remaining
    response["deploy_url"] = "https://piata-ai.ro/agency/"
    return response


# === Agent Provisioning ===

@app.post("/api/agency/agents")
async def provision_agent(req: ProvisionRequest, _user: dict = Depends(get_current_user)):
    """Provision a new AI agent. Returns webhook URL and API key."""
    try:
        result = provisioner.provision(
            name=req.name,
            purpose=req.purpose,
            tools=req.tools,
            model=req.model,
            system_prompt=req.system_prompt,
            owner_email=req.owner_email,
            owner_name=req.owner_name,
            mcp_servers=req.mcp_servers,
        )
        logger.info(f"Provisioned agent {result['agent_id']}: {req.name}")
        return _public_provision_result(result)
    except Exception as e:
        logger.error(f"Provisioning failed: {e}")
        raise HTTPException(500, f"Provisioning failed: {e}")


@app.post("/api/agency/agents/{agent_id}/checkout")
async def agent_checkout(agent_id: str, request: Request, _user: dict = Depends(get_current_user)):
    """Create a Stripe checkout session for an agent via the Stripe MCP server."""
    # Verify agent exists
    agent = provisioner.get_agent(agent_id)
    if not agent:
        raise HTTPException(404, "Agent not found")

    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    plan_id = (body or {}).get("plan_id") or "starter"
    interval = (body or {}).get("interval") or "monthly"
    customer_email = (body or {}).get("customer_email")

    if plan_id not in BILLING_PLANS:
        raise HTTPException(400, "Invalid plan")
    if interval not in ("monthly", "yearly"):
        raise HTTPException(400, "Invalid interval")

    # Lightweight ownership check: if both agent and request provide an email, they must match
    agent_email = agent.get("owner_email")
    if agent_email and customer_email and agent_email.lower() != customer_email.lower():
        raise HTTPException(403, "Email does not match agent owner")

    try:
        mcp_result = await mcp_call_tool(
            "stripe",
            "create_checkout_session",
            {
                "agent_id": agent_id,
                "plan_id": plan_id,
                "interval": interval,
                "customer_email": customer_email,
            },
        )
        if mcp_result.get("error"):
            logger.error(f"Stripe MCP checkout failed for {agent_id}: {mcp_result['error']}")
            raise HTTPException(502, f"Stripe MCP error: {mcp_result['error']}")

        # MCP result wraps the tool response in result.content
        payload = None
        try:
            content = mcp_result.get("result", {}).get("content", [{}])
            if content and isinstance(content, list) and content[0].get("text"):
                payload = json.loads(content[0]["text"])
        except Exception:
            payload = None
        if not payload:
            payload = mcp_result.get("result", {})

        session_id = payload.get("id")
        if session_id:
            create_checkout_record(
                session_id=session_id,
                agent_id=agent_id,
                plan_id=plan_id,
                interval=interval,
                status=payload.get("status", "pending"),
            )

        return {
            "url": payload.get("url"),
            "id": session_id,
            "status": payload.get("status", "created"),
            "agent_id": agent_id,
            "plan_id": plan_id,
            "interval": interval,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Checkout failed for {agent_id}: {e}")
        raise HTTPException(500, f"Checkout failed: {e}")


@app.get("/api/agency/agents")
async def list_agents(owner_email: Optional[str] = None,
                      status: Optional[str] = None,
                      _user: dict = Depends(get_current_user)):
    """List all agents."""
    return {"agents": [_public_agent(a) for a in provisioner.list_agents(owner_email, status)]}


@app.get("/api/agency/agents/{agent_id}")
async def get_agent(agent_id: str, _user: dict = Depends(get_current_user)):
    """Get agent details."""
    agent = provisioner.get_agent(agent_id)
    if not agent:
        raise HTTPException(404, "Agent not found")
    return _public_agent(agent)


@app.get("/api/agency/agents/{agent_id}/history")
async def get_history(agent_id: str, limit: int = 50, _user: dict = Depends(get_current_user)):
    """Get chat history for an agent."""
    agent = provisioner.get_agent(agent_id)
    if not agent:
        raise HTTPException(404, "Agent not found")
    return {"history": provisioner.get_chat_history(agent_id, limit)}


@app.post("/api/agency/agents/{agent_id}/pause")
async def pause_agent(agent_id: str, _user: dict = Depends(get_current_user)):
    """Pause an agent."""
    agent = provisioner.get_agent(agent_id)
    if not agent:
        raise HTTPException(404, "Agent not found")
    provisioner.update_agent_status(agent_id, "paused")
    return {"status": "paused", "agent_id": agent_id}


@app.post("/api/agency/agents/{agent_id}/resume")
async def resume_agent(agent_id: str, _user: dict = Depends(get_current_user)):
    """Resume a paused agent."""
    agent = provisioner.get_agent(agent_id)
    if not agent:
        raise HTTPException(404, "Agent not found")
    provisioner.update_agent_status(agent_id, "active")
    return {"status": "active", "agent_id": agent_id}


# === Chat via API Key ===

@app.post("/api/agency/agents/{agent_id}/chat")
async def chat_api(agent_id: str, req: ChatRequest,
                   authorization: Optional[str] = Header(None),
                   x_api_key: Optional[str] = Header(None),
                   x_dashboard_token: Optional[str] = Header(None),
                   request: Request = None):
    """Chat with an agent using API key auth or from admin dashboard.
    Auth (any one of):
      1. Agent's own API key (pia_... via Authorization: Bearer or X-API-Key)
      2. AGENCY_BEARER_TOKEN / ADMIN_API_KEY (global access)
      3. Dashboard token (X-Dashboard-Token)
      4. Localhost (127.0.0.1)
    """
    agent = provisioner.get_agent(agent_id)
    if not agent:
        raise HTTPException(404, "Agent not found")
    if agent["status"] != "active":
        raise HTTPException(403, f"Agent is {agent['status']}")

    client_host = request.client.host if request.client else ""
    key = None
    if authorization and authorization.lower().startswith("bearer "):
        key = authorization[7:].strip()
    if not key and x_api_key:
        key = x_api_key.strip()

    # Auth check — any one wins
    authenticated = False

    # 1. Agent's own API key (stored hashed; check_secret supports legacy plaintext too)
    if key and check_secret(agent.get("api_key", ""), key):
        authenticated = True

    # 2. Global agency bearer token
    if not authenticated and AGENCY_BEARER_TOKEN and key and secrets.compare_digest(key, AGENCY_BEARER_TOKEN):
        authenticated = True

    # 3. Admin API keys
    if not authenticated and ADMIN_API_KEYS and key in ADMIN_API_KEYS:
        authenticated = True

    # 4. Dashboard token
    if not authenticated and _verify_dashboard_token(x_dashboard_token):
        authenticated = True

    # 5. Localhost
    if not authenticated and client_host in ("127.0.0.1", "::1", "localhost"):
        authenticated = True

    if not authenticated:
        raise HTTPException(401, "Authentication required — provide agent API key, agency bearer token, or access from localhost")

    response = await chat_service.run_agent_chat(agent_id, req.message, req.history, agent)
    return response


# === Webhook Endpoint ===

# Per-agent simple in-memory rate limiter (token bucket).
# Protects webhooks from being hammered if a -wh_ token leaks.
_webhook_rate_state: dict[str, list] = {}  # agent_id -> [timestamps]
_WEBHOOK_RATE_WINDOW_SECONDS = float(os.environ.get("PIATA_WEBHOOK_RATELIMIT_SECONDS", "60"))
_WEBHOOK_RATE_MAX_CALLS = int(os.environ.get("PIATA_WEBHOOK_RATELIMIT_MAX", "30"))


def _webhook_rate_check(agent_id: str) -> bool:
    """Returns True if the call is allowed, False if rate-limited."""
    now = time.time()
    window = _WEBHOOK_RATE_WINDOW_SECONDS
    max_calls = _WEBHOOK_RATE_MAX_CALLS
    history = _webhook_rate_state.get(agent_id, [])
    # Drop timestamps outside the window
    history = [t for t in history if now - t < window]
    if len(history) >= max_calls:
        _webhook_rate_state[agent_id] = history
        return False
    history.append(now)
    _webhook_rate_state[agent_id] = history
    return True


@app.post("/webhook/{agent_id}/{token}")
async def webhook_chat(agent_id: str, token: str, req: WebhookRequest):
    """Webhook endpoint — external systems send messages here."""
    agent = provisioner.get_agent(agent_id)
    if not agent:
        raise HTTPException(404, "Agent not found")
    # Constant-time comparison to avoid timing side-channels on webhook tokens.
    stored_token = agent.get("webhook_token") or ""
    if not (stored_token and secrets.compare_digest(stored_token, token)):
        raise HTTPException(401, "Invalid webhook token")
    if agent["status"] != "active":
        raise HTTPException(403, f"Agent is {agent['status']}")

    if not _webhook_rate_check(agent_id):
        raise HTTPException(429, f"Rate limit exceeded: max {_WEBHOOK_RATE_MAX_CALLS} calls / {_WEBHOOK_RATE_WINDOW_SECONDS:.0f}s per agent")

    message = req.message or req.text or req.body or ""
    if not message:
        raise HTTPException(400, "No message provided")

    response = await chat_service.run_agent_chat(agent, message, agent_id=agent_id)
    return response


@app.get("/api/agency/billing/plans")
async def list_billing_plans(_user: dict = Depends(get_current_user)):
    """Return available billing plans."""
    from payments import STRIPE_PUBLISHABLE_KEY
    return {"plans": BILLING_PLANS, "stripe_pk": os.getenv("STRIPE_PUBLISHABLE_KEY") or STRIPE_PUBLISHABLE_KEY}


@app.get("/api/agency/billing/usage/{agent_id}")
async def get_billing_usage(agent_id: str, _user: dict = Depends(get_current_user)):
    """Return usage stats for an agent."""
    summary = get_usage_summary(agent_id)
    return summary


@app.get("/api/agency/billing/status/{agent_id}")
async def get_agent_billing_status(agent_id: str, _user: dict = Depends(get_current_user)):
    """Return payment status for an agent (pending/paid/canceled/failed)."""
    checkouts = get_checkout_by_agent(agent_id, limit=1)
    if not checkouts:
        return {"agent_id": agent_id, "status": "none", "has_checkout": False}
    latest = checkouts[0]
    return {
        "agent_id": agent_id,
        "status": latest["status"],
        "has_checkout": True,
        "session_id": latest["session_id"],
        "plan_id": latest["plan_id"],
        "interval": latest["interval"],
        "created_at": latest["created_at"],
    }


@app.post("/api/agency/billing/checkout")
async def create_billing_checkout(request: Request, _user: dict = Depends(get_current_user)):
    """Create a Stripe checkout session for plan upgrade.

    Accepts JSON body: { plan_id, interval, agent_id?, customer_email? }
    """
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    plan_id = (body or {}).get("plan_id") or "starter"
    interval = (body or {}).get("interval") or "monthly"
    agent_id = (body or {}).get("agent_id") or "global"
    customer_email = (body or {}).get("customer_email")
    if plan_id not in BILLING_PLANS:
        raise HTTPException(400, "Invalid plan")
    result = create_checkout_session(agent_id, plan_id, interval, customer_email=customer_email)
    return result


@app.post("/api/factory/checkout")
async def factory_checkout(request: Request):
    """Compatibility endpoint for frontend selectAgent() -> /api/factory/checkout."""
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    agent_id = (body or {}).get("agent_id") or "global"
    plan_id = (body or {}).get("plan_id", "starter")
    result = create_checkout_session(agent_id, plan_id, "monthly")
    # Frontend expects checkout_url
    if "url" in result and "checkout_url" not in result:
        result["checkout_url"] = result["url"]
    return result

@app.post("/api/agency/billing/portal")
async def billing_portal(request: Request, _user: dict = Depends(get_current_user)):
    """Create Stripe customer portal session."""
    body = await request.json()
    customer_id = body.get("customer_id")
    if not customer_id:
        raise HTTPException(400, "customer_id required")
    result = create_customer_portal_session(customer_id)
    return result


@app.post("/api/agency/billing/cancel")
async def billing_cancel(request: Request, _user: dict = Depends(get_current_user)):
    """Cancel subscription for an agent."""
    body = await request.json()
    agent_id = body.get("agent_id")
    if not agent_id:
        raise HTTPException(400, "agent_id required")
    success = cancel_subscription(agent_id)
    return {"canceled": success}


@app.post("/webhook/stripe")
async def stripe_webhook(request: Request):
    """Handle Stripe webhooks."""
    payload = await request.body()
    signature = request.headers.get("stripe-signature", "")
    event = verify_webhook_signature(payload, signature)
    if not event:
        raise HTTPException(400, "Invalid signature")
    payments_handle_webhook_event(event)
    return {"status": "ok"}


# === Chatbox UI ===

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the admin dashboard with relative paths anchored to current location."""
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard", "index.html")
    with open(dashboard_path) as f:
        html = f.read()
    if "<base " not in html:
        base_tag = "<head>\n  <base href=\"./\">"
        html = html.replace("<head>", base_tag, 1)
    # Inject Firebase config for Auth
    firebase_config = {
        "FIREBASE_API_KEY": os.environ.get("FIREBASE_API_KEY", ""),
        "FIREBASE_AUTH_DOMAIN": os.environ.get("FIREBASE_AUTH_DOMAIN", ""),
        "FIREBASE_PROJECT_ID": os.environ.get("FIREBASE_PROJECT_ID", ""),
        "FIREBASE_STORAGE_BUCKET": os.environ.get("FIREBASE_STORAGE_BUCKET", ""),
        "FIREBASE_MESSAGING_SENDER_ID": os.environ.get("FIREBASE_MESSAGING_SENDER_ID", ""),
        "FIREBASE_APP_ID": os.environ.get("FIREBASE_APP_ID", ""),
    }
    config_script = "<script>\n"
    for key, value in firebase_config.items():
        if value:
            config_script += f"  window.{key} = '{value}';\n"
    config_script += "</script>\n"
    html = html.replace("</head>", config_script + "</head>")
    return html


# ── News-Spike Dashboard (integrated from qwen-bots) ────────────
from fastapi.responses import HTMLResponse as _NewsHTMLResponse
import html as _news_html
import json as _news_json


@app.get("/api/news-spike/latest")
async def _piata_news_spike_latest():
    """Return the latest news-spike report from Common Brain."""
    from news_spike_dashboard import get_latest_report
    report = get_latest_report(window=3600)
    if report is None:
        return JSONResponse({"error": "no recent news-spike report"}, status_code=404)
    return report


@app.get("/api/news-spike/history")
async def _piata_news_spike_history(limit: int = Query(20, ge=1, le=200)):
    """Return recent news-spike reports."""
    rows = _news_spike_history(limit=limit)
    if rows and rows[0].get("error"):
        return JSONResponse({"error": rows[0]["error"]}, status_code=500)
    return {"history": rows}


@app.get("/news-spike", response_class=HTMLResponse)
async def _piata_news_spike_dashboard():
    """Serve the news-spike HTML dashboard under piata-ai.ro."""
    from news_spike_dashboard import get_latest_report as _get_latest, get_history as _get_history
    report = _get_latest(window=3600) or {}
    assets = report.get("assets", {})
    macro = report.get("macro_sentiment", "unknown")
    timestamp = _news_fmt_ts(report.get("timestamp", "N/A"))

    spike_count = sum(1 for a in assets.values() if a.get("label") in ("bearish_spike", "bullish_spike"))
    asset_count = len(assets)

    rows = []
    for asset, data in assets.items():
        label = _news_html.escape(str(data.get("label", "calm")))
        rows.append(
            f"<tr><td><b>{_news_html.escape(str(asset))}</b></td>"
            f"<td><span class=\"badge {label}\">{label.replace('_', ' ')}</span></td>"
            f"<td>{data.get('total', 0)}</td>"
            f"<td>{data.get('bullish', 0)}</td>"
            f"<td>{data.get('bearish', 0)}</td>"
            f"<td>{data.get('neutral', 0)}</td>"
            f"<td>{data.get('net_sentiment', 0)}</td></tr>"
        )

    history_rows = _get_history(limit=10)
    history_html = []
    for h in history_rows:
        if "error" in h:
            continue
        h_macro = h.get("macro_sentiment", "neutral")
        side = "bearish" if h_macro == "risk_off" else ("bullish" if h_macro == "risk_on" else "")
        assets_dict = h.get("assets") or {}
        if not isinstance(assets_dict, dict):
            assets_dict = {}
        assets_short = ", ".join(
            f"{_news_html.escape(str(k))}={_news_html.escape(str(v.get('label', 'calm')))}" for k, v in assets_dict.items()
        )
        history_html.append(
            f"<div class=\"history-item {_news_html.escape(side)}\">"
            f"<div><b>{_news_html.escape(h_macro.upper().replace('_', ' '))}</b> — {assets_short}</div>"
            f"<div class=\"history-meta\">{_news_html.escape(_news_fmt_ts(h.get('timestamp', '')))}</div></div>"
        )

    page = _NEWS_SPIKE_TEMPLATE.replace("{{macro}}", _news_html.escape(macro.replace("_", " ").upper()))
    page = page.replace("{{macro_class}}", _news_html.escape(macro))
    page = page.replace("{{timestamp}}", _news_html.escape(timestamp))
    page = page.replace("{{spike_count}}", _news_html.escape(str(spike_count)))
    page = page.replace("{{asset_count}}", _news_html.escape(str(asset_count)))
    page = page.replace("{{rows}}", "\n".join(rows) if rows else "<tr><td colspan=\"7\">No data</td></tr>")
    page = page.replace("{{history}}", "\n".join(history_html) if history_html else "<div class='history-meta'>No history yet.</div>")
    return HTMLResponse(page)


@app.get("/styles.css", response_class=HTMLResponse)
async def _serve_styles():
    css_path = os.path.join(os.path.dirname(__file__), "dashboard", "styles.css")
    if not os.path.exists(css_path):
        raise HTTPException(404, "styles.css not found")
    with open(css_path) as f:
        return HTMLResponse(f.read(), media_type="text/css")


@app.get("/agents/styles.css", response_class=HTMLResponse)
async def _serve_agents_styles():
    css_path = os.path.join(os.path.dirname(__file__), "dashboard", "styles.css")
    if not os.path.exists(css_path):
        raise HTTPException(404, "styles.css not found")
    with open(css_path) as f:
        return HTMLResponse(f.read(), media_type="text/css")


@app.get("/agents/", response_class=HTMLResponse)
async def agents_dashboard():
    html_path = os.path.join(os.path.dirname(__file__), "dashboard", "index.html")
    if not os.path.exists(html_path):
        raise HTTPException(404, "dashboard index not found")
    with open(html_path) as f:
        return HTMLResponse(f.read())


@app.get("/chat/{agent_id}", response_class=HTMLResponse)
async def chatbox_ui(agent_id: str, token: Optional[str] = None):
    """Serve the chatbox UI for an agent."""
    agent = provisioner.get_agent(agent_id)
    if not agent:
        raise HTTPException(404, "Agent not found")

    return CHATBOX_HTML.replace("{{AGENT_ID}}", agent_id) \
                       .replace("{{AGENT_NAME}}", agent["name"]) \
                       .replace("{{WEBHOOK_TOKEN}}", agent["webhook_token"]) \
                       .replace("{{PURPOSE}}", agent["purpose"])


@app.get("/agents/chat/{agent_id}", response_class=HTMLResponse)
async def agents_chatbox_ui(agent_id: str, token: Optional[str] = None):
    return await chatbox_ui(agent_id, token)




# === UNIFIED BRIDGE (Sacred API + Internal Tools) ===
# All sacred infrastructure exposed through Piata Agency :8080
# PUBLIC routes:  /api/sacred/*    (safe for frontend, no scraping exposed)
# INTERNAL routes: /api/internal/* (require X-Internal-Key header, scraping etc)

import os as _ub_os
import hmac as _ub_hmac
import hashlib as _ub_hash
from typing import Optional as _ub_Optional
from pathlib import Path as _ub_Path
from fastapi import Header as _ub_Header, HTTPException as _ub_HTTPException

SACRED_API = _ub_os.environ.get("SACRED_API_URL", "http://127.0.0.1:9100")
INTERNAL_API_KEY = _ub_os.environ.get("INTERNAL_API_KEY", "sacred-bhairava-2026-internal-only")


def _ub_internal_required(x_internal_key: _ub_Optional[str] = _ub_Header(None)):
    """Verify internal API key. Used on /api/internal/* routes."""
    if not x_internal_key or not _ub_hmac.compare_digest(x_internal_key, INTERNAL_API_KEY):
        raise _ub_HTTPException(403, "Internal API key required (X-Internal-Key header)")


def _ub_proxy(target_url: str, method: str = "GET", body: dict = None, timeout: int = 30):
    """Proxy HTTP request to internal service."""
    import urllib.request, urllib.error, json as _ub_json
    data = _ub_json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        target_url,
        data=data,
        headers={"Content-Type": "application/json"} if body else {},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return _ub_json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return {"error": f"upstream {e.code}", "body": e.read().decode()[:300]}
    except Exception as e:
        return {"error": str(e)[:200]}


@app.get("/api/sacred/health")
async def sacred_health_public():
    """PUBLIC — Sacred knowledge graph + kanban health (no scraping exposed)."""
    return _ub_proxy(f"{SACRED_API}/health")


@app.get("/api/sacred/nodes")
async def sacred_nodes_public(limit: int = 20):
    """PUBLIC — List sacred knowledge graph nodes (filtered, no internal metadata)."""
    data = _ub_proxy(f"{SACRED_API}/api/nodes")
    nodes = data.get("nodes", []) if isinstance(data, dict) else data
    # Strip internal-only fields
    safe = []
    for n in nodes[:limit]:
        safe.append({
            "id": n.get("id"),
            "label": n.get("label"),
            "type": n.get("type"),
            "description": n.get("metadata", {}).get("description", "")[:200],
            "total_tools": n.get("metadata", {}).get("total_tools"),
        })
    return {"nodes": safe, "count": len(safe)}


@app.get("/api/sacred/kanban")
async def sacred_kanban_public():
    """PUBLIC — Kanban board overview (task titles only, no internal fields)."""
    data = _ub_proxy(f"{SACRED_API}/api/kanban/board")
    if isinstance(data, dict) and "tasks" in data:
        return {
            "board": data.get("board", "default"),
            "task_count": len(data.get("tasks", [])),
            "tasks": [{"id": t.get("id"), "title": t.get("title"), "status": t.get("status")} for t in data.get("tasks", [])[:50]],
        }
    return data


@app.get("/api/sacred/relations")
async def sacred_relations_public(limit: int = 30):
    """PUBLIC — Relation types in the sacred graph."""
    data = _ub_proxy(f"{SACRED_API}/api/relation-types")
    return data if isinstance(data, dict) else {"relations": data}


# === INTERNAL ROUTES (require X-Internal-Key) ===

@app.get("/api/internal/sacred/raw")
async def sacred_internal_raw(x_internal_key: _ub_Optional[str] = _ub_Header(None)):
    """INTERNAL — Full unfiltered sacred API response (scraping, librarian, etc)."""
    _ub_internal_required(x_internal_key)
    return _ub_proxy(f"{SACRED_API}/api/nodes")


@app.post("/api/internal/sacred/scrape")
async def sacred_internal_scrape(req: dict, x_internal_key: _ub_Optional[str] = _ub_Header(None)):
    """INTERNAL — Trigger scraping into sacred graph (research only)."""
    _ub_internal_required(x_internal_key)
    return _ub_proxy(f"{SACRED_API}/api/nodes", method="POST", body=req)


@app.post("/api/internal/sacred/librarian/check")
async def sacred_internal_librarian(req: dict, x_internal_key: _ub_Optional[str] = _ub_Header(None)):
    """INTERNAL — Direct librarian check (research only)."""
    _ub_internal_required(x_internal_key)
    return _ub_proxy(f"{SACRED_API}/api/librarian/check", method="POST", body=req)


@app.post("/api/internal/babyagi/trigger")
async def babyagi_trigger(req: dict, x_internal_key: _ub_Optional[str] = _ub_Header(None)):
    """INTERNAL — Send a task to babyAGI daemon (research/exploration)."""
    _ub_internal_required(x_internal_key)
    # babyAGI runs as PM2 daemon, listens via signal/cycle
    # For now, just log and return success
    import json as _babyagi_json
    log_path = _ub_Path("/root/AgentX/babyagi_triggered.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(_babyagi_json.dumps({"timestamp": __import__("datetime").datetime.utcnow().isoformat()+"Z", "task": req}) + "\n")
    return {"status": "logged", "file": str(log_path), "note": "babyAGI will pick this up in next cycle (≤60s)"}


@app.get("/api/internal/council/state")
async def council_state(x_internal_key: _ub_Optional[str] = _ub_Header(None)):
    """INTERNAL — Current state of council_bhairavas vote weights."""
    _ub_internal_required(x_internal_key)
    # Council is at :5012 (council-service), not exposed publicly
    return _ub_proxy("http://127.0.0.1:5012/health")


@app.get("/api/agency/marketplace/health")
async def marketplace_health_enhanced():
    """Composite health with sacred knowledge graph stats."""
    sacred = _ub_proxy(f"{SACRED_API}/health")
    return {
        "status": "ok",
        "executor": "claude-code-2.1.177",
        "sacred_api": sacred,
        "sacred_api_url_internal": SACRED_API,
        "internal_api_key_required_for": "/api/internal/*",
        "timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z",
    }


# === Persona Orchestrator (Claude Code powered) ===
# Endpoints exposing the 10 marketplace personas and their proof-of-work artifacts.
# Personas are NOT separate processes — they are execution contexts for Claude Code
# (the AI in this session), which switches persona via system prompt + awareness.
from pathlib import Path as _PowPath
import json as _pow_json
import sqlite3 as _pow_sq
import subprocess as _pow_sp
from datetime import datetime as _pow_dt

_POW_DIR = _PowPath("/root/piata-agency/pow")
_PERSONAS_DIR = _PowPath("/root/piata-agency/personas")


@app.get("/api/agency/personas")
async def list_personas():
    """List all marketplace personas (Claude Code execution contexts)."""
    personas = []
    for path in sorted(_PERSONAS_DIR.glob("*.yaml")):
        try:
            import yaml as _yaml
            data = _yaml.safe_load(path.read_text())
            personas.append({
                "agent_id": data.get("agent_id", path.stem),
                "name": data.get("name", path.stem),
                "purpose": data.get("purpose", ""),
                "tools": data.get("tools", []),
                "owner": data.get("context", {}).get("owner", "unknown"),
                "platform": data.get("context", {}).get("platform", "piata-ai.ro"),
                "schedule": data.get("schedule", "on-demand"),
            })
        except Exception as e:
            personas.append({"slug": path.stem, "error": str(e)})
    return {
        "personas": personas,
        "executor": "claude-code-2.1.177",
        "count": len(personas),
    }


@app.get("/api/agency/personas/{agent_id}/education")
async def persona_education(agent_id: str, _user: dict = Depends(get_current_user)):
    """Return Kimi Academy education recommendations for a persona."""
    from personalities.kimiraikonen import KimiRaikonen
    from personalities.base import load_persona, PERSONAS_DIR

    # Find matching persona YAML by agent_id prefix
    persona_yaml = None
    for path in PERSONAS_DIR.glob("*.yaml"):
        if path.stem in agent_id:
            persona_yaml = path
            break

    if not persona_yaml or not persona_yaml.exists():
        raise HTTPException(404, "Persona not found")

    try:
        ctx = load_persona(persona_yaml.stem)
    except Exception as e:
        raise HTTPException(500, f"Failed to load persona: {e}")

    kimi = KimiRaikonen()
    result = kimi.educate_agent(ctx.agent_id, ctx.purpose, ctx.tools)
    return {"agent_id": ctx.agent_id, "education": result}


@app.get("/api/agency/personas/{agent_id}/artifacts")
async def list_persona_artifacts(agent_id: str, limit: int = 5, _user: dict = Depends(get_current_user)):
    """Recent proof-of-work artifacts for a persona."""
    agent_dir = _POW_DIR / agent_id
    if not agent_dir.exists():
        return {"artifacts": [], "agent_id": agent_id, "count": 0}
    files = sorted(agent_dir.glob("*.json"), key=lambda p: -p.stat().st_mtime)[:limit]
    artifacts = []
    for f in files:
        try:
            data = _pow_json.loads(f.read_text())
            artifacts.append({
                "timestamp_iso": data.get("timestamp_iso"),
                "task": data.get("task", "")[:100],
                "executor": data.get("executor"),
                "content_preview": _pow_json.dumps(data.get("response", {}))[:200],
                "signature_valid": True,
                "content_sha256": data.get("content_sha256"),
            })
        except Exception as e:
            artifacts.append({"file": str(f), "error": str(e)})
    return {"agent_id": agent_id, "artifacts": artifacts, "count": len(artifacts)}


@app.get("/api/agency/personas/all/status")
async def all_personas_status(_user: dict = Depends(get_current_user)):
    """Status snapshot across all marketplace personas."""
    personas = await list_personas()
    statuses = []
    for p in personas.get("personas", []):
        s = await persona_status(p["agent_id"])
        s["name"] = p.get("name")
        s["purpose"] = (p.get("purpose") or "")[:80]
        statuses.append(s)
    fresh_count = sum(1 for s in statuses if s.get("fresh"))
    return {
        "personas": statuses,
        "count": len(statuses),
        "fresh_count": fresh_count,
        "stale_count": len(statuses) - fresh_count,
        "executor": "piata-agency-personality-runner/1.0",
        "checked_at": _pow_dt.utcnow().isoformat() + "Z",
    }


@app.get("/api/agency/personas/{agent_id}/status")
async def persona_status(agent_id: str, _user: dict = Depends(get_current_user)):
    """Live status for a persona: latest PoW artifact, freshness, signature verification."""
    agent_dir = _POW_DIR / agent_id
    persona_yaml = _PERSONAS_DIR / f"{agent_id.replace('agt_', '')}.yaml"
    # The persona yaml filenames don't match agent_id exactly — try to find by name
    out = {"agent_id": agent_id, "artifacts_total": 0, "latest": None, "fresh": False, "age_s": None}
    if not agent_dir.exists():
        out["note"] = "no PoW directory yet — runner not executed for this agent"
        return out
    files = sorted(agent_dir.glob("*.json"), key=lambda p: -p.stat().st_mtime)
    out["artifacts_total"] = len(files)
    if not files:
        return out
    latest = files[0]
    out["latest"] = {
        "path": str(latest),
        "timestamp_iso": None,
        "content_sha256": None,
        "executor": None,
        "size_bytes": latest.stat().st_size,
        "age_s": int((_pow_dt.utcnow().timestamp() - latest.stat().st_mtime)),
    }
    try:
        data = _pow_json.loads(latest.read_text())
        out["latest"]["timestamp_iso"] = data.get("timestamp_iso")
        out["latest"]["content_sha256"] = data.get("content_sha256")
        out["latest"]["executor"] = data.get("executor")
        out["latest"]["task"] = (data.get("task") or "")[:120]
        # Verify signature using the same scheme as the runner
        import hashlib as _hl, hmac as _hm
        secret_file = _PowPath("/root/piata-agency/.orchestrator_secret")
        if secret_file.exists():
            sig_input = {k: v for k, v in data.items()
                         if k not in ("content_sha256", "artifact_sha256", "orchestrator_signature")}
            body = _pow_json.dumps(sig_input, sort_keys=True, ensure_ascii=False).encode()
            expected_artifact_hash = _hl.sha256(body).hexdigest()
            expected_sig = _hm.new(secret_file.read_bytes(), expected_artifact_hash.encode(), _hl.sha256).hexdigest()
            out["latest"]["signature_valid"] = expected_sig == data.get("orchestrator_signature")
            out["latest"]["artifact_sha256"] = expected_artifact_hash
    except Exception as e:
        out["latest"]["read_error"] = str(e)
    out["fresh"] = out["latest"]["age_s"] is not None and out["latest"]["age_s"] < 1800  # < 30 min
    return out


@app.get("/api/agency/personas/all/status")
async def all_personas_status():
    """Status snapshot across all marketplace personas."""
    personas = await list_personas()
    statuses = []
    for p in personas.get("personas", []):
        s = await persona_status(p["agent_id"])
        s["name"] = p.get("name")
        s["purpose"] = (p.get("purpose") or "")[:80]
        statuses.append(s)
    fresh_count = sum(1 for s in statuses if s.get("fresh"))
    return {
        "personas": statuses,
        "count": len(statuses),
        "fresh_count": fresh_count,
        "stale_count": len(statuses) - fresh_count,
        "executor": "piata-agency-personality-runner/1.0",
        "checked_at": _pow_dt.utcnow().isoformat() + "Z",
    }


@app.post("/api/agency/personas/run")
async def run_personas(slug: Optional[str] = None, all: bool = False, _user: dict = Depends(get_current_user)):
    """Trigger a persona run via the Python runner. If `slug` is given, run that one; if `all`, run all personas."""
    import subprocess as _sp
    agency_root = "/root/piata-agency"
    cmd = ["python3", "-m", "personalities.runner"]
    if slug:
        cmd.append(slug)
    try:
        r = _sp.run(cmd, cwd=agency_root, capture_output=True, text=True, timeout=180)
        return {
            "ok": r.returncode == 0,
            "returncode": r.returncode,
            "stdout_tail": r.stdout[-2000:],
            "stderr_tail": r.stderr[-2000:],
            "command": " ".join(cmd),
        }
    except _sp.TimeoutExpired:
        return {"ok": False, "error": "runner timeout (180s)"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/agency/marketplace/stats")
async def marketplace_health():
    """Composite stats across personas + DB + disk."""
    conn = _pow_sq.connect("/root/piata-agency/agency.db")
    cur = conn.cursor()
    cur.execute("SELECT count(*), sum(case when status='active' then 1 else 0 end) FROM agents")
    total_agents, active_agents = cur.fetchone()
    cur.execute("SELECT COALESCE(SUM(chat_count),0) FROM agents")
    total_chats = cur.fetchone()[0]
    conn.close()

    persona_count = len(list(_PERSONAS_DIR.glob("*.yaml")))
    pow_count = sum(1 for d in _POW_DIR.iterdir() if d.is_dir() for f in d.glob("*.json"))

    try:
        df_out = _pow_sp.check_output(["df", "-h", "/"]).decode()
        disk_use_pct = int(df_out.splitlines()[1].split()[4].rstrip("%"))
    except Exception:
        disk_use_pct = None

    return {
        "status": "ok",
        "executor": "claude-code-2.1.177",
        "marketplace_url": "https://piata-ai.ro/agency/",
        "personas": {
            "total": persona_count,
            "executor_engine": "Claude Code acting as personas with marketplace awareness",
            "aware_in_marketplace": True,
        },
        "agents": {
            "total": total_agents,
            "active": active_agents,
            "total_chats": total_chats,
        },
        "proof_of_work": {
            "total_artifacts": pow_count,
            "verification": "HMAC-SHA256 with rotating orchestrator secret",
        },
        "disk_use_pct": disk_use_pct,
        "alert_if_disk_gt_80": disk_use_pct is not None and disk_use_pct > 80,
        "timestamp": _pow_dt.utcnow().isoformat() + "Z",
    }


# === Agent Chat Runtime ===

async def _run_agent_chat_demo(agent: dict, message: str, history: list = None) -> dict:
    """Run chat without DB logging — used for demo agent and public previews."""
    import aiohttp

    agent_id = agent["agent_id"]
    if isinstance(agent["tools"], str):
        tools_on_agent = [t.strip() for t in agent["tools"].split(",") if t.strip()] if agent["tools"] else []
    else:
        tools_on_agent = agent["tools"] or []
    tools = tools_on_agent

    tool_desc = []
    openai_tools = []
    for t in tools:
        for schema in TOOL_SCHEMAS.get(t, []):
            params = schema.get("parameters", {}).get("properties", {})
            name = schema.get("name", "")
            desc = schema.get("description", "")
            param_str = " | params: " + ", ".join(f"{k}:{v}" for k, v in params.items()) if params else ""
            tool_desc.append(f"- **{name}**: {desc}{param_str}")
            openai_properties = {}
            required = schema.get("parameters", {}).get("required", [])
            for pname, ptype_schema in params.items():
                if isinstance(ptype_schema, dict):
                    openai_properties[pname] = ptype_schema
                else:
                    openai_properties[pname] = {"type": "string", "description": str(ptype_schema)}
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": {
                        "type": "object",
                        "properties": openai_properties,
                        **({"required": list(required)} if required else {}),
                    }
                }
            })

    system_content = f"""{agent['system_prompt']}

## Available Tools
{chr(10).join(tool_desc)}

You have access to tools. Use them when needed to fulfill requests.
IMPORTANT: When you need to use a tool, output a JSON block in this EXACT format:
```json
{"tool": "tool_name", "args": {"param1": "value1"}}
```
Do NOT use native function calling. Use the JSON block format above.
"""

    messages = [{"role": "system", "content": system_content}]
    if history:
        messages.extend(history[-20:])
    messages.append({"role": "user", "content": message})

    proxy_url = os.environ.get("LLM_PROXY_URL", "http://127.0.0.1:5007")
    api_key = os.environ.get("OPENAI_API_KEY", "***")
    model = _resolve_model(agent["model"])

    try:
        async with aiohttp.ClientSession() as session:
            req_body = {
                "model": model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2000,
            }
            if openai_tools:
                req_body["tools"] = openai_tools
                req_body["tool_choice"] = "auto"

            async with session.post(
                f"{proxy_url}/v1/chat/completions",
                json=req_body,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                data = await resp.json()
                if "error" in data:
                    error_msg = data["error"].get("message", str(data["error"]))
                    raise Exception(error_msg)

                choice = data.get("choices", [{}])[0]
                msg = choice.get("message", {})
                content = msg.get("content", "")
                tool_calls = msg.get("tool_calls", None)

                has_valid_tool_calls = False
                if tool_calls:
                    has_valid_tool_calls = all(
                        json.loads(tc["function"]["arguments"]) != {}
                        if isinstance(tc.get("function", {}).get("arguments"), str)
                        else False
                        for tc in tool_calls
                    )

                if has_valid_tool_calls:
                    result = await _execute_tool_calls_and_respond(
                        tool_calls, messages, content, agent, proxy_url, api_key, model, session
                    )
                    return result

                extracted_tool_calls = _extract_tool_calls(content)
                if extracted_tool_calls:
                    result = await _execute_tools_and_respond(
                        extracted_tool_calls, messages, content, agent, proxy_url, api_key, model, session
                    )
                    return result

                return {
                    "role": "assistant",
                    "content": content,
                    "agent_id": agent_id,
                    "agent_name": agent["name"],
                }
    except Exception as e:
        logger.error(f"Demo chat error: {e}")
        return {"error": str(e), "agent_id": agent_id}


async def _run_agent_chat(agent: dict, message: str, history: list = None, agent_id: str = None) -> dict:
    """Run a chat through the agent's LLM with native OpenAI function calling."""
    import aiohttp

    # Parse tools (support both JSON array and comma-separated string)
    def parse_tools(s):
        if not s:
            return []
        s = s.strip()
        if not s:
            return []
        if s.startswith("["):
            try:
                return json.loads(s)
            except:
                return []
        return [t.strip() for t in s.split(",") if t.strip()]

    tools_on_agent = parse_tools(agent["tools"]) if isinstance(agent["tools"], str) else (agent["tools"] or [])
    
    # Enforce billing limits
    ok, reason = check_plan_limits(agent_id, requested_tools=tools_on_agent)
    if not ok:
        return {"error": reason, "agent_id": agent_id, "role": "error"}

    tools = tools_on_agent

    # Build tool descriptions for the system prompt
    tool_desc = []
    openai_tools = []
    for t in tools:
        for schema in TOOL_SCHEMAS.get(t, []):
            params = schema.get("parameters", {}).get("properties", {})
            name = schema.get("name", "")
            desc = schema.get("description", "")
            param_str = " | params: " + ", ".join(f"{k}:{v}" for k,v in params.items()) if params else ""
            tool_desc.append(f"- **{name}**: {desc}{param_str}")
            # Build OpenAI function calling format
            openai_properties = {}
            required = schema.get("parameters", {}).get("required", [])
            for pname, ptype_schema in params.items():
                if isinstance(ptype_schema, dict):
                    openai_properties[pname] = ptype_schema
                else:
                    openai_properties[pname] = {"type": "string", "description": str(ptype_schema)}
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": {
                        "type": "object",
                        "properties": openai_properties,
                        **({"required": list(required)} if required else {}),
                    }
                }
            })

    # When OpenAI tools are declared, prefer native function calling; the
    # JSON-in-text fallback is kept only as a last-resort escape hatch for
    # models/proxies that don't implement tool_calls correctly.
    if openai_tools:
        tools_guidance = (
            "You have access to tools. Call them via the native function-calling "
            "interface when needed. Only if the runtime cannot expose tool calls, "
            "emit a fenced JSON block in this exact format:\n"
            "```json\n{\"tool\": \"tool_name\", \"args\": {\"param\": \"value\"}}\n```"
        )
    else:
        tools_guidance = ""

    system_content = f"""{agent['system_prompt']}

## Available Tools
{chr(10).join(tool_desc)}

{tools_guidance}
""".rstrip() + "\n"

    messages = [{"role": "system", "content": system_content}]
    if history:
        messages.extend(history[-20:])
    messages.append({"role": "user", "content": message})

    # Log user message
    provisioner.log_chat(agent_id, "user", message)

    proxy_url = os.environ.get("LLM_PROXY_URL", "http://127.0.0.1:5007")
    api_key = os.environ.get("OPENAI_API_KEY", "***")
    model = _resolve_model(agent["model"])

    try:
        async with aiohttp.ClientSession() as session:
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
                llm_resp = await _llm_chat_completion(session, proxy_url, api_key, req_body)
            except RuntimeError as e:
                logger.error(f"Agent chat LLM call failed: {e}")
                err = "Chat backend is currently unavailable. See logs."
                provisioner.log_chat(agent_id, "assistant", err)
                return {"role": "assistant", "agent_id": agent_id,
                        "agent_name": agent["name"], "content": err}

            content = llm_resp["content"]
            tool_calls = llm_resp["tool_calls"]

            # Check if native tool_calls have valid args (models like minimax
            # often return tool_calls with empty arguments but write JSON in text)
            def _tc_args_nonempty(tc: dict) -> bool:
                """True if this tool_call has a non-empty JSON arguments string.
                Crash-safe: malformed JSON or non-str args return False so the
                JSON-in-text fallback can still recover the call."""
                args = (tc.get("function") or {}).get("arguments")
                if not isinstance(args, str) or not args.strip():
                    return False
                try:
                    parsed = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    return False
                return bool(parsed) and parsed != {}

            has_valid_tool_calls = bool(tool_calls) and all(_tc_args_nonempty(tc) for tc in tool_calls)

            if has_valid_tool_calls:
                result = await _execute_tool_calls_and_respond(
                    tool_calls, messages, content, agent, proxy_url, api_key, model, session
                )
                provisioner.log_chat(agent_id, "assistant", result["content"],
                                    json.dumps([t["function"]["name"] for t in tool_calls]))
                return result

            # Fallback: check for JSON-in-text format (native was empty or absent)
            extracted_tool_calls = _extract_tool_calls(content)
            if extracted_tool_calls:
                result = await _execute_tools_and_respond(
                    extracted_tool_calls, messages, content, agent, proxy_url, api_key, model, session
                )
                provisioner.log_chat(agent_id, "assistant", result["content"],
                                    json.dumps(extracted_tool_calls))
                return result

            provisioner.log_chat(agent_id, "assistant", content)
            return {
                "role": "assistant",
                "content": content,
                "agent_id": agent_id,
                "agent_name": agent["name"],
            }
    except Exception as e:
        logger.error(f"Agent chat error: {e}")
        # Don't leak raw exception strings (which may contain secrets) to callers.
        safe_msg = "I encountered an error processing that request. See server logs."
        provisioner.log_chat(agent_id, "assistant", safe_msg)
        return {"role": "assistant", "agent_id": agent_id,
                "agent_name": agent.get("name", ""), "content": safe_msg}


async def _execute_tool_calls_and_respond(tool_calls, messages, llm_content, agent,
                                          proxy_url, api_key, model, session):
    """Execute native OpenAI tool_calls and get final response."""
    import aiohttp
    from piata_tools.registry import get_registry
    registry = get_registry()
    results = []

    for tc in tool_calls:
        function_name = tc["function"]["name"]
        try:
            function_args = json.loads(tc["function"]["arguments"])
        except (json.JSONDecodeError, KeyError):
            function_args = {}
        try:
            # Bound each tool execution so a misbehaving tool/MCP server can't
            # stall the whole chat coroutine forever.
            result = await asyncio.wait_for(
                registry.call(function_name, **function_args),
                timeout=float(os.environ.get("PIATA_TOOL_TIMEOUT_SECONDS", "30")),
            )
            results.append({"tool": function_name, "result": result})
        except asyncio.TimeoutError:
            results.append({"tool": function_name, "error": "tool execution timed out"})
        except Exception as e:
            results.append({"tool": function_name, "error": str(e)})

    # Send tool results back to LLM
    tool_msg_lines = []
    for r in results:
        r_json = json.dumps(r.get("result", r.get("error")), indent=2, default=str)
        tool_msg_lines.append(f"**{r['tool']}** result:\n```json\n{r_json}\n```")

    tool_msg = "\n\n".join(tool_msg_lines)

    messages.append({"role": "assistant", "content": llm_content,
                     "tool_calls": [{"id": tc.get("id", f"call_{i}"),
                                       "type": "function",
                                       "function": {"name": tc["function"]["name"],
                                                      "arguments": tc["function"]["arguments"]}}
                                      for i, tc in enumerate(tool_calls)]})
    for i, r in enumerate(results):
        tool_call_id = tool_calls[i].get("id", f"call_{i}") if i < len(tool_calls) else f"call_{i}"
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps(r.get("result", r.get("error")), default=str)
        })

    try:
        resp = await _llm_chat_completion(
            session, proxy_url, api_key,
            {"model": model, "messages": messages, "temperature": 0.7, "max_tokens": 2000},
        )
    except RuntimeError as e:
        logger.error(f"Tool-followup LLM call failed: {e}")
        return {
            "role": "assistant",
            "content": "Error generating final response after tool calls. See logs.",
            "tool_calls": results,
            "agent_id": agent["agent_id"],
            "agent_name": agent["name"],
        }
    return {
        "role": "assistant",
        "content": resp["content"],
        "tool_calls": results,
        "agent_id": agent["agent_id"],
        "agent_name": agent["name"],
    }

def _extract_tool_calls(content: str) -> list:
    """Extract one or more JSON tool-call objects from LLM text.

    LLMs often return nested arguments, prose around the JSON, or multiple
    calls in one response. A greedy regex cannot reliably balance nested
    braces, so use JSONDecoder.raw_decode at each object boundary instead.
    Invalid/prose objects are ignored without aborting later valid calls.
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
            parsed.get("arguments", parsed.get("params", parsed.get("parameters", {})))
        )
        if not isinstance(tool_args, dict):
            continue
        calls.append(parsed)
    return calls


async def _execute_tools_and_respond(calls, messages, llm_content, agent,
                                     proxy_url, api_key, model, session):
    """Execute tool calls and get final response."""
    import aiohttp
    registry = get_registry()
    results = []

    def normalize_piata_args(tool_name: str, args: dict) -> dict:
        """Normalize common parameter name variations for piata tools."""
        if tool_name != "piata_post_ad":
            return args
        
        normalized = dict(args)
        
        # location -> city
        if "location" in normalized and "city" not in normalized:
            normalized["city"] = normalized.pop("location")
        
        # contact_info -> phone
        if "contact_info" in normalized and "phone" not in normalized:
            normalized["phone"] = normalized.pop("contact_info")
        
        # categories array -> category + subcategory
        if "categories" in normalized and isinstance(normalized["categories"], list):
            cats = normalized.pop("categories")
            if len(cats) >= 1 and "category" not in normalized:
                normalized["category"] = cats[0]
            if len(cats) >= 2 and "subcategory" not in normalized:
                normalized["subcategory"] = cats[1]
        
        # price string -> number
        if "price" in normalized and isinstance(normalized["price"], str):
            import re
            price_str = normalized["price"]
            # Extract numeric value from strings like "6200 EUR" or "€6,200"
            match = re.search(r'[\d,.]+', price_str.replace(',', ''))
            if match:
                normalized["price"] = float(match.group())
        
        return normalized

    for call in calls:
        tool_name = call["tool"]
        tool_args = call.get(
            "args",
            call.get("arguments", call.get("params", call.get("parameters", {})))
        )
        if not isinstance(tool_name, str) or not isinstance(tool_args, dict):
            results.append({"tool": str(tool_name), "error": "invalid tool call shape"})
            continue
        # Normalize piata tool arguments
        tool_args = normalize_piata_args(tool_name, tool_args)
        try:
            result = await asyncio.wait_for(
                registry.call(tool_name, **tool_args),
                timeout=float(os.environ.get("PIATA_TOOL_TIMEOUT_SECONDS", "30")),
            )
            results.append({"tool": tool_name, "result": result})
        except asyncio.TimeoutError:
            results.append({"tool": tool_name, "error": "tool execution timed out"})
        except Exception as e:
            results.append({"tool": tool_name, "error": str(e)})

    # Send results back to LLM
    tool_msg = "\n\n".join(
        f"**{r['tool']}** result:\n```json\n{json.dumps(r.get('result', r.get('error')), indent=2)}\n```"
        for r in results
    )

    messages.append({"role": "assistant", "content": llm_content})
    messages.append({"role": "user", "content": f"Tool results:\n{tool_msg}\n\nProvide your final answer."})

    try:
        resp = await _llm_chat_completion(
            session, proxy_url, api_key,
            {"model": model, "messages": messages, "temperature": 0.7, "max_tokens": 2000},
        )
    except RuntimeError as e:
        logger.error(f"Tool-followup LLM call failed (JSON-text path): {e}")
        return {
            "role": "assistant",
            "content": "Error generating final response after tool calls. See logs.",
            "tool_calls": results,
            "agent_id": agent["agent_id"],
            "agent_name": agent["name"],
        }
    return {
        "role": "assistant",
        "content": resp["content"],
        "tool_calls": results,
        "agent_id": agent["agent_id"],
        "agent_name": agent["name"],
    }


# === Legal Pages (for TikTok App Review) ===

TOS_HTML = """<!DOCTYPE html>
<html lang="ro">
<head><meta charset="UTF-8"><title>Termeni și Condiții — Piața AI</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>body{font-family:Inter,sans-serif;max-width:720px;margin:40px auto;padding:0 20px;background:#020202;color:#ddd;line-height:1.7}a{color:#f97316}h1{color:#fff}h2{color:#f97316;margin-top:32px}</style>
</head>
<body>
<h1>Termeni și Condiții — Piața AI</h1>
<p><em>Ultima actualizare: 9 iulie 2026</em></p>
<h2>1. Serviciul</h2>
<p>Piața AI (piata-ai.ro) este un marketplace online care folosește inteligență artificială pentru a facilita vânzarea și cumpărarea de produse.</p>
<h2>2. Conturi</h2>
<p>Pentru a publica anunțuri, utilizatorii trebuie să își creeze un cont. Datele personale sunt gestionate conform Politicii de Confidențialitate.</p>
<h2>3. Anunțuri</h2>
<p>Utilizatorii sunt responsabili pentru conținut. Este interzisă publicarea de produse ilegale sau care încalcă drepturile de proprietate intelectuală.</p>
<h2>4. Taxe</h2>
<p>Publicarea anunțurilor de bază este gratuită. Planurile premium încep de la €14/lună. Plățile se procesează prin Stripe.</p>
<h2>5. Contact</h2>
<p><a href="mailto:legal@piata-ai.ro">legal@piata-ai.ro</a></p>
</body>
</html>"""

PRIVACY_HTML = """<!DOCTYPE html>
<html lang="ro">
<head><meta charset="UTF-8"><title>Politica de Confidențialitate — Piața AI</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>body{font-family:Inter,sans-serif;max-width:720px;margin:40px auto;padding:0 20px;background:#020202;color:#ddd;line-height:1.7}a{color:#f97316}h1{color:#fff}h2{color:#f97316;margin-top:32px}</style>
</head>
<body>
<h1>Politica de Confidențialitate — Piața AI</h1>
<p><em>Ultima actualizare: 9 iulie 2026</em></p>
<h2>1. Date Colectate</h2>
<p>Nume, email, telefon (opțional), adresă IP. Plățile sunt procesate de Stripe — nu stocăm detalii card bancar.</p>
<h2>2. Scopul</h2>
<p>Gestionare cont, facilitare tranzacții, îmbunătățire serviciu, comunicări.</p>
<h2>3. Partajare</h2>
<p>Nu vindem date personale. Stripe procesează plățile conform propriei politici.</p>
<h2>4. Drepturile Tale</h2>
<p>Acces, corectare, ștergere date. Solicită exportul datelor oricând la <a href="mailto:privacy@piata-ai.ro">privacy@piata-ai.ro</a>.</p>
</body>
</html>"""


@app.get("/termeni")
async def termeni():
    return HTMLResponse(TOS_HTML)


@app.get("/privacy")
async def privacy():
    return HTMLResponse(PRIVACY_HTML)


# === TikTok App Icon ===

@app.get("/piata-ai-icon.png")
async def app_icon():
    from fastapi.responses import FileResponse
    icon_path = "/root/piata-agency/static/piata-ai-icon.png"
    if os.path.exists(icon_path):
        return FileResponse(icon_path, media_type="image/png")
    return JSONResponse(status_code=404, content={"error": "Icon not found"})


# === TikTok Domain Verification ===

@app.get("/tiktokYptO9qazzWaB1pAvyIY1whctWXQmaXWr.txt")
async def tiktok_verify_file():
    """TikTok domain verification file — root path."""
    return "2"


@app.get("/api/tiktok/auth/callback/")
async def tiktok_callback_verify():
    """TikTok verification at callback path."""
    return "2"


# === TikTok OAuth ===

TIKTOK_CLIENT_KEY = "awke87f3ntwrus14"
TIKTOK_CLIENT_SECRET = "IQdC2ODgZVyhG8ojHZKuRZE2e0etpgNl"
TIKTOK_REDIRECT_URI = os.environ.get("TIKTOK_REDIRECT_URI", "https://piata-ai.ro/api/tiktok/auth/callback")

# In-memory CSRF state store
_tiktok_states: dict = {}


@app.get("/api/tiktok/auth")
async def tiktok_auth():
    """Return TikTok OAuth2 authorization URL. User visits this to grant access."""
    from tiktok_oauth import build_auth_url
    result = build_auth_url(TIKTOK_REDIRECT_URI)
    _tiktok_states[result["state"]] = time.time()
    return {
        "auth_url": result["auth_url"],
        "redirect_uri": result["redirect_uri"],
        "scopes": result["scopes"],
        "note": "Visit the auth_url in a browser to authorize piata-ai.ro on TikTok.",
    }


@app.get("/api/tiktok/auth/callback")
async def tiktok_callback(request: Request):
    """Handle TikTok OAuth callback — exchange code for tokens."""
    from tiktok_oauth import exchange_code, get_token_status

    code = request.query_params.get("code")
    state = request.query_params.get("state")
    error = request.query_params.get("error")

    if error:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": f"TikTok returned error: {error}"}
        )

    if not code:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Missing authorization code"}
        )

    if state and state in _tiktok_states:
        del _tiktok_states[state]

    result = await exchange_code(code, TIKTOK_REDIRECT_URI)
    if result.get("success"):
        logger.info("TikTok OAuth successful! Tokens saved.")
        status = get_token_status()
        return {
            "success": True,
            "message": "TikTok authorization successful! Tokens saved and ready for posting.",
            "token_status": status,
        }
    else:
        return JSONResponse(status_code=400, content=result)


@app.get("/api/tiktok/auth/status")
async def tiktok_status():
    """Check TikTok OAuth token status."""
    from tiktok_oauth import get_token_status
    return get_token_status()


# === Chatbox HTML Template ===

CHATBOX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{AGENT_NAME}} — Piata AI</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0a0a; color: #e0e0e0; height: 100vh; display: flex; flex-direction: column; }
.header { background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 16px 24px; border-bottom: 1px solid #333; display: flex; align-items: center; gap: 12px; }
.header .logo { font-size: 24px; }
.header .info h2 { font-size: 16px; color: #fff; }
.header .info p { font-size: 12px; color: #888; }
.header .status { margin-left: auto; display: flex; align-items: center; gap: 6px; font-size: 12px; color: #4ade80; }
.header .status::before { content: ''; width: 8px; height: 8px; background: #4ade80; border-radius: 50%; animation: pulse 2s infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
.chat { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 16px; }
.msg { max-width: 80%; padding: 12px 16px; border-radius: 16px; line-height: 1.5; font-size: 14px; animation: fadeIn 0.3s; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
.msg.user { align-self: flex-end; background: linear-gradient(135deg, #667eea, #764ba2); color: #fff; border-bottom-right-radius: 4px; }
.msg.assistant { align-self: flex-start; background: #1e1e2e; border: 1px solid #333; border-bottom-left-radius: 4px; }
.msg.assistant .name { font-size: 11px; color: #667eea; margin-bottom: 4px; font-weight: 600; }
.msg.tool { align-self: flex-start; background: #1a1a1a; border: 1px solid #444; font-family: monospace; font-size: 12px; color: #fbbf24; max-width: 90%; }
.msg.tool .label { color: #888; font-size: 10px; margin-bottom: 4px; }
.msg pre { background: #0d0d0d; padding: 8px; border-radius: 8px; overflow-x: auto; margin: 8px 0; }
.msg code { font-family: 'Fira Code', monospace; font-size: 13px; }
.input-area { padding: 16px 24px; border-top: 1px solid #333; background: #111; display: flex; gap: 12px; }
.input-area input { flex: 1; background: #1e1e2e; border: 1px solid #444; border-radius: 12px; padding: 12px 16px; color: #fff; font-size: 14px; outline: none; transition: border-color 0.2s; }
.input-area input:focus { border-color: #667eea; }
.input-area button { background: linear-gradient(135deg, #667eea, #764ba2); border: none; border-radius: 12px; padding: 12px 24px; color: #fff; font-weight: 600; cursor: pointer; transition: opacity 0.2s; }
.input-area button:hover { opacity: 0.9; }
.input-area button:disabled { opacity: 0.5; cursor: not-allowed; }
.typing { display: none; align-self: flex-start; padding: 12px 16px; }
.typing span { display: inline-block; width: 8px; height: 8px; background: #667eea; border-radius: 50%; animation: bounce 1.4s infinite; margin: 0 2px; }
.typing span:nth-child(2) { animation-delay: 0.2s; }
.typing span:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce { 0%, 80%, 100% { transform: translateY(0); } 40% { transform: translateY(-8px); } }
.welcome { text-align: center; padding: 40px; color: #666; }
.welcome h3 { color: #667eea; margin-bottom: 8px; }
</style>
</head>
<body>
<div class="header">
  <div class="logo">🤖</div>
  <div class="info">
    <h2>{{AGENT_NAME}}</h2>
    <p>{{PURPOSE}}</p>
  </div>
  <div class="status">Online</div>
</div>
<div class="chat" id="chat">
  <div class="welcome">
    <h3>Welcome to {{AGENT_NAME}}</h3>
    <p>Powered by Piata AI — your custom AI agent with MCP tools</p>
  </div>
</div>
<div class="typing" id="typing"><span></span><span></span><span></span></div>
<div class="input-area">
  <input type="text" id="input" placeholder="Type a message..." autocomplete="off" />
  <button id="send" onclick="send()">Send</button>
</div>
<script>
const AGENT_ID = "{{AGENT_ID}}";
const TOKEN = "{{WEBHOOK_TOKEN}}";
const chat = document.getElementById("chat");
const input = document.getElementById("input");
const typing = document.getElementById("typing");
const sendBtn = document.getElementById("send");
let history = [];

input.addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } });

function addMsg(role, content, extra) {
  const div = document.createElement("div");
  div.className = "msg " + role;
  if (role === "assistant" && extra) {
    div.innerHTML = '<div class="name">' + extra + '</div>' + formatContent(content);
  } else if (role === "tool") {
    div.innerHTML = '<div class="label">🔧 Tool Result</div><pre><code>' + escapeHtml(content) + '</code></pre>';
  } else {
    div.innerHTML = formatContent(content);
  }
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function formatContent(text) {
  return text.replace(/```(\\w*)\\n([\\s\\S]*?)```/g, '<pre><code>$2</code></pre>')
             .replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>')
             .replace(/\\n/g, '<br>');
}

function escapeHtml(t) { return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

async function send() {
  const msg = input.value.trim();
  if (!msg) return;
  input.value = "";
  addMsg("user", msg);
  history.push({role: "user", content: msg});

  typing.style.display = "block";
  sendBtn.disabled = true;

  try {
    const resp = await fetch(`/webhook/${AGENT_ID}/${TOKEN}`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({message: msg}),
    });
    const data = await resp.json();
    typing.style.display = "none";
    sendBtn.disabled = false;

    if (data.error) {
      addMsg("assistant", "Error: " + data.error, "System");
    } else {
      if (data.tool_calls) {
        for (const tc of data.tool_calls) {
          addMsg("tool", JSON.stringify(tc.result || tc.error, null, 2));
        }
      }
      addMsg("assistant", data.content, data.agent_name || "{{AGENT_NAME}}");
      history.push({role: "assistant", content: data.content});
    }
  } catch (e) {
    typing.style.display = "none";
    sendBtn.disabled = false;
    addMsg("assistant", "Connection error: " + e.message, "System");
  }
}

// Focus input on load
input.focus();
</script>
</body>
</html>"""


# === Main ===

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
