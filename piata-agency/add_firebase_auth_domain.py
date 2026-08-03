#!/usr/bin/env python3
"""
add_firebase_auth_domain.py — add Firebase Auth authorized domains via the
Identity Toolkit Admin API using the project service account.

Why this exists: Firebase's signInWithPopup / signInWithRedirect flows reject
any page origin that is not in the project's "Authorized domains" list, and
that check is enforced server-side by Google — no app code can bypass it.
This script performs the console step (Firebase Console → Authentication →
Settings → Authorized domains) programmatically, so any piata subdomain can be
enabled in seconds without opening the console.

Usage:
    python3 add_firebase_auth_domain.py real.piata-ai.ro www.real.piata-ai.ro
    python3 add_firebase_auth_domain.py --remove old.piata-ai.ro

Requires: PyJWT (installed — server.py depends on it) and a service account
with Firebase Admin / Owner access (default /root/.firebase/piata-agency-sa.json,
override with FIREBASE_SERVICE_ACCOUNT_PATH).
"""
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import jwt

SA_PATH = os.environ.get(
    "FIREBASE_SERVICE_ACCOUNT_PATH",
    os.path.expanduser("/root/.firebase/piata-agency-sa.json"),
)
PROJECT = os.environ.get("FIREBASE_PROJECT_ID", "gen-lang-client-0680189072")
SCOPES = (
    "https://www.googleapis.com/auth/cloud-platform "
    "https://www.googleapis.com/auth/firebase "
    "https://www.googleapis.com/auth/identitytoolkit"
)


def _access_token(sa: dict) -> str:
    now = int(time.time())
    assertion = jwt.encode(
        {
            "iss": sa["client_email"],
            "scope": SCOPES,
            "aud": sa["token_uri"],
            "iat": now,
            "exp": now + 3600,
        },
        sa["private_key"],
        algorithm="RS256",
    )
    body = urllib.parse.urlencode(
        {"grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer", "assertion": assertion}
    ).encode()
    req = urllib.request.Request(
        sa["token_uri"], data=body, headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    return json.loads(urllib.request.urlopen(req, timeout=20).read())["access_token"]


def _api(token: str, method: str, url: str, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=20)
        return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code}: {e.read().decode()[:600]}") from e


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    remove = "--remove" in sys.argv
    if not args:
        print(__doc__)
        return 2

    if not os.path.exists(SA_PATH):
        print(f"Service account not found: {SA_PATH}")
        return 1
    sa = json.load(open(SA_PATH))
    token = _access_token(sa)

    base = f"https://identitytoolkit.googleapis.com/admin/v2/projects/{PROJECT}/config"
    _, cfg = _api(token, "GET", base)
    domains = list(cfg.get("authorizedDomains", []))

    changed = []
    for d in args:
        d = d.strip().lower()
        if remove:
            if d in domains:
                domains.remove(d)
                changed.append(("-", d))
        else:
            if d not in domains:
                domains.append(d)
                changed.append(("+", d))

    if not changed:
        print("No changes needed — requested domains already in the desired state.")
        return 0

    _, out = _api(token, "PATCH", base + "?updateMask=authorizedDomains", {"authorizedDomains": domains})
    result = out.get("authorizedDomains", [])
    for sign, d in changed:
        print(f"{sign} {d}")
    print(f"Done — authorized domains now: {len(result)} total")
    if "real.piata-ai.ro" not in result and not remove:
        print("NOTE: changes may take a minute to propagate to the SDK endpoint.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
