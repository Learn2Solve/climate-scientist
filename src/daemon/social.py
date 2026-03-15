"""Agent-to-agent communication - local relay and federation.

Mode 1: Local relay - direct SQLite writes between agents on the same machine.
Mode 2: Federation - HTTP relay for cross-machine communication.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ulid import ULID

log = logging.getLogger("daemon.social")


# ---------------------------------------------------------------------------
# Mode 1: Local relay (same machine, direct SQLite access)
# ---------------------------------------------------------------------------


def send_local(sender_name: str, recipient_db_path: Path, content: str) -> str:
    """Write a message directly to another agent's inbox_messages table."""
    if not recipient_db_path.exists():
        raise ValueError(f"Recipient database not found: {recipient_db_path}")

    msg_id = str(ULID())
    conn = sqlite3.connect(str(recipient_db_path))
    try:
        conn.execute(
            "INSERT OR IGNORE INTO inbox_messages (id, from_agent, content, received_at) "
            "VALUES (?, ?, ?, ?)",
            (msg_id, sender_name, content, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()

    return msg_id


def poll_inbox(db: Any, limit: int = 10) -> list[dict[str, Any]]:
    """Get unprocessed messages from own inbox."""
    return db.get_unprocessed_inbox(limit=limit)


def mark_processed(db: Any, message_id: str) -> None:
    """Mark a message as processed."""
    db.mark_inbox_processed(message_id)


# ---------------------------------------------------------------------------
# Mode 2: Federation (cross-machine HTTP)
# ---------------------------------------------------------------------------


def send_remote(
    sender_name: str,
    recipient_url: str,
    recipient_name: str,
    content: str,
    shared_secret: str = "",
    topic: str = "",
) -> dict[str, Any]:
    """Send a message to a remote agent via HTTP POST."""
    import urllib.request
    import urllib.error

    timestamp = datetime.now(timezone.utc).isoformat()
    payload = {
        "from": sender_name,
        "to": recipient_name,
        "content": content,
        "topic": topic,
        "timestamp": timestamp,
        "reply_to": None,
    }

    # Sign with HMAC-SHA256 if shared secret provided
    if shared_secret:
        sig_data = f"{sender_name}:{recipient_name}:{content}:{timestamp}"
        signature = hmac.new(
            shared_secret.encode("utf-8"),
            sig_data.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        payload["signature"] = signature

    body = json.dumps(payload).encode("utf-8")
    url = f"{recipient_url.rstrip('/')}/api/v1/messages"

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        log.warning("Failed to send message to %s: %s", url, exc)
        return {"ok": False, "error": str(exc)}


def broadcast_to_peers(
    sender_name: str,
    db: Any,
    knowledge_entry: dict[str, Any],
) -> list[str]:
    """Broadcast a knowledge finding to all known peers. Returns list of peer names sent to."""
    peers = db.get_peers()
    sent_to: list[str] = []

    content = json.dumps({
        "type": "finding",
        "knowledge_id": knowledge_entry.get("id", ""),
        "topic": knowledge_entry.get("topic", ""),
        "content": knowledge_entry.get("content", ""),
        "source": knowledge_entry.get("source", ""),
        "confidence": knowledge_entry.get("confidence", 0.5),
    })

    for peer in peers:
        url = peer.get("url", "")
        if not url:
            continue

        try:
            if url.startswith("local:"):
                # Local peer
                db_path = Path(url.removeprefix("local:"))
                send_local(sender_name, db_path, content)
            else:
                # Remote peer
                send_remote(sender_name, url, peer["name"], content)
            sent_to.append(peer["name"])
        except Exception as exc:
            log.warning("Failed to broadcast to peer %s: %s", peer["name"], exc)

    return sent_to


# ---------------------------------------------------------------------------
# Federation HTTP server (for receiving messages)
# ---------------------------------------------------------------------------


class SocialRelay:
    """Simple HTTP server for receiving messages from remote agents.

    Start with relay.start() to listen on a port.
    Other agents POST messages to /api/v1/messages.
    """

    def __init__(self, listen_port: int, db: Any, shared_secret: str = "") -> None:
        self.port = listen_port
        self.db = db
        self.shared_secret = shared_secret
        self._server = None

    async def start(self) -> None:
        """Start the HTTP relay server."""
        try:
            from aiohttp import web
        except ImportError:
            log.warning("aiohttp not installed - federation relay disabled")
            return

        app = web.Application()
        app.router.add_post("/api/v1/messages", self._handle_message)
        app.router.add_get("/api/v1/peers", self._handle_list_peers)
        app.router.add_post("/api/v1/reviews", self._handle_review)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self.port)
        await site.start()
        self._server = runner
        log.info("Social relay listening on port %d", self.port)

    async def stop(self) -> None:
        if self._server:
            await self._server.cleanup()
            self._server = None

    async def _handle_message(self, request: Any) -> Any:
        from aiohttp import web

        try:
            data = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid JSON"}, status=400)

        sender = data.get("from", "unknown")
        content = data.get("content", "")
        timestamp = data.get("timestamp", datetime.now(timezone.utc).isoformat())
        reply_to = data.get("reply_to")

        # Verify signature if shared secret configured
        if self.shared_secret and "signature" in data:
            sig_data = f"{sender}:{data.get('to', '')}:{content}:{timestamp}"
            expected = hmac.new(
                self.shared_secret.encode("utf-8"),
                sig_data.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            if not hmac.compare_digest(data["signature"], expected):
                return web.json_response({"ok": False, "error": "invalid signature"}, status=403)

        msg_id = str(ULID())
        self.db.insert_inbox_message({
            "id": msg_id,
            "from_agent": sender,
            "content": content,
            "received_at": timestamp,
            "reply_to": reply_to,
        })

        # Update peer last_seen
        peer = self.db.get_peer_by_name(sender)
        if peer:
            self.db.upsert_peer({**peer, "last_seen": timestamp})

        return web.json_response({"ok": True, "id": msg_id})

    async def _handle_list_peers(self, request: Any) -> Any:
        from aiohttp import web

        peers = self.db.get_peers()
        return web.json_response({
            "ok": True,
            "peers": [
                {
                    "name": p["name"],
                    "topic": p.get("topic", ""),
                    "url": p["url"],
                }
                for p in peers
            ],
        })

    async def _handle_review(self, request: Any) -> Any:
        from aiohttp import web

        try:
            data = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid JSON"}, status=400)

        review_id = str(ULID())
        self.db.insert_review({
            "id": review_id,
            "from_agent": data.get("from", "unknown"),
            "knowledge_id": data.get("knowledge_id", ""),
            "score": float(data.get("score", 0.5)),
            "comment": data.get("comment", ""),
        })

        # Update knowledge confidence based on reviews
        kid = data.get("knowledge_id", "")
        if kid:
            reviews = self.db.get_reviews_for_knowledge(kid)
            if reviews:
                avg_score = sum(r["score"] for r in reviews) / len(reviews)
                # Update the knowledge entry confidence (via direct SQL)
                self.db._conn.execute(
                    "UPDATE knowledge SET confidence = ? WHERE id = ?",
                    (avg_score, kid),
                )
                self.db._conn.commit()

        return web.json_response({"ok": True, "id": review_id})
