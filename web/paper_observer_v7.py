#!/usr/bin/env python3
"""Proculus V7: localhost-only, paper EVIDENCE observer. No trading/control writes.

Copy Proculus_Immersive_v7.html into web/ and run:
    python web/paper_observer_v7.py --repo . --port 8096
Then visit http://127.0.0.1:8096/.
Python standard library only. Never expose this development observer to the internet.
"""
from __future__ import annotations
from collections import deque
from datetime import datetime, timezone
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, unquote
import argparse
import json

ALLOW = {
    "setup_edge_stats": "reports/setup_edge_stats_latest.json",
    "edge_calibration_proposals": "reports/edge_calibration_proposals_latest.json",
    "edge_events_tail": "reports/edge_events.jsonl",
}
SECRETS = ("key", "secret", "password", "passphrase", "token", "private", "credential", "webhook", "authorization")
MAX_SIZE = 2 * 1024 * 1024

def mask(data, key="", depth=0):
    if any(x in key.lower() for x in SECRETS) or depth >= 16:
        return "[REDACTED]"
    if isinstance(data, dict):
        return {str(k): mask(v, str(k), depth + 1) for k, v in data.items()}
    if isinstance(data, list):
        return [mask(x, key, depth + 1) for x in data[:500]]
    return data if isinstance(data, (int, float, bool, type(None))) else str(data)[:1024]

def json_report(repo: Path, rel: str):
    target = (repo / rel).resolve()
    if not target.is_relative_to(repo) or not target.is_file():
        return {"available": False, "source": rel}
    if target.stat().st_size > MAX_SIZE:
        return {"available": False, "source": rel, "error": "oversized"}
    try:
        raw = target.read_bytes()
        obj = json.loads(raw)
        return {
            "available": True, "source": rel, "sha256": sha256(raw).hexdigest(),
            "modified_utc": datetime.fromtimestamp(target.stat().st_mtime, timezone.utc).isoformat(),
            "data": mask(obj),
        }
    except (OSError, UnicodeError, ValueError):
        return {"available": False, "source": rel, "error": "not_readable_or_invalid"}

def recent_events(repo: Path):
    rel = ALLOW["edge_events_tail"]
    target = (repo / rel).resolve()
    if not target.is_relative_to(repo) or not target.is_file():
        return {"available": False, "source": rel}
    rows, invalid = deque(maxlen=60), 0
    try:
        with target.open("rb") as f:
            for raw in f:
                if len(raw) > 16384:
                    invalid += 1
                    continue
                try:
                    item = json.loads(raw)
                    if isinstance(item, dict):
                        rows.append(mask(item))
                except (UnicodeError, ValueError):
                    invalid += 1
        return {"available": True, "source": rel, "entries": list(rows), "invalid_line_count": invalid}
    except OSError:
        return {"available": False, "source": rel, "error": "read_failed"}

def serve(repo: Path, site: Path, port: int):
    repo, site = repo.resolve(strict=True), site.resolve(strict=True)

    class Handler(BaseHTTPRequestHandler):
        server_version = "ProculusPaperReadOnly/0.3"
        def reply(self, data, code=200):
            raw = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(raw)
        def denied(self):
            self.reply({"error": "paper_read_only", "writes": False}, 405)
        do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = denied
        def do_GET(self):
            allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            if self.headers.get("Host") not in allowed_hosts:
                return self.reply({"error": "invalid_host"}, 403)
            origin = self.headers.get("Origin", "")
            if origin and origin not in {"http://" + host for host in allowed_hosts}:
                return self.reply({"error": "cross_origin_rejected"}, 403)
            u = urlsplit(self.path)
            if u.query or u.fragment:
                return self.reply({"error": "queries_disabled"}, 400)
            path = unquote(u.path)
            if path in {"/", "/Proculus_Immersive_v7.html"}:
                website = site / "Proculus_Immersive_v7.html"
                if not website.is_file():
                    return self.reply({"error": "copy_v7_website_to_web_folder"}, 404)
                raw = website.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("Content-Security-Policy", "default-src 'self' https://cdn.jsdelivr.net; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline'; connect-src 'self' https://cdn.jsdelivr.net; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'")
                self.end_headers()
                return self.wfile.write(raw)
            prefix = "/api/studio/v1/"
            if not path.startswith(prefix):
                return self.reply({"error": "not_found"}, 404)
            method = path[len(prefix):]
            if method == "health":
                return self.reply({"status": "ok", "mode": "paper_read_only", "utc": datetime.now(timezone.utc).isoformat()})
            if method == "capabilities":
                return self.reply({"mode": "local_paper_read_only", "verified": True, "writes": False,
                                   "live_exchange_actions": False,
                                   "capabilities": ["health", "config_redacted", "snapshot", "setup_edge_stats",
                                                    "edge_calibration_proposals", "edge_events_tail"]})
            if method == "config":
                report = json_report(repo, "config.json")
                if report.get("available") and isinstance(report["data"], dict):
                    report["data"] = mask({k: report["data"].get(k)
                          for k in ("paper_trading", "edge_learning", "live_readiness",
                                    "performance_evidence", "stochrsi_parallel", "llm_contract", "live_safety")})
                return self.reply(report)
            if method == "snapshot":
                return self.reply({"mode": "paper_read_only",
                          "observed_at_utc": datetime.now(timezone.utc).isoformat(),
                          "reports": {k: json_report(repo, ALLOW[k])
                                      for k in ("setup_edge_stats", "edge_calibration_proposals")},
                          "real_exchange_balance": None, "active_orders": None})
            if method == "edge_events_tail":
                return self.reply(recent_events(repo))
            if method in ALLOW:
                return self.reply(json_report(repo, ALLOW[method]))
            return self.reply({"error": "capability_not_allowlisted"}, 404)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Proculus paper observer running on http://127.0.0.1:{port}/ (GET only)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--site", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--port", type=int, default=8096)
    args = parser.parse_args()
    if not (args.repo / "config.json").is_file():
        parser.error("Expected Proculus checkout with config.json")
    serve(args.repo, args.site, args.port)
