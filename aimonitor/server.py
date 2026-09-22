import json
import mimetypes
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .sdk import dumps

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

TEXT_TYPES = ("text/", "application/javascript", "application/json", "image/svg+xml")


def serve(monitor):
    class Server(ThreadingHTTPServer):
        allow_reuse_address = False
        daemon_threads = True

    class Handler(BaseHTTPRequestHandler):
        server_version = "aimonitor/1.0"

        def _send(self, body, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj):
            try:
                body = dumps(obj).encode("utf-8")
            except Exception as e:
                body = json.dumps({"error": str(e)}).encode("utf-8")
            self._send(body, "application/json; charset=utf-8")

        def _asset(self, path):
            name = os.path.basename(path.split("?", 1)[0]) or "index.html"
            full = os.path.join(ASSETS, name)
            if not os.path.isfile(full):
                full = os.path.join(ASSETS, "index.html")
            try:
                with open(full, "rb") as f:
                    body = f.read()
            except OSError:
                return self._send("前端资源缺失".encode("utf-8"), "text/plain; charset=utf-8")
            ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
            if ctype.startswith(TEXT_TYPES):
                ctype += "; charset=utf-8"
            self._send(body, ctype)

        def do_GET(self):  # noqa: N802
            p = self.path.split("?", 1)[0]
            if p == "/api/state":
                return self._json(monitor.build_state())
            if p == "/api/graph":
                return self._json(monitor.build_graph())
            if p == "/api/ping":
                return self._json({"ok": True, "title": monitor.title})
            return self._asset(p)

        def do_POST(self):  # noqa: N802
            p = self.path.split("?", 1)[0]
            try:
                if p == "/api/control/start":
                    changed = monitor.run_start()
                elif p == "/api/control/stop":
                    changed = monitor.run_stop()
                else:
                    return self._json({"ok": False, "msg": "未知接口"})
                self._json({"ok": True, "changed": changed, "training": monitor.is_training()})
            except Exception as e:
                self._json({"ok": False, "msg": str(e)})

        def log_message(self, *a):
            return

    srv = None
    port = monitor.port
    for _ in range(20):
        try:
            srv = Server((monitor.host, port), Handler)
            break
        except OSError:
            port += 1
    if srv is None:
        raise OSError("无法绑定端口 %d" % monitor.port)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, port
