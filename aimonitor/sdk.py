import collections
import hashlib
import json
import logging
import math
import threading
import time

import numpy as np

from . import imaging


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _resolve(v):
    return _safe(v) if callable(v) else v


def json_default(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, np.bool_):
        return bool(v)
    return str(v)


def json_finite(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, np.floating):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return json_finite(o.tolist())
    if isinstance(o, dict):
        return dict((k, json_finite(v)) for k, v in o.items())
    if isinstance(o, (list, tuple)):
        return [json_finite(v) for v in o]
    return o


def dumps(obj):
    return json.dumps(json_finite(obj), ensure_ascii=False, default=json_default, allow_nan=False)


def _num(v):
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return f if np.isfinite(f) else None
    return None


class LogBuffer:
    def __init__(self, capacity=400):
        self._lines = collections.deque(maxlen=int(capacity))
        self._lock = threading.Lock()
        self._seq = 0

    def write(self, *parts):
        text = " ".join(str(p) for p in parts)
        with self._lock:
            self._seq += 1
            self._lines.append({"i": self._seq, "t": time.time(), "text": text})
        return self

    def __call__(self, *parts):
        return self.write(*parts)

    def clear(self):
        with self._lock:
            self._lines.clear()

    def get(self):
        with self._lock:
            return list(self._lines)

    def __len__(self):
        with self._lock:
            return len(self._lines)


class _LogHandler(logging.Handler):
    def __init__(self, buffer, fmt="%(asctime)s %(message)s", datefmt="%H:%M:%S"):
        logging.Handler.__init__(self)
        self.buffer = buffer
        self.setFormatter(logging.Formatter(fmt, datefmt))

    def emit(self, record):
        try:
            self.buffer.write(self.format(record))
        except Exception:
            pass


class Monitor:
    def __init__(self, title="AI 训练监控", host="127.0.0.1", port=80, interval=1.0, history=600):
        self.title = title
        self.host = host
        self.port = port
        self.interval = interval
        self.history = history
        self._metrics = []
        self._health = []
        self._images = []
        self._hists = []
        self._tables = []
        self._matrices = []
        self._scatters = []
        self._bars = []
        self._logs = []
        self._alerts = []
        self._watchdogs = []
        self._wd = {}
        self._lastfinite = {}
        self._nanflag = {}
        self._graph_fn = None
        self._surface_fn = None
        self._params_fn = None
        self._info_fn = None
        self._start_fn = None
        self._stop_fn = None
        self._lock = threading.Lock()
        self._training = True
        self._series = {}
        self._hcache = {}
        self._gcache = {"t": 0.0, "version": 0, "summary": "", "data": None}
        self.actual_port = None
        self._server = None

    def metric(self, key, fn, label=None, color=None, fmt="auto", chart=True):
        self._metrics.append({
            "key": key, "label": label or key, "fn": fn,
            "color": color, "fmt": fmt, "chart": chart,
        })
        return self

    def health(self, key, fn, label=None, unit="", fmt="auto"):
        self._health.append({
            "key": key, "label": label or key, "fn": fn, "unit": unit, "fmt": fmt,
        })
        return self

    def image(self, key, fn, label=None, cmap=None, scale=4, group="图片"):
        self._images.append({
            "key": key, "label": label or key, "fn": fn,
            "cmap": cmap, "scale": scale, "group": group,
        })
        return self

    def histogram(self, key, fn, label=None, bins=40, refresh=5.0):
        self._hists.append({
            "key": key, "label": label or key, "fn": fn, "bins": bins, "refresh": refresh,
        })
        return self

    def table(self, key, fn, label=None, columns=None):
        self._tables.append({
            "key": key, "label": label or key, "fn": fn, "columns": columns,
        })
        return self

    def matrix(self, key, fn, label=None, cmap=None, xlabels=None, ylabels=None,
               fmt="f3", refresh=2.0):
        self._matrices.append({
            "key": key, "label": label or key, "fn": fn, "cmap": cmap,
            "xlabels": xlabels, "ylabels": ylabels, "fmt": fmt, "refresh": refresh,
        })
        return self

    def scatter(self, key, fn, label=None, xlabel="x", ylabel="y"):
        self._scatters.append({
            "key": key, "label": label or key, "fn": fn,
            "xlabel": xlabel, "ylabel": ylabel,
        })
        return self

    def bars(self, key, fn, label=None):
        self._bars.append({"key": key, "label": label or key, "fn": fn})
        return self

    def log(self, key, fn, label=None, limit=400):
        self._logs.append({
            "key": key, "label": label or key, "fn": fn, "limit": int(limit),
        })
        return self

    def attach_logger(self, logger, key="log", label="训练日志", level=logging.INFO, limit=400,
                      fmt="%(asctime)s %(message)s", datefmt="%H:%M:%S"):
        buf = LogBuffer(capacity=max(100, limit))
        handler = _LogHandler(buf, fmt=fmt, datefmt=datefmt)
        handler.setLevel(level)
        logger.addHandler(handler)
        self.log(key, buf.get, label=label, limit=limit)
        return buf

    def alert(self, key, fn, label=None, level="warn"):
        self._alerts.append({
            "key": key, "fn": fn, "label": label or key, "level": level,
        })
        return self

    def watchdog(self, key, fn, label=None, timeout=30.0, level="error"):
        self._watchdogs.append({
            "key": key, "fn": fn, "label": label or key,
            "timeout": float(timeout), "level": level,
        })
        return self

    def graph(self, fn):
        self._graph_fn = fn
        return self

    def surface(self, fn):
        self._surface_fn = fn
        return self

    def params(self, fn):
        self._params_fn = fn
        return self

    def info(self, fn):
        self._info_fn = fn
        return self

    def control(self, start=None, stop=None):
        self._start_fn = start
        self._stop_fn = stop
        return self

    def set_training(self, flag):
        with self._lock:
            self._training = bool(flag)

    def is_training(self):
        with self._lock:
            return self._training

    def run_start(self):
        if self._start_fn is None:
            return False
        changed = bool(self._start_fn())
        self.set_training(True)
        return changed

    def run_stop(self):
        if self._stop_fn is None:
            return False
        changed = bool(self._stop_fn())
        self.set_training(False)
        return changed

    def graph_meta(self):
        now = time.time()
        c = self._gcache
        if c["data"] is not None and now - c["t"] < max(self.interval, 0.5):
            return c
        if self._graph_fn is None:
            return c
        data = _safe(self._graph_fn)
        if not data:
            return c
        layers = data.get("layers") or []
        if len(layers) < 2:
            return c
        edges = data.get("edges") or []
        meta = [
            [l.get("name"), l.get("nodeCount"), l.get("params"), l.get("realCount"),
             l.get("detail"), l.get("out")]
            for l in layers
        ]
        shapes = [list(np.asarray(e).shape) for e in edges]
        key = json.dumps([meta, shapes], ensure_ascii=False, sort_keys=True)
        version = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)
        total_edges = 0
        for i in range(len(layers) - 1):
            total_edges += int(layers[i].get("nodeCount", 0)) * int(layers[i + 1].get("nodeCount", 0))
        total_nodes = sum(int(l.get("nodeCount", 0)) for l in layers)
        out = dict(data)
        out["version"] = version
        out["totalEdges"] = total_edges
        out["totalNodes"] = total_nodes
        c = {
            "t": now,
            "version": version,
            "summary": "%d 神经元 / %d 连线" % (total_nodes, total_edges),
            "data": out,
        }
        self._gcache = c
        return c

    def _images_payload(self):
        out = []
        for im in self._images:
            raw = _safe(im["fn"])
            if raw is None:
                continue
            try:
                arr = np.asarray(raw)
                lo = hi = None
                if arr.dtype != np.uint8:
                    arr, lo, hi = imaging.normalize(arr)
                out.append({
                    "key": im["key"], "label": im["label"], "scale": im["scale"],
                    "group": im.get("group") or "图片",
                    "w": int(arr.shape[1]), "h": int(arr.shape[0]),
                    "lo": lo, "hi": hi,
                    "png": imaging.to_png_data_uri(arr, im["cmap"]),
                })
            except Exception:
                continue
        return out

    def _cache_get(self, kind, key, refresh):
        c = self._hcache.get((kind, key))
        if c is not None and time.time() - c["t"] < refresh:
            return c["v"]
        return None

    def _cache_old(self, kind, key):
        c = self._hcache.get((kind, key))
        return None if c is None else c["v"]

    def _cache_put(self, kind, key, v):
        self._hcache[(kind, key)] = {"t": time.time(), "v": v}
        return v

    def _hists_payload(self):
        out = []
        for hs in self._hists:
            key = hs["key"]
            hit = self._cache_get("hist", key, hs.get("refresh", 5.0))
            if hit is not None:
                out.append(hit)
                continue
            raw = _safe(hs["fn"])
            if raw is None:
                old = self._cache_old("hist", key)
                if old is not None:
                    out.append(old)
                continue
            try:
                v = np.asarray(raw, np.float64).ravel()
                v = v[np.isfinite(v)]
                if v.size == 0:
                    continue
                cnt, edges = np.histogram(v, bins=hs["bins"])
                item = {
                    "key": key, "label": hs["label"],
                    "counts": cnt.tolist(),
                    "edges": [round(float(x), 6) for x in edges],
                    "min": float(v.min()), "max": float(v.max()),
                    "mean": float(v.mean()), "std": float(v.std()),
                    "n": int(v.size),
                }
                out.append(self._cache_put("hist", key, item))
            except Exception:
                continue
        return out

    def _tables_payload(self):
        out = []
        for t in self._tables:
            raw = _safe(t["fn"])
            if raw is None:
                continue
            try:
                if isinstance(raw, dict):
                    rows = raw.get("rows") or []
                    cols = _resolve(raw.get("columns")) or _resolve(t["columns"])
                else:
                    rows = list(raw)
                    cols = _resolve(t["columns"])
                rows = [list(r) if isinstance(r, (list, tuple, np.ndarray)) else [r] for r in rows]
                width = max([len(r) for r in rows] or [0])
                if width == 0:
                    continue
                for r in rows:
                    while len(r) < width:
                        r.append(None)
                if not cols:
                    cols = ["列 %d" % (i + 1) for i in range(width)]
                cols = [str(c) for c in list(cols)[:width]]
                while len(cols) < width:
                    cols.append("列 %d" % (len(cols) + 1))
                out.append({
                    "key": t["key"], "label": t["label"],
                    "columns": cols, "rows": rows, "n": len(rows),
                })
            except Exception:
                continue
        return out

    def _matrices_payload(self):
        out = []
        for mt in self._matrices:
            key = mt["key"]
            hit = self._cache_get("matrix", key, mt.get("refresh", 2.0))
            if hit is not None:
                out.append(hit)
                continue
            raw = _safe(mt["fn"])
            if raw is None:
                old = self._cache_old("matrix", key)
                if old is not None:
                    out.append(old)
                continue
            try:
                a = np.asarray(raw, np.float64)
                if a.ndim != 2 or a.size == 0:
                    continue
                a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
                xs = _resolve(mt["xlabels"])
                ys = _resolve(mt["ylabels"])
                item = {
                    "key": key, "label": mt["label"],
                    "h": int(a.shape[0]), "w": int(a.shape[1]),
                    "data": np.round(a, 6).tolist(),
                    "min": float(a.min()), "max": float(a.max()),
                    "xlabels": None if xs is None else [str(v) for v in xs],
                    "ylabels": None if ys is None else [str(v) for v in ys],
                    "cmap": mt["cmap"], "fmt": mt["fmt"],
                }
                out.append(self._cache_put("matrix", key, item))
            except Exception:
                continue
        return out

    def _scatters_payload(self):
        out = []
        for sc in self._scatters:
            raw = _safe(sc["fn"])
            if raw is None:
                continue
            try:
                groups = []
                if isinstance(raw, dict):
                    groups = [str(g) for g in (raw.get("groups") or [])]
                    src = raw.get("points") or []
                else:
                    src = raw
                points = []
                for p in src:
                    if isinstance(p, dict):
                        x = _num(p.get("x"))
                        y = _num(p.get("y"))
                        g = p.get("group")
                    else:
                        p = list(p)
                        x = _num(p[0]) if len(p) > 0 else None
                        y = _num(p[1]) if len(p) > 1 else None
                        g = p[2] if len(p) > 2 else 0
                    if x is None or y is None:
                        continue
                    if isinstance(g, str):
                        if g not in groups:
                            groups.append(g)
                        gi = groups.index(g)
                    else:
                        gi = int(g or 0)
                    points.append([x, y, gi])
                    if len(points) >= 40000:
                        break
                if not points:
                    continue
                if not groups:
                    groups = ["点"]
                out.append({
                    "key": sc["key"], "label": sc["label"],
                    "xlabel": sc["xlabel"], "ylabel": sc["ylabel"],
                    "points": points, "groups": groups, "n": len(points),
                })
            except Exception:
                continue
        return out

    def _bars_payload(self):
        out = []
        for b in self._bars:
            raw = _safe(b["fn"])
            if raw is None:
                continue
            try:
                items = raw.get("bars") if isinstance(raw, dict) else list(raw)
                bars = []
                for k, it in enumerate(items or []):
                    if isinstance(it, dict):
                        lab = it.get("label")
                        val = _num(it.get("value"))
                    elif isinstance(it, (list, tuple, np.ndarray)) and len(it) >= 2:
                        lab = it[0]
                        val = _num(it[1])
                    else:
                        lab = k
                        val = _num(it)
                    if val is None:
                        continue
                    bars.append({"label": str(lab), "value": val})
                if not bars:
                    continue
                out.append({"key": b["key"], "label": b["label"], "bars": bars, "n": len(bars)})
            except Exception:
                continue
        return out

    def _logs_payload(self):
        out = []
        for lg in self._logs:
            raw = _safe(lg["fn"])
            if raw is None:
                continue
            try:
                lines = []
                for it in list(raw)[-lg["limit"]:]:
                    if isinstance(it, dict):
                        lines.append({
                            "i": int(it.get("i", 0)),
                            "t": float(it.get("t", 0.0)),
                            "text": str(it.get("text", "")),
                        })
                    else:
                        lines.append({"i": 0, "t": 0.0, "text": str(it)})
                out.append({
                    "key": lg["key"], "label": lg["label"],
                    "lines": lines, "n": len(lines),
                })
            except Exception:
                continue
        return out

    def _alerts_payload(self, metric_alerts):
        now = time.time()
        out = list(metric_alerts)
        for a in self._alerts:
            msg = _safe(a["fn"])
            if msg:
                out.append({
                    "key": a["key"], "label": a["label"],
                    "level": a["level"], "msg": str(msg),
                })
        training = self.is_training()
        for w in self._watchdogs:
            st = self._wd.get(w["key"])
            if st is None:
                st = {"v": None, "t": now, "stall": 0.0}
                self._wd[w["key"]] = st
            v = _num(_safe(w["fn"]))
            if v is None:
                continue
            if st["v"] is None or v != st["v"]:
                st["v"] = v
                st["t"] = now
                st["stall"] = 0.0
            else:
                st["stall"] = now - st["t"]
            if training and st["stall"] >= w["timeout"]:
                out.append({
                    "key": "watchdog:" + w["key"], "label": w["label"], "level": w["level"],
                    "msg": "%s 已 %.0f 秒没有变化（阈值 %.0f 秒），训练可能卡死" % (
                        w["label"], st["stall"], w["timeout"]),
                })
        return out

    def build_state(self):
        now = time.time()
        metrics = []
        metric_alerts = []
        for m in self._metrics:
            v = _num(_safe(m["fn"]))
            if v is None:
                if m["key"] in self._lastfinite and not self._nanflag.get(m["key"]):
                    self._nanflag[m["key"]] = True
                    metric_alerts.append({
                        "key": "metric_nan:" + m["key"], "label": m["label"], "level": "error",
                        "msg": "%s 变成非法值（NaN / Inf），最近一次有效值 %s" % (
                            m["label"], self._lastfinite[m["key"]]),
                    })
            else:
                self._nanflag[m["key"]] = False
                self._lastfinite[m["key"]] = v
            metrics.append({
                "key": m["key"], "label": m["label"], "value": v,
                "color": m["color"], "fmt": m["fmt"], "chart": m["chart"],
            })
            if v is not None:
                s = self._series.setdefault(m["key"], [])
                s.append([round(now, 2), v])
                if len(s) > self.history:
                    del s[0:len(s) - self.history]
        g = self.graph_meta()
        acts = []
        if g["data"]:
            acts = [l.get("nodes") or [] for l in g["data"].get("layers") or []]
        return {
            "title": self.title,
            "version": 1,
            "interval": self.interval,
            "training": self.is_training(),
            "info": (_safe(self._info_fn) if self._info_fn else None) or {},
            "metrics": metrics,
            "series": self._series,
            "health": [
                {"key": h["key"], "label": h["label"], "unit": h["unit"],
                 "fmt": h["fmt"], "value": _num(_safe(h["fn"]))}
                for h in self._health
            ],
            "images": self._images_payload(),
            "histograms": self._hists_payload(),
            "tables": self._tables_payload(),
            "matrices": self._matrices_payload(),
            "scatters": self._scatters_payload(),
            "bars": self._bars_payload(),
            "logs": self._logs_payload(),
            "surface": _safe(self._surface_fn) if self._surface_fn else None,
            "params": _safe(self._params_fn) if self._params_fn else None,
            "graph": {"version": g["version"], "summary": g["summary"]},
            "acts": acts,
            "alerts": self._alerts_payload(metric_alerts),
        }

    def build_graph(self):
        g = self.graph_meta()
        return g["data"] or {"ready": 0, "version": 0}

    def start(self):
        from .server import serve

        self._server, self.actual_port = serve(self)
        return self.actual_port

    def stop(self):
        if self._server is not None:
            self._server.shutdown()
            self._server = None
