import hashlib
import json
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

    def _hists_payload(self):
        now = time.time()
        out = []
        for hs in self._hists:
            key = hs["key"]
            c = self._hcache.get(key)
            if c is not None and now - c["t"] < hs.get("refresh", 5.0):
                out.append(c["v"])
                continue
            raw = _safe(hs["fn"])
            if raw is None:
                if c is not None:
                    out.append(c["v"])
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
                self._hcache[key] = {"t": now, "v": item}
                out.append(item)
            except Exception:
                continue
        return out

    def build_state(self):
        now = time.time()
        metrics = []
        for m in self._metrics:
            v = _num(_safe(m["fn"]))
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
            "surface": _safe(self._surface_fn) if self._surface_fn else None,
            "params": _safe(self._params_fn) if self._params_fn else None,
            "graph": {"version": g["version"], "summary": g["summary"]},
            "acts": acts,
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
