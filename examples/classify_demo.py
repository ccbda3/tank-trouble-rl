import ctypes
import logging
import os
import sys
import threading
import time

import numpy as np

if not ctypes.windll.kernel32.GetModuleHandleW("msvcp140.dll"):
    _sys32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    try:
        ctypes.WinDLL(os.path.join(_sys32, "msvcp140.dll"))
    except OSError:
        pass

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aimonitor import LogBuffer, Monitor

CLASSES = ["螺旋 A", "螺旋 B", "螺旋 C", "螺旋 D"]
CLASS_RGB = np.array([[235, 90, 80], [90, 200, 125], [95, 155, 240], [235, 190, 85]], np.uint8)
PORT = 8080
STEPS = 4000
BATCH = 256
LR = 3e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("demo")


def make_spiral(n_per_class=300, noise=0.09, turns=1.6, seed=7):
    rs = np.random.RandomState(seed)
    xs, ys = [], []
    k = len(CLASSES)
    for c in range(k):
        t = np.sqrt(rs.rand(n_per_class)) * turns * 2 * np.pi
        r = t / (turns * 2 * np.pi)
        ang = t + c * (2 * np.pi / k)
        x = r * np.cos(ang) + rs.randn(n_per_class) * noise
        y = r * np.sin(ang) + rs.randn(n_per_class) * noise
        xs.append(np.stack([x, y], 1))
        ys.append(np.full(n_per_class, c, np.int64))
    X = np.concatenate(xs).astype(np.float32)
    Y = np.concatenate(ys)
    idx = rs.permutation(len(X))
    return X[idx], Y[idx]


class Net(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, len(CLASSES))

    def forward(self, x):
        feat = self.body(x)
        return self.head(feat), feat


class Demo:
    def __init__(self):
        self.X, self.Y = make_spiral()
        self.net = Net().to(DEVICE)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.loss_fn = nn.CrossEntropyLoss()
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = None
        self.step = 0
        self.loss = 0.0
        self.acc = 0.0
        self.lr = LR
        self.grad_norm = 0.0
        self.snapshot = {}
        self.pred = None
        self.eval_at = 0.0
        self.events = LogBuffer(capacity=200)

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            return False
        self.stop.clear()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        self.events.write("手动启动训练（从第 %d 步继续）" % self.step)
        return True

    def halt(self):
        if self.thread is None or not self.thread.is_alive():
            return False
        self.stop.set()
        self.thread.join(timeout=5)
        self.events.write("手动停止训练（当前第 %d 步）" % self.step)
        return True

    def loop(self):
        x = torch.from_numpy(self.X).to(DEVICE)
        y = torch.from_numpy(self.Y).to(DEVICE)
        log.info("开始训练：%d 样本 · %d 类 · 设备 %s", len(x), len(CLASSES), DEVICE)
        while not self.stop.is_set() and self.step < STEPS:
            idx = torch.randint(0, len(x), (BATCH,), device=DEVICE)
            xb, yb = x[idx], y[idx]
            out, _ = self.net(xb)
            loss = self.loss_fn(out, yb)
            self.opt.zero_grad()
            loss.backward()
            self.grad_norm = float(torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0))
            self.opt.step()
            self.step += 1
            with torch.no_grad():
                a = float((out.argmax(1) == yb).float().mean())
            with self.lock:
                self.loss = float(loss.detach())
                self.acc = a
            if self.step % 25 == 0:
                self.snapshot = {n: p.grad.detach().to("cpu", copy=True).float()
                                 for n, p in self.net.named_parameters() if p.grad is not None}
            if time.time() - self.eval_at > 2.0:
                self.eval_at = time.time()
                self.evaluate(log_it=True)
            time.sleep(0.001)
        self.evaluate(log_it=True)
        log.info("训练结束：步数 %d · 全量准确率 %.4f", self.step, self.acc)

    def evaluate(self, log_it=False):
        self.net.eval()
        with torch.no_grad():
            out, feat = self.net(torch.from_numpy(self.X).to(DEVICE))
            pred = out.argmax(1).cpu().numpy()
            self.pred = pred
            self.acc = float((pred == self.Y).mean())
            self.feat = feat.cpu().numpy()
        self.net.train()
        if log_it:
            log.info("step %d · loss %.4f · acc %.4f", self.step, self.loss, self.acc)

    def confusion(self):
        if self.pred is None:
            return None
        k = len(CLASSES)
        cm = np.zeros((k, k), np.int64)
        for t, p in zip(self.Y, self.pred):
            cm[t, p] += 1
        return cm

    def report(self):
        cm = self.confusion()
        if cm is None:
            return []
        rows = []
        for c in range(len(CLASSES)):
            tp = float(cm[c, c])
            fp = float(cm[:, c].sum() - tp)
            fn = float(cm[c, :].sum() - tp)
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            rows.append([CLASSES[c], int(cm[c].sum()), prec, rec, f1])
        return rows

    def projection(self):
        if not hasattr(self, "feat"):
            return []
        f = self.feat - self.feat.mean(0, keepdims=True)
        u, s, vt = np.linalg.svd(f, full_matrices=False)
        pts = f @ vt[:2].T
        return [[float(pts[i, 0]), float(pts[i, 1]), CLASSES[int(self.Y[i])]] for i in range(len(pts))]

    def boundary(self, res=120):
        pad = 0.6
        x0, x1 = self.X[:, 0].min() - pad, self.X[:, 0].max() + pad
        y0, y1 = self.X[:, 1].min() - pad, self.X[:, 1].max() + pad
        gx, gy = np.meshgrid(np.linspace(x0, x1, res), np.linspace(y0, y1, res))
        grid = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)
        self.net.eval()
        with torch.no_grad():
            logits, _ = self.net(torch.from_numpy(grid).to(DEVICE))
            conf = torch.softmax(logits, 1).max(1).values.cpu().numpy()
            cls = logits.argmax(1).cpu().numpy()
        self.net.train()
        rgb = CLASS_RGB[cls].astype(np.float32)
        shade = (0.45 + 0.55 * conf)[:, None]
        img = (rgb * shade).reshape(res, res, 3).astype(np.uint8)
        return img[::-1]

    def grad_bars(self):
        return [[n if len(n) < 22 else n[:21] + "…", float(g.pow(2).sum().sqrt())]
                for n, g in self.snapshot.items()]

    def layer_table(self):
        rows = []
        for n, p in self.net.named_parameters():
            pf = p.detach().float()
            g = self.snapshot.get(n)
            rows.append([n, "×".join(str(int(s)) for s in tuple(p.shape)), int(p.numel()),
                         float(pf.mean()), float(pf.std()),
                         float(g.pow(2).sum().sqrt()) if g is not None else 0.0])
        return rows


def build(demo):
    mon = Monitor(title="图像分类 · 训练监控示例", port=PORT, interval=1.0, history=900)
    mon.attach_logger(log, key="train_log", label="训练日志（logging 实时镜像）")

    mon.info(lambda: {
        "任务": "四类二维螺旋分类",
        "样本": "%d" % len(demo.X),
        "设备": str(DEVICE).upper(),
        "模型": "MLP 2-64-64-4",
    })
    mon.metric("loss", lambda: demo.loss, label="交叉熵 Loss", fmt="f4", color="#ff8f5a")
    mon.metric("acc", lambda: demo.acc, label="全量准确率", fmt="pct", color="#3ddc84")
    mon.metric("step", lambda: demo.step, label="已训练步数", fmt="int")
    mon.metric("lr", lambda: demo.lr, label="学习率", fmt="sci", color="#8fb9ff")
    mon.metric("gnorm", lambda: demo.grad_norm, label="梯度范数", fmt="f3", color="#ffd36a")

    mon.health("acc", lambda: demo.acc, label="准确率", fmt="pct")
    mon.health("loss", lambda: demo.loss, label="当前 Loss", fmt="f4")
    mon.health("params", lambda: sum(p.numel() for p in demo.net.parameters()),
               label="参数量", fmt="int")
    mon.health("err", lambda: 1.0 - demo.acc, label="错误率", fmt="pct")

    mon.image("boundary", demo.boundary, label="决策边界（网格逐点前向的真实预测）", scale=3, group="图片")

    mon.matrix("confusion", demo.confusion, label="混淆矩阵（行=真实 列=预测）",
               xlabels=CLASSES, ylabels=CLASSES, fmt="int", refresh=2.0)
    mon.matrix("corr", lambda: np.corrcoef(demo.net.head.weight.detach().cpu().numpy()),
               label="分类头权重的相关性矩阵", xlabels=CLASSES, ylabels=CLASSES, fmt="f3", refresh=4.0)

    mon.table("report", demo.report, label="每类精确率 / 召回率 / F1",
              columns=["类别", "样本数", "精确率", "召回率", "F1"])
    mon.table("layers", demo.layer_table, label="逐层权重统计",
              columns=["参数", "形状", "参数量", "均值", "标准差", "梯度范数"])

    mon.scatter("proj", demo.projection, label="倒数第二层特征 PCA 投影（按真实类别着色）",
                xlabel="主成分 1", ylabel="主成分 2")
    mon.bars("grad", demo.grad_bars, label="逐层梯度范数（真实值）")

    mon.log("events", demo.events.get, label="训练事件（LogBuffer 手工写入）", limit=200)
    mon.control(start=demo.start, stop=demo.halt)
    return mon


def main():
    demo = Demo()
    demo.evaluate()
    demo.events.write("示例进程启动，数据与模型均为现场真实计算")
    mon = build(demo)
    port = mon.start()
    log.info("监控面板: http://127.0.0.1:%d", port)
    demo.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        demo.halt()


if __name__ == "__main__":
    main()
