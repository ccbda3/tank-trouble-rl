import ctypes
import logging
import os
import subprocess
import sys
import threading
import time
from collections import deque

import numpy as np
import pygame

if not ctypes.windll.kernel32.GetModuleHandleW("msvcp140.dll"):
    _sys32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    try:
        ctypes.WinDLL(os.path.join(_sys32, "msvcp140.dll"))
    except OSError:
        pass

import torch

try:
    import pynvml

    pynvml.nvmlInit()
    _gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    _gpu_handle = None

import config
import env
import snake
from agent import Agent, Counter, EnvWorker
from buffer import PrioritizedReplayBuffer
from network import DQN, DEVICE
from snake_monitor import SnakeMonitor
from torch.utils.tensorboard import SummaryWriter
from trainer import LandscapeProbe, Trainer, load_optimizer

logging.basicConfig(
    filename=config.LOG,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("snake")


def avg(stats):
    with stats["lock"]:
        if not stats["recent"]:
            return 0.0
        return sum(stats["recent"]) / len(stats["recent"])


def max_score(stats):
    with stats["lock"]:
        return max(stats["recent"]) if stats["recent"] else 0


def gpu_util():
    if _gpu_handle is None:
        return None
    try:
        return pynvml.nvmlDeviceGetUtilizationRates(_gpu_handle).gpu
    except Exception:
        return None


def draw_scene(screen, g, scene_px):
    gd = config.GRID
    img = np.zeros((gd, gd, 3), np.uint8)
    img[g.apple[0], g.apple[1]] = (220, 40, 40)
    for i, (x, y) in enumerate(g.body):
        if i == 0:
            img[x, y] = (120, 255, 140)
        else:
            img[x, y] = (40, 170, 70)
    arr = np.ascontiguousarray(img.transpose(1, 0, 2))
    surf = pygame.surfarray.make_surface(arr)
    scaled = pygame.transform.scale(surf, (scene_px, scene_px))
    screen.blit(scaled, (0, 0))
    pygame.draw.rect(screen, (60, 70, 90), (0, 0, scene_px, scene_px), 2)


def load_partial(module, state):
    own = module.state_dict()
    kept = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    own.update(kept)
    module.load_state_dict(own)
    return len(kept), len(own)


def main():
    net = DQN(config.FRAME_STACK, config.GRID, config.ACTIONS).to(DEVICE)
    target_net = DQN(config.FRAME_STACK, config.GRID, config.ACTIONS).to(DEVICE)
    target_net.load_state_dict(net.state_dict())

    buffer = PrioritizedReplayBuffer(config.REPLAY_CAP, config.FRAME_STACK, config.GRID)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.LR)
    counter = Counter()
    stats = {"lock": threading.Lock(), "recent": deque(maxlen=200), "episodes": 0, "loss": 0.0}

    resumed = False
    saved = {"train_steps": 0, "updates": 0}
    if os.path.exists(config.FULL_STATE):
        try:
            d = torch.load(config.FULL_STATE, map_location=DEVICE)
            kept, total = load_partial(net, d["net"])
            load_partial(target_net, d["target"])
            if load_optimizer(optimizer, d["opt"]):
                log.info("已恢复优化器状态")
            else:
                log.warning("优化器状态与当前网络不匹配，已重新初始化")
            counter.set(int(d.get("steps", 0)))
            if kept == total:
                saved["train_steps"] = int(d.get("train_steps", 0))
                saved["updates"] = int(d.get("updates", 0))
            else:
                log.warning("网络结构有变化（%d/%d 层迁移），决策层重学、探索进度重置", kept, total)
            log.info("续训成功：迁移 %d/%d 层权重，已累计步伐 %d、梯度更新 %d",
                     kept, total, counter.get(), saved["train_steps"])
            resumed = True
        except Exception as e:
            log.warning("续训失败，回退普通恢复: %s", e)
    if not resumed and os.path.exists(config.CHECKPOINT):
        try:
            kept, total = load_partial(net, torch.load(config.CHECKPOINT, map_location="cpu"))
            target_net.load_state_dict(net.state_dict())
            log.info("已加载检查点：迁移 %d/%d 层权重", kept, total)
        except Exception as e:
            log.warning("load failed: %s", e)

    log.info("device: %s grid=%d actions=%d", DEVICE, config.GRID, config.ACTIONS)

    log_dir = os.path.abspath("runs")
    writer = SummaryWriter(os.path.join(log_dir, "snake"))
    try:
        subprocess.Popen(
            [sys.executable, "-m", "tensorboard.main", "--logdir", log_dir, "--host", "127.0.0.1", "--port", "6006"],
            creationflags=0x208,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log.info("tensorboard: http://127.0.0.1:6006")
    except Exception as e:
        log.warning("tensorboard launch failed: %s", e)

    trainer = Trainer(net, target_net, buffer, optimizer, counter, config, stats, writer=writer)
    trainer.train_steps = saved["train_steps"]
    trainer.updates = saved["updates"]
    probe = LandscapeProbe(net, target_net, buffer, config)
    per_worker = max(1, config.AGENTS // config.WORKERS)
    run = {"stop": None, "threads": []}

    def start_training():
        if run["stop"] is not None:
            return False
        ev = threading.Event()
        threads = [
            threading.Thread(target=trainer.loop, args=(ev,), daemon=True),
            threading.Thread(target=probe.loop, args=(ev,), daemon=True),
        ]
        for _ in range(config.WORKERS):
            envs = [env.Env(config.GRID, config.FRAME_STACK) for _ in range(per_worker)]
            worker = EnvWorker(net, buffer, envs, counter, config, stats, trainer)
            threads.append(threading.Thread(target=worker.loop, args=(ev,), daemon=True))
        for t in threads:
            t.start()
        run["stop"] = ev
        run["threads"] = threads
        log.info("训练已启动：%d 个采集线程 + 训练线程 + 曲面探针", config.WORKERS)
        return True

    def stop_training(save=True):
        ev = run["stop"]
        if ev is None:
            return False
        ev.set()
        for t in run["threads"]:
            t.join(timeout=5.0)
        run["stop"] = None
        run["threads"] = []
        if save:
            torch.save(net.state_dict(), config.CHECKPOINT)
            trainer.save_full()
            log.info("训练已停止并保存：步伐 %d、梯度更新 %d", counter.get(), trainer.train_steps)
        return True

    start_training()

    pygame.init()
    info = pygame.display.Info()
    cell = min(12, max(6, (info.current_h - 120) // config.GRID))
    scene_px = cell * config.GRID
    screen = pygame.display.set_mode((scene_px, scene_px))
    pygame.display.set_caption("贪吃蛇 - DQN 强化学习")
    clock = pygame.time.Clock()

    r_env = env.Env(config.GRID, config.FRAME_STACK)
    r_agent = Agent(net, buffer, r_env, counter, config, stats, trainer)

    mon = SnakeMonitor(
        net, target_net, buffer, counter, trainer, stats, probe, r_env,
        start_training, stop_training, lambda: r_agent.epsilon(), gpu_util,
    ).build(port=config.MONITOR_PORT)
    mon_port = mon.start()
    log.info("监控面板: http://127.0.0.1:%d", mon_port)
    s = r_env.reset()
    state = {"running": True}

    if os.environ.get("AT_QUIT_SECONDS"):
        def auto_quit():
            time.sleep(float(os.environ["AT_QUIT_SECONDS"]))
            state["running"] = False

        threading.Thread(target=auto_quit, daemon=True).start()

    last_log = time.time()
    while state["running"]:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                state["running"] = False

        a = r_agent.act(s)
        s2, r, done = r_env.step(a)
        buffer.push(s, a, r, s2, done)
        counter.add(1)
        s = s2
        if done:
            with stats["lock"]:
                stats["recent"].append(r_env.score)
                stats["episodes"] += 1
            s = r_env.reset()

        draw_scene(screen, r_env.g, scene_px)
        pygame.display.flip()
        clock.tick(60)

        now = time.time()
        if now - last_log > 5:
            last_log = now
            gstep = trainer.updates
            log.info(
                "steps=%d eps=%.3f loss=%.4f episodes=%d avg_len=%.1f best=%d",
                counter.get(),
                r_agent.epsilon(),
                stats["loss"],
                stats["episodes"],
                avg(stats),
                max_score(stats),
            )
            writer.add_scalar("metric/eps", r_agent.epsilon(), gstep)
            writer.add_scalar("metric/avg_len", avg(stats), gstep)
            writer.add_scalar("metric/best_len", max_score(stats), gstep)
            writer.add_scalar("metric/episodes", stats["episodes"], gstep)
            g = gpu_util()
            if g is not None:
                writer.add_scalar("metric/gpu", g, gstep)

    if stop_training(save=False):
        log.info("退出前停止训练线程")
    torch.save(net.state_dict(), config.CHECKPOINT)
    trainer.save_full()
    writer.close()
    log.info("已保存（下次自动完整续训）。权重=%s 完整状态=%s，总步法=%d",
             config.CHECKPOINT, config.FULL_STATE, counter.get())
    pygame.quit()


if __name__ == "__main__":
    main()