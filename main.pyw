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

    _nvml_ok = pynvml.nvmlInit() == 0
except Exception:
    _nvml_ok = False

if _nvml_ok:
    try:
        _gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        _gpu_handle = None
else:
    _gpu_handle = None

import config
import env
import snake
from agent import Agent, Counter
from buffer import PrioritizedReplayBuffer
from monitor import Monitor, act_stats
from network import DQN, DEVICE
from torch.utils.tensorboard import SummaryWriter
from trainer import Trainer

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


def main():
    net = DQN(config.FRAME_STACK, config.GRID, config.ACTIONS).to(DEVICE)
    target_net = DQN(config.FRAME_STACK, config.GRID, config.ACTIONS).to(DEVICE)
    target_net.load_state_dict(net.state_dict())

    buffer = PrioritizedReplayBuffer(config.REPLAY_CAP)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.LR)
    counter = Counter()
    stats = {"lock": threading.Lock(), "recent": deque(maxlen=200), "episodes": 0, "loss": 0.0}

    resumed = False
    if os.path.exists(config.FULL_STATE):
        try:
            d = torch.load(config.FULL_STATE, map_location=DEVICE)
            net.load_state_dict(d["net"])
            target_net.load_state_dict(d["target"])
            optimizer.load_state_dict(d["opt"])
            counter.set(int(d.get("steps", 0)))
            log.info("完整续训成功：加载 %s，已累计步伐 %d", config.FULL_STATE, counter.get())
            resumed = True
        except Exception as e:
            log.warning("完整续训失败，回退普通恢复: %s", e)
    if not resumed and os.path.exists(config.CHECKPOINT):
        try:
            net.load_state_dict(torch.load(config.CHECKPOINT, map_location="cpu"))
            target_net.load_state_dict(net.state_dict())
            log.info("loaded %s", config.CHECKPOINT)
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

    stop_event = threading.Event()
    trainer = Trainer(net, target_net, buffer, optimizer, counter, config, stats, writer=writer)
    threading.Thread(target=trainer.loop, args=(stop_event,), daemon=True).start()

    for _ in range(config.AGENTS - 1):
        ag = Agent(net, buffer, env.Env(config.GRID, config.FRAME_STACK), counter, config, stats)
        threading.Thread(target=ag.loop, args=(stop_event,), daemon=True).start()

    pygame.init()
    info = pygame.display.Info()
    cell = min(12, max(6, (info.current_h - 120) // config.GRID))
    scene_px = cell * config.GRID
    screen = pygame.display.set_mode((scene_px, scene_px))
    pygame.display.set_caption("贪吃蛇 - DQN 强化学习")
    clock = pygame.time.Clock()

    r_env = env.Env(config.GRID, config.FRAME_STACK)
    r_agent = Agent(net, buffer, r_env, counter, config, stats)

    mon_port = [0]

    def state_fn():
        return {
            "port": mon_port[0],
            "steps": counter.get(),
            "eps": r_agent.epsilon(),
            "trainSteps": trainer.train_steps,
            "buffer": len(buffer),
            "loss": stats["loss"],
            "score": r_env.score,
            "avgScore": avg(stats),
            "bestScore": max_score(stats),
            "gpu": gpu_util() or -1,
            "episodes": stats["episodes"],
            "agents": config.AGENTS,
            "acts": self_acts(),
        }

    def self_acts():
        try:
            r = act_stats(net, r_env._stack())
            r["iteration"] = counter.get()
            r["samples"] = stats["episodes"]
            return r
        except Exception:
            return None

    mon_port[0] = Monitor(state_fn).start()
    log.info("监控面板: http://127.0.0.1:%d (中文)", mon_port[0])
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

    stop_event.set()
    torch.save(net.state_dict(), config.CHECKPOINT)
    trainer.save_full()
    writer.close()
    log.info("已保存（下次自动完整续训）。权重=%s 完整状态=%s，总步法=%d",
             config.CHECKPOINT, config.FULL_STATE, counter.get())
    pygame.quit()


if __name__ == "__main__":
    main()