import ctypes
import logging
import os
import threading
import time
from collections import deque

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
import game
from agent import Agent, Counter
from buffer import PrioritizedReplayBuffer
from network import DQN, DEVICE
from trainer import Trainer

logging.basicConfig(
    filename=config.LOG,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")


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


def draw_game(screen, g, canvas, scene_w, scene_h):
    canvas.fill((16, 20, 30))
    for r in range(game.ROWS):
        for c in range(game.COLS):
            if not g.bricks[r][c]:
                continue
            x0 = game.BRICK_LEFT + c * (game.BRICK_W + game.BRICK_GAP)
            y0 = game.BRICK_TOP + r * (game.BRICK_H + game.BRICK_GAP)
            pygame.draw.rect(canvas, game.COLORS[r % len(game.COLORS)], (x0, y0, game.BRICK_W, game.BRICK_H))
            pygame.draw.rect(canvas, (10, 12, 18), (x0, y0, game.BRICK_W, game.BRICK_H), 2)
    px = int(g.paddle_x - game.PADDLE_W / 2)
    pygame.draw.rect(canvas, (150, 190, 255), (px, game.PADDLE_Y, game.PADDLE_W, game.PADDLE_H))
    pygame.draw.rect(canvas, (10, 12, 18), (px, game.PADDLE_Y, game.PADDLE_W, game.PADDLE_H), 2)
    pygame.draw.circle(canvas, (255, 235, 160), (int(g.ball_x), int(g.ball_y)), game.BALL_R)
    pygame.draw.circle(canvas, (10, 12, 18), (int(g.ball_x), int(g.ball_y)), game.BALL_R, 1)
    scaled = pygame.transform.smoothscale(canvas, (scene_w, scene_h))
    screen.blit(scaled, (0, 0))


def draw_plot(screen, font_small, stats, x, y, w, h):
    pygame.draw.rect(screen, (26, 32, 46), (x, y, w, h))
    pygame.draw.rect(screen, (80, 92, 118), (x, y, w, h), 1)
    with stats["lock"]:
        scores = list(stats["recent"])
    if not scores:
        return
    maxv = max(1, max(scores))
    n = len(scores)
    mean = []
    run = 0.0
    for i, v in enumerate(scores):
        run += v
        mean.append(run / (i + 1))
    if n == 1:
        px = x + w // 2
        py = y + h - int((mean[0] / maxv) * h)
        pygame.draw.circle(screen, (255, 120, 80), (px, py), 2)
    else:
        pts = []
        for i, v in enumerate(mean):
            px = x + int((i / (n - 1)) * w)
            py = y + h - int((v / maxv) * h)
            pts.append((px, py))
        pygame.draw.lines(screen, (255, 120, 80), False, pts, 2)
    text = font_small.render(f"平均分曲线  当前均值 {mean[-1]:.1f}", True, (220, 230, 245))
    screen.blit(text, (x + 6, y + 6))


def draw(screen, fonts, stats, counter, agent, buffer, trainer, display_size, current_score):
    font_title, font, font_small = fonts
    panel_x = display_size
    panel_w = screen.get_width() - panel_x

    lines = [
        ("打砖块 · DQN 强化学习", font_title, (255, 255, 255)),
        (f"后台智能体  {config.AGENTS} 个（共用一张网络）", font_small, (170, 200, 255)),
        (f"计算设备  {'CUDA (GPU)' if DEVICE.type == 'cuda' else 'CPU'}", font_small, (170, 200, 255)),
        (f"训练吞吐  {counter.get():,} 步 / 有效批量 {config.BATCH * config.ACCUM:,}", font_small, (170, 200, 255)),
        (f"全局步数  {counter.get():,}", font, (230, 235, 245)),
        (f"探索率 ε  {agent.epsilon():.3f}", font, (230, 235, 245)),
        (f"经验池  {len(buffer):,} / {config.REPLAY_CAP:,}", font, (230, 235, 245)),
        (f"训练步数  {trainer.train_steps:,}", font, (230, 235, 245)),
        (f"当前得分  {current_score}", font, (255, 220, 120)),
        (f"平均得分  {avg(stats):.1f}", font, (255, 220, 120)),
        (f"最高得分  {max_score(stats)}", font, (255, 220, 120)),
        (f"Loss  {trainer.loss:.4f}", font, (230, 235, 245)),
    ]

    if DEVICE.type == "cuda":
        u = gpu_util()
        if u is not None:
            lines.append((f"GPU 利用率  {u}%", font, (120, 255, 160)))

    y = 12
    for text, f, color in lines:
        surf = f.render(text, True, color)
        screen.blit(surf, (panel_x + 12, y))
        y += f.get_height() + 10

    draw_plot(screen, font_small, stats, panel_x + 10, y + 8, panel_w - 20, 200)

    hint = font_small.render("关闭窗口 = 保存模型并退出", True, (140, 150, 170))
    screen.blit(hint, (panel_x + 12, screen.get_height() - 28))


def main():
    net = DQN(config.FRAME_STACK, config.SIZE, 3).to(DEVICE)
    target_net = DQN(config.FRAME_STACK, config.SIZE, 3).to(DEVICE)
    target_net.load_state_dict(net.state_dict())
    if os.path.exists(config.CHECKPOINT):
        try:
            net.load_state_dict(torch.load(config.CHECKPOINT, map_location="cpu"))
            target_net.load_state_dict(net.state_dict())
            log.info("loaded %s", config.CHECKPOINT)
        except Exception as e:
            log.warning("load failed: %s", e)

    buffer = PrioritizedReplayBuffer(config.REPLAY_CAP)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.LR)
    counter = Counter()
    stats = {"lock": threading.Lock(), "recent": deque(maxlen=200), "episodes": 0, "loss": 0.0}
    log.info("device: %s", DEVICE)

    stop_event = threading.Event()
    trainer = Trainer(net, target_net, buffer, optimizer, counter, config, stats)
    threading.Thread(target=trainer.loop, args=(stop_event,), daemon=True).start()

    for _ in range(config.AGENTS - 1):
        ag = Agent(net, buffer, env.Env(config.FRAME_STACK, config.SIZE), counter, config, stats)
        threading.Thread(target=ag.loop, args=(stop_event,), daemon=True).start()

    pygame.init()
    info = pygame.display.Info()
    panel_w = 300
    scene_h = min(info.current_h - 90, info.current_w - 360)
    scene_w = int(scene_h * game.W // game.H)
    canvas = pygame.Surface((game.W, game.H))
    screen = pygame.display.set_mode((scene_w + panel_w, scene_h))
    pygame.display.set_caption("打砖块 - 像素输入 CNN DQN 强化学习")
    clock = pygame.time.Clock()
    font_title = pygame.font.SysFont("microsoftyahei", 20)
    font = pygame.font.SysFont("microsoftyahei", 17)
    font_small = pygame.font.SysFont("microsoftyahei", 13)
    fonts = (font_title, font, font_small)

    r_env = env.Env(config.FRAME_STACK, config.SIZE)
    r_agent = Agent(net, buffer, r_env, counter, config, stats)
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
                stats["recent"].append(r_env.g.score)
                stats["episodes"] += 1
            s = r_env.reset()

        draw_game(screen, r_env.g, canvas, scene_w, scene_h)
        draw(screen, fonts, stats, counter, r_agent, buffer, trainer, scene_w, r_env.g.score)
        pygame.display.flip()
        clock.tick(60)

        now = time.time()
        if now - last_log > 5:
            last_log = now
            log.info(
                "steps=%d eps=%.3f buffer=%d loss=%.4f episodes=%d avg=%.1f best=%d",
                counter.get(),
                r_agent.epsilon(),
                len(buffer),
                stats["loss"],
                stats["episodes"],
                avg(stats),
                max_score(stats),
            )

    stop_event.set()
    torch.save(net.state_dict(), config.CHECKPOINT)
    log.info("saved %s, total steps=%d", config.CHECKPOINT, counter.get())
    pygame.quit()


if __name__ == "__main__":
    main()
