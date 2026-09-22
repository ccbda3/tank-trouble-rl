import time

import torch

from network import DEVICE


class Trainer:
    def __init__(self, net, target_net, buffer, optimizer, counter, config, stats):
        self.net = net
        self.target_net = target_net
        self.buffer = buffer
        self.optimizer = optimizer
        self.counter = counter
        self.config = config
        self.stats = stats
        self.train_steps = 0
        self.loss = 0.0

    def loop(self, stop_event):
        last_save = time.time()
        self.optimizer.zero_grad()
        while not stop_event.is_set():
            if len(self.buffer) < self.config.BATCH:
                time.sleep(0.05)
                continue
            batch, weights, idxs = self.buffer.sample(self.config.BATCH)
            s = torch.tensor([x[0] for x in batch], dtype=torch.float32, device=DEVICE)
            a = torch.tensor([x[1] for x in batch], dtype=torch.long, device=DEVICE).unsqueeze(1)
            r = torch.tensor([x[2] for x in batch], dtype=torch.float32, device=DEVICE).unsqueeze(1)
            s2 = torch.tensor([x[3] for x in batch], dtype=torch.float32, device=DEVICE)
            d = torch.tensor([x[4] for x in batch], dtype=torch.float32, device=DEVICE).unsqueeze(1)
            w = torch.tensor(weights, dtype=torch.float32, device=DEVICE).unsqueeze(1)

            with torch.no_grad():
                a_argmax = self.net(s2).max(1, keepdim=True).indices
                target = r + self.config.GAMMA * (1 - d) * self.target_net(s2).gather(1, a_argmax)
            q = self.net(s).gather(1, a)
            td = (q - target).abs()
            loss = (w * torch.nn.functional.smooth_l1_loss(q, target, reduction="none")).mean()
            self.buffer.update_priorities(idxs, td.squeeze(1).cpu().tolist())

            loss = loss / self.config.ACCUM
            loss.backward()
            self.train_steps += 1
            with self.stats["lock"]:
                self.stats["loss"] = float(loss.item() * self.config.ACCUM)

            if self.train_steps % self.config.ACCUM == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.train_steps % self.config.TARGET_SYNC == 0:
                    self.target_net.load_state_dict(self.net.state_dict())
                if self.train_steps % self.config.SAVE_EVERY == 0:
                    torch.save(self.net.state_dict(), self.config.CHECKPOINT)
                if time.time() - last_save > self.config.CHECKPOINT_INTERVAL:
                    last_save = time.time()
                    torch.save(self.net.state_dict(), self.config.CHECKPOINT)
