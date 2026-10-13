# 坦克动荡 · 四人混战自博弈强化学习

用 Dueling Double DQN + 自我对弈，让 AI 从零学会打「坦克动荡」（Tank Trouble）四人混战。附带一套**零依赖的网页版训练监控面板**，训练时能实时看到网络内部的真实状态。

> 目录名 `打砖块` 是早期版本沿用下来的，与当前内容无关，可自行改名。

## 亮点

| 项目 | 说明 |
|---|---|
| 训练速度 | 1024 局 / 8 进程环境池，实测 **7.7 万步/秒**（未池化时约 2 万） |
| 环境实现 | 纯 numpy 全向量化，一次推进 256 局 × 4 座位 |
| 进程通信 | 共享内存零拷贝 + 双缓冲，一帧 1.8 MB 观测不走队列序列化 |
| 算法 | Dueling Double DQN + 优先经验回放(PER) + n-step 回报 + PFSP 对手池 |
| 观测压缩 | 8 通道按位打包，1 像素仅 1 字节，比逐通道 float32 省 32 倍内存 |
| 监控面板 | 手写 PNG 编码，零第三方依赖，浏览器里看全量神经元 / 梯度分布 / 损失曲面 |

## 项目构成

这个仓库有两部分，可以分开使用：

1. **坦克自博弈训练**（`selfplay.py` 及其依赖）——RL 主体
2. **aimonitor 训练监控面板**（`aimonitor/`）——可复用的通用库，任何 PyTorch 训练都能接入

## 游戏环境

复刻 Tank Trouble 的四人对战规则，全部物理在 `tank.py` 里定义，训练环境 `tank_vec.py` 直接复用同一份常量，保证「看得见的游戏」和「练出来的环境」规则永远一致。

**地图**
- 19 × 13 格迷宫，墙厚 7 px、通道净宽 36 px
- `braid` 参数控制环路连通度，可做迷宫难度课程学习（从简单到复杂线性退火）

**坦克**
- 前进 170 px/s，后退更慢(108)，转向上限 3 rad/s
- 正方形刚体碰撞，用分离轴定理(SAT)做重叠判定

**武器**（8 种）

| 编号 | 武器 | 特性 |
|---|---|---|
| 0 | 普通弹 | 450 px/s，最多反弹 7 次，同屏上限 5 发 |
| 1 | 加特林 | 高射速 540 px/s |
| 2 | 追踪导弹 | 飞行中自动转向，寿命 9 秒 |
| 3 | 破片弹 | 撞墙爆炸，溅射碎片 |
| 4 | 双管炮 | 一次两发 |
| 5 | 激光 | 瞬间命中，无视弹道 |
| 6 | 护盾 | 短时无敌 |
| 7 | 地雷 | 布设后延时引爆 |

**回合规则**
- 4 人混战，60 fps，单回合上限 1800 帧（30 秒）
- 双方都苟着不打则判平局（都拿 0 奖励），避免回合无限拖长
- 子弹每帧位移切 4 个子步推进，防止高速弹「跳过」薄墙

## 强化学习设计

### 观测

**8 通道按位打包的低分辨率图**（1 像素 1 字节，8 个通道压在位里）：

```
bit0 墙 | bit1 自机 | bit2 对手(合并) | bit3 保留未用
bit4 子弹 | bit5 地雷 | bit6 道具 | bit7 激光
```

物理世界每 16×16 像素合并成观测图上 1 个像素（`OBS_RES=16`），一张约 51×35 的小图就能覆盖全场。按位打包让回放池装下 20 万条经验成为可能。

**81 维标量特征**：自机状态、最近 8 颗子弹的方向距离、武器剩余时间、名次等。

### 动作

**18 个离散动作** = 转向(-1/0/1) × 油门(前进/停/后退) × 开火(0/1)。

### 奖励

稀疏为主，只有终局才有实质信号：

| 项 | 值 |
|---|---|
| 每步基线 | -0.001 |
| 拾取道具 | +0.05 |
| 被淘汰 | -1.0 |
| 最后存活 | +1.0 |

步惩罚的作用是让「磨蹭不出手」有代价，否则纯稀疏奖励下最优行为容易退化成原地躲着不动，同等胜率下更偏好快速解决战斗。

### 算法

- **Dueling 结构**：`Q = V + A - mean(A)`，减均值是为了消除 V/A 的不可辨识性，不约束两个头会一起漂移
- **Double DQN**：动作由在线网络挑、价值由目标网络给，避免高估
- **PER**：容量 20 万，按 |TD 误差| 回写采样概率，配重要性采样权重
- **n-step**：`NSTEP=5` 压缩回报，兼顾偏差与方差
- **目标网络**：每 20000 步硬拷贝
- **ε-贪心**：1.0 线性衰减到 0.05，共 3000 万步

### 自博弈设计

**四个座位共用一张网络**：座位规则完全对称（无先后手、无座位优势），最优策略本身就是同一个函数。拆成 4 张网会让它们各自朝对方的最优反应跑，收敛期互相追着震荡。共用一张既省显存，又天然避免「追尾巴」。

**既做自博弈又留历史对手**：全用当前策略自博弈容易停在「两个同水平选手互相啄」的平台期；把历史快照冻结成对手，提供不会退化的固定压力。

**训练数据混合**（256 个环境分 4 组）：

- 第 0 组：4 个座位全由当前策略驱动，纯自博弈，提供大规模水平匹配的数据
- 第 1~3 组：每组只有 1 个「受测座位」是当前策略，另外 3 个座位来自**同一份历史快照**，专门制造「我必须打赢这个老版本」的压力；受测座位在组间轮换，保证所有组合被公平覆盖

**PFSP 对手池**：每 50 万步冻结一份快照，最多保留 12 个。权重由「受测座位对该快照的平均名次」回写：`clamp((平均名次 - 1) / 3, 0.05, 1.0)`（名次越差权重越高），抽样时再取 2 次幂放大差距后归一化，实现「越是打不过的对手越要多打」。没交手过的快照保持初始权重，会被优先探索到。

## 工程实现

### 多进程环境池（`tank_pool.py`）

环境推演是纯 numpy，单核跑满也就那样；机器有 24 个逻辑核却只忙 1 个。把环境按局切成 W 份交给子进程并行推演。

**实测线程不行**：numpy 这些小数组运算以 Python 层调用为主，GIL 根本不放行，4 线程反而慢 3 倍；只有真进程才能真正并行。

**共享内存零拷贝**：一帧要交换 1.8 MB 观测图，走队列/管道要 pickle 一遍再传一遍，省下的时间会被序列化全吃掉。改成主进程开一块共享内存，按固定布局摆好数组，子进程直接往对应行区间写，主进程直接读，全程零序列化。帧号只用两组信号量握手，一次约几微秒。

**双缓冲**：子进程写第 t 帧的同时主进程还要读第 t-1 帧算经验，单缓冲会被覆盖，两块轮换两边都不用拷贝。

**镜像状态**：监控面板和观战窗口要读整局物理状态，子进程额外抄一份进共享内存，主进程用 `MirrorEnv` 包装成「接口一样、只是只读」的对象，于是面板和观战代码一行都不用改。

### GPU 端观测解包

观测在 CPU 侧是打包字节，送卷积前要逐位展开成 float32 并按通道排好。同样的活在 numpy 里要 49 ms/千行（写 58 MB float32 全是主机内存带宽），放到 GPU 上只要 12 ms，而且上卡的只有 1/32 大小的原始字节。

## 训练监控面板（aimonitor）

零依赖的网页端训练监控。训练脚本只加几行代码，就能在浏览器里看到神经网络的真实内部状态。

> 约定：**页面上出现的每一个数字都来自真实运行时的张量**，没有演示数据、没有占位图、没有抽样降采样。显示不出来的一律标注「无数据」，不做假。

| 模块 | 内容 |
|---|---|
| 实时指标 | 任意标量自动成卡片 + 折线图，历史存服务端，刷新不丢 |
| 训练健康度 | 梯度范数、权重范数、学习率、训练/采样比、参数量、GPU 利用率 |
| 神经网络全量图 | 全部神经元逐个显示（512 个 FC 神经元就是 512 个），连线为真实权重矩阵，3D 可旋转 / 2D 平面 / 悬停查单个神经元 |
| 特征图热力图 | 每个卷积层的全部通道拼成一张图，原像素放大不插值 |
| 权重/梯度直方图 | 每个张量的完整分布（800 万个数也是全量统计），判断梯度消失/爆炸 |
| 损失曲面 3D | 在参数空间沿两个随机方向扰动采样，高度是实测损失，叠加真实优化器轨迹 |
| 指标明细表 | 每类精确率/召回率/F1、top-k 预测，全量数值不四舍五入 |
| 矩阵热力图 | 混淆矩阵、注意力权重、相关性矩阵，悬停读单格 |
| 散点投影 | PCA / t-SNE 降维、预测-真值分布、PR 曲线，按类别着色 |
| 对比条图 | 逐层梯度范数、各类准确率，自动按绝对值排序 |
| 训练日志 | 实时日志流，`attach_logger()` 一行接入 `logging`，自动标红 ERROR / 标黄 WARN |
| 异常告警 | `alert()` 自定义条件 + `watchdog()` 看门狗 + NaN/Inf 自动检测，顶部红条分级提示 |
| 评估与泛化 | 独立 ε=0 评估进程分开「策略真实水平」与「带探索的训练均值」，另给中位数、P75、崩局率、达标占比 |
| 曲线交互 | 悬停十字准线读单点、1/10/全部时间窗口、平滑、导出 CSV |
| 训练控制 | 网页上直接「启动训练 / 结束训练并保存」 |

面板是**注册制**的：注册了就显示，没注册的区块自动隐藏。同一套前端既能当强化学习面板，也能当分类/NLP/检测的面板。

### 接入自己的项目

```python
from aimonitor import Monitor

mon = Monitor(title="我的模型训练监控", port=8080, interval=1.0, history=900)

mon.info(lambda: {"设备": "CUDA", "数据集": "cifar10"})
mon.metric("loss", lambda: loss_value, label="Loss", fmt="f4")
mon.metric("acc",  lambda: acc,        label="准确率", fmt="pct")
mon.health("grad_norm", lambda: grad_norm, label="梯度范数", fmt="sci")

from aimonitor import montage
mon.image("conv1", lambda: montage(list(featmaps[0]), cols=8),
          label="conv1 全部通道", cmap="viridis", group="特征图")

mon.histogram("fc.weight", lambda: model.fc.weight.detach().cpu().numpy(), bins=48)
mon.matrix("confusion", lambda: cm, label="混淆矩阵", xlabels=class_names, ylabels=class_names)
mon.bars("grad_norm", lambda: [[n, float(p.grad.norm())] for n, p in model.named_parameters()])

buf = mon.attach_logger(logger, key="train_log", label="训练日志")
mon.watchdog("progress", counter.get, label="总步数", timeout=30.0)

port = mon.start()
print("监控面板 http://127.0.0.1:%d" % port)
```

完整示例见 [examples/classify_demo.py](examples/classify_demo.py)。

### SDK API

| 方法 | 说明 |
|---|---|
| `Monitor(title, host, port, interval, history)` | `interval` 采样间隔秒；`history` 服务端保留的采样点数 |
| `info(fn)` | 顶部信息条，返回 `{键: 值}` |
| `metric(key, fn, label, color, fmt, chart)` | 标量指标；`chart=False` 只出卡片不出曲线 |
| `health(key, fn, label, unit, fmt)` | 健康度卡片 |
| `image(key, fn, label, cmap, scale, group)` | 真实图片；`group` 决定归到「特征图」还是「图片」 |
| `histogram(key, fn, label, bins, refresh)` | 直方图；`refresh` 是重算间隔秒（大张量建议 ≥5） |
| `table(key, fn, label, columns)` | 表格；返回行列表或 `{"columns": [...], "rows": [...]}` |
| `matrix(key, fn, label, cmap, xlabels, ylabels, fmt, refresh)` | 二维矩阵热力图；含负数时自动用发散色图 |
| `scatter(key, fn, label, xlabel, ylabel)` | 散点；返回 `[[x, y, 类别], ...]`，上限 4 万点 |
| `bars(key, fn, label)` | 条图；返回 `[[标签, 数值], ...]` |
| `log(key, fn, label, limit)` | 日志流 |
| `attach_logger(logger, key, label, level, limit)` | 把 `logging.Logger` 输出实时镜像到网页 |
| `alert(key, fn, label, level)` | 异常告警；`fn()` 返回字符串即告警 |
| `watchdog(key, fn, label, timeout, level)` | 看门狗；数字超过 `timeout` 秒没变化就告警 |
| `graph(fn)` | 返回 `{"layers": [...], "edges": [...]}` |
| `surface(fn)` / `params(fn)` | 损失曲面 / 底部参数表 |
| `control(start, stop)` | 网页启停按钮回调 |
| `start()` | 启动服务，返回实际端口 |

`fmt` 可选：`auto` / `int` / `f1`~`f6` / `pct` / `sci`。

辅助函数：`montage(maps, cols, pad)` 铺图、`normalize(a, lo, hi)` 归一化、`colormap(u8, name)` 上色（内置 viridis / magma / blue / red / diverging / gray）、`encode_png(arr)` 无依赖编码 PNG、`LogBuffer(capacity)` 环形日志缓冲。

### 数据协议

前端是纯静态文件，只依赖三个接口，你也可以自己写前端：

| 接口 | 频率 | 内容 |
|---|---|---|
| `GET /api/state` | 默认 1 秒 | 指标、曲线、图片、直方图、表格、矩阵、散点、条图、日志、损失曲面、参数表、当前激活值 |
| `GET /api/graph` | 仅当 `graph.version` 变化时 | 全量神经元与连线矩阵（可能很大，故按模型版本缓存） |
| `POST /api/control/start`、`/api/control/stop` | 手动 | 网页启停训练 |

### 实现说明

- **零第三方依赖**：只用标准库 + numpy，PNG 编码、色图、直方图全部自己实现，不需要 matplotlib / Pillow
- **前端可热改**：`assets/` 下文件每次请求都从磁盘读，改完刷新浏览器即生效，不用重启训练
- **对大张量友好**：连线矩阵按模型版本缓存；直方图有独立刷新间隔；非线性数值一律转成 `null` 而不是输出非法 JSON 的 `NaN`
- **不阻塞训练**：所有采样都在独立线程里做，训练主循环不参与

## 目录结构

```
aimonitor/                训练监控面板（可复用库，零依赖）
  sdk.py                  注册式 API 与采样线程
  server.py               HTTP 服务
  imaging.py              无依赖 PNG 编码 / 色图 / 拼图
  assets/                 前端静态文件（index.html / style.css / app.js）

examples/
  classify_demo.py        监控面板的分类任务示例

tank.py                   可玩本体：物理、地图生成、武器、脚本 AI(Laika)
tank_vec.py               纯 numpy 向量化环境（一次推进 N 局 4 人混战）
tank_pool.py              多进程环境池：共享内存零拷贝 + 双缓冲
tank_play.pyw             手动试玩（pygame）
tank_view.py              观战窗口（读训练共享内存 / --follow 本地推演）
tank_monitor.py           把坦克训练接入 aimonitor 面板
selfplay.py               训练主循环（自博弈）
network.py                Dueling Double DQN 网络
buffer.py                 优先经验回放 + n-step 队列
config.py                 全部超参数
```

## 快速开始

```bash
pip install numpy torch pygame
```

**训练**

```bash
python selfplay.py --minutes 10                      # 跑 10 分钟
python selfplay.py --resume --monitor --view         # 续训 + 网页面板 + 开放观战共享内存
```

`--monitor` 启动后打开 <http://127.0.0.1:8080>（端口见 `config.MONITOR_PORT`）。

**观战**

```bash
python tank_view.py              # 读训练进程的共享内存镜像（训练时要带 --view）
python tank_view.py --follow     # 本地自己推演一局给人看，不需要训练进程
python tank_view.py --env 3      # 看第 3 个环境
```

**手动试玩**

```bash
python tank_play.pyw
```

**只跑监控面板示例**

```bash
python examples/classify_demo.py     # 图像分类任务，端口 8080
```

### 命令行参数

| 参数 | 说明 |
|---|---|
| `--minutes` | 本次训练跑多少墙钟分钟（默认 10） |
| `--envs` | 并行环境数（默认 256） |
| `--workers` | 环境推演的子进程数（默认 4） |
| `--monitor` | 开启网页监控面板 |
| `--view` | 开放观战共享内存（配合 `tank_view.py`） |
| `--resume` | 从检查点续训 |
| `--ckpt` / `--log` | 覆盖检查点与日志路径，跑 A/B 对比时用来隔离存档 |
| `--steps-max` | 累计步数上限（0 表示不限，只按时间跑）。A/B 对比时按步数封顶能把变量钉死 |
| `--braid-from` / `--braid-to` / `--braid-steps` | 迷宫连通度课程：从 from 线性退火到 to，跨 steps 步走完 |

## 性能调优记录

这些都是实测数据，不是估算：

| 改动 | 效果 |
|---|---|
| 环境推演改多进程池（1024 局 / 8 进程） | 2 万步/秒 → **7.7 万步/秒** |
| `TRAIN_EVERY` 由 2 改为 1 | 梯度更新频率 18 次/秒 → **32 次/秒**（GPU 始终喂饱） |
| 观测每帧只算一次，复用末帧 | 省 11% |
| 观测解包从 numpy 搬到 GPU | 49 ms → 12 ms（每千行） |
| `EVAL_EVERY` 由 300 万步提到 1000 万步 | 评测开销从近 50% 压到 15% 以内 |

`EVAL_EVERY` 那条值得说明：评测是单进程跑的、一次 30 秒上下，按步数触发。训练一快，同样步数间隔对应的墙钟时间变短，评测占比反而变大，所以提速后必须同步调大间隔。

## 硬件

开发机为单张 RTX 5060，训练与推理都在 GPU 上。几百个并行环境每步都要前向，全靠 CPU 喂不上。

## 许可

MIT，详见 [LICENSE](LICENSE)
