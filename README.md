# aimonitor · 强化学习 / 深度学习训练实时监控

一个**零依赖**的网页端训练监控系统。训练脚本只加几行代码，就能在浏览器里看到神经网络的**真实**内部状态：每个神经元、每条连线、每张特征图、每个梯度分布。

> 设计约定：**页面上出现的每一个数字都来自真实运行时的张量，没有演示数据、没有占位图、没有抽样降采样。** 显示不出来的一律标注"无数据"，不做假。

## 特性

| 模块 | 内容 |
|---|---|
| 实时指标 | 任意标量指标自动成卡片 + 折线图，历史保存在服务端，刷新页面不丢 |
| 训练健康度 | 梯度范数、权重范数、梯度最大绝对值、学习率、训练/采样比、参数量 |
| 神经网络全量图 | **全部神经元逐个显示**（512 个 FC 神经元也是 512 个），连线为真实权重矩阵，3D 可旋转 / 2D 平面 / 悬停查单个神经元 |
| 特征图热力图 | 每个卷积层的**全部通道**拼成一张图，原像素放大不插值 |
| 真实图片 | 网络输入输出按原始像素显示 |
| 权重/梯度直方图 | 每个张量的**完整**分布（800 万个数也是全量统计），用于判断梯度消失/爆炸 |
| 损失曲面 3D | 在参数空间沿两个固定随机方向扰动采样，高度是**实测**损失，叠加真实优化器轨迹 |
| 训练控制 | 网页上直接"启动训练 / 结束训练并保存" |

## 目录结构

```
aimonitor/              监控系统（可复用包）
  sdk.py                Monitor 类与注册式 API
  server.py             HTTP 服务（/api/state、/api/graph、/api/control/*）
  imaging.py            无依赖 PNG 编码、色图、特征图拼图
  assets/               前端静态文件，改完刷新浏览器即生效（不用重启训练）
    index.html
    style.css
    app.js
snake_monitor.py        示例：把贪吃蛇 DQN 项目接入 SDK
main.pyw                示例：训练主程序
config.py / network.py / env.py / snake.py / agent.py / buffer.py / trainer.py
```

## 快速开始

```bash
pip install numpy torch        # 另外需要 pygame（仅示例游戏窗口用）
pythonw main.pyw               # 无控制台启动（Windows）
```

浏览器打开 <http://127.0.0.1:80>。

端口在 `config.py` 的 `MONITOR_PORT` 里改；被占用时会自动顺延并在启动日志里写出实际端口。

## 接入你自己的项目

```python
from aimonitor import Monitor

mon = Monitor(title="我的模型训练监控", port=80, interval=1.0, history=900)

# 1) 顶部信息条
mon.info(lambda: {"设备": "CUDA", "数据集": "cifar10"})

# 2) 标量指标：自动出卡片 + 曲线
mon.metric("loss", lambda: loss_value, label="Loss", fmt="f4", color="#ff8f5a")
mon.metric("acc",  lambda: acc,        label="准确率", fmt="pct")

# 3) 训练健康度
mon.health("grad_norm", lambda: grad_norm, label="梯度范数", fmt="sci")

# 4) 真实图片（numpy HxW 或 HxWx3，float 会自动归一化）
mon.image("input", lambda: batch_images[0], label="输入样本", cmap="gray", group="图片")

# 5) 特征图拼图（把所有通道铺成一张图）
from aimonitor import montage
mon.image("conv1", lambda: montage(list(featmaps[0]), cols=8),
          label="conv1 全部通道", cmap="viridis", group="特征图")

# 6) 权重/梯度直方图
for name, p in model.named_parameters():
    mon.histogram(name, (lambda n: (lambda: dict(model.named_parameters())[n].detach().cpu().numpy()))(name),
                  label=name + " 权重", bins=48)

# 7) 神经元全量图 + 精确连线矩阵
mon.graph(lambda: {
    "layers": [
        {"name": "输入", "detail": "784", "nodeCount": 784, "params": 0,
         "realCount": 784, "nodes": x.flatten().tolist()},
        {"name": "隐层", "detail": "256", "nodeCount": 256, "params": 200960,
         "realCount": 256, "nodes": h.tolist()},
        {"name": "输出", "detail": "10", "nodeCount": 10, "params": 2570,
         "realCount": 10, "nodes": out.tolist()},
    ],
    # edges[i][t][s] = 第 i 段里 目标神经元 t 与 源神经元 s 的真实权重
    "edges": [abs(w1.numpy()), abs(w2.numpy())],
})

# 8) 任意 3D 曲面（可选）
mon.surface(lambda: {"ready": 1, "res": 15, "ext": 0.5, "zmin": 0.1, "zmax": 0.4,
                     "values": grid_flat.tolist(), "path": [{"a": 0, "b": 0, "z": 0.2}]})

# 9) 参数表（可选）
mon.params(lambda: [{"title": "结构", "layers": [...], "rows": [["设备", "CUDA"]]}])

# 10) 网页上的启停按钮（可选）
mon.control(start=start_training, stop=stop_training)

port = mon.start()
print("监控面板 http://127.0.0.1:%d" % port)
```

## SDK API

| 方法 | 说明 |
|---|---|
| `Monitor(title, host, port, interval, history)` | `interval` 采样间隔秒；`history` 服务端保留的采样点数 |
| `info(fn)` | 顶部信息条，`fn()` 返回 `{键: 值}` |
| `metric(key, fn, label, color, fmt, chart)` | 标量指标；`chart=False` 只出卡片不出曲线 |
| `health(key, fn, label, unit, fmt)` | 健康度卡片 |
| `image(key, fn, label, cmap, scale, group)` | 真实图片；`cmap` 为空按灰度/原色，`group` 决定归到"特征图"还是"图片" |
| `histogram(key, fn, label, bins, refresh)` | 直方图；`refresh` 是重算间隔秒（大张量建议 ≥5） |
| `graph(fn)` | 返回 `{"layers": [...], "edges": [...]}`；`edges[i]` 形状必须是 `(目标层节点数, 源层节点数)` |
| `surface(fn)` | 损失曲面数据 |
| `params(fn)` | 底部参数表，返回 `[{"title": ..., "layers": [...], "rows": [[键, 值]]}]` |
| `control(start, stop)` | 网页启停按钮回调，返回 `True` 表示状态真的变了 |
| `start()` | 启动服务，返回实际端口 |
| `set_training(bool)` | 手动同步训练状态（用 `control` 时自动维护） |

`fmt` 可选：`auto` / `int` / `f1`~`f6` / `pct` / `sci`。

辅助函数：`montage(maps, cols, pad)` 铺图、`normalize(a, lo, hi)` 归一化、`colormap(u8, name)` 上色（内置 viridis / magma / blue / red / diverging / gray）、`encode_png(arr)` 无依赖编码 PNG。

## 数据协议

前端是纯静态文件，只依赖下面三个接口。你也可以自己写前端，只要遵守协议。

**`GET /api/state`** — 高频（默认 1 秒）

```jsonc
{
  "title": "...", "interval": 1.0, "training": true,
  "info": {"设备": "CUDA"},
  "metrics":   [{"key":"loss","label":"Loss","value":0.12,"fmt":"f4","color":"#ff8f5a","chart":true}],
  "series":    {"loss": [[1712345678.1, 0.15], [1712345679.1, 0.12]]},
  "health":    [{"key":"grad_norm","label":"梯度范数","value":0.19,"unit":"","fmt":"sci"}],
  "images":    [{"key":"input","label":"输入","png":"data:image/png;base64,...","w":32,"h":32,"scale":6,"group":"图片","lo":0,"hi":255}],
  "histograms":[{"key":"fc.weight","label":"fc.weight 权重","counts":[...],"edges":[...],"min":-0.1,"max":0.1,"mean":0.0,"std":0.02,"n":8388608}],
  "surface":   {"ready":1,"res":15,"ext":0.05,"zmin":0.3,"zmax":0.6,"values":[...],"path":[{"a":..,"b":..,"z":..}]},
  "params":    [{"title":"网络结构","layers":[{"name":"FC","detail":"512","params":8389120,"realCount":512}],"rows":[["设备","CUDA"]]}],
  "graph":     {"version": 3790918552, "summary": "680 神经元 / 41088 连线"},
  "acts":      [[...每层节点激活...]]
}
```

**`GET /api/graph`** — 低频（仅当 `graph.version` 变化时才请求，因为连线矩阵可能很大）

```jsonc
{ "version": 3790918552, "totalNodes": 680, "totalEdges": 41088,
  "layers": [{"name":"FC","detail":"512 个神经元","nodeCount":512,"params":8389120,"realCount":512}],
  "edges":  [ [[...], ...] ]   // edges[i][目标节点][源节点] = 真实权重
}
```

**`POST /api/control/start`** / **`POST /api/control/stop`**

```jsonc
{ "ok": true, "changed": true, "training": true }
```

## 实现说明

- **零第三方依赖**：只用到标准库 + numpy。PNG 编码、色图、直方图全部自己实现，不需要 matplotlib / Pillow。
- **前端可热改**：`assets/` 下的文件每次请求都从磁盘读取，改完刷新浏览器即可，不用重启训练进程。
- **对大张量友好**：连线矩阵按"模型版本"缓存，只在模型结构变化时传输；直方图有独立刷新间隔；非线性数值一律转成 `null` 而不是输出非法 JSON 的 `NaN`。
- **不阻塞训练**：所有采样都在独立线程里做，训练主循环不参与。

## 许可

MIT
