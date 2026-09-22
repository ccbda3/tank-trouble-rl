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
| 指标明细表 | 任意表格：每类精确率/召回率/F1、top-k 预测、数据集统计，全量数值不四舍五入 |
| 矩阵热力图 | 混淆矩阵、注意力权重、相关性矩阵；颜色按真实值映射，悬停读单格 |
| 散点投影 | 嵌入降维（PCA/t-SNE）、预测-真值分布、PR 曲线，按类别着色 + 图例 |
| 对比条图 | 逐层梯度范数、各类准确率、任意分组对比，自动按绝对值排序 |
| 训练日志 | 实时日志流，`attach_logger()` 一行把 `logging` 接进来（自动标红 ERROR / 标黄 WARN） |
| 异常告警 | `alert()` 自定义条件 + `watchdog()` 看门狗 + 指标 NaN/Inf 自动检测，顶部红条分级提示 |
| 曲线交互 | 悬停十字准线读单点数值、1/10/全部时间窗口、平滑、导出 CSV |
| 训练控制 | 网页上直接"启动训练 / 结束训练并保存" |

面板是**注册制**的：注册了就显示，没注册的区块自动隐藏。所以同一套前端既能当强化学习面板，也能当分类/NLP/检测的训练面板。

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
examples/
  classify_demo.py      示例：图像分类（混淆矩阵 / 每类指标 / 特征投影 / 梯度范数 / 日志）
snake_monitor.py        示例：把贪吃蛇 DQN 项目接入 SDK
main.pyw                示例：训练主程序
config.py / network.py / env.py / snake.py / agent.py / buffer.py / trainer.py
```

## 快速开始

```bash
pip install numpy torch        # 另外需要 pygame（仅示例游戏窗口用）

python examples/classify_demo.py   # 例1：分类任务，端口 8080
pythonw main.pyw                   # 例2：贪吃蛇 DQN，端口 80（无控制台启动）
```

浏览器打开 <http://127.0.0.1:8080> 或 <http://127.0.0.1:80>。

示例里的数据、模型、张量全部现场真实计算（合成数据现场生成、每个数字现场前向），不是录制的假数据。

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

# 11) 表格：每类精确率/召回率/F1（返回行列表，或 {"columns": [...], "rows": [...]}）
mon.table("report", lambda: [["cat", 0.91, 120], ["dog", 0.88, 96]],
          label="每类指标", columns=["类别", "精确率", "样本数"])

# 12) 矩阵热力图：混淆矩阵 / 注意力 / 相关性（xlabels、ylabels 可传列表，也可传函数）
mon.matrix("confusion", lambda: confusion_matrix(y_true, y_pred, labels=range(10)),
           label="混淆矩阵（行=真实 列=预测）", xlabels=class_names, ylabels=class_names, fmt="int")

# 13) 散点：嵌入投影（点 = [x, y, 类别]，类别可以是字符串）
mon.scatter("emb", lambda: [[float(p[0]), float(p[1]), CLASSES[int(y[i])]]
                            for i, p in enumerate(project(features))],
            label="倒数第二层特征 PCA 投影", xlabel="主成分 1", ylabel="主成分 2")

# 14) 对比条图：逐层梯度范数（判断梯度消失/爆炸）
mon.bars("grad_norm", lambda: [[n, float(p.grad.norm())] for n, p in model.named_parameters()],
         label="逐层梯度范数")

# 15) 日志流：一行接管 logging（返回的 LogBuffer 也可以自己 write）
buf = mon.attach_logger(logger, key="train_log", label="训练日志", limit=400)
buf.write("这一行会出现在网页上")

# 16) 异常告警：fn() 返回字符串就弹红条，返回 None 表示正常
mon.alert("grad", lambda: ("范数 %.1f，疑似爆炸" % g) if g > 100 else None, label="梯度", level="error")

# 17) 看门狗：这个数字超过 timeout 秒没变化就报警（训练卡死、采集线程死掉都能抓到）
mon.watchdog("progress", counter.get, label="总步数", timeout=30.0)

port = mon.start()
print("监控面板 http://127.0.0.1:%d" % port)
```

## 面板交互

| 操作 | 效果 |
|---|---|
| 悬停曲线 | 十字准线 + 显示该采样点的时间与精确数值 |
| 曲线右上按钮 | 切换 1 分钟 / 10 分钟 / 全部时间窗口 |
| 平滑 | 对曲线做指数滑动平均，看趋势用 |
| 导出 CSV | 把所有曲线导成宽表 CSV（UTF-8 BOM，Excel 直接打开） |
| 点击区块标题 | 折叠 / 展开区块，状态记在浏览器里 |
| 顶部导航 | 按「训练概览 / 模型内部 / 诊断分析 / 配置」分组跳转 |
| 顶部告警条 | 有告警才出现，红色 = 严重，黄色 = 警告 |

## 异常告警

告警由服务端每秒评估一次，前端只在有告警时显示顶部红条，不需要人工盯着。

| 来源 | 触发条件 |
|---|---|
| `watchdog` | 监控的数字在 `timeout` 秒内没有变化（训练线程/采集线程卡死、进程僵死）。训练停止时自动不报警 |
| `alert` | 自定义条件，`fn()` 返回非空字符串即告警 |
| 指标非法值 | 任何 `metric` 从有限值变成 NaN / Inf 时自动告警一次，恢复正常后自动解除 |

## 其他 AI 领域怎么用

同一套面板换几个注册项就是另一种训练监控，参考 [classify_demo.py](examples/classify_demo.py)：

| 领域 | 组合 |
|---|---|
| 图像分类 | `matrix` 混淆矩阵 + `table` 每类 P/R/F1 + `scatter` 特征 PCA 投影 + `bars` 逐层梯度范数 + `image` 决策边界 |
| 目标检测 / 分割 | `image` 原图叠预测框 + `table` 每类 AP + `scatter` PR 曲线（点 = [recall, precision]）+ `log` |
| NLP / Transformer | `matrix` 注意力矩阵（`cmap="viridis"`）+ `table` top-k 预测 + `scatter` token 嵌入投影 + `log` 生成样例 |
| LLM 预训练 | `metric` loss/lr/吞吐 + `log` 实时日志 + `bars` 各层梯度范数 + `histogram` 权重分布 |
| 表格 / 回归 | `scatter` 预测值-真值散点 + `bars` 特征重要度 + `matrix` 特征相关性（自动用发散色图） |
| 强化学习 | 见 [snake_monitor.py](snake_monitor.py)：奖励曲线 + 损失曲面 + 神经元全量图 + 逐层梯度 |

## SDK API

| 方法 | 说明 |
|---|---|
| `Monitor(title, host, port, interval, history)` | `interval` 采样间隔秒；`history` 服务端保留的采样点数 |
| `info(fn)` | 顶部信息条，`fn()` 返回 `{键: 值}` |
| `metric(key, fn, label, color, fmt, chart)` | 标量指标；`chart=False` 只出卡片不出曲线 |
| `health(key, fn, label, unit, fmt)` | 健康度卡片 |
| `image(key, fn, label, cmap, scale, group)` | 真实图片；`cmap` 为空按灰度/原色，`group` 决定归到"特征图"还是"图片" |
| `histogram(key, fn, label, bins, refresh)` | 直方图；`refresh` 是重算间隔秒（大张量建议 ≥5） |
| `table(key, fn, label, columns)` | 表格；`fn()` 返回行列表 `[[...], ...]` 或 `{"columns": [...], "rows": [...]}`，行不足自动补空 |
| `matrix(key, fn, label, cmap, xlabels, ylabels, fmt, refresh)` | 二维矩阵热力图；`fn()` 返回 2D 数组（numpy/list 均可）。`cmap` 为空时含负数自动用发散色图 |
| `scatter(key, fn, label, xlabel, ylabel)` | 散点图；`fn()` 返回 `[[x, y, 类别], ...]` 或 `{"groups": [...], "points": [...]}`，上限 4 万点 |
| `bars(key, fn, label)` | 条图；`fn()` 返回 `[[标签, 数值], ...]` / `[数值, ...]` / `[{"label":.., "value":..}, ...]` |
| `log(key, fn, label, limit)` | 日志流；`fn()` 返回字符串列表或 `LogBuffer.get()` |
| `attach_logger(logger, key, label, level, limit)` | 把某个 `logging.Logger` 的输出实时镜像到网页，返回可继续 `write()` 的 `LogBuffer` |
| `alert(key, fn, label, level)` | 异常告警；`fn()` 返回字符串即告警，返回 `None` / `False` 表示正常；`level` 取 `"warn"` 或 `"error"` |
| `watchdog(key, fn, label, timeout, level)` | 看门狗；`fn()` 返回的数字在 `timeout` 秒内没有变化就告警 |
| `graph(fn)` | 返回 `{"layers": [...], "edges": [...]}`；`edges[i]` 形状必须是 `(目标层节点数, 源层节点数)` |
| `surface(fn)` | 损失曲面数据 |
| `params(fn)` | 底部参数表，返回 `[{"title": ..., "layers": [...], "rows": [[键, 值]]}]` |
| `control(start, stop)` | 网页启停按钮回调，返回 `True` 表示状态真的变了 |
| `start()` | 启动服务，返回实际端口 |
| `set_training(bool)` | 手动同步训练状态（用 `control` 时自动维护） |

`fmt` 可选：`auto` / `int` / `f1`~`f6` / `pct` / `sci`。

辅助函数：`montage(maps, cols, pad)` 铺图、`normalize(a, lo, hi)` 归一化、`colormap(u8, name)` 上色（内置 viridis / magma / blue / red / diverging / gray）、`encode_png(arr)` 无依赖编码 PNG、`LogBuffer(capacity)` 环形日志缓冲。

约定：`key` 相同的面板会被视为同一个；所有 `fn` 都在采样线程里调用，异常会被吞掉（页面保持上一帧），不影响训练。

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
  "tables":    [{"key":"report","label":"每类指标","columns":["类别","精确率"],"rows":[["cat",0.91]],"n":1}],
  "matrices":  [{"key":"cm","label":"混淆矩阵","h":10,"w":10,"data":[[205,37],[51,186]],"min":2,"max":208,
                 "xlabels":["cat","dog"],"ylabels":["cat","dog"],"cmap":null,"fmt":"int"}],
  "scatters":  [{"key":"emb","label":"特征投影","xlabel":"主成分 1","ylabel":"主成分 2",
                 "points":[[2.02,-0.36,0]],"groups":["螺旋 A","螺旋 B"],"n":1}],
  "bars":      [{"key":"grad","label":"逐层梯度范数","bars":[{"label":"body.0.weight","value":0.42}],"n":1}],
  "logs":      [{"key":"train_log","label":"训练日志","lines":[{"i":10,"t":1712345678.1,"text":"step 100 loss 0.8"}],"n":1}],
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
