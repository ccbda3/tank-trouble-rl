import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch

import config

_EDGE_CACHE = {"t": 0.0, "v": None}


def group_avg(a, n):
    a = np.asarray(a, dtype=float).ravel()
    if len(a) <= n:
        return [float(x) for x in a]
    idx = np.linspace(0, len(a), n + 1).astype(int)
    return [float(a[idx[i]:idx[i + 1]].mean()) for i in range(n)]


def bucket_edges(weight, conn, n_src, n_dst):
    w = torch.abs(weight)
    while w.dim() > 2:
        w = w.mean(-1)
    W = w.detach().cpu().numpy().astype(np.float32)
    t_n, s_n = W.shape
    gs = np.clip(((np.arange(s_n) + 0.5) * conn / n_src).astype(np.int64), 0, conn - 1)
    gt = np.clip(((np.arange(t_n) + 0.5) * conn / n_dst).astype(np.int64), 0, conn - 1)
    idx = (gt[:, None] * conn + gs[None, :]).ravel()
    acc = np.bincount(idx, weights=W.ravel(), minlength=conn * conn)
    cnt = np.bincount(idx, minlength=conn * conn)
    out = np.zeros(conn * conn, np.float32)
    np.divide(acc, cnt, out=out, where=cnt > 0)
    return out


def act_stats(net, obs):
    x = torch.tensor(np.asarray(obs, dtype=np.float32) / 255.0).unsqueeze(0).to(
        net.fc[0].weight.device
    )
    with torch.no_grad():
        c1a = net.net[1](net.net[0](x))
        c2a = net.net[3](net.net[2](c1a))
        c3a = net.net[5](net.net[4](c2a))
        c1 = c1a[0].mean((1, 2)).cpu().numpy()
        c2 = c2a[0].mean((1, 2)).cpu().numpy()
        c3 = c3a[0].mean((1, 2)).cpu().numpy()
        pool = net.net[6](c3a)
        f = net.fc[1](net.fc[0](torch.flatten(pool, 1)))
        fc = f[0].cpu().numpy()
        out = net.fc[2](f)[0].cpu().numpy()

    gray = np.asarray(obs, dtype=float).mean(0)
    rows, cols = 6, 8
    rh = gray.shape[0] // rows
    cw = gray.shape[1] // cols
    in_arr = []
    for r in range(rows):
        for c in range(cols):
            in_arr.append(float(gray[r * rh:(r + 1) * rh, c * cw:(c + 1) * cw].mean()))

    conn = 16

    def layer(name, detail, shown, tdim, params, outsz):
        arr = np.asarray(shown, dtype=float)
        return {
            "name": name,
            "detail": detail,
            "nodeCount": len(arr),
            "trueDim": int(tdim),
            "nodes": [float(v) for v in arr],
            "params": int(params),
            "out": outsz,
            "actMin": float(arr.min()) if arr.size else 0.0,
            "actMax": float(arr.max()) if arr.size else 0.0,
            "actMean": float(arr.mean()) if arr.size else 0.0,
        }

    def nparam(m):
        return sum(p.numel() for p in m.parameters())

    k1, k2, k3 = net.net[0].out_channels, net.net[2].out_channels, net.net[4].out_channels
    s1, s2, s3 = c1a.shape[-1], c2a.shape[-1], c3a.shape[-1]

    layers = [
        layer("输入", f"{config.GRID}×{config.GRID} 灰度 · 融合 {config.FRAME_STACK} 帧",
              in_arr, config.GRID * config.GRID, 0, f"{rows}×{cols} 网格采样"),
        layer("C1", f"{k1} 通道 @ {s1}×{s1} 卷积", group_avg(c1, 24), k1, nparam(net.net[0]),
              f"{k1} @ {s1}×{s1}"),
        layer("C2", f"{k2} 通道 @ {s2}×{s2} 卷积", group_avg(c2, 24), k2, nparam(net.net[2]),
              f"{k2} @ {s2}×{s2}"),
        layer("C3", f"{k3} 通道 @ {s3}×{s3} 卷积", group_avg(c3, 24), k3, nparam(net.net[4]),
              f"{k3} @ {s3}×{s3}"),
        layer("FC", f"全连接 {net.fc[0].out_features} · 展平 {int(net.fc[0].in_features)}",
              group_avg(fc, 24), net.fc[0].out_features, nparam(net.fc[0]),
              f"{net.fc[0].out_features} 特征"),
        layer("输出", "四个动作 Q 值", out.tolist(), config.ACTIONS, nparam(net.fc[2]),
              "上 / 下 / 左 / 右"),
    ]

    now = time.time()
    if _EDGE_CACHE["v"] is None or now - _EDGE_CACHE["t"] > 5.0:
        _EDGE_CACHE["v"] = [
            [float(net.net[0].weight.abs().mean().item())] * (conn * conn),
            bucket_edges(net.net[2].weight, conn, 24, 24).tolist(),
            bucket_edges(net.net[4].weight, conn, 24, 24).tolist(),
            bucket_edges(net.fc[0].weight, conn, 24, 24).tolist(),
            bucket_edges(net.fc[2].weight, conn, 24, 24).tolist(),
        ]
        _EDGE_CACHE["t"] = now
    edges = _EDGE_CACHE["v"]

    return {
        "ready": 1,
        "layers": layers,
        "edges": edges,
        "conn": conn,
        "value": float(out.max()) if out.size else 0.0,
    }


CFG = {
    "grid": config.GRID,
    "frames": config.FRAME_STACK,
    "actions": config.ACTIONS,
    "agents": config.AGENTS,
    "workers": config.WORKERS,
    "tickInterval": config.TICK_INTERVAL,
    "gamma": config.GAMMA,
    "lr": config.LR,
    "batch": config.BATCH,
    "accum": config.ACCUM,
    "replay": config.REPLAY_CAP,
    "targetSync": config.TARGET_SYNC,
    "alpha": config.ALPHA,
    "epsStart": config.EPS_START,
    "epsEnd": config.EPS_END,
    "epsDecay": config.EPS_DECAY,
    "betaFrames": config.BETA_FRAMES,
    "saveEvery": config.SAVE_EVERY,
    "ckptEvery": config.CHECKPOINT_INTERVAL,
    "connBucket": 16,
}

PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark">
<title>贪吃蛇 DQN 训练详情</title>
<style>
*{box-sizing:border-box}
html,body{margin:0;background:#0f1216;color:#c8d0da;font:12px/1.35 "Microsoft YaHei UI","Segoe UI",system-ui,sans-serif;overflow-x:hidden}
header{position:sticky;top:0;z-index:5;display:flex;align-items:center;gap:14px;flex-wrap:wrap;padding:7px 12px;background:#141920;border-bottom:1px solid #232a33}
.brand{font-size:13px;font-weight:700;color:#e8eef5;letter-spacing:.5px}
.stats{display:flex;gap:16px;flex-wrap:wrap;flex:1;min-width:0}
.st{display:flex;align-items:baseline;gap:5px}
.st em{font-style:normal;font-size:11px;color:#79828f}
.st b{font-size:13px;font-weight:700;color:#e2e9f1;font-variant-numeric:tabular-nums}
.conn{display:flex;align-items:center;gap:6px;font-size:11px;color:#79828f;white-space:nowrap}
.conn .dot{width:8px;height:8px;border-radius:50%;background:#4a5563}
.conn.ok .dot{background:#3ddc84;box-shadow:0 0 6px #3ddc84}
.conn.ok{color:#9fe6bf}
.conn.bad .dot{background:#ff5f56;box-shadow:0 0 6px #ff5f56}
.conn.bad{color:#ff9d97}
main{display:flex;flex-direction:column;gap:10px;padding:10px;min-width:0}
.card{min-width:0;background:#151a21;border:1px solid #232a33;border-radius:8px;padding:8px 10px 10px}
.chead{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;margin-bottom:8px}
.ctitle{font-size:13px;font-weight:700;color:#e8eef5;letter-spacing:.3px}
.csub{font-size:11px;color:#79828f;font-variant-numeric:tabular-nums}
.chint{margin-left:auto;font-size:11px;color:#5f6a78}
.charts{display:grid;gap:8px;grid-template-columns:repeat(5,minmax(0,1fr))}
@media(max-width:1500px){.charts{grid-template-columns:repeat(4,minmax(0,1fr))}}
@media(max-width:1180px){.charts{grid-template-columns:repeat(3,minmax(0,1fr))}}
@media(max-width:860px){.charts{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:560px){.charts{grid-template-columns:minmax(0,1fr)}}
.ch{min-width:0;overflow:hidden;background:#12171d;border:1px solid #1f262f;border-radius:6px;padding:5px 6px 4px;display:flex;flex-direction:column;gap:3px}
.cht{display:flex;align-items:baseline;gap:6px;font-size:11px;white-space:nowrap}
.chn{color:#93a0af}
.chv{margin-left:auto;font-weight:700;color:#dfe7ef;font-variant-numeric:tabular-nums}
.chc{display:block;width:100%;height:76px}
.chf{display:flex;gap:9px;font-size:10px;color:#66707d;font-variant-numeric:tabular-nums;white-space:nowrap}
.chf b{color:#8d99a7;font-weight:600}
.cards{display:grid;gap:8px;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));margin-bottom:0}
.cd{min-width:0;background:#12171d;border:1px solid #1f262f;border-radius:6px;padding:6px 9px}
.cd .ck{font-size:11px;color:#79828f}
.cd .cv{font-size:21px;font-weight:700;color:#dfe7ef;font-variant-numeric:tabular-nums;margin-top:3px}
.cd .cv.acc{color:#ff8f5a}.cd .cv.good{color:#5fd08a}.cd .cv.cy{color:#8fb9ff}
.panels{display:grid;gap:8px;grid-template-columns:minmax(0,1fr) minmax(0,1fr) minmax(0,1fr);align-items:start}
@media(max-width:1240px){.panels{grid-template-columns:minmax(0,1fr)}}
.pnl{min-width:0;background:#12171d;border:1px solid #1f262f;border-radius:6px;padding:6px 8px 8px}
.pnlh{font-size:11px;font-weight:700;color:#b9c4d1;margin:0 0 5px;border-bottom:1px solid #202730;padding-bottom:4px}
.pnlh.sub{margin-top:9px}
.row{display:flex;align-items:baseline;gap:10px;font-size:11px;padding:1px 0;min-width:0}
.row .rk{color:#79828f;white-space:nowrap}
.row .rv{margin-left:auto;color:#cbd4de;font-variant-numeric:tabular-nums;text-align:right;min-width:0;word-break:break-word}
.row .rv b{font-weight:700;color:#a9c8ee}
.row .rv b.dim{color:#8b95a3;font-weight:600}
.lyrs{display:flex;flex-direction:column;gap:1px}
.lyr{display:grid;grid-template-columns:52px minmax(0,1fr) 66px 66px;gap:6px;align-items:baseline;font-size:11px;font-variant-numeric:tabular-nums;padding:1px 0}
.lyr.hd{color:#5f6a78;font-size:10px;border-bottom:1px solid #1c232b;padding-bottom:3px;margin-bottom:2px}
.lyr.hd span:not(:first-child),.lyr span:not(.lnm):not(.lsh){text-align:right}
.lyr.tot{border-top:1px solid #232a33;margin-top:3px;padding-top:4px;color:#e8eef5;font-weight:700}
.lnm{color:#9db4cd;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.lsh{color:#8b95a3;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.lpr{color:#cbb98f;text-align:right;font-variant-numeric:tabular-nums}
.lot{color:#7f8b99;text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.vbtn{font:12px "Microsoft YaHei UI","Segoe UI",sans-serif;color:#9fb2c6;background:#151b22;border:1px solid #263039;border-radius:5px;padding:3px 10px;margin-left:6px;cursor:pointer}
.vbtn:hover{border-color:#3b4a58;color:#cfd8e3}
.vbtn.on{color:#0d1218;background:#5fb0e8;border-color:#5fb0e8;font-weight:700}
.netwrap{position:relative;width:100%;height:560px;background:radial-gradient(120% 100% at 50% 0%,#141b24 0%,#0d1218 72%);border:1px solid #1c232b;border-radius:6px;overflow:hidden}
#netcv{display:block;width:100%;height:100%;cursor:crosshair}
.nettip{position:absolute;display:none;z-index:4;max-width:264px;padding:6px 9px;background:rgba(11,15,20,.95);border:1px solid #2b3644;border-radius:5px;font-size:11px;line-height:1.55;color:#b9c4d1;pointer-events:none;box-shadow:0 8px 20px rgba(0,0,0,.55)}
.nettip.on{display:block}
.nettip .th{font-weight:700;color:#a9c8ee;margin-bottom:2px}
.nettip .th.sub{margin-top:5px;border-top:1px solid #202730;padding-top:4px}
.nettip .tr{display:flex;gap:10px;font-variant-numeric:tabular-nums;white-space:nowrap}
.nettip .tr i{margin-left:auto;font-style:normal;color:#8fd7f2}
.netlegend{display:flex;gap:16px;flex-wrap:wrap;margin-top:7px;font-size:11px;color:#79828f}
.netlegend i{display:inline-block;width:14px;height:3px;border-radius:2px;vertical-align:middle;margin-right:5px}
.gdwrap{position:relative;width:100%;height:340px;background:radial-gradient(130% 100% at 50% 0%,#141b24 0%,#0d1218 72%);border:1px solid #1c232b;border-radius:6px;overflow:hidden}
#gdcv{display:block;width:100%;height:100%;cursor:grab;touch-action:none}
.hint{font-size:11px;color:#79828f;margin-top:6px;display:flex;gap:16px;flex-wrap:wrap}
.hint i{display:inline-block;width:14px;height:3px;border-radius:2px;vertical-align:middle;margin-right:5px}
.help{font-size:12px;color:#79828f;line-height:1.9;margin-top:6px}
.help b{color:#c8d0da}
@media(max-width:860px){.netwrap{height:440px}}
</style>
</head>
<body>
<header>
<span class="brand">贪吃蛇 · DQN 强化学习训练监控</span>
<div class="stats" id="stats"></div>
<div class="conn" id="conn"><i class="dot"></i><span id="connTxt">连接中</span></div>
</header>
<main>
<section class="card">
<div class="chead"><span class="ctitle">实时指标</span><span class="csub" id="cardSub">等待数据…</span></div>
<div class="cards" id="cards"></div>
</section>
<section class="card">
<div class="chead"><span class="ctitle">训练曲线</span><span class="csub" id="chartSub">等待数据…</span></div>
<div class="charts" id="charts"></div>
</section>
<section class="card">
<div class="chead"><span class="ctitle">神经网络连接强度</span><span class="csub" id="netSub">等待数据…</span><span class="chint">每秒取一次真实前向快照 · 连线亮度 = 权重 × 源激活 · 3D 可拖拽旋转/滚轮缩放 · 悬停节点看细节</span><button class="vbtn on" id="v-3d" data-view="3d">3D 立体</button><button class="vbtn" id="v-2d" data-view="2d">2D 平面</button></div>
<div class="netwrap" id="netwrap">
<canvas id="netcv"></canvas>
<div class="nettip" id="nettip"></div>
</div>
<div class="netlegend">
<span><i style="background:linear-gradient(90deg,#101e2a,#2f6f97,#8ce0fa)"></i>节点颜色 = 该层激活强度（按本层节点值 min~max 归一）</span>
<span><i style="background:linear-gradient(90deg,rgba(126,170,216,.05),rgba(126,170,216,.9))"></i>连线明暗与粗细 = 权重 × 源节点激活，越强越亮</span>
<span><i style="background:#ffb020"></i>最强通路</span>
<span>每秒随训练快照刷新一次（拖动/缩放时即时重绘）</span>
</div>
</section>
<section class="card">
<div class="chead"><span class="ctitle">详细参数</span><span class="csub" id="paramSub">等待数据…</span></div>
<div class="panels" id="panels"><div class="pnl"><div class="pnlh">参数</div><div class="row"><span class="rk">状态</span><span class="rv">等待数据…</span></div></div></div>
</section>
<section class="card">
<div class="chead"><span class="ctitle">损失曲面 3D · 真实测量</span><span class="csub" id="gdSub">等待首次测量…</span><span class="chint">每 20 秒在真实权重周围扰动采样一次网格 · 拖动旋转 · 滚轮缩放</span></div>
<div class="gdwrap" id="gdwrap"><canvas id="gdcv"></canvas></div>
<div class="hint">
<span><i style="background:linear-gradient(90deg,#182c48,#4a96c6,#a6e0f6)"></i>曲面高度 = 在固定评估批上实测的 TD 损失，低暗高亮</span>
<span><i style="background:#ffb020"></i>橙色轨迹 = 真实优化器走过的权重位移，投影到这两个固定随机方向上</span>
<span><i style="background:#3ddc84"></i>绿点 = 当前权重位置</span>
</div>
</section>
<section class="card">
<div class="chead"><span class="ctitle">通俗解释 · 怎么看这些数字</span></div>
<div class="help" id="help"></div>
</section>
</main>
<script>
(function(){
"use strict";
function el(id){return document.getElementById(id);}
function fInt(v){var n=Number(v);return isFinite(n)?Math.round(n).toLocaleString("en-US"):"-";}
function fFix(v,d){var n=Number(v);return isFinite(n)?n.toFixed(d):"-";}
function fSci(v,sig){var n=Number(v);if(!isFinite(n))return"-";if(n===0)return"0";var e=Math.floor(Math.log(Math.abs(n))/Math.LN10);var m=n/Math.pow(10,e);return m.toFixed((sig||3)-1)+"e"+e;}
function fSmart(v){var n=Number(v);if(!isFinite(n))return"-";if(n===0)return"0";var a=Math.abs(n);if(a<1e-3)return fSci(n,3);if(a>=1e5)return fInt(n);if(a>=1e3)return n.toLocaleString("en-US",{maximumFractionDigits:1});return String(+n.toFixed(5));}
function esc(s){return String(s===null||s===undefined?"":s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");}
function row(k,v){return '<div class="row"><span class="rk">'+k+'</span><span class="rv">'+v+'</span></div>';}
function setT(id,txt){var e=el(id);if(e) e.textContent=txt;}
var charts=[];
function buildCharts(){
 var host=el("charts"),h="",i;
 for(i=0;i<METRICS.length;i++){
  h+='<div class="ch"><div class="cht"><span class="chn">'+METRICS[i].n+'</span><b class="chv">-</b></div><canvas class="chc"></canvas>'
     +'<div class="chf"><span>最小 <b class="f0">-</b></span><span>最大 <b class="f1">-</b></span><span>平均 <b class="f2">-</b></span></div></div>';
 }
 host.innerHTML=h;
 var nodes=host.querySelectorAll(".ch");
 for(i=0;i<nodes.length;i++){
  var n=nodes[i],cv=n.querySelector(".chc");
  charts.push({m:METRICS[i],cv:cv,ctx:cv.getContext("2d"),v:n.querySelector(".chv"),f0:n.querySelector(".f0"),f1:n.querySelector(".f1"),f2:n.querySelector(".f2"),win:[]});
 }
}
function drawChart(ch){
 var cv=ch.cv,ctx=ch.ctx,w=cv.clientWidth|0,h=cv.clientHeight|0;
 if(w<24||h<24)return;
 var dpr=Math.min(window.devicePixelRatio||1,2),nw=Math.round(w*dpr),nh=Math.round(h*dpr);
 if(cv.width!==nw||cv.height!==nh){cv.width=nw;cv.height=nh;}
 ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
 var win=ch.win||[],n=win.length;
 if(n<1)return;
 var i,ymin=Infinity,ymax=-Infinity,xmin=win[0][0],xmax=win[n-1][0];
 for(i=0;i<n;i++){var v=win[i][1];if(v<ymin)ymin=v;if(v>ymax)ymax=v;}
 if(ymin===ymax){var e=Math.abs(ymin)*0.05+0.5;ymin-=e;ymax+=e;}
 var pad=(ymax-ymin)*0.12;ymin-=pad;ymax+=pad;
 if(xmax===xmin)xmax=xmin+1;
 var L=2,R=w-2,T=6,B=h-12;
 function px(it){return L+(it-xmin)/(xmax-xmin)*(R-L);}
 function py(v){return B-(v-ymin)/(ymax-ymin)*(B-T);}
 ctx.strokeStyle="rgba(255,255,255,.045)";ctx.lineWidth=1;
 for(i=0;i<=3;i++){var gy=T+(B-T)*i/3;ctx.beginPath();ctx.moveTo(L,Math.round(gy)+.5);ctx.lineTo(R,Math.round(gy)+.5);ctx.stroke();}
 ctx.beginPath();
 for(i=0;i<n;i++){var X=px(win[i][0]),Y=py(win[i][1]);if(i===0)ctx.moveTo(X,Y);else ctx.lineTo(X,Y);}
 var last=win[n-1],lx=px(last[0]),ly=py(last[1]);
 var grad=ctx.createLinearGradient(0,T,0,B);
 grad.addColorStop(0,ch.m.c+"2e");grad.addColorStop(1,ch.m.c+"00");
 ctx.lineTo(lx,B);ctx.lineTo(px(win[0][0]),B);ctx.closePath();ctx.fillStyle=grad;ctx.fill();
 ctx.beginPath();
 for(i=0;i<n;i++){var X2=px(win[i][0]),Y2=py(win[i][1]);if(i===0)ctx.moveTo(X2,Y2);else ctx.lineTo(X2,Y2);}
 ctx.strokeStyle=ch.m.c;ctx.lineWidth=1.5;ctx.lineJoin="round";ctx.lineCap="round";ctx.stroke();
 ctx.beginPath();ctx.arc(lx,ly,2.6,0,Math.PI*2);ctx.fillStyle=ch.m.c;ctx.fill();
 ctx.font='9px "Microsoft YaHei UI","Segoe UI",sans-serif';ctx.fillStyle="#5b6672";
 ctx.textAlign="left";ctx.fillText(ch.m.f(ymax),3,T+8);ctx.fillText(ch.m.f(ymin),3,B-2);
 ctx.textAlign="left";ctx.fillText(fInt(xmin),3,h-3);ctx.textAlign="right";ctx.fillText(fInt(xmax),w-2,h-3);
}
var METRICS=[
 {k:"loss",n:"Loss",c:"#ff8f5a",f:fSmart},
 {k:"avg",n:"平均蛇长",c:"#3ddc84",f:function(v){var n=Number(v);return isFinite(n)?n.toFixed(2):"-";}},
 {k:"best",n:"最长蛇身",c:"#ffd36a",f:function(v){var n=Number(v);return isFinite(n)?n.toFixed(1):"-";}},
 {k:"eps",n:"探索率 ε",c:"#8fb9ff",f:function(v){var n=Number(v);return isFinite(n)?n.toFixed(3):"-";}},
 {k:"gpu",n:"GPU 利用率",c:"#cf7fff",f:function(v){var n=Number(v);return isFinite(n)?n.toFixed(0)+"%":"-";}},
 {k:"dps",n:"决策/秒",c:"#7fd0e0",f:function(v){var n=Number(v);return isFinite(n)?n.toLocaleString("en-US"):"-";}}
];
var KEYS=[["steps","总步数",""],["eps","探索率 ε","cy"],["trainSteps","训练步数",""],["buffer","经验池容量",""],
 ["loss","Loss","acc"],["score","本局蛇长","good"],["avgScore","平均蛇长","good"],["bestScore","最长蛇身","good"],
 ["dps","决策/秒",""],["gpu","GPU 利用率","acc"],["episodes","完成局数",""]];
function fmt(k,d){var v=d[k];switch(k){case"eps":return Number(v).toFixed(3);case"loss":return Number(v).toFixed(4);case"gpu":return Number(v)>0?Number(v)+"%":"CPU";case"dps":return Number(v).toLocaleString("en-US");default:return Number(v).toLocaleString("en-US");}}
var statRefs=[];
function renderStats(d){
 var h="";
 for(var i=0;i<KEYS.length;i++){
  var k=KEYS[i][0];h+='<div class="cd"><div class="ck">'+KEYS[i][1]+'</div><div class="cv '+KEYS[i][2]+'">'+fmt(k,d)+'</div></div>';
 }
 el("cards").innerHTML=h;
 el("cardSub").textContent="第 "+fInt(d.episodes)+" 局 · 总步数 "+fInt(d.steps)+" · 并行 "+fInt(d.agents)+" 局 / "+fInt((d.config||{}).workers||0)+" 线程";
 el("help").innerHTML=[
  ["Loss（损失）","网络猜错的程度，越低越会玩；训练就是让小球往谷底滚把 Loss 压低。"],
  ["探索率 ε","从 1 慢慢降到接近 0。前期疯狂乱走试错，后期主要靠学到的策略。"],
  ["平均蛇长 / 最长蛇身","这局蛇吃到几个苹果、长到多长，越大越会玩，是吃苹果能力的进度条。"],
  ["神经网络图","训练时网络在思考：节点颜色=本层激活强弱，亮的连线=正在算的数据通路。"],
  ["决策/秒","所有后台智能体每秒作出的动作数，越大说明并行利用越满。"]
 ].map(function(x){return"<div><b>"+x[0]+"</b>："+x[1]+"</div>";}).join("");
}
var HIST=[];
function pushHistory(d){
 HIST.push({s:d.steps,loss:d.loss,avg:d.avgScore,best:d.bestScore,eps:d.eps,gpu:d.gpu,dps:d.dps});
 if(HIST.length>240)HIST.splice(0,HIST.length-240);
}
function renderCharts(){
 for(var i=0;i<charts.length;i++){
  var ch=charts[i],win=[],j;
  for(j=0;j<HIST.length;j++){var v=Number(HIST[j][ch.m.k]);if(ch.m.k==="gpu"&&v<0)continue;if(isFinite(v))win.push([HIST[j].s,v]);}
  ch.win=win;drawChart(ch);
  var n=win.length;
  if(n){var s=0,mn=Infinity,mx=-Infinity,k;for(k=0;k<n;k++){s+=win[k][1];if(win[k][1]<mn)mn=win[k][1];if(win[k][1]>mx)mx=win[k][1];}
   ch.v.textContent=ch.m.f(win[n-1][1]);ch.f0.textContent=ch.m.f(mn);ch.f1.textContent=ch.m.f(mx);ch.f2.textContent=ch.m.f(s/n);}
  else{ch.v.textContent="-";ch.f0.textContent="-";ch.f1.textContent="-";ch.f2.textContent="-";}
 }
 el("chartSub").textContent="历史 "+fInt(HIST.length)+" 点 · 每 1 秒刷新";
}
var panelSig=null;
function panelSigOf(d){var c=d.config||{},L=(d.acts&&d.acts.layers)||[];return JSON.stringify([c,c.L?0:L.map(function(l){return l.params+"/"+l.trueDim;})]);}
function buildPanels(d){
 var c=d.config||{},L=(d.acts&&d.acts.layers)||[];
 var h='<div class="pnl"><div class="pnlh">网络结构 · Double DQN</div><div class="lyrs">';
 h+='<div class="lyr hd"><span>层</span><span>结构</span><span>参数量</span><span>输出</span></div>';
 var i,tot=0;
 for(i=0;i<L.length;i++){var ly=L[i];tot+=ly.params;
  h+='<div class="lyr"><span class="lnm" title="'+esc(ly.name)+'">'+esc(ly.name)+'</span><span class="lsh" title="'+esc(ly.detail)+'">'+esc(ly.detail)+'</span><span class="lpr">'+fInt(ly.params)+'</span><span class="lot">'+esc(ly.out)+'</span></div>';
  if(i<L.length-1){h+='<div class="lyr"><span class="lnm">↘ 连接</span><span class="lsh">'+fInt(ly.nodeCount)+' → '+fInt(L[i+1].nodeCount)+' 桶 · 权重绝对均值采样</span><span class="lpr">-</span><span class="lot">桶化 '+fInt(c.connBucket)+'×'+fInt(c.connBucket)+'</span></div>';}
 }
 h+='<div class="lyr tot"><span class="lnm">合计</span><span class="lsh">'+fInt(L.length)+' 层</span><span class="lpr">'+fInt(tot)+'</span><span class="lot">'+fInt(c.actions)+' 动作</span></div>';
 h+='</div><div class="row" style="margin-top:5px"><span class="rk">算力</span><span class="rv"><b>'+(d.gpu>0?"GPU 加速":"CPU")+'</b></span></div></div>';
 h+='<div class="pnl"><div class="pnlh">DQN 超参数</div>';
 h+=row("Double DQN",'<b class="yes">启用</b>');
 h+=row("优先级经验池","SumTree · α="+fFix(c.alpha,2));
 h+=row("折扣因子 gamma",fFix(c.gamma,3));
 h+=row("学习率 lr",fSci(c.lr,3));
 h+=row("批大小 × 梯度累积",fInt(c.batch)+" × "+fInt(c.accum));
 h+=row("目标网络同步",fInt(c.targetSync)+" 步");
 h+=row("经验池容量",fInt(c.replay));
 h+=row("探索率区间",fFix(c.epsStart,2)+" → "+fFix(c.epsEnd,2));
 h+=row("探索衰减总更新数",fInt(c.epsDecay)+" 次梯度更新");
 h+=row("当前探索率",'<b id="p-eps">-</b>');
 h+=row("当前经验池",'<b id="p-buf">-</b>');
 h+='</div>';
 h+='<div class="pnl"><div class="pnlh">运行配置</div>';
 h+=row("棋盘 GRID",fInt(c.grid)+" × "+fInt(c.grid));
 h+=row("堆叠帧",fInt(c.frames)+" 帧");
 h+=row("动作数",fInt(c.actions));
 h+=row("并行环境 × 采集线程",fInt(c.agents)+" 局 × "+fInt(c.workers)+" 线程");
 h+=row("采集节流",(Number(c.tickInterval)*1000).toFixed(1)+" ms/批 · 约 "+fInt(Number(c.agents)/Math.max(Number(c.tickInterval),1e-6))+" 步/秒");
 h+=row("帧堆叠 / 采样率","DQN 每帧决策");
 h+=row("保存权重间隔",fInt(c.ckptEvery)+" 秒");
 h+=row("完整续训保存",fInt(c.saveEvery)+" 步 + 60 秒 + 关窗");
 h+=row("β 退火帧数",fInt(c.betaFrames));
 h+='<div class="row"><span class="rk">训练步数</span><span class="rv"><b id="p-train">-</b></span></div>';
 h+='<div class="row"><span class="rk">训练/采样比</span><span class="rv"><b id="p-ratio">-</b></span></div>';
 h+='<div class="row"><span class="rk">Loss</span><span class="rv"><b id="p-loss">-</b></span></div>';
 h+='<div class="row"><span class="rk">GPU</span><span class="rv"><b id="p-gpu">-</b></span></div>';
 h+='</div>';
 el("panels").innerHTML=h;
}
function updatePanels(d){
 setT("p-eps",fFix(d.eps,3));
 setT("p-buf",fInt(d.buffer));
 setT("p-train",fInt(d.trainSteps));
 var cc=d.config||{},bs=Number(cc.batch)||1,st=Number(d.steps)||1;
 setT("p-ratio",(Number(d.trainSteps)*bs/st).toFixed(2)+" ×");
 setT("p-loss",fFix(d.loss,4));
 setT("p-gpu",d.gpu>0?d.gpu+"%":"CPU 训练");
 el("paramSub").textContent="第 "+fInt(d.episodes)+" 局 · 步数 "+fInt(d.steps)+" · "+fFix(d.loss,4)+" Loss";
}
var NET={ready:false,layers:[],edges:[],conn:0,act:[],pick:[],maxS:1,hover:null,hoverKey:"",view:null,mode:"3d",iter:0,samples:0,value:0,budget:480,cv:null,ctx:null,pairs:[[0,1],[1,2],[2,3],[3,4],[4,5]],cam:{yaw:-1.05,pitch:0.30,dist:40.0,drag:false,lx:0,ly:0}};
var NET_FONT='10px "Microsoft YaHei UI","Segoe UI",sans-serif';
function netBucket(b,n,conn){var v=Math.floor((b+0.5)*conn/n);return v<0?0:(v>conn-1?conn-1:v);}
function netColorRange(ly){var cnt=Math.max(1,ly.nodeCount|0),raw=ly.nodes||[],lo=Infinity,hi=-Infinity,j,v;for(j=0;j<cnt;j++){v=Number(raw[j]);if(!isFinite(v))continue;if(v<lo)lo=v;if(v>hi)hi=v;}return isFinite(lo)&&isFinite(hi)?[lo,hi]:[0,0];}
function netColor(t){t=t<0?0:(t>1?1:t);var r,g,b;if(t<0.5){var u=t/0.5;r=16+(47-16)*u;g=30+(111-30)*u;b=42+(151-42)*u;}else{var v=(t-0.5)/0.5;r=47+(140-47)*v;g=111+(224-111)*v;b=151+(250-151)*v;}return "rgb("+(r|0)+","+(g|0)+","+(b|0)+")";}
function netNodeAct(ly){var cnt=Math.max(1,ly.nodeCount|0),raw=ly.nodes||[],sc=netColorRange(ly),span=sc[1]-sc[0],out=[],j,v;for(j=0;j<cnt;j++){v=Number(raw[j]);out.push(isFinite(v)?(span>1e-12?((v-sc[0])/span<0?0:((v-sc[0])/span>1?1:(v-sc[0])/span)):0.5):0.5);}return out;}
function buildNetPick(){
 var L=NET.layers,E=NET.edges,conn=NET.conn,n=L.length,i,e,j,k;
 NET.act=[];
 for(i=0;i<n;i++)NET.act.push(netNodeAct(L[i]));
 var pairs=Math.max(1,NET.pairs.length),per=Math.max(20,Math.round(NET.budget/pairs)),all=[];
 for(e=0;e<pairs;e++){
  var pr=NET.pairs[e]||[e,e+1];
  if(!L[pr[0]]||!L[pr[1]])continue;
  var src=NET.act[pr[0]],sN=Math.max(1,L[pr[0]].nodeCount|0),tN=Math.max(1,L[pr[1]].nodeCount|0);
  var M=E[e]||[],cand=[];
  for(j=0;j<sN;j++){
   var sj=netBucket(j,sN,conn),sa=src[j];
   for(k=0;k<tN;k++){
    var w=Number(M[netBucket(k,tN,conn)*conn+sj]);
    if(!isFinite(w)||w<=0)continue;
    cand.push({e:e,s:pr[0],t:pr[1],j:j,i:k,w:w,a:sa,v:w*(0.28+0.72*sa)});
   }
  }
  cand.sort(function(p,q){return q.v-p.v;});
  var lim=Math.min(per,cand.length);
  for(j=0;j<lim;j++)all.push(cand[j]);
 }
 all.sort(function(p,q){return p.v-q.v;});
 NET.pick=all;
 NET.strong=all.length?all[all.length-1]:null;
 NET.maxS=1e-9;
 for(i=0;i<all.length;i++)if(all[i].v>NET.maxS)NET.maxS=all[i].v;
}
function prepNet(d){
 NET.ready=false;NET.layers=[];NET.edges=[];NET.conn=0;NET.act=[];NET.pick=[];NET.strong=null;
 if(!d||Number(d.ready)!==1)return;
 var L=d.layers||[],conn=Number(d.conn)|0;
 if(!L.length||conn<2)return;
 NET.layers=L;NET.edges=d.edges||[];NET.conn=conn;
 NET.iter=Number(d.iteration);NET.samples=Number(d.samples);NET.value=Number(d.value);
 NET.ready=true;buildNetPick();
}
function netEmptyFrame(ctx,w,h){
 ctx.save();ctx.setLineDash([6,5]);ctx.strokeStyle="rgba(120,140,168,.24)";ctx.lineWidth=1;ctx.strokeRect(15.5,15.5,w-31,h-31);ctx.restore();
 ctx.textAlign="center";ctx.textBaseline="middle";ctx.font='13px "Microsoft YaHei UI","Segoe UI",sans-serif';ctx.fillStyle="#5f6a78";ctx.fillText("正在采集网络激活…",w*0.5,h*0.5-11);
 ctx.font='11px "Microsoft YaHei UI","Segoe UI",sans-serif';ctx.fillStyle="#4a545f";ctx.fillText("等待训练循环产生前向快照",w*0.5,h*0.5+11);ctx.textBaseline="alphabetic";
}
function netCam(yaw,pitch,dist){var cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);var ex=cp*cy,ey=cp*sy,ez=sp;return {ox:ex*dist,oy:ey*dist,oz:ez*dist,fx:-ex,fy:-ey,fz:-ez,rx:-sy,ry:cy,rz:0,ux:-cy*sp,uy:-sy*sp,uz:cp};}
function netProj(c,x,y,z){var vx=x-c.ox,vy=y-c.oy,vz=z-c.oz;var cz=vx*c.fx+vy*c.fy+vz*c.fz;if(cz<0.15)cz=0.15;var sx=vx*c.rx+vy*c.ry+vz*c.rz;var uy=vx*c.ux+vy*c.uy+vz*c.uz;return [sx/cz,uy/cz,cz];}
function netGrid(count){if(count<=1)return[1,1];var cols=Math.max(1,Math.ceil(Math.sqrt(count*1.9)));return[cols,Math.ceil(count/cols)];}
function netLayout(){var L=NET.layers,n=L.length,i,j,gap=2.6,pitch=0.40,zSpan=(n-1)*gap,boxes=[];for(i=0;i<n;i++){var cnt=Math.max(1,L[i].nodeCount|0);var cr=netGrid(cnt),cols=cr[0],rows=cr[1];var x=i*gap-zSpan*0.5;var pts=[];for(j=0;j<cnt;j++){var g=Math.floor(j/cols),c=j%cols;pts.push([(c-(cols-1)*0.5)*pitch,((rows-1)*0.5-g)*pitch]);}boxes.push({x:x,cols:cols,rows:rows,pts:pts,cnt:cnt,halfY:(cols-1)*0.5*pitch,halfZ:(rows-1)*0.5*pitch});}return {boxes:boxes,gap:gap,zSpan:zSpan};}
function netFitFocal(lay,w,h){var c=netCam(NET.cam.yaw,NET.cam.pitch,40.0);var mx=1e-4,my=1e-4,i,k;for(i=0;i<lay.boxes.length;i++){var b=lay.boxes[i],ey=b.halfY+0.30,ez=b.halfZ+0.30;var cand=[[b.x,ey,ez],[b.x,-ey,ez],[b.x,ey,-ez],[b.x,-ey,-ez]];for(k=0;k<4;k++){var p=netProj(c,cand[k][0],cand[k][1],cand[k][2]);if(Math.abs(p[0])>mx)mx=Math.abs(p[0]);if(Math.abs(p[1])>my)my=Math.abs(p[1]);}}return Math.min(w*0.46/mx,h*0.42/my);}
function drawNet3D(){
 var cv=NET.cv,ctx=NET.ctx;if(!cv||!ctx)return;
 var w=cv.clientWidth|0,h=cv.clientHeight|0;if(w<40||h<40)return;
 var dpr=Math.min(window.devicePixelRatio||1,2),nw=Math.round(w*dpr),nh=Math.round(h*dpr);
 if(cv.width!==nw||cv.height!==nh){cv.width=nw;cv.height=nh;}
 ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
 if(!NET.ready){NET.view=null;netEmptyFrame(ctx,w,h);return;}
 var L=NET.layers,n=L.length,i,j,lay=netLayout();
 var FOCAL=netFitFocal(lay,w,h),c=netCam(NET.cam.yaw,NET.cam.pitch,NET.cam.dist);
 var ox=w*0.5,oy=h*0.52,act=NET.act||[],scr=[],flat=[];
 for(i=0;i<n;i++){var b=lay.boxes[i],A=act[i]||netNodeAct(L[i]),col=[];
  for(j=0;j<b.cnt;j++){var pt=b.pts[j],p=netProj(c,b.x,pt[0],pt[1]),nd={x:ox+p[0]*FOCAL,y:oy-p[1]*FOCAL,z:p[2],r:(0.055*FOCAL/p[2])<1?1:((0.055*FOCAL/p[2])>6.5?6.5:0.055*FOCAL/p[2]),li:i,bi:j,a:A[j]};col.push(nd);flat.push(nd);}
  scr.push(col);}
 NET.view={nodes:flat,cols:null};
 var zmin=Infinity,zmax=-Infinity;for(i=0;i<flat.length;i++){if(flat[i].z<zmin)zmin=flat[i].z;if(flat[i].z>zmax)zmax=flat[i].z;}var zspan=(zmax-zmin)||1;
 function depthF(z){var q=(z-zmin)/zspan;return q<0?0:(q>1?1:q);}
 var pick=NET.pick,edges=[];
 for(i=0;i<pick.length;i++){var q=pick[i],A2=scr[q.s]&&scr[q.s][q.j],B2=scr[q.t]&&scr[q.t][q.i];if(!A2||!B2)continue;var st=q.v/(NET.maxS||1);if(st<0.03)continue;edges.push({A:A2,B:B2,st:st,z:(A2.z+B2.z)*0.5});}
 edges.sort(function(p,q2){return q2.z-p.z;});
 ctx.lineCap="round";
 for(i=0;i<edges.length;i++){
  var ed=edges[i],fz=depthF(ed.z),fade=1-0.55*fz;
  ctx.strokeStyle="rgba(150,205,246,"+((0.06+0.68*ed.st)*fade).toFixed(3)+")";
  ctx.lineWidth=0.5+1.7*ed.st;
  ctx.beginPath();ctx.moveTo(ed.A.x,ed.A.y);ctx.lineTo(ed.B.x,ed.B.y);ctx.stroke();
 }
 ctx.lineCap="butt";
 var order=[];for(i=0;i<flat.length;i++)order.push(i);
 order.sort(function(p,q2){return flat[q2].z-flat[p].z;});
 for(var k=0;k<order.length;k++){var nd2=flat[order[k]],fade2=1-0.45*depthF(nd2.z);
  if(nd2.a>0.5){ctx.beginPath();ctx.arc(nd2.x,nd2.y,nd2.r+1.2+2.4*nd2.a,0,Math.PI*2);ctx.fillStyle="rgba(150,214,255,"+(0.05+0.24*nd2.a*fade2).toFixed(3)+")";ctx.fill();}
  ctx.beginPath();ctx.arc(nd2.x,nd2.y,nd2.r,0,Math.PI*2);ctx.fillStyle=netColor(nd2.a);ctx.fill();}
 var st2=NET.strong;
 if(st2){var SA=scr[st2.s]&&scr[st2.s][st2.j],SB=scr[st2.t]&&scr[st2.t][st2.i];if(SA&&SB){ctx.save();ctx.strokeStyle="rgba(255,176,32,.85)";ctx.lineWidth=1.8;ctx.shadowColor="rgba(255,176,32,.8)";ctx.shadowBlur=9;ctx.beginPath();ctx.moveTo(SA.x,SA.y);ctx.lineTo(SB.x,SB.y);ctx.stroke();ctx.restore();}}
 ctx.textAlign="center";ctx.textBaseline="alphabetic";
 for(i=0;i<n;i++){var bb=lay.boxes[i],lyr=L[i],lp=netProj(c,bb.x,0,bb.halfZ+0.42);var lx=ox+lp[0]*FOCAL,ly=oy-lp[1]*FOCAL;var al2=0.55+0.45*(1-depthF(lp[2]));ctx.font='700 12px "Microsoft YaHei UI","Segoe UI",sans-serif';ctx.fillStyle="rgba(223,231,239,"+al2.toFixed(2)+")";ctx.fillText(String(lyr.name||""),lx,ly);ctx.font=NET_FONT;ctx.fillStyle="rgba(139,149,163,"+al2.toFixed(2)+")";ctx.fillText(String(lyr.detail||""),lx,ly+13);ctx.fillStyle="rgba(102,112,125,"+al2.toFixed(2)+")";ctx.fillText("激活均值 "+fSmart(Number(lyr.actMean)),lx,ly+25);}
 var hv=NET.hover;
 if(hv&&scr[hv.li]&&scr[hv.li][hv.bi]){var hn=scr[hv.li][hv.bi];ctx.strokeStyle="#e8eef5";ctx.lineWidth=1.4;ctx.beginPath();ctx.arc(hn.x,hn.y,hn.r+3.4,0,Math.PI*2);ctx.stroke();}
 ctx.font=NET_FONT;ctx.textAlign="left";ctx.fillStyle="rgba(170,208,240,.8)";
 ctx.fillText("3D 立体 · 连线亮度 = 权重 × 源激活 · 每秒刷新 · 拖动旋转 · 滚轮缩放",8,14);
}
function drawNet2D(){
 var cv=NET.cv,ctx=NET.ctx;if(!cv||!ctx)return;
 var w=cv.clientWidth|0,h=cv.clientHeight|0;if(w<40||h<40)return;
 var dpr=Math.min(window.devicePixelRatio||1,2),nw=Math.round(w*dpr),nh=Math.round(h*dpr);
 if(cv.width!==nw||cv.height!==nh){cv.width=nw;cv.height=nh;}
 ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
 if(!NET.ready){NET.view=null;netEmptyFrame(ctx,w,h);return;}
 var L=NET.layers,n=L.length,i,j;
 var padX=Math.max(34,Math.min(74,w*0.05));
 var step=n>1?(w-padX*2)/(n-1):0;
 var yTop=60,yBot=h-80;if(yBot-yTop<100){yTop=48;yBot=h-64;}
 var cols=[],nodes=[];
 for(i=0;i<n;i++){var ly=L[i],cnt=Math.max(1,ly.nodeCount|0),x=padX+step*i;var sp=cnt>1?(yBot-yTop)/(cnt-1):0;var rad=cnt>1?((sp*0.34)<1.5?1.5:(sp*0.34>5.2?5.2:sp*0.34)):6.5;var ys=[];for(j=0;j<cnt;j++)ys.push(cnt>1?yTop+sp*j:(yTop+yBot)*0.5);var col={x:x,ys:ys,r:rad,act:NET.act[i]||netNodeAct(ly),sc:netColorRange(ly),ly:ly};cols.push(col);for(j=0;j<ys.length;j++)nodes.push({x:x,y:ys[j],r:rad,li:i,bi:j,a:col.act[j]});}
 NET.view={nodes:nodes,cols:cols};
 var pick=NET.pick,maxS=NET.maxS||1;
 ctx.lineCap="round";
 for(i=0;i<pick.length;i++){
  var q=pick[i],qa=cols[q.s],qb=cols[q.t];
  if(!qa||!qb)continue;
  var al=q.v/maxS;
  if(al<0.03)continue;
  ctx.strokeStyle="rgba(150,205,246,"+(0.06+0.70*al).toFixed(3)+")";
  ctx.lineWidth=0.5+1.7*al;
  ctx.beginPath();ctx.moveTo(qa.x+qa.r,qa.ys[q.j]);ctx.lineTo(qb.x-qb.r,qb.ys[q.i]);ctx.stroke();
 }
 ctx.lineCap="butt";
 for(i=0;i<n;i++){var cc=cols[i];
  for(j=0;j<cc.ys.length;j++){var a=cc.act[j];
   if(a>0.5){ctx.beginPath();ctx.arc(cc.x,cc.ys[j],cc.r+1.2+2.6*a,0,Math.PI*2);ctx.fillStyle="rgba(150,214,255,"+(0.05+0.26*a).toFixed(3)+")";ctx.fill();}
   ctx.beginPath();ctx.arc(cc.x,cc.ys[j],cc.r,0,Math.PI*2);ctx.fillStyle=netColor(a);ctx.fill();}}
 var st=NET.strong;
 if(st){var sa=cols[st.s],sb=cols[st.t];var x0=sa.x+sa.r,y0=sa.ys[st.j],x1=sb.x-sb.r,y1=sb.ys[st.i];ctx.save();ctx.strokeStyle="#ffb020";ctx.lineWidth=2;ctx.shadowColor="rgba(255,176,32,.85)";ctx.shadowBlur=11;ctx.beginPath();ctx.moveTo(x0,y0);ctx.lineTo(x1,y1);ctx.stroke();ctx.restore();ctx.strokeStyle="rgba(255,213,138,.95)";ctx.lineWidth=1.4;ctx.beginPath();ctx.arc(sa.x,y0,sa.r+3.4,0,Math.PI*2);ctx.stroke();ctx.beginPath();ctx.arc(sb.x,y1,sb.r+3.4,0,Math.PI*2);ctx.stroke();ctx.textAlign="left";ctx.textBaseline="middle";ctx.font=NET_FONT;ctx.fillStyle="#ffd58a";ctx.fillText("最强通路 强度 "+st.v.toFixed(3),"6",h-8);ctx.textBaseline="alphabetic";}
 ctx.textAlign="center";
 for(i=0;i<n;i++){var cl=cols[i],lyr=cl.ly;ctx.font='700 12px "Microsoft YaHei UI","Segoe UI",sans-serif';ctx.fillStyle="#dfe7ef";ctx.fillText(String(lyr.name||""),cl.x,16);ctx.font=NET_FONT;ctx.fillStyle="#8b95a3";ctx.fillText(String(lyr.detail||""),cl.x,30);ctx.font=NET_FONT;ctx.fillStyle="#66707d";ctx.textAlign="center";ctx.fillText("真实宽度 "+(lyr.trueDim|0),cl.x,43);var bw2=(step>0?step*0.62:90)<40?40:((step>0?step*0.62:90)>150?150:step>0?step*0.62:90);var gx=cl.x-bw2*0.5,gy=h-62;for(j=0;j<24;j++){ctx.fillStyle=netColor(j/23);ctx.fillRect(gx+bw2*j/24,gy,bw2/24+1,7);}ctx.strokeStyle="rgba(220,235,250,.2)";ctx.lineWidth=1;ctx.strokeRect(gx+.5,gy+.5,bw2,7);ctx.font=NET_FONT;ctx.fillStyle="#8794a3";ctx.textAlign="left";ctx.fillText(fSmart(cl.sc[0]),gx,gy+21);ctx.textAlign="right";ctx.fillText(fSmart(cl.sc[1]),gx+bw2,gy+21);ctx.textAlign="center";ctx.fillStyle="#79828f";ctx.fillText("均值 "+fSmart(Number(lyr.actMean)),cl.x,gy+36);ctx.fillStyle="#66707d";ctx.fillText("全层 "+fSmart(Number(lyr.actMin))+" ~ "+fSmart(Number(lyr.actMax)),cl.x,gy+48);}
 if(NET.hover&&cols[NET.hover.li]){var hc=cols[NET.hover.li],hy=hc.ys[NET.hover.bi];if(hy!==undefined){ctx.strokeStyle="#e8eef5";ctx.lineWidth=1.4;ctx.beginPath();ctx.arc(hc.x,hy,hc.r+3.6,0,Math.PI*2);ctx.stroke();}}
 ctx.font=NET_FONT;ctx.textAlign="left";ctx.textBaseline="alphabetic";ctx.fillStyle="rgba(170,208,240,.8)";ctx.fillText("2D 平面 · 连线亮度 = 权重 × 源激活 · 每秒刷新",8,14);
}
function drawNet(){if(NET.mode==="3d"){drawNet3D();return;}drawNet2D();}
var netRaf=0;
function requestNet(){if(netRaf)return;netRaf=requestAnimationFrame(function(){netRaf=0;drawNet();});}
function netHit(mx,my){var v=NET.view;if(!v)return null;var best=null,bd=Infinity,i,nd,d,rr;for(i=0;i<v.nodes.length;i++){nd=v.nodes[i];d=(nd.x-mx)*(nd.x-mx)+(nd.y-my)*(nd.y-my);rr=nd.r+4;if(d<=rr*rr&&d<bd){bd=d;best=nd;}}return best;}
function netTipHTML(hit){
 var L=NET.layers,ly=L[hit.li],j=hit.bi;
 var cnt=Math.max(1,ly.nodeCount|0),td=Math.max(1,ly.trueDim|0);
 var raw=Number((ly.nodes||[])[j]);
 var h='<div class="th">'+esc(ly.name)+' · 节点 '+(j+1)+' / '+cnt+'</div>';
 h+='<div class="tr"><span>激活值</span><i>'+fSmart(raw)+'</i></div>';
 h+='<div class="tr"><span>覆盖真实索引</span><i>'+Math.floor(j*td/cnt)+' ~ '+Math.floor((j+1)*td/cnt)+'</i></div>';
 h+='<div class="tr"><span>层真实宽度</span><i>'+td+'</i></div>';
 h+='<div class="tr"><span>本层范围</span><i>'+fSmart(ly.actMin)+' ~ '+fSmart(ly.actMax)+'</i></div>';
 var gi=-1,kk;
 for(kk=0;kk<NET.pairs.length;kk++){if(NET.pairs[kk][0]===hit.li){gi=kk;break;}}
 if(gi>=0){
  var tgt=NET.pairs[gi][1],conn=NET.conn,M=NET.edges[gi]||[],sj=netBucket(j,cnt,conn),arr=[],k,wv;
  for(k=0;k<conn;k++){wv=Number(M[k*conn+sj]);if(isFinite(wv))arr.push([k,wv]);}
  arr.sort(function(p,q){return q[1]-p[1];});
  h+='<div class="th sub">输出连接最强 5 条 → '+esc(L[tgt].name)+'</div>';
  for(k=0;k<Math.min(5,arr.length);k++){h+='<div class="tr"><span>目标桶 '+arr[k][0]+'</span><i>'+fFix(arr[k][1],3)+'</i></div>';}
 }else{h+='<div class="tr"><span>输出层</span><i>无后续连接</i></div>';}
 return h;
}
function netTipShow(hit,cx,cy){var tip=el("nettip"),wrap=el("netwrap");if(!tip||!wrap)return;tip.innerHTML=netTipHTML(hit);tip.className="nettip on";var ww=wrap.clientWidth,wh=wrap.clientHeight,tw=tip.offsetWidth,th=tip.offsetHeight;var x=cx+16,y=cy+14;if(x+tw>ww-6)x=cx-tw-16;if(x<6)x=6;if(y+th>wh-6)y=cy-th-14;if(y<6)y=6;tip.style.left=Math.round(x)+"px";tip.style.top=Math.round(y)+"px";}
function setNetMode(v){NET.mode=(v==="2d")?"2d":"3d";var a=el("v-3d"),b=el("v-2d");
 if(a)a.className="vbtn"+(NET.mode==="3d"?" on":"");
 if(b)b.className="vbtn"+(NET.mode==="2d"?" on":"");
 if(NET.cv)NET.cv.style.cursor=(NET.mode==="3d")?"grab":"default";requestNet();}
function renderNetwork(dd){prepNet(dd);var tip=el("nettip");if(NET.ready){NET.value=(dd&&Number(dd.value))||0;NET.iter=(dd&&Number(dd.iteration))||0;NET.samples=(dd&&Number(dd.samples))||0;
 el("netSub").textContent="第 "+fInt(NET.iter)+" 步快照 · "+fInt(NET.samples)+" 局 · 最优 Q "+fFix(NET.value,4)+" · 边缘矩阵 "+NET.conn+"×"+NET.conn+" · 可见连线 "+fInt(NET.pick.length)+" · 最强强度 "+fFix(NET.maxS,3);
 }else{el("netSub").textContent="等待网络快照…";NET.hover=null;NET.hoverKey="";if(tip){tip.className="nettip";tip.innerHTML="";}}
 requestNet();}
function bindNet(){
 var cv=NET.cv,tip=el("nettip");if(!cv)return;
 cv.addEventListener("pointerdown",function(e){if(NET.mode!=="3d")return;NET.cam.drag=true;NET.cam.lx=e.clientX;NET.cam.ly=e.clientY;cv.style.cursor="grabbing";if(cv.setPointerCapture)cv.setPointerCapture(e.pointerId);});
 cv.addEventListener("pointermove",function(e){if(NET.mode!=="3d"||!NET.cam.drag)return;NET.cam.yaw-=(e.clientX-NET.cam.lx)*0.010;NET.cam.pitch=(NET.cam.pitch+(e.clientY-NET.cam.ly)*0.008)<-1.35?-1.35:((NET.cam.pitch+(e.clientY-NET.cam.ly)*0.008)>1.35?1.35:NET.cam.pitch+(e.clientY-NET.cam.ly)*0.008);NET.cam.lx=e.clientX;NET.cam.ly=e.clientY;});
 cv.addEventListener("pointerup",function(){NET.cam.drag=false;cv.style.cursor=(NET.mode==="3d")?"grab":"default";});
 cv.addEventListener("pointercancel",function(){NET.cam.drag=false;cv.style.cursor=(NET.mode==="3d")?"grab":"default";});
 cv.addEventListener("wheel",function(e){if(NET.mode!=="3d")return;e.preventDefault();NET.cam.dist=(NET.cam.dist*Math.exp(e.deltaY*0.0011))<12?12:((NET.cam.dist*Math.exp(e.deltaY*0.0011))>90?90:NET.cam.dist*Math.exp(e.deltaY*0.0011));},{passive:false});
 cv.addEventListener("mousemove",function(ev){if(NET.cam.drag)return;var rect=cv.getBoundingClientRect(),mx=ev.clientX-rect.left,my=ev.clientY-rect.top,hit=netHit(mx,my),key=hit?(hit.li+":"+hit.bi):"";if(key===NET.hoverKey){if(hit)netTipShow(hit,mx,my);return;}NET.hoverKey=key;NET.hover=hit;if(hit)netTipShow(hit,mx,my);else if(tip){tip.className="nettip";tip.innerHTML="";}requestNet();});
 cv.addEventListener("mouseleave",function(){if(!NET.hoverKey)return;NET.hoverKey="";NET.hover=null;if(tip){tip.className="nettip";tip.innerHTML="";}requestNet();});
}
function bindNetView(){var list=document.querySelectorAll("[data-view]"),i;for(i=0;i<list.length;i++){list[i].addEventListener("click",function(){setNetMode(this.getAttribute("data-view"));});}}
var GD={yaw:-0.72,pitch:0.62,dist:3.1,drag:false,lx:0,ly:0,cv:null,ctx:null,raf:0,land:null,MESH:34};
function gdProj(c,x,y,z){var vx=x-c.ox,vy=y-c.oy,vz=z-c.oz;var cz=vx*c.fx+vy*c.fy+vz*c.fz;if(cz<0.08)cz=0.08;var cx=vx*c.rx+vy*c.ry+vz*c.rz,cyv=vx*c.ux+vy*c.uy+vz*c.uz;return[cx/cz,cyv/cz,cz];}
function gdCam(){var cy=Math.cos(GD.yaw),sy=Math.sin(GD.yaw),cp=Math.cos(GD.pitch),sp=Math.sin(GD.pitch);var ex=cp*cy,ey=cp*sy,ez=sp,d=GD.dist;return{ox:ex*d,oy:ey*d,oz:ez*d,fx:-ex,fy:-ey,fz:-ez,rx:-sy,ry:cy,rz:0,ux:-cy*sp,uy:-sy*sp,uz:cp};}
function gdColor(t){t=t<0?0:(t>1?1:t);var r,g,b;if(t<0.35){var u=t/0.35;r=20+(42-20)*u;g=40+(100-40)*u;b=66+(150-66)*u;}else if(t<0.7){var u2=(t-0.35)/0.35;r=42+(86-42)*u2;g=100+(170-100)*u2;b=150+(210-150)*u2;}else{var u3=(t-0.7)/0.3;r=86+(190-86)*u3;g=170+(230-170)*u3;b=210+(252-210)*u3;}return[r|0,g|0,b|0];}
function gdSample(L,ti,tj){
 var res=L.res,fi=ti*(res-1),fj=tj*(res-1);
 var i0=Math.min(res-1,Math.max(0,Math.floor(fi))),j0=Math.min(res-1,Math.max(0,Math.floor(fj)));
 var i1=Math.min(res-1,i0+1),j1=Math.min(res-1,j0+1),tx=fi-i0,ty=fj-j0;
 function g(i,j){var v=Number(L.values[i*res+j]);return isFinite(v)?v:L.zmin;}
 var a=g(i0,j0),b=g(i1,j0),cc=g(i1,j1),dd=g(i0,j1);
 return (a*(1-tx)+b*tx)*(1-ty)+(dd*(1-tx)+cc*tx)*ty;
}
function gdEmpty(ctx,w,h,msg,sub){
 ctx.save();ctx.setLineDash([6,5]);ctx.strokeStyle="rgba(120,140,168,.24)";ctx.lineWidth=1;ctx.strokeRect(15.5,15.5,w-31,h-31);ctx.restore();
 ctx.textAlign="center";ctx.textBaseline="middle";ctx.font='13px "Microsoft YaHei UI","Segoe UI",sans-serif';ctx.fillStyle="#5f6a78";ctx.fillText(msg,w*0.5,h*0.5-11);
 ctx.font='11px "Microsoft YaHei UI","Segoe UI",sans-serif';ctx.fillStyle="#4a545f";ctx.fillText(sub,w*0.5,h*0.5+11);ctx.textBaseline="alphabetic";
}
function drawGD(t){
 var cv=GD.cv,ctx=GD.ctx;if(!cv||!ctx)return;
 var w=cv.clientWidth|0,h=cv.clientHeight|0;if(w<40||h<40)return;
 var dpr=Math.min(window.devicePixelRatio||1,2),nw=Math.round(w*dpr),nh=Math.round(h*dpr);
 if(cv.width!==nw||cv.height!==nh){cv.width=nw;cv.height=nh;}
 ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
 var L=GD.land;
 if(!L){gdEmpty(ctx,w,h,"正在测量损失曲面…","每隔一段时间会在真实权重周围采样网格点，首次需要稍等");return;}
 var res=L.res,ext=Number(L.ext)||1,zmin=Number(L.zmin),zmax=Number(L.zmax);
 var range=(zmax-zmin)||1,dcnt=Number(L.probeCount)||0;
 var zh=0.34;
 var c=gdCam(),FOCAL=Math.min(w,h)*1.02,ox=w*0.5,oy=h*0.56;
 var M=GD.MESH,i,j,tx,ty;
 var sx=new Float64Array(M*M),sy=new Float64Array(M*M),sd=new Float64Array(M*M),zn=new Float64Array(M*M),wx=new Float64Array(M*M),wy=new Float64Array(M*M),wz=new Float64Array(M*M);
 for(i=0;i<M;i++){
  tx=i/(M-1);var uu=-1+2*tx;
  for(j=0;j<M;j++){
   ty=j/(M-1);var vv=-1+2*ty;
   var loss=gdSample(L,tx,ty);
   var nt=(loss-zmin)/range;if(nt<0)nt=0;if(nt>1)nt=1;
   var z=nt*zh-zh*0.5;
   var p=gdProj(c,uu,vv,z),idx=i*M+j;
   sx[idx]=ox+p[0]*FOCAL;sy[idx]=oy-p[1]*FOCAL;sd[idx]=p[2];
   zn[idx]=nt;wx[idx]=uu;wy[idx]=vv;wz[idx]=z;
  }
 }
 var axisC="rgba(158,190,220,.30)";
 ctx.strokeStyle=axisC;ctx.lineWidth=1;
 var bp=[];
 for(i=0;i<8;i++)bp.push(gdProj(c,-1+2*i/7,-1,-zh*0.5));
 for(i=1;i<8;i++)bp.push(gdProj(c,-1,-1+2*i/7,-zh*0.5));
 bp.push(gdProj(c,-1,-1,-zh*0.5-0.12));
 ctx.beginPath();ctx.moveTo(ox+bp[0][0]*FOCAL,oy-bp[0][1]*FOCAL);
 for(i=1;i<8;i++)ctx.lineTo(ox+bp[i][0]*FOCAL,oy-bp[i][1]*FOCAL);
 for(i=7;i<14;i++)ctx.lineTo(ox+bp[i][0]*FOCAL,oy-bp[i][1]*FOCAL);
 ctx.stroke();
 ctx.beginPath();ctx.moveTo(ox+bp[0][0]*FOCAL,oy-bp[0][1]*FOCAL);ctx.lineTo(ox+bp[14][0]*FOCAL,oy-bp[14][1]*FOCAL);ctx.stroke();
 var quads=[];
 for(i=0;i<M-1;i++)for(j=0;j<M-1;j++){var a0=i*M+j,b0=(i+1)*M+j,cc=(i+1)*M+j+1,dd=i*M+j+1;quads.push([a0,b0,cc,dd,(sd[a0]+sd[b0]+sd[cc]+sd[dd])*0.25,(zn[a0]+zn[b0]+zn[cc]+zn[dd])*0.25]);}
 quads.sort(function(p,q){return q[4]-p[4];});
 var light=[0.45,0.25,0.86];
  for(var k=0;k<quads.length;k++){
   var q=quads[k],a=q[0],b=q[1],cc2=q[2],dd=q[3];
   var e1x=wx[b]-wx[a],e1y=wy[b]-wy[a],e1z=wz[b]-wz[a],e2x=wx[dd]-wx[a],e2y=wy[dd]-wy[a],e2z=wz[dd]-wz[a];
   var nxv=e1y*e2z-e1z*e2y,nyv=e1z*e2x-e1x*e2z,nzv=e1x*e2y-e1y*e2x;
   var nl=Math.sqrt(nxv*nxv+nyv*nyv+nzv*nzv)||1,dot=(nxv*light[0]+nyv*light[1]+nzv*light[2])/nl;
   if(dot<0)dot=-dot;
   var sh=0.70+0.42*dot,rgb=gdColor(q[5]);
   ctx.beginPath();ctx.moveTo(sx[a],sy[a]);ctx.lineTo(sx[b],sy[b]);ctx.lineTo(sx[cc2],sy[cc2]);ctx.lineTo(sx[dd],sy[dd]);ctx.closePath();
   ctx.fillStyle="rgb("+Math.min(255,rgb[0]*sh|0)+","+Math.min(255,rgb[1]*sh|0)+","+Math.min(255,rgb[2]*sh|0)+")";
   ctx.fill();
   ctx.strokeStyle="rgba(200,226,250,.06)";ctx.lineWidth=0.5;ctx.stroke();
  }
 var path=L.path||[],pn=path.length;
  var spanU=0,spanV=0;
  if(pn>0){
   var last=path[pn-1],ca=Number(last.a)||0,cb=Number(last.b)||0;
   for(i=0;i<pn;i++){var da=Math.abs((Number(path[i].a)||0)-ca),db=Math.abs((Number(path[i].b)||0)-cb);if(da>spanU)spanU=da;if(db>spanV)spanV=db;}
   var scl=0.86/Math.max(spanU,spanV,1e-12);
   function toWorld(pp){
    var du=((Number(pp.a)||0)-ca)*scl,dv=((Number(pp.b)||0)-cb)*scl;
    du=du<-0.98?-0.98:(du>0.98?0.98:du);dv=dv<-0.98?-0.98:(dv>0.98?0.98:dv);
    var nt=(Number(pp.z)-zmin)/range;if(nt<0)nt=0;if(nt>1)nt=1;
    return [du,dv,nt*zh-zh*0.5+0.02];
   }
   var pts=[];
   for(i=0;i<pn;i++){var wpt=toWorld(path[i]),sp=gdProj(c,wpt[0],wpt[1],wpt[2]);
    pts.push({x:ox+sp[0]*FOCAL,y:oy-sp[1]*FOCAL,z:Number(path[i].z)});}
   ctx.save();ctx.strokeStyle="#ffb020";ctx.lineWidth=2.2;ctx.lineJoin="round";ctx.lineCap="round";ctx.shadowColor="rgba(255,176,32,.7)";ctx.shadowBlur=9;
   ctx.beginPath();
   for(i=0;i<pts.length;i++){if(i===0)ctx.moveTo(pts[i].x,pts[i].y);else ctx.lineTo(pts[i].x,pts[i].y);}
   ctx.stroke();ctx.restore();
   var every=Math.max(1,Math.floor(pts.length/40));
   for(i=0;i<pts.length-1;i++){
    if(i%every!==0)continue;
    ctx.beginPath();ctx.arc(pts[i].x,pts[i].y,2.6,0,Math.PI*2);ctx.fillStyle="rgba(255,211,106,.85)";ctx.fill();
   }
   var lp=pts[pts.length-1];
   ctx.beginPath();ctx.arc(lp.x,lp.y,7,0,Math.PI*2);ctx.strokeStyle="#3ddc84";ctx.lineWidth=2;ctx.stroke();
   ctx.beginPath();ctx.arc(lp.x,lp.y,3.4,0,Math.PI*2);ctx.fillStyle="#3ddc84";ctx.fill();
   ctx.font='10px "Microsoft YaHei UI","Segoe UI",sans-serif';ctx.textAlign="left";ctx.fillStyle="#8fe0b2";
   ctx.fillText("当前权重",lp.x+11,lp.y+10);
   ctx.textAlign="right";ctx.fillStyle="#ffd58a";
   ctx.fillText("损失 "+fSmart(Number(last.z)),w-16,16);
   ctx.fillStyle="rgba(120,140,168,.75)";
   ctx.fillText("轨迹跨度 u "+fSmart(spanU)+" · v "+fSmart(spanV)+"（已等比缩放填满窗口）",w-16,30);
  }
 ctx.textAlign="left";ctx.font='10px "Microsoft YaHei UI","Segoe UI",sans-serif';
 ctx.fillStyle="rgba(170,208,240,.75)";
 ctx.fillText("真实测量损失曲面 · 网格 "+res+"×"+res+" · 窗口 ±"+fFix(ext,2)+" · 已测 "+fInt(dcnt)+" 次 · 轨迹 "+fInt(pn)+" 点",8,14);
 ctx.fillStyle="rgba(120,140,168,.65)";
 ctx.fillText("拖动旋转 · 滚轮缩放 · 高度=固定评估批上的真实 TD 损失",8,h-8);
 var cbx=w-14,cby=34,cbh=84,cbw=8;
 for(i=0;i<32;i++){var t2=1-i/31,rgb2=gdColor(t2);ctx.fillStyle="rgb("+rgb2[0]+","+rgb2[1]+","+rgb2[2]+")";ctx.fillRect(cbx,cby+cbh*i/32,cbw,cbh/32+1);}
 ctx.strokeStyle="rgba(220,235,250,.2)";ctx.lineWidth=1;ctx.strokeRect(cbx+.5,cby+.5,cbw,cbh);
 ctx.textAlign="right";ctx.fillStyle="#8794a3";ctx.fillText(fSmart(zmax),cbx-5,cby+8);ctx.fillText(fSmart(zmin),cbx-5,cby+cbh);
 ctx.fillStyle="#5f6a78";ctx.fillText("损失",cbx-5,cby-8);
}
function gdFrame(t){GD.raf=requestAnimationFrame(gdFrame);drawGD(t);}
function bindGD(){
 var cv=GD.cv;cv.addEventListener("pointerdown",function(e){GD.drag=true;GD.lx=e.clientX;GD.ly=e.clientY;cv.style.cursor="grabbing";if(cv.setPointerCapture)cv.setPointerCapture(e.pointerId);});
 cv.addEventListener("pointermove",function(e){if(!GD.drag)return;GD.yaw-=(e.clientX-GD.lx)*0.011;GD.pitch=(GD.pitch+(e.clientY-GD.ly)*0.009)<0.08?0.08:((GD.pitch+(e.clientY-GD.ly)*0.009)>1.45?1.45:GD.pitch+(e.clientY-GD.ly)*0.009);GD.lx=e.clientX;GD.ly=e.clientY;});
 function stop(){if(!GD.drag)return;GD.drag=false;cv.style.cursor="grab";}
 cv.addEventListener("pointerup",stop);cv.addEventListener("pointercancel",stop);cv.addEventListener("pointerleave",stop);
 cv.addEventListener("wheel",function(e){e.preventDefault();GD.dist=(GD.dist*Math.exp(e.deltaY*0.0012))<1.7?1.7:((GD.dist*Math.exp(e.deltaY*0.0012))>11?11:GD.dist*Math.exp(e.deltaY*0.0012));},{passive:false});
}
function setConn(ok){var e=el("conn"),t=el("connTxt");if(!ok){e.className="conn bad";t.textContent="连接断开";return;}e.className="conn ok";t.textContent="已连接";}
var PREV=null;
async function poll(){
 try{
  var d=await (await fetch("/stats",{cache:"no-store"})).json();
  var now=Date.now(),dps=-1;
  if(PREV&&PREV.ts>0){var dt=(now-PREV.ts)/1000;if(dt>0)dps=Math.round((Number(d.steps)-PREV.steps)/dt);}
  PREV={steps:Number(d.steps),ts:now};
  if(dps<0)dps=0;
  d.dps=dps;
  setConn(true);
  renderStats(d);
  pushHistory(d);
  renderCharts();
  renderNetwork(d.acts||null);
  var ld=d.landscape||null;
  GD.land=(ld&&Number(ld.ready)===1&&ld.values&&ld.values.length>=4)?ld:null;
  var lsub=el("gdSub");
  if(lsub)lsub.textContent=GD.land?("网格 "+GD.land.res+"×"+GD.land.res+" · 窗口 ±"+fFix(GD.land.ext,2)+" · 已测 "+fInt(GD.land.probeCount)+" 次 · 轨迹 "+fInt((GD.land.path||[]).length)+" 点"):"等待首次测量…";
  var sig=(d.config||{});var sigS=JSON.stringify(sig)+"|"+((d.acts&&d.acts.layers||[]).map(function(l){return l.params+"/"+l.trueDim;})).join(",");
  if(sigS!==panelSig){panelSig=sigS;buildPanels(d);}
  updatePanels(d);
 }catch(e){setConn(false);}
 setTimeout(poll,1000);
}
NET.cv=el("netcv");NET.ctx=NET.cv.getContext("2d");
GD.cv=el("gdcv");GD.ctx=GD.cv.getContext("2d");
buildCharts();
bindNet();
bindNetView();
setNetMode("3d");
bindGD();
gdFrame();
renderNetwork(null);
poll();
window.addEventListener("resize",function(){var i;for(i=0;i<charts.length;i++)drawChart(charts[i]);requestNet();});
})();
</script>
</body>
</html>
"""


class Monitor:
    def __init__(self, state_fn, port=8787):
        self.port = port
        self.state_fn = state_fn

    def start(self):
        state_fn = self.state_fn
        port = self.port

        class H(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                if self.path == "/stats":
                    d = dict(state_fn())
                    d["config"] = CFG
                    body = json.dumps(d).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    body = PAGE.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

            def log_message(self, *a):
                return

        srv = None
        for _ in range(20):
            try:
                srv = ThreadingHTTPServer(("127.0.0.1", port), H)
                break
            except OSError:
                port += 1
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return port