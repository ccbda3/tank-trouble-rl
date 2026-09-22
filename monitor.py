import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch


def group_avg(a, n):
    a = np.asarray(a, dtype=float).ravel()
    if len(a) <= n:
        return [float(x) for x in a]
    idx = np.linspace(0, len(a), n + 1).astype(int)
    return [float(a[idx[i]:idx[i + 1]].mean()) for i in range(n)]


def act_stats(net, obs):
    x = torch.tensor(np.asarray(obs, dtype=np.float32) / 255.0).unsqueeze(0).to(
        net.fc[0].weight.device
    )
    with torch.no_grad():
        z = net.net[1](net.net[0](x))
        c1 = z.mean((2, 3))[0].cpu().numpy()
        z = net.net[3](net.net[2](z))
        c2 = z.mean((2, 3))[0].cpu().numpy()
        z = net.net[5](net.net[4](z))
        c3 = z.mean((2, 3))[0].cpu().numpy()
        f = net.fc[1](net.fc[0](torch.flatten(z, 1)))
        fc = f[0].cpu().numpy()
        out = net.fc[2](f)[0].cpu().numpy()
    gray = np.asarray(obs, dtype=float).mean(0)
    h, w = gray.shape
    in_arr = [float(gray[int(i * h / 10):int((i + 1) * h / 10)].mean()) for i in range(10)]
    wf = np.array([torch.mean(torch.abs(t)).item() for t in [
        net.net[0].weight,
        net.net[2].weight,
        net.net[4].weight,
        net.fc[0].weight,
        net.fc[2].weight,
    ]])
    wf = (wf / (wf.max() or 1.0)).tolist()
    return {
        "in": in_arr,
        "c1": group_avg(c1, 12),
        "c2": group_avg(c2, 14),
        "c3": group_avg(c3, 12),
        "fc": group_avg(fc, 12),
        "out": group_avg(out, 3),
        "wf": wf,
    }


PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>超级马里奥 · 强化学习监控面板</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0f1219;--card:#171c27;--line:#262e3f;--txt:#c9d4e3;--dim:#7d8aa0;--acc:#ff8f5a;--acc2:#5ac8ff;--good:#6bff8f}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif}
header{padding:16px 22px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:14px;flex-wrap:wrap}
h1{font-size:20px;margin:0;font-weight:600}
.dot{width:9px;height:9px;border-radius:50%;background:#5ff25f;box-shadow:0 0 8px #5ff25f;animation:p 1.6s infinite}
@keyframes p{50%{opacity:.35}}
.meta{color:var(--dim);font-size:13px;margin-left:auto}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;padding:16px 22px 6px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:13px 16px}
.card .k{font-size:12px;color:var(--dim);margin-bottom:6px}
.card .v{font-size:23px;font-weight:600;font-variant-numeric:tabular-nums}
.card .v.acc{color:var(--acc)}.card .v.good{color:var(--good)}.card .v.cy{color:var(--acc2)}
section{padding:6px 22px 24px;display:grid;grid-template-columns:1fr 1fr;gap:14px}
.panel.rowwide{grid-column:1/-1}
@media(max-width:900px){section{grid-template-columns:1fr}}
.panel{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.panel h2{font-size:14px;margin:0 0 10px;color:var(--dim);font-weight:600}
canvas{width:100%;display:block;background:#0b0e14;border-radius:8px}
.lg{display:flex;gap:14px;font-size:11px;color:var(--dim);margin-top:6px;flex-wrap:wrap}
.lg i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
.help{font-size:12px;color:var(--dim);line-height:1.8}
.help b{color:var(--txt)}
</style>
</head>
<body>
<header><span class="dot"></span><h1>超级马里奥 · 强化学习监控面板</h1><span class="meta" id="meta">连接中…</span></header>
<div class="grid" id="cards"></div>
<section>
  <div class="panel rowwide"><h2>神经网络 · 逐层神经元连接（连线亮 = 真实权重 × 两端激活）</h2><canvas id="nn" height="430"></canvas>
    <div class="lg"><span><i style="background:#1d7fd1"></i>输入</span><span><i style="background:#e76a2e"></i>卷积/FC</span><span><i style="background:#6bff8f"></i>输出</span><span style="opacity:.85">活跃神经元变大、连线随推理脉动 —— 当前画面哪条通路被点亮一目了然</span></div>
  </div>
  <div class="panel"><h2>梯度下降 · 3D 损失面</h2><canvas id="gd" height="300"></canvas>
    <div class="lg"><span><i style="background:#3a465c"></i>损失面</span><span><i style="background:#ff8f5a"></i>小球沿梯度滚向谷底 = 让 Loss 最小化</span></div>
  </div>
  <div class="panel"><h2>训练曲线（Loss 与 平均最远通关距离）</h2><canvas id="chart" height="300"></canvas></div>
  <div class="panel rowwide"><h2>通俗解释 · 怎么看这些数字</h2>
    <div class="help" id="help">
      <div><b>Loss（损失）</b>：网络"猜错的程度"，越高越糊涂、越低越会玩。训练就是让小球往谷底滚、把 Loss 压下去。</div>
      <div><b>探索率 ε</b>：从 1 慢慢降到接近 0。前期疯狂乱按试错，后期主要靠学到的策略。</div>
      <div><b>平均最远 / 历史最远</b>：这局马里奥跑了多远，越大越会玩，是给"通关能力"打的进度条。</div>
      <div><b>神经连接图</b>：训练时网络大脑在"思考"，亮的线就是正在算的数据通路；权重大、激活强的连接更亮。</div>
    </div>
  </div>
</section>
<script>
const $=id=>document.getElementById(id);
const num=v=>Math.round(v||0).toLocaleString();
function pick(k,d){switch(k){
 case"步数":return num(d.steps);case"探索率 ε":return d.eps.toFixed(3);case"训练步数":return num(d.trainSteps);
 case"经验池":return num(d.buffer);case"Loss":return d.loss.toFixed(4);case"本局最远":return num(d.score);
 case"平均最远":return num(d.avgScore);case"历史最远":return num(d.bestScore);case"GPU 利用率":return d.gpu+"%";
 case"局数":return num(d.episodes);default:return"—";}}
function cls(k){return(k==="平均最远"||k==="历史最远"||k==="GPU 利用率")?" good":(k==="探索率 ε"?" cy":"");}
const keys=["步数","探索率 ε","训练步数","经验池","Loss","本局最远","平均最远","历史最远","GPU 利用率","局数"];
const LN=[["in","输入",10],["c1","C1·32",12],["c2","C2·64",14],["c3","C3·64",12],["fc","FC·512",12],["out","输出",3]];
let A={in:new Array(10).fill(0),c1:new Array(12).fill(0),c2:new Array(14).fill(0),c3:new Array(12).fill(0),fc:new Array(12).fill(0),out:new Array(3).fill(0),wf:[0.5,0.5,0.5,0.5,0.5]};
function norm(v){const m=Math.max.apply(null,v);return m>0?v.map(x=>Math.max(0,x)/m):v;}
function drawNN(t){
 const c=$("nn"),x=c.getContext("2d"),W=c.clientWidth,H=c.clientHeight||430;
 c.width=W;c.height=H;x.clearRect(0,0,W,H);
 const pad=40;const xs=LN.map((_,i)=>pad+i*(W-2*pad)/(LN.length-1));
 const pos=LN.map(f=>{const n=f[2],ys=[];for(let i=0;i<n;i++)ys.push(H/2+(i-(n-1)/2)*(H-80)/n);return ys;});
 for(let li=0;li<LN.length-1;li++){
  const aarr=norm(A[LN[li][0]]),barr=norm(A[LN[li+1][0]]);
  const wfv=(A.wf&&A.wf[li])||0;
  for(let i=0;i<LN[li][2];i++)for(let j=0;j<LN[li+1][2];j++){
   const a=aarr[i],b=barr[j],amp=a*b,pulse=.45+.55*Math.abs(Math.sin(t/300+li*1.7+i*0.4+j*0.25));
   const al=Math.min(1,(amp*(0.5+wfv*1.8)+wfv*0.3)*pulse);
   if(al<0.03)continue;
   x.beginPath();x.strokeStyle="rgba(255,143,90,"+al.toFixed(2)+")";
   x.lineWidth=Math.min(2.2,.6+al*1.8);
   if(al>0.5){x.shadowColor="rgba(255,143,90,"+al.toFixed(2)+")";x.shadowBlur=8;}else x.shadowBlur=0;
   x.moveTo(xs[li],pos[li][i]);x.lineTo(xs[li+1],pos[li+1][j]);x.stroke();
  }
  x.shadowBlur=0;
 }
 for(let li=0;li<LN.length;li++){
  const aarr=norm(A[LN[li][0]]);
  for(let i=0;i<LN[li][2];i++){
   const v=aarr[i],col=li===0?"#1d7fd1":(li===LN.length-1?"#6bff8f":"#e76a2e");
   const r=2.5+v*6;
   x.beginPath();x.arc(xs[li],pos[li][i],r,0,7);x.fillStyle=col;x.globalAlpha=.35+.65*v;x.shadowBlur=10;x.shadowColor=col;x.fill();x.globalAlpha=1;x.shadowBlur=0;
  }
  x.font="11px sans-serif";x.fillStyle="#7d8aa0";x.textAlign="center";x.fillText(LN[li][1],xs[li],H-6);
 }
}
function drawGD(t){
 const c=$("gd"),x=c.getContext("2d"),W=c.clientWidth,H=300;
 c.width=W;c.height=H;x.clearRect(0,0,W,H);
 const cx=W/2,cy=H/2+8,S=Math.min(W,H)*0.30,rot=t/2600,rot2=Math.sin(t/5200)*0.5;
 function pr(u,v,h){let X=(u*Math.cos(rot)-v*Math.sin(rot));let Z=(u*Math.sin(rot)+v*Math.cos(rot));const Y=h + Math.sin(rot2)*X*0.5 + Math.cos(rot2)*Z*0.5;const per=(1/(1.6+Y*0.06));return[cx+X*S*per,cy-Y*S*per];}
 x.strokeStyle="#2a3245";x.lineWidth=1.1;
 for(let i=0;i<=24;i++){const u=-1+2*i/24;x.beginPath();for(let j=0;j<=24;j++){const p=pr(u,-1+2*j/24,(u*u+(-1+2*j/24)*(-1+2*j/24))/2);j?x.lineTo(p[0],p[1]):x.moveTo(p[0],p[1]);}x.stroke();x.beginPath();for(let j=0;j<=24;j++){const p=pr(-1+2*j/24,u,((-1+2*j/24)*(-1+2*j/24)+u*u)/2);j?x.lineTo(p[0],p[1]):x.moveTo(p[0],p[1]);}x.stroke();}
 const th=(t/3200)%1;const bu=Math.cos(th*2)*0.35,ba=Math.sin(th*2)*0.35;const bh=(bu*bu+ba*ba)/2;const bp=pr(bu,ba,bh);
 x.beginPath();x.arc(bp[0],bp[1],9,0,7);x.fillStyle="#ff8f5a";x.shadowColor="#ff8f5a";x.shadowBlur=16;x.fill();x.shadowBlur=0;
 x.strokeStyle="#ff8f5a";x.lineWidth=2;x.globalAlpha=.4;
 x.beginPath();for(let i=0;i<=40;i++){const tt=Math.max(0,th-i*0.01);const p=pr(Math.cos(tt*2)*0.4,Math.sin(tt*2)*0.4,0.16);i?x.lineTo(p[0],p[1]):x.moveTo(p[0],p[1]);}x.stroke();x.globalAlpha=1;
 x.font="12px sans-serif";x.fillStyle="#7d8aa0";x.fillText("小球沿梯度向谷底滚动 = Loss 越来越小",10,18);
}
let loss=[],avg=[];
function drawChart(){
 const c=$("chart"),x=c.getContext("2d"),W=c.clientWidth,H=300,pad=24;
 c.width=W;c.height=H;x.clearRect(0,0,W,H);
 x.strokeStyle="#2a3245";for(let i=1;i<4;i++){const yy=pad+(H-2*pad)*i/4;x.beginPath();x.moveTo(pad,yy);x.lineTo(W-pad,yy);x.stroke();}
 const N=Math.max(loss.length,200),nl=loss.slice(-N),na=avg.slice(-N);
 const maxl=Math.max.apply(null,nl)||1,maxa=Math.max.apply(null,na)||1;
 const line=(arr,max,col)=>{x.beginPath();x.strokeStyle=col;x.lineWidth=2;for(let i=0;i<arr.length;i++){const px=pad+(i/(N-1))*(W-2*pad),py=H-pad-(arr[arr.length-1-i]/max)*(H-2*pad);i?x.lineTo(px,py):x.moveTo(px,py);}x.stroke();};
 line(nl,maxl,"#ff8f5a");line(na,maxa,"#6bff8f");
 x.fillStyle="#7d8aa0";x.font="12px sans-serif";x.fillText("0",4,H-8);x.fillStyle="#5ac8ff";x.fillText("Loss",W-44,12);x.fillStyle="#6bff8f";x.fillText("通关距离",W-60,26);
 if(loss.length>1){x.fillStyle="#6bff8f";x.font="13px sans-serif";x.fillText("最近平均通关距离 "+num(avg[avg.length-1]),pad+8,18);}
}
function loop(){requestAnimationFrame(loop);const t=performance.now();drawNN(t);drawGD(t);}
async function poll(){
 try{
  const d=await (await fetch("/stats")).json();
  $("meta").textContent="在线 · 每 1 秒刷新 · http://127.0.0.1:"+d.port;
  if(d.acts){A=d.acts;}else for(const k in A)A[k]=new Array(A[k].length).fill(0);
  loss.push(d.loss);avg.push(d.avgScore||0);if(loss.length>400){loss.shift();avg.shift();}
  $("cards").innerHTML=keys.map(k=>'<div class="card"><div class="k">'+k+'</div><div class="v'+cls(k)+'">'+pick(k,d)+'</div></div>').join("");
  drawChart();
  $("help").innerHTML=[
   ["Loss（损失）","网络猜错的程度，越低越会玩；训练就是让它往谷底滚把 Loss 压低。"],
   ["探索率 ε","从 1 降到接近 0：前期乱按试错，后期用学到的策略。"],
   ["平均最远 / 历史最远","这局马里奥跑了多远，越大越会玩，是通关能力的进度条。"],
   ["神经连接图","亮的线 = 正在算的数据通路；真实权重 × 激活 越强越亮。"]
  ].map(x=>'<div><b>'+x[0]+'</b>：'+x[1]+'</div>').join("");
 }catch(e){$("meta").textContent="离线 / 连接断开";}
 setTimeout(poll,1000);
}
requestAnimationFrame(loop);poll();drawChart();
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
                    body = json.dumps(state_fn()).encode("utf-8")
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