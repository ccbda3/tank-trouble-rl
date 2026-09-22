import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>DQN 训练监控 · 超级马里奥</title>
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
@media(max-width:900px){section{grid-template-columns:1fr}}
.panel{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.panel h2{font-size:14px;margin:0 0 10px;color:var(--dim);font-weight:600}
canvas{width:100%;display:block;background:#0b0e14;border-radius:8px}
.net{display:flex;align-items:stretch;gap:8px;overflow-x:auto;padding:8px 2px}
.lay{flex:0 0 auto;border:1px solid var(--line);border-radius:10px;padding:12px 13px;background:#10151f;min-width:92px;text-align:center}
.lay .n{font-size:12px;color:var(--dim)}
.lay .s{font-size:15px;margin-top:6px;font-weight:600}
.lay .p{font-size:11px;color:var(--good);margin-top:4px}
.arr{align-self:center;color:var(--dim);font-size:20px;flex:none}
#info{font-size:13px;line-height:2.1;color:var(--dim)}
#info b{color:var(--txt);font-weight:600;margin-right:8px}
</style>
</head>
<body>
<header><span class="dot"></span><h1>DQN 训练监控 · 超级马里奥</h1><span class="meta" id="meta">连接中…</span></header>
<div class="grid" id="cards"></div>
<section>
  <div class="panel"><h2>训练曲线（Loss ↑ · 平均最远距离 ↓）</h2><canvas id="chart" height="220"></canvas></div>
  <div class="panel"><h2>梯度下降 · 损失曲面模拟</h2><canvas id="gd" height="220"></canvas></div>
  <div class="panel"><h2>神经网络结构（CNN DQN）</h2><div id="netmap"></div></div>
  <div class="panel"><h2>工程信息</h2><div id="info"></div></div>
</section>
<script>
const $=id=>document.getElementById(id);
const num=v=>Math.round(v||0).toLocaleString();
function pick(k,d){
 switch(k){
  case"步数":return num(d.steps);
  case"探索率 ε":return d.eps.toFixed(3);
  case"训练步数":return num(d.trainSteps);
  case"经验池":return num(d.buffer);
  case"Loss":return d.loss.toFixed(4);
  case"本局最远":return num(d.score);
  case"平均最远":return num(d.avgScore);
  case"历史最远":return num(d.bestScore);
  case"GPU 利用率":return d.gpu+"%";
  case"局数":return num(d.episodes);
  default:return"—";
 }
}
function cls(k){
 if(k==="平均最远"||k==="历史最远"||k==="GPU 利用率")return" good";
 if(k==="探索率 ε")return" cy";
 return"";
}
const keys=["步数","探索率 ε","训练步数","经验池","Loss","本局最远","平均最远","历史最远","GPU 利用率","局数"];
function drawChart(loss,avgScore){
 const c=$("chart"),cs=198,x=c.getContext("2d");
 const W=c.clientWidth,H=cs,pad=24;
 c.width=W;c.height=H;x.clearRect(0,0,W,H);
 x.strokeStyle="#2a3245";x.lineWidth=1;
 for(let i=1;i<4;i++){const yy=pad+(H-2*pad)*i/4;x.beginPath();x.moveTo(pad,yy);x.lineTo(W-pad,yy);x.stroke();}
 const N=Math.max(loss.length,200),L=loss.length;
 const maxl=Math.max.apply(null,loss.slice(-N))||1,maxa=Math.max.apply(null,avgScore.slice(-N))||1;
 const line=(arr,max,color)=>{
  x.beginPath();x.strokeStyle=color;x.lineWidth=2;
  for(let i=0;i<arr.length;i++){
   const px=pad+(i/(N-1))*(W-2*pad);
   const py=H-pad-(arr[arr.length-1-i]/max)*(H-2*pad);
   i?x.lineTo(px,py):x.moveTo(px,py);
  }
  x.stroke();
 };
 line(loss.slice(-N),maxl,"#ff8f5a");
 line(avgScore.slice(-N),maxa,"#6bff8f");
 x.fillStyle="#7d8aa0";x.font="12px sans-serif";
 x.fillText("0",4,H-8);
 x.fillStyle="#5ac8ff";x.fillText("Loss 轴",W-70,12);
 x.fillStyle="#6bff8f";x.fillText("均距 轴",W-70,26);
}
function gd(){
 const c=$("gd"),x=c.getContext("2d");
 const W=c.clientWidth,H=208,cx=W/2,cy=H/2-4;
 function frame(t){
  c.width=W;c.height=H;x.clearRect(0,0,W,H);
  x.strokeStyle="#2a3245";x.lineWidth=1.5;
  const tilt=Math.sin(t/1500)*0.12,cam=t*0.0016;
  for(let ring=0;ring<6;ring++){
   const r0=ring*W/13+10;
   x.beginPath();
   for(let i=0;i<=80;i++){
    const a=cam+(i/80)*Math.PI*2;
    const r=r0*(1+0.06*ring);
    const px=cx+r*Math.cos(a);
    const py=cy+r*Math.sin(a)- (ring*ring/8)*(tilt*80);
    i?x.lineTo(px,py):x.moveTo(px,py);
   }
   x.closePath();x.stroke();
  }
  const dec=Math.min(1,t/16000);
  const ball=8+5*Math.E**(3*(dec-1))*3-0;
  const rr=Math.max(8,W/5*(1-dec));
  const bx=cx+rr*Math.cos(Math.PI/2),by=cy+rr*Math.sin(Math.PI/2)-(1-dec)*(1-dec)*26;
  x.beginPath();x.arc(cx,cy-((1-dec)*30),rr,0,7);x.strokeStyle="#ff8f5a";x.stroke();
  x.fillStyle="#ff8f5a";x.shadowColor="#ff8f5a";x.shadowBlur=14;
  x.beginPath();x.arc(cx,cy-(1-dec)*30,6,0,7);x.fill();x.shadowBlur=0;
  x.fillStyle="#7d8aa0";x.font="12px sans-serif";
  x.fillText("小球沿梯度从高处滚向谷底（Loss 最小化）",10,16);
  requestAnimationFrame(frame);
 }
 requestAnimationFrame(frame);
}
function renderNet(d){
 const parts=[];
 parts.push(["Input","4×96×96","灰度4帧"]);
 [["Conv1",32,"8·S4"],["Conv2",64,"4·S2"],["Conv3",64,"3·S1"]].forEach(x=>parts.push(["Conv2d "+x[0],x[1]+" 通道","核 "+x[2]]));
 parts.push(["FC 512","512","ReLU"]);
 parts.push(["Output",d.actions+" 动作","Q 值"]);
 $("netmap").innerHTML='<div class="net">'+parts.map(p=>'<div class="lay"><div class="n">'+p[0]+'</div><div class="s">'+p[1]+'</div><div class="p">'+p[2]+'</div></div>').join('<span class="arr">→</span>')+'<div class="lay"><div class="n">参数量</div><div class="s">'+num(d.params)+'</div><div class="p">训练中最优</div></div></div>';
}
let loss=[],avg=[];
async function poll(){
 try{
  const d=await (await fetch("/stats")).json();
  $("meta").textContent="在线 · 每 1 秒刷新  ·  http://127.0.0.1:"+d.port;
  loss.push(d.loss);avg.push(d.avgScore||0);
  if(loss.length>400){loss.shift();avg.shift();}
  $("cards").innerHTML=keys.map(k=>'<div class="card"><div class="k">'+k+'</div><div class="v'+cls(k)+'">'+pick(k,d)+'</div></div>').join("");
  drawChart(loss,avg);
  $("info").innerHTML=[
   ["环境",d.mark],["有效批量",(d.batch*d.accum)+"（batch "+d.batch+" × 累积 "+d.accum+"）"],
   ["智能体数",d.agents],["算法","Double DQN + 优先级经验池"],["设备",d.device],
   ["动作空间","7（右/跳/跑…）SIMPLE_MOVEMENT"],["帧输入","4 帧 × 96 × 96 灰度"],["探索衰减","ε: "+d.eps+" → 随步数线性下降"]
  ].map(x=>'<div><b>'+x[0]+'</b>'+x[1]+'</div>').join("");
 }catch(e){$("meta").textContent="离线 / 连接断开";}
 setTimeout(poll,1000);
}
poll();gd();setTimeout(()=>renderNet({actions:7,params:2177187}),60);
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