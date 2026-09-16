import os, time, math, json
import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()
GAMMA = "https://gamma-api.polymarket.com"
BINANCE = "https://api.binance.com/api/v3/klines"

def sigmoid(x):
    return 1 / (1 + math.exp(-max(-20, min(20, x))))

def window_start():
    return (int(time.time()) // 300) * 300

def parse_field(v):
    if isinstance(v, list):
        return v
    try:
        return json.loads(v)
    except Exception:
        return []

async def get_market(client, start):
    slug = f"btc-updown-5m-{start}"
    r = await client.get(f"{GAMMA}/markets", params={"slug": slug}, timeout=8)
    r.raise_for_status()
    data = r.json()
    return data[0] if isinstance(data, list) and data else None

async def get_klines(client):
    r = await client.get(BINANCE, params={
        "symbol": "BTCUSDT", "interval": "1m", "limit": 60
    }, timeout=8)
    r.raise_for_status()
    return r.json()

def make_signal(klines, market):
    closes = [float(x[4]) for x in klines]
    if len(closes) < 21:
        return {"signal":"NO SIGNAL","model_up":0.5,"confidence":"LOW","btc":closes[-1] if closes else None}

    p = closes[-1]
    r1 = p/closes[-2]-1
    r3 = p/closes[-4]-1
    r5 = p/closes[-6]-1
    r10 = p/closes[-11]-1
    r20 = p/closes[-21]-1
    rets = [closes[i]/closes[i-1]-1 for i in range(1,len(closes))]
    vol = math.sqrt(sum(x*x for x in rets[-20:])/20)

    z = (0.30*r1/0.0008 + 0.35*r3/0.0016 + 0.25*r5/0.0024
         + 0.15*r10/0.004 + 0.10*r20/0.006)
    z *= max(0.30, min(1.0, 0.0015/max(vol,0.0004)))
    momentum_up = sigmoid(z)

    market_up = None
    if market:
        outcomes = parse_field(market.get("outcomes","[]"))
        prices = parse_field(market.get("outcomePrices","[]"))
        for o, pr in zip(outcomes, prices):
            if str(o).lower() == "up":
                market_up = float(pr)
                break

    final_up = momentum_up if market_up is None else 0.60*market_up + 0.40*momentum_up
    edge = abs(final_up-0.5)
    if edge < 0.055:
        signal, confidence = "NO SIGNAL", "LOW"
    else:
        signal = "UP" if final_up > 0.5 else "DOWN"
        confidence = "MEDIUM" if edge < 0.10 else "HIGH"

    return {
        "signal":signal, "model_up":final_up, "momentum_up":momentum_up,
        "market_up":market_up, "confidence":confidence, "btc":p
    }

@app.get("/api/signal")
async def signal():
    start = window_start()
    end = start + 300
    remaining = max(0, end-int(time.time()))
    async with httpx.AsyncClient(headers={"User-Agent":"BTC5M-Predictor/1.0"}) as c:
        market = None
        try:
            market = await get_market(c, start)
        except Exception:
            pass
        try:
            result = make_signal(await get_klines(c), market)
        except Exception as e:
            return {"ok":False,"error":str(e),"remaining":remaining}
    result.update({
        "ok":True, "remaining":remaining, "window_start":start,
        "window_end":end, "slug":f"btc-updown-5m-{start}",
        "polymarket_found":market is not None
    })
    return result

HTML = '''<!doctype html>
<html lang="uk"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BTC 5M Predictor</title>
<style>
body{margin:0;background:#101217;color:#eee;font-family:-apple-system,BlinkMacSystemFont,Arial}
.wrap{max-width:680px;margin:auto;padding:18px}.card{background:#191c22;border-radius:18px;padding:20px;margin:12px 0}
.signal{font-size:52px;font-weight:900;margin:12px 0}.up{color:#35d07f}.down{color:#ff6473}.none{color:#aaa}
.price{font-size:30px;font-weight:700}.prob{font-size:22px}.timer{font-size:34px}
.small{color:#9aa1ad;font-size:13px;line-height:1.5}.row{display:flex;justify-content:space-between;margin:10px 0}
table{width:100%;border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #2a2e35;text-align:left}
button{width:100%;padding:12px;border:0;border-radius:12px;background:#303642;color:white}
</style></head><body><div class="wrap">
<h1>BTC 5M Predictor</h1>
<div class="card"><div class="small">BTCUSDT</div><div class="price" id="btc">—</div>
<div class="small" id="pm">Polymarket: перевірка…</div></div>
<div class="card"><div class="small">СИГНАЛ</div><div id="sig" class="signal none">LOADING</div>
<div id="prob" class="prob">—</div><div id="conf" class="small">—</div>
<div class="small">До кінця</div><div id="timer" class="timer">—</div></div>
<div class="card"><div class="row"><span>Market UP</span><b id="mkt">—</b></div>
<div class="row"><span>Momentum UP</span><b id="mom">—</b></div></div>
<div class="card"><h3>Історія</h3><div class="small">Зберігається локально в Safari.</div>
<table><thead><tr><th>Час</th><th>Сигнал</th><th>UP</th></tr></thead><tbody id="hist"></tbody></table>
<br><button onclick="localStorage.removeItem('btc5m');draw()">Очистити</button></div>
<div class="card small"><b>Увага:</b> % — модельна оцінка, не гарантія. Settlement Polymarket BTC 5M
використовує Chainlink BTC/USD TWAP. Бот ставки не робить.</div>
</div>
<script>
let last=null;
function pc(x){return x==null?'—':(x*100).toFixed(1)+'%'}
function tm(s){s=Math.max(0,Math.ceil(s));return String(Math.floor(s/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0')}
function draw(){let a=JSON.parse(localStorage.getItem('btc5m')||'[]');
document.getElementById('hist').innerHTML=a.slice(-50).reverse().map(x=>'<tr><td>'+
new Date(x.t).toLocaleTimeString()+'</td><td>'+x.s+'</td><td>'+pc(x.p)+'</td></tr>').join('')}
function save(x){if(!x.ok||x.window_start===last)return;last=x.window_start;
let a=JSON.parse(localStorage.getItem('btc5m')||'[]');a.push({t:Date.now(),s:x.signal,p:x.model_up});
localStorage.setItem('btc5m',JSON.stringify(a.slice(-200)));draw()}
async function refresh(){try{let x=await fetch('/api/signal?t='+Date.now()).then(r=>r.json());
if(!x.ok){document.getElementById('sig').textContent='ERROR';return}
document.getElementById('btc').textContent='$'+Number(x.btc).toLocaleString(undefined,{maximumFractionDigits:2});
let e=document.getElementById('sig');e.textContent=x.signal;
e.className='signal '+(x.signal==='UP'?'up':x.signal==='DOWN'?'down':'none');
document.getElementById('prob').textContent=pc(x.model_up)+' UP / '+pc(1-x.model_up)+' DOWN';
document.getElementById('conf').textContent='Confidence: '+x.confidence;
document.getElementById('timer').textContent=tm(x.remaining);
document.getElementById('mkt').textContent=pc(x.market_up);
document.getElementById('mom').textContent=pc(x.momentum_up);
document.getElementById('pm').textContent=x.polymarket_found?'Polymarket market знайдено':'Polymarket market не знайдено';
save(x)}catch(e){document.getElementById('sig').textContent='CONNECTING'}}
draw();refresh();setInterval(refresh,5000);
</script></body></html>'''

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML
