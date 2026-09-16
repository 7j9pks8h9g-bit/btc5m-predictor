import json
import math
import time
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="BTC 5M Predictor")

GAMMA = "https://gamma-api.polymarket.com"
COINBASE = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
KRAKEN = "https://api.kraken.com/0/public/OHLC"
BINANCE = "https://api.binance.com/api/v3/klines"

_http_headers = {"User-Agent": "BTC5M-Predictor/2.0"}
_market_cache: dict[str, Any] = {"ts": 0.0, "market": None, "error": None}


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, x))))


def window_start() -> int:
    return (int(time.time()) // 300) * 300


def parse_jsonish(value: Any) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def market_text(market: dict) -> str:
    return " ".join(
        str(market.get(k, ""))
        for k in ("question", "slug", "description", "title")
    ).lower()


def is_btc_5m_market(market: dict) -> bool:
    text = market_text(market)
    btc = "bitcoin" in text or "btc" in text
    updown = ("up" in text and "down" in text) or "up or down" in text
    five_min = any(x in text for x in ("5m", "5-min", "5 min", "5 minute", "5-minute"))
    return btc and updown and five_min


def extract_up_price(market: Optional[dict]) -> Optional[float]:
    if not market:
        return None
    outcomes = parse_jsonish(market.get("outcomes"))
    prices = parse_jsonish(market.get("outcomePrices"))
    for outcome, price in zip(outcomes, prices):
        if str(outcome).strip().lower() == "up":
            try:
                return float(price)
            except (TypeError, ValueError):
                return None
    # Some records may expose bestAsk/bestBid instead of outcomePrices.
    for key in ("bestAsk", "lastTradePrice"):
        try:
            value = market.get(key)
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            pass
    return None


def market_end_timestamp(market: dict) -> Optional[float]:
    raw = market.get("endDate") or market.get("endDateIso")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


async def get_market(client: httpx.AsyncClient, start: int) -> tuple[Optional[dict], Optional[str]]:
    # First try the known 5-minute slug pattern around the current window.
    for ts in (start, start - 300, start + 300):
        slug = f"btc-updown-5m-{ts}"
        try:
            r = await client.get(f"{GAMMA}/markets", params={"slug": slug}, timeout=5)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list) and data:
                return data[0], None
        except Exception:
            continue

    # Fallback: discover active markets and choose the BTC 5m market closest to now.
    try:
        r = await client.get(
            f"{GAMMA}/markets",
            params={"active": "true", "closed": "false", "limit": 500, "offset": 0},
            timeout=8,
        )
        r.raise_for_status()
        data = r.json()
        markets = data if isinstance(data, list) else data.get("data", [])
        candidates = [m for m in markets if isinstance(m, dict) and is_btc_5m_market(m)]
        now = time.time()
        future = [m for m in candidates if (market_end_timestamp(m) or 0) >= now - 30]
        if future:
            future.sort(key=lambda m: abs((market_end_timestamp(m) or now) - (start + 300)))
            return future[0], None
        if candidates:
            candidates.sort(key=lambda m: abs((market_end_timestamp(m) or now) - (start + 300)))
            return candidates[0], None
        return None, "No active BTC 5M market found"
    except Exception as exc:
        return None, f"Polymarket API: {type(exc).__name__}: {exc}"


async def get_coinbase_candles(client: httpx.AsyncClient) -> tuple[list[list[float]], str]:
    r = await client.get(
        COINBASE,
        params={"granularity": 60, "limit": 60},
        timeout=8,
    )
    r.raise_for_status()
    data = r.json()
    # Coinbase: [time, low, high, open, close, volume]
    rows = sorted(data, key=lambda x: x[0])
    return rows, "Coinbase"


async def get_kraken_candles(client: httpx.AsyncClient) -> tuple[list[list[float]], str]:
    r = await client.get(KRAKEN, params={"pair": "XBTUSD", "interval": 1}, timeout=8)
    r.raise_for_status()
    data = r.json()
    if data.get("error"):
        raise RuntimeError(str(data["error"]))
    result = data.get("result", {})
    pair_key = next((k for k in result if k != "last"), None)
    if not pair_key:
        raise RuntimeError("Kraken returned no OHLC data")
    # Kraken rows: [time, open, high, low, close, vwap, volume, count]
    rows = []
    for x in result[pair_key][-60:]:
        rows.append([x[0], x[3], x[2], x[1], x[4], x[6]])
    return rows, "Kraken"


async def get_binance_candles(client: httpx.AsyncClient) -> tuple[list[list[float]], str]:
    r = await client.get(
        BINANCE,
        params={"symbol": "BTCUSDT", "interval": "1m", "limit": 60},
        timeout=8,
    )
    r.raise_for_status()
    data = r.json()
    return data, "Binance"


async def get_candles(client: httpx.AsyncClient) -> tuple[list[list[float]], str, list[str]]:
    errors = []
    for fn in (get_coinbase_candles, get_kraken_candles, get_binance_candles):
        try:
            rows, source = await fn(client)
            if len(rows) >= 21:
                return rows, source, errors
        except Exception as exc:
            errors.append(f"{fn.__name__}: {type(exc).__name__}: {exc}")
    raise RuntimeError(" | ".join(errors) or "No candle provider returned enough data")


def make_signal(rows: list[list[float]], market: Optional[dict]) -> dict:
    closes = [float(x[4]) for x in rows]
    if len(closes) < 21:
        raise RuntimeError(f"Only {len(closes)} candles received; need at least 21")

    p = closes[-1]
    r1 = p / closes[-2] - 1
    r3 = p / closes[-4] - 1
    r5 = p / closes[-6] - 1
    r10 = p / closes[-11] - 1
    r20 = p / closes[-21] - 1
    rets = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]
    vol = math.sqrt(sum(x * x for x in rets[-20:]) / 20)

    z = (
        0.30 * r1 / 0.0008
        + 0.35 * r3 / 0.0016
        + 0.25 * r5 / 0.0024
        + 0.15 * r10 / 0.004
        + 0.10 * r20 / 0.006
    )
    z *= max(0.30, min(1.0, 0.0015 / max(vol, 0.0004)))
    momentum_up = sigmoid(z)

    market_up = extract_up_price(market)
    # Market price is informative but not treated as ground truth.
    final_up = momentum_up if market_up is None else 0.60 * market_up + 0.40 * momentum_up
    edge = abs(final_up - 0.5)

    if edge < 0.055:
        signal, confidence = "NO SIGNAL", "LOW"
    elif edge < 0.10:
        signal, confidence = ("UP" if final_up > 0.5 else "DOWN"), "MEDIUM"
    else:
        signal, confidence = ("UP" if final_up > 0.5 else "DOWN"), "HIGH"

    return {
        "signal": signal,
        "model_up": final_up,
        "momentum_up": momentum_up,
        "market_up": market_up,
        "confidence": confidence,
        "btc": p,
    }


@app.get("/api/signal")
async def signal():
    start = window_start()
    end = start + 300
    remaining = max(0, end - int(time.time()))

    async with httpx.AsyncClient(headers=_http_headers, follow_redirects=True) as client:
        market = None
        market_error = None
        now = time.time()
        if now - _market_cache["ts"] < 20:
            market = _market_cache["market"]
            market_error = _market_cache["error"]
        else:
            market, market_error = await get_market(client, start)
            _market_cache.update(ts=now, market=market, error=market_error)

        try:
            rows, candle_source, provider_errors = await get_candles(client)
            result = make_signal(rows, market)
        except Exception as exc:
            return {
                "ok": False,
                "error": f"Market data error: {type(exc).__name__}: {exc}",
                "market_error": market_error,
                "remaining": remaining,
            }

    result.update(
        {
            "ok": True,
            "remaining": remaining,
            "window_start": start,
            "window_end": end,
            "market_slug": market.get("slug") if market else None,
            "polymarket_found": market is not None,
            "market_error": market_error,
            "candle_source": candle_source,
            "provider_errors": provider_errors,
        }
    )
    return result


HTML = r'''<!doctype html>
<html lang="uk"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BTC 5M Predictor</title>
<style>
body{margin:0;background:#101217;color:#eee;font-family:-apple-system,BlinkMacSystemFont,Arial}
.wrap{max-width:680px;margin:auto;padding:18px}.card{background:#191c22;border-radius:18px;padding:20px;margin:12px 0}
.signal{font-size:52px;font-weight:900;margin:12px 0}.up{color:#35d07f}.down{color:#ff6473}.none{color:#aaa}.price{font-size:30px;font-weight:700}
.prob{font-size:22px}.timer{font-size:34px}.small{color:#9aa1ad;font-size:13px;line-height:1.5}.row{display:flex;justify-content:space-between;margin:10px 0}
table{width:100%;border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #2a2e35;text-align:left}
button{width:100%;padding:12px;border:0;border-radius:12px;background:#303642;color:white}
.err{background:#2b1c20;color:#ffb4bd;border-radius:12px;padding:12px;margin-top:12px;display:none;white-space:pre-wrap;word-break:break-word}
</style></head><body><div class="wrap">
<h1>BTC 5M Predictor</h1>
<div class="card"><div class="small">BTC</div><div class="price" id="btc">—</div>
<div class="small" id="source">Data source: —</div><div class="small" id="pm">Polymarket: перевірка…</div><div id="err" class="err"></div></div>
<div class="card"><div class="small">СИГНАЛ</div><div id="sig" class="signal none">LOADING</div>
<div id="prob" class="prob">—</div><div id="conf" class="small">—</div>
<div class="small">До кінця</div><div id="timer" class="timer">—</div></div>
<div class="card"><div class="row"><span>Market UP</span><b id="mkt">—</b></div>
<div class="row"><span>Momentum UP</span><b id="mom">—</b></div></div>
<div class="card"><h3>Історія</h3><div class="small">Зберігається локально в Safari.</div>
<table><thead><tr><th>Час</th><th>Сигнал</th><th>UP</th></tr></thead><tbody id="hist"></tbody></table>
<br><button onclick="localStorage.removeItem('btc5m');draw()">Очистити</button></div>
<div class="card small"><b>Важливо:</b> це евристична модель, а не гарантований прогноз. Вона не робить ставки автоматично. Дані BTC беруться з публічних біржових API; ціна Polymarket використовується лише як додатковий сигнал. Розрахунок результату ринку Polymarket може використовувати інший референс, тому біржова ціна не є тотожною ціною розрахунку.</div>
</div>
<script>
let last=null;
function pc(x){return x==null?'—':(x*100).toFixed(1)+'%'}
function tm(s){s=Math.max(0,Math.ceil(s));return String(Math.floor(s/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0')}
function draw(){let a=JSON.parse(localStorage.getItem('btc5m')||'[]');document.getElementById('hist').innerHTML=a.slice(-50).reverse().map(x=>'<tr><td>'+new Date(x.t).toLocaleTimeString()+'</td><td>'+x.s+'</td><td>'+pc(x.p)+'</td></tr>').join('')}
function save(x){if(!x.ok||x.window_start===last)return;last=x.window_start;let a=JSON.parse(localStorage.getItem('btc5m')||'[]');a.push({t:Date.now(),s:x.signal,p:x.model_up});localStorage.setItem('btc5m',JSON.stringify(a.slice(-200)));draw()}
async function refresh(){
try{
 let r=await fetch('/api/signal?t='+Date.now(),{cache:'no-store'}); let x=await r.json();
 let err=document.getElementById('err'); err.style.display='none'; err.textContent='';
 if(!x.ok){document.getElementById('sig').textContent='ERROR';document.getElementById('sig').className='signal none';err.textContent=x.error||'Невідома помилка';err.style.display='block';return}
 document.getElementById('btc').textContent='$'+Number(x.btc).toLocaleString(undefined,{maximumFractionDigits:2});
 document.getElementById('source').textContent='Data source: '+x.candle_source;
 document.getElementById('pm').textContent=x.polymarket_found?'Polymarket: market знайдено':'Polymarket: market не знайдено (модель працює без нього)';
 let e=document.getElementById('sig');e.textContent=x.signal;e.className='signal '+(x.signal==='UP'?'up':x.signal==='DOWN'?'down':'none');
 document.getElementById('prob').textContent=pc(x.model_up)+' UP / '+pc(1-x.model_up)+' DOWN';
 document.getElementById('conf').textContent='Confidence: '+x.confidence;
 document.getElementById('timer').textContent=tm(x.remaining);
 document.getElementById('mkt').textContent=pc(x.market_up);document.getElementById('mom').textContent=pc(x.momentum_up);save(x)
}catch(e){let err=document.getElementById('err');err.textContent='Помилка з’єднання з сервером: '+e;err.style.display='block';document.getElementById('sig').textContent='CONNECTING'}}
draw();refresh();setInterval(refresh,5000);
</script></body></html>'''


@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML
