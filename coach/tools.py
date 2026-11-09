"""教練 agent 可以呼叫的工具：折溢價檢查、通路選擇＋持有成本體檢。

邏輯來自 weekly/1019 的 premium_check.py 與 weekly/1102 的 channel_check.py，
這裡改成回傳 dict 給 LLM 讀，不直接 print。費用數字資料日期 115/9/30；僅供學習，非投資建議。
"""
import json
import math
import ssl
import statistics
import time
import urllib.request
from datetime import date, datetime

URL = "https://mis.twse.com.tw/stock/data/all_etf.txt"
HISTORY_URL = "https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date={ym}01&stockNo={code}"
UA = "Mozilla/5.0 (course-prototype etf-coach)"

PREMIUM_THRESHOLD = 3.0  # 折溢價警示門檻 %
FEE_RATE = 0.001425      # 證券手續費上限 0.1425%
TAX_RATE = 0.001         # ETF 賣出證交稅 0.1%
FEEDER_CUTOFF = "13:00"  # 基富通境內基金單筆申購截止時間
FEEDER_EXTRA = 0.0011    # 低估法：連結基金這一層的年費用率（2025 年 0.11%）
# 實際報酬差法：連結基金比 0050 每年少賺多少（近 1／3／5 年換算）
ACTUAL_GAP = {1: 0.0037, 3: 0.0049, 5: 0.00276}
HAS_FEEDER = {"0050": "元大台灣卓越50 ETF連結基金", "0056": "元大台灣高股息 ETF連結基金",
              "006201": "元大富櫃50 ETF連結基金"}
MAINTENANCE = 1.30       # 融資維持率門檻（粗算：市值 ÷ 借款）
MAX_LEVERAGE = 2.5       # 自備 4 成、借 6 成
HISTORY_MONTHS = 6       # 用近幾個月的收盤價估波動
TRADING_DAYS = 252
MAX_DAILY_MOVE = 0.11    # 對數報酬，略高於漲跌幅 10%
Z_95 = 1.645             # 常態分配單尾 95%
_closes_cache = {}       # 歷史資料一天只抓一次，證交所有頻率限制


def to_float(x):
    """證交所欄位有時是數字、有時是含逗號的字串，統一轉成 float；空值或 '-' 回傳 None。"""
    try:
        v = float(str(x).replace(",", ""))
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def _get_json(url):
    ctx = ssl.create_default_context()
    # 證交所憑證缺少 Subject Key Identifier，只關掉嚴格模式，仍會驗證憑證鏈與網域
    ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=10, context=ctx) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_closes(code):
    """抓近幾個月的每日收盤價，由舊到新回傳 [(民國日期, 收盤價), ...]；同一天重複查用快取。"""
    key = (code, date.today())
    if key in _closes_cache:
        return _closes_cache[key]
    today = date.today()
    closes = []
    for back in range(HISTORY_MONTHS - 1, -1, -1):
        months = today.year * 12 + today.month - 1 - back
        rows = _get_json(HISTORY_URL.format(ym=f"{months // 12}{months % 12 + 1:02d}", code=code)).get("data") or []
        closes += [(r[0], to_float(r[6])) for r in rows if to_float(r[6])]
        if back:
            time.sleep(2)
    if len(closes) >= 60:  # 資料不完整時不快取，下次重抓
        _closes_cache[key] = closes
    return closes


def market_note(now=None):
    """非交易時段回傳提醒（沒有扣掉國定假日）；盤中回傳 None。"""
    now = now or datetime.now()
    if now.weekday() < 5 and "09:00" <= now.strftime("%H:%M") <= "13:30":
        return None
    return "現在不是台股交易時段，這是最後一筆資料，不是即時價格"


def fetch_twse():
    """抓證交所 ETF 預估淨值 JSON，回傳 {代號: 資料}。"""
    data = _get_json(URL)
    out = {}
    for group in data.get("a1", []):
        for row in group.get("msgArray", []):
            # a 代號, b 名稱, e 市價, f 預估淨值, g 預估折溢價%, h 前一營業日淨值, i 日期, j 時間（依官網頁面推得）
            est, prev = to_float(row.get("f")), to_float(row.get("h"))
            out[row["a"]] = {
                "name": row.get("b", "").split("(")[0],
                "price": to_float(row.get("e")),
                "nav": est or prev,
                "nav_type": "預估" if est else "前日",
                "time": f"{row.get('i', '')} {row.get('j', '')}",
            }
    return out


def _quote(code):
    """取一檔 ETF 的市價、淨值和折溢價率；抓不到就丟 ValueError，由 agent 回報給模型。"""
    d = fetch_twse().get(code)
    if not d or not d["price"] or not d["nav"]:
        raise ValueError(f"{code} 查無市價或淨值（可能不是上市 ETF 代號，或尚未開盤成交）")
    return d, (d["price"] - d["nav"]) / d["nav"]


def round_trip_cost(amount, discount, min_fee):
    """0050 一買一賣的成本（比例）：手續費兩次（有最低手續費）＋證交稅一次。"""
    fee = max(amount * FEE_RATE * discount, min_fee)
    return (2 * fee) / amount + TAX_RATE


def feeder_drag(years, method):
    """連結基金持有 n 年總共比 0050 少賺多少（比例）。"""
    if method == "low":
        return FEEDER_EXTRA * years
    rate = ACTUAL_GAP[1] if years <= 1 else ACTUAL_GAP[3] if years <= 3 else ACTUAL_GAP[5]
    return rate * years


def check_premium(code):
    """工具 1：折溢價檢查。"""
    d, prem = _quote(code)
    pct = prem * 100
    if pct > PREMIUM_THRESHOLD:
        status = "溢價偏高，建議這次先別追價"
    elif pct < -PREMIUM_THRESHOLD:
        status = "折價偏大，先查原因再決定"
    else:
        status = "正常範圍"
    return {"code": code, "name": d["name"], "price": d["price"], "nav": d["nav"], "nav_type": d["nav_type"],
            "premium_pct": round(pct, 2), "threshold_pct": PREMIUM_THRESHOLD, "status": status, "data_time": d["time"],
            "market_note": market_note()}


def compare_channels(code, amount, years, discount=1.0, min_fee=1.0, method="actual"):
    """工具 2：比較證券戶買 ETF 和基富通買連結基金，算損益兩平溢價 p* = D(n) − RT。"""
    if amount <= 0 or years <= 0:
        raise ValueError("amount 和 years 都要大於 0")
    if not 0 < discount <= 1:
        raise ValueError("discount 要介於 0 和 1 之間，例如 2 折填 0.2")
    if min_fee < 0:
        raise ValueError("min_fee 不能是負數")
    if method not in ("low", "actual"):
        raise ValueError("method 只能是 low 或 actual")
    if code not in HAS_FEEDER:
        return {"code": code, "has_feeder_fund": False,
                "advice": "沒有連結基金，沒有另一條路；溢價偏高就先暫停買進、等溢價收斂"}
    if code != "0050":
        return {"code": code, "has_feeder_fund": True, "supported": False,
                "advice": f"有連結基金（{HAS_FEEDER[code]}），但費用資料還沒查，目前只支援 0050"}

    d, prem = _quote(code)
    rt = round_trip_cost(amount, discount, min_fee)
    drag = feeder_drag(years, method)
    p_star = drag - rt
    saving = abs((rt + prem) - drag)
    cheaper = "連結基金" if prem > p_star else "ETF（證券戶）"
    # 原因和定義由程式寫好交給模型，實測發現讓模型自己解釋會把因果講反
    result = {"reason": f"持有 {years:g} 年，連結基金累積比 0050 少賺 {drag:.3%}；ETF 買賣成本 {rt:.3%} 加上現在折溢價 "
                        f"{prem:+.3%}，合計 {rt + prem:.3%}。比較兩邊後，{cheaper}的總成本較低。",
              "principle": "ETF 買賣成本只付一次，放越久攤越薄；連結基金少賺的部分每年累積。"
                           "所以短期偏向連結基金、長期偏向 ETF。金額很小時，最低手續費會讓 ETF 的成本比例變高。",
              "feeder_drag_meaning": ("用連結基金實際報酬落後 0050 的幅度估算，不是經理費" if method == "actual"
                                      else "只算連結基金這一層的費用率，可能低估") + f"；基金這一層的費用率每年 {FEEDER_EXTRA:.2%}",
              "assumptions": [f"券商手續費用上限 {FEE_RATE:.4%} 乘以折數 {discount:g}",
                              f"每筆最低手續費用 {min_fee:g} 元（各券商不同）",
                              "基富通申購、買回手續費 0 元（持有超過 7 日）"]}
    result |= {"code": code, "has_feeder_fund": True, "supported": True, "amount": amount, "years": years,
              "premium_pct": round(prem * 100, 3), "data_time": d["time"], "market_note": market_note(),
              "etf_round_trip_cost_pct": round(rt * 100, 3), "feeder_drag_pct": round(drag * 100, 3),
              "break_even_premium_pct": round(p_star * 100, 3),
              "cheaper_channel": cheaper,
              "saving_pct": round(saving * 100, 3), "saving_twd": round(amount * saving),
              "cost_method": "只算費用率" if method == "low" else "實際報酬差",
              "fee_data_date": "115/9/30"}
    if prem > p_star and datetime.now().strftime("%H:%M") >= FEEDER_CUTOFF:
        result["warning"] = f"基富通單筆今天 {FEEDER_CUTOFF} 已截止，現在下單會用下一營業日淨值，到時溢價可能不一樣"
    return result


def margin_call_drop(leverage):
    """開 leverage 倍時，價格跌多少（比例）會碰到維持率門檻。"""
    return 1 - MAINTENANCE * (1 - 1 / leverage)


def touch_probability(drop, sigma):
    """一年內碰到追繳線的機率：無漂移隨機漫步的反射原理 2Φ(ln(1−drop)/σ)。"""
    return 2 * statistics.NormalDist().cdf(math.log(1 - drop) / sigma)


def margin_risk(code, leverage):
    """工具 3：風險係數。用近期波動估開槓桿時被追繳的風險。"""
    if not 1 < leverage <= MAX_LEVERAGE:
        raise ValueError(f"leverage 要大於 1、最多 {MAX_LEVERAGE} 倍（自備 4 成、借 6 成）")
    closes = fetch_closes(code)
    if len(closes) < 20:
        raise ValueError(f"{code} 查無足夠的歷史收盤價，無法估波動")
    prices = [p for _, p in closes]
    # 台股單日漲跌幅上限 10%，超過多半是分割或除權息造成的，不算波動
    returns = [r for r in (math.log(b / a) for a, b in zip(prices, prices[1:])) if abs(r) <= MAX_DAILY_MOVE]
    daily = statistics.stdev(returns)
    sigma = daily * math.sqrt(TRADING_DAYS)
    rows = [{"leverage": x, "margin_call_drop_pct": round(margin_call_drop(x) * 100, 1),
             "touch_probability_1y_pct": round(touch_probability(margin_call_drop(x), sigma) * 100, 1)}
            for x in sorted({1.5, 2.0, 2.5, float(leverage)})]
    picked = next(r for r in rows if r["leverage"] == float(leverage))
    return {"code": code, "leverage": leverage,
            "risk_coefficient_annual_volatility_pct": round(sigma * 100, 1),
            "daily_volatility_pct": round(daily * 100, 2),
            "sample": f"{closes[0][0]}～{closes[-1][0]}，{len(closes)} 個交易日",
            "margin_call_drop_pct": picked["margin_call_drop_pct"],
            "touch_probability_1y_pct": picked["touch_probability_1y_pct"],
            "one_day_95pct_loss_on_own_capital_pct": round(leverage * Z_95 * daily * 100, 1),
            "comparison": rows,
            "meaning": "風險係數在這裡用年化波動度表示，代表價格平常晃動多大；波動越大，同樣的槓桿越容易被追繳。"
                       "追繳跌幅是從買進時的價格算起，不是從高點。"
                       "一天 95% 損失是指約 95% 的日子，一天虧損不超過本金的這個比例。",
            "limits": ["只用近半年的樣本，波動低的時期也可能突然暴跌", "沒算跳空和長期漂移，是粗估", "收盤價沒有還原配息",
                       f"維持率用 {MAINTENANCE:.0%} 粗算，實際算法要看券商規定"],
            "ask_user": "如果真的跌到追繳線，你有現金可以補，還是只能被賣掉？"}


TOOLS = {"check_premium": check_premium, "compare_channels": compare_channels, "margin_risk": margin_risk}

DECLARATIONS = [
    {
        "name": "check_premium",
        "description": "查一檔台灣上市 ETF 現在的市價、預估淨值和折溢價率。使用者問某檔 ETF 現在貴不貴、有沒有溢價時使用。",
        "parameters": {
            "type": "OBJECT",
            "properties": {"code": {"type": "STRING", "description": "ETF 代號，例如 0050、00878"}},
            "required": ["code"],
        },
    },
    {
        "name": "compare_channels",
        "description": "比較在證券戶買 ETF 和在基富通買連結基金哪條路划算，並做持有成本體檢。"
                       "需要金額和打算持有的年數，使用者沒講就先追問，不要自己假設。",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "code": {"type": "STRING", "description": "ETF 代號，例如 0050"},
                "amount": {"type": "NUMBER", "description": "這次要投入的金額（新台幣）"},
                "years": {"type": "NUMBER", "description": "打算持有幾年"},
                "discount": {"type": "NUMBER", "description": "券商手續費折數，2 折填 0.2；使用者沒講就不要填"},
                "min_fee": {"type": "NUMBER", "description": "零股每筆最低手續費（元）；使用者沒講就不要填"},
            },
            "required": ["code", "amount", "years"],
        },
    },
    {
        "name": "margin_risk",
        "description": "用近半年真實波動估算開槓桿（融資、質借再買股）的風險：風險係數、跌多少會被追繳、一年內碰到的機率。"
                       "使用者問槓桿、融資、質借、追繳時使用。需要槓桿倍數，使用者沒講就先追問，不要自己假設。",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "code": {"type": "STRING", "description": "ETF 代號，例如 0050"},
                "leverage": {"type": "NUMBER", "description": "槓桿倍數，大於 1、最多 2.5，例如 2"},
            },
            "required": ["code", "leverage"],
        },
    },
]
