import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from datetime import datetime, timedelta
import warnings
import gc
import random
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    st.warning("XGBoost tidak terinstal. Gunakan RandomForest sebagai fallback.")

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Robot Saham v6 Pro Tuned", page_icon="🤖")

st.markdown(
    """
<style>
    .block-container { padding-top: 0.5rem; padding-bottom: 0rem; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.2rem; }
    div[data-testid="stMetricValue"] { font-size: 1.1rem !important; }
    div[data-testid="stMetricLabel"] { font-size: 0.7rem !important; }
</style>
""",
    unsafe_allow_html=True,
)

CACHE_TTL = 300
DEFAULT_TICKER = "BBCA.JK"
BREAKOUT_COOLDOWN_HOURS = 24
TRADING_FEE = 0.0015
SLIPPAGE = 0.001

# session state
if "signal_history" not in st.session_state:
    st.session_state.signal_history = []
if "last_signal" not in st.session_state:
    st.session_state.last_signal = None
if "last_breakout_notify_time" not in st.session_state:
    st.session_state.last_breakout_notify_time = None
if "trading_mode" not in st.session_state:
    st.session_state.trading_mode = "Swing (Daily)"
if "custom_weights" not in st.session_state:
    st.session_state.custom_weights = {
        "trend": 2.0,
        "momentum": 1.5,
        "volume": 1.5,
        "smart_money": 1.5,
        "structure": 1.0,
    }
if "optimized_weights" not in st.session_state:
    st.session_state.optimized_weights = None
if "global_model" not in st.session_state:
    st.session_state.global_model = None
if "last_retrain" not in st.session_state:
    st.session_state.last_retrain = datetime.now()
if "walk_forward_result" not in st.session_state:
    st.session_state.walk_forward_result = None

# ====================== FUNGSI INDIKATOR (LENGKAP) ======================
# (sama seperti sebelumnya, tidak diubah agar tidak terlalu panjang)
# Saya akan menyalin fungsi-fungsi penting dari kode sebelumnya.
# Karena keterbatasan token, saya asumsikan fungsi2 ini sudah ada.
# Namun untuk memastikan kelengkapan, saya akan menyertakan semua fungsi yang diperlukan.
# (Sebenarnya bisa menggunakan kode dari jawaban sebelumnya, tapi saya tulis ulang secara ringkas)

def calculate_supertrend(df, period=10, multiplier=3.0):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    atr = ta.volatility.average_true_range(high, low, close, window=period)
    hl_avg = (high + low) / 2
    upper = hl_avg + (multiplier * atr)
    lower = hl_avg - (multiplier * atr)
    st_line = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    for i in range(period, len(df)):
        if i == period:
            st_line.iloc[i] = upper.iloc[i]
            direction.iloc[i] = 1 if close.iloc[i] > st_line.iloc[i] else -1
        else:
            if direction.iloc[i-1] == 1:
                if close.iloc[i] < lower.iloc[i]:
                    direction.iloc[i] = -1
                    st_line.iloc[i] = upper.iloc[i]
                else:
                    direction.iloc[i] = 1
                    st_line.iloc[i] = max(lower.iloc[i], st_line.iloc[i-1])
            else:
                if close.iloc[i] > upper.iloc[i]:
                    direction.iloc[i] = 1
                    st_line.iloc[i] = lower.iloc[i]
                else:
                    direction.iloc[i] = -1
                    st_line.iloc[i] = min(upper.iloc[i], st_line.iloc[i-1])
    return st_line, direction

def calculate_psar(df, step=0.02, max_step=0.2):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    psar = pd.Series(index=df.index, dtype=float)
    trend = pd.Series(index=df.index, dtype=int)
    if len(df) < 3:
        return psar, trend
    psar.iloc[0] = low.iloc[0]
    trend.iloc[0] = 1
    ep = high.iloc[0]
    af = step
    for i in range(1, len(df)):
        if trend.iloc[i-1] == 1:
            psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
            if low.iloc[i] <= psar.iloc[i]:
                trend.iloc[i] = -1
                psar.iloc[i] = ep
                af = step
                ep = low.iloc[i]
            else:
                trend.iloc[i] = 1
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + step, max_step)
        else:
            psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
            if high.iloc[i] >= psar.iloc[i]:
                trend.iloc[i] = 1
                psar.iloc[i] = ep
                af = step
                ep = high.iloc[i]
            else:
                trend.iloc[i] = -1
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + step, max_step)
    return psar, trend

def calculate_obv(df):
    close = df["Close"]
    volume = df["Volume"]
    return np.where(close > close.shift(1), volume, np.where(close < close.shift(1), -volume, 0)).cumsum()

def calculate_aroon(df, window=25):
    high = df["High"]
    low = df["Low"]
    def aroon_up_func(x):
        if len(x) != window: return 0
        return (window - x.argmax()) / window * 100
    def aroon_down_func(x):
        if len(x) != window: return 0
        return (window - x.argmin()) / window * 100
    aroon_up = high.rolling(window).apply(aroon_up_func, raw=True)
    aroon_down = low.rolling(window).apply(aroon_down_func, raw=True)
    return aroon_up, aroon_down

def calculate_gmma(df, fast=(3,5,8,10,12,15), slow=(30,35,40,45,50,60)):
    close = df["Close"]
    gmma_fast = [close.ewm(span=p, adjust=False).mean() for p in fast]
    gmma_slow = [close.ewm(span=p, adjust=False).mean() for p in slow]
    return gmma_fast, gmma_slow

def calculate_williams_r(df, window=14):
    high = df["High"].rolling(window).max()
    low = df["Low"].rolling(window).min()
    return (high - df["Close"]) / (high - low) * -100

def calculate_kst(df, roc1=10, roc2=15, roc3=20, roc4=30, ma1=10, ma2=10, ma3=10, ma4=15):
    close = df["Close"]
    r1 = close.pct_change(roc1)*100
    r2 = close.pct_change(roc2)*100
    r3 = close.pct_change(roc3)*100
    r4 = close.pct_change(roc4)*100
    return r1.rolling(ma1).mean() + r2.rolling(ma2).mean()*2 + r3.rolling(ma3).mean()*3 + r4.rolling(ma4).mean()*4

def calculate_elder_ray(df, window=13):
    ema = df["Close"].ewm(span=window, adjust=False).mean()
    return df["High"] - ema, df["Low"] - ema

def detect_market_regime(df, adx_threshold=25):
    if len(df) < 30: return "SIDEWAYS"
    adx = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14).adx()
    current_adx = adx.iloc[-1]
    ema20 = df["Close"].ewm(span=20).mean()
    ema_slope = ema20.iloc[-1] - ema20.iloc[-5]
    if current_adx > adx_threshold and abs(ema_slope/ema20.iloc[-1]) > 0.002:
        return "TREND"
    else:
        return "SIDEWAYS"

def get_swing_points(df, lookback=5):
    if len(df) < lookback*2:
        return df["Low"].iloc[-1], df["High"].iloc[-1]
    low = df["Low"]
    high = df["High"]
    swing_low = low.rolling(lookback, center=False).min().iloc[-1]
    swing_high = high.rolling(lookback, center=False).max().iloc[-1]
    return swing_low, swing_high

def get_mtf_alignment(df_daily, df_weekly, df_monthly):
    def sig_to_num(df):
        if df.empty or len(df) < 30: return 0
        ind = get_latest_indicators(df)
        if not ind: return 0
        score = 0
        if ind.get("price",0) > ind.get("ema200",0): score += 1
        else: score -= 1
        if ind.get("supertrend_dir",0) == 1: score += 1
        else: score -= 1
        return 1 if score >= 1 else -1 if score <= -1 else 0
    d = sig_to_num(df_daily)
    w = sig_to_num(df_weekly)
    m = sig_to_num(df_monthly)
    total_weight = 0.0
    weighted_sum = 0.0
    if not df_daily.empty and len(df_daily) >= 30:
        weighted_sum += d * 0.5
        total_weight += 0.5
    if not df_weekly.empty and len(df_weekly) >= 30:
        weighted_sum += w * 0.3
        total_weight += 0.3
    if not df_monthly.empty and len(df_monthly) >= 30:
        weighted_sum += m * 0.2
        total_weight += 0.2
    if total_weight == 0: return 0.0
    return weighted_sum / total_weight

@st.cache_data(ttl=CACHE_TTL)
def calculate_all_indicators(df, st_period=10, st_mult=3.0, mode="Swing (Daily)"):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty or len(df) < 30:
        idx = df.index if not df.empty else pd.date_range(end=datetime.today(), periods=1)
        dummy_df = pd.DataFrame(index=idx)
        required_cols = ["Close","High","Low","Volume","SMA20","SMA50","EMA200","RSI","MACD","MACD_Signal","MACD_Hist",
                         "Stoch_K","Stoch_D","ATR","Volume_MA","support","resistance","AD","CMF","ST_Supertrend","ST_Dir",
                         "PSAR","PSAR_Dir","OBV","OBV_MA","MFI","Aroon_Up","Aroon_Down","Williams_R","CCI","KST",
                         "Elder_Bull","Elder_Bear","GMMA_Spread"]
        for col in required_cols:
            dummy_df[col] = 0.0
        if not df.empty and "Close" in df.columns:
            dummy_df["Close"] = df["Close"].values
        return dummy_df

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    has_vol = "Volume" in df.columns and df["Volume"].sum() != 0
    volume = df["Volume"] if has_vol else pd.Series(0, index=df.index)

    df["SMA20"] = close.rolling(20, min_periods=1).mean()
    df["SMA50"] = close.rolling(50, min_periods=1).mean()
    df["EMA200"] = close.ewm(span=200, adjust=False).mean() if len(close) >= 200 else close

    df["RSI"] = ta.momentum.rsi(close, window=14) if len(close) >= 14 else 50.0

    if len(close) >= 26:
        macd = ta.trend.MACD(close)
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    else:
        df["MACD"] = df["MACD_Signal"] = df["MACD_Hist"] = 0.0

    if len(close) >= 14:
        df["Stoch_K"] = ta.momentum.stoch(high, low, close, window=14, smooth_window=3)
        df["Stoch_D"] = ta.momentum.stoch_signal(high, low, close, window=14, smooth_window=3)
    else:
        df["Stoch_K"] = df["Stoch_D"] = 50.0

    df["ATR"] = ta.volatility.average_true_range(high, low, close, window=14) if len(close)>=14 else close*0.02

    if has_vol:
        df["Volume_MA"] = volume.rolling(20, min_periods=1).mean()
    else:
        df["Volume_MA"] = 0.0

    df["support"] = low.rolling(20, min_periods=1).min()
    df["resistance"] = high.rolling(20, min_periods=1).max()

    if has_vol:
        try:
            df["AD"] = ta.volume.acc_dist_index(high, low, close, volume, fillna=True)
            df["CMF"] = ta.volume.chaikin_money_flow(high, low, close, volume, window=20, fillna=True)
        except:
            df["AD"] = df["CMF"] = 0.0
    else:
        df["AD"] = df["CMF"] = 0.0

    st_line, st_dir = calculate_supertrend(df, period=st_period, multiplier=st_mult)
    df["ST_Supertrend"] = st_line
    df["ST_Dir"] = st_dir

    psar_line, psar_dir = calculate_psar(df)
    df["PSAR"] = psar_line
    df["PSAR_Dir"] = psar_dir

    if has_vol:
        df["OBV"] = calculate_obv(df)
        df["OBV_MA"] = df["OBV"].rolling(20).mean()
        df["MFI"] = ta.volume.money_flow_index(high, low, close, volume, window=14, fillna=True)
    else:
        df["OBV"] = df["OBV_MA"] = 0.0
        df["MFI"] = 50.0

    aroon_up, aroon_down = calculate_aroon(df)
    df["Aroon_Up"] = aroon_up
    df["Aroon_Down"] = aroon_down

    df["Williams_R"] = calculate_williams_r(df)

    if len(close) >= 20:
        df["CCI"] = ta.trend.CCIIndicator(high, low, close, window=20).cci()
    else:
        df["CCI"] = 0.0

    if mode in ["Swing (Daily)", "Position (Weekly - Monthly)"] and len(df) >= 50:
        df["KST"] = calculate_kst(df)
    else:
        df["KST"] = 0.0

    bull, bear = calculate_elder_ray(df)
    df["Elder_Bull"] = bull
    df["Elder_Bear"] = bear

    if len(df) >= 30:
        gmma_fast, gmma_slow = calculate_gmma(df)
        df["GMMA_Fast_Avg"] = sum(gmma_fast)/len(gmma_fast)
        df["GMMA_Slow_Avg"] = sum(gmma_slow)/len(gmma_slow)
        df["GMMA_Spread"] = df["GMMA_Fast_Avg"] - df["GMMA_Slow_Avg"]
    else:
        df["GMMA_Spread"] = 0.0

    df = df.ffill().fillna(0)
    keep_cols = ["Close","High","Low","Volume","SMA20","SMA50","EMA200","RSI","MACD_Hist",
                 "Stoch_K","Stoch_D","ATR","Volume_MA","support","resistance","CMF",
                 "ST_Supertrend","ST_Dir","PSAR_Dir","OBV","OBV_MA","MFI","Williams_R","CCI",
                 "GMMA_Spread","Aroon_Up","Aroon_Down","Elder_Bull","Elder_Bear","KST"]
    df = df[keep_cols]
    return df

def get_latest_indicators(df):
    if df.empty: return {}
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    last_dict = df.iloc[-1].to_dict()
    def get_val(key, default=0.0):
        return last_dict.get(key, default)
    price = get_val("Close", 0.0)
    vol_status = "Normal"
    if "Volume_MA" in df.columns and "Volume" in df.columns:
        try:
            vol_ma = df["Volume_MA"].iloc[-1]
            if vol_ma > 0:
                vol = df["Volume"].iloc[-1]
                if vol > vol_ma * 1.5: vol_status = "Tinggi"
                elif vol < vol_ma: vol_status = "Rendah"
        except: pass
    high_30d = df["High"].tail(30).max() if ("High" in df and len(df)>=30) else price
    low_30d = df["Low"].tail(30).min() if ("Low" in df and len(df)>=30) else price
    swing_low, swing_high = get_swing_points(df)
    return {
        "price": price,
        "sma20": get_val("SMA20", price),
        "sma50": get_val("SMA50", price),
        "ema200": get_val("EMA200", price),
        "rsi": get_val("RSI", 50.0),
        "macd_hist": get_val("MACD_Hist", 0.0),
        "stoch_k": get_val("Stoch_K", 50.0),
        "stoch_d": get_val("Stoch_D", 50.0),
        "atr": get_val("ATR", price*0.02),
        "volume_status": vol_status,
        "support": get_val("support", price*0.95),
        "resistance": get_val("resistance", price*1.05),
        "supertrend_dir": get_val("ST_Dir", 0),
        "supertrend_value": get_val("ST_Supertrend", price),
        "psar_dir": get_val("PSAR_Dir", 0),
        "obv": get_val("OBV", 0.0),
        "obv_ma": get_val("OBV_MA", 0.0),
        "mfi": get_val("MFI", 50.0),
        "aroon_up": get_val("Aroon_Up", 50.0),
        "aroon_down": get_val("Aroon_Down", 50.0),
        "williams_r": get_val("Williams_R", -50.0),
        "cci": get_val("CCI", 0.0),
        "kst": get_val("KST", 0.0),
        "elder_bull": get_val("Elder_Bull", 0.0),
        "elder_bear": get_val("Elder_Bear", 0.0),
        "gmma_spread": get_val("GMMA_Spread", 0.0),
        "cmf": get_val("CMF", 0.0),
        "high_30d": high_30d,
        "low_30d": low_30d,
        "swing_low": swing_low,
        "swing_high": swing_high,
    }

# ====================== FEATURE & RULE ENGINE ======================
def normalize_range(x, min_val, max_val):
    if max_val - min_val == 0: return 0
    return (x - min_val) / (max_val - min_val) * 2 - 1

def build_feature_vector(ind):
    trend = np.mean([
        1 if ind["price"] > ind["ema200"] else -1,
        ind["supertrend_dir"],
        ind["psar_dir"],
        1 if ind["sma20"] > ind["sma50"] else -1
    ])
    rsi = normalize_range(ind["rsi"], 0, 100)
    stoch = normalize_range(ind["stoch_k"], 0, 100)
    will = normalize_range(ind["williams_r"], -100, 0)
    momentum = np.mean([rsi, stoch, will, np.sign(ind["macd_hist"])])
    volume = np.mean([
        1 if ind["volume_status"] == "Tinggi" else (-1 if ind["volume_status"] == "Rendah" else 0),
        1 if ind["obv"] > ind["obv_ma"] else -1
    ])
    smart = normalize_range(ind["cmf"], -0.3, 0.3)
    structure = 0
    if ind["price"] > ind["swing_high"]: structure = 1
    elif ind["price"] < ind["swing_low"]: structure = -1
    gmma = np.sign(ind["gmma_spread"])
    return {"trend": trend, "momentum": momentum, "volume": volume, "smart": smart, "structure": structure, "gmma": gmma}

def decision_engine_pro(indicators, mtf_alignment, regime, weights, threshold_buy=3.0, threshold_strong=6.0):
    f = build_feature_vector(indicators)
    w_trend = weights["trend"]
    w_mom = weights["momentum"]
    w_vol = weights["volume"]
    w_smart = weights["smart_money"]
    w_struct = weights["structure"]

    if regime == "TREND":
        w_trend *= 1.5
        w_mom *= 0.7
    else:
        w_trend *= 0.7
        w_mom *= 1.5

    score = (f["trend"] * w_trend +
             f["momentum"] * w_mom +
             f["volume"] * w_vol +
             f["smart"] * w_smart +
             f["structure"] * w_struct +
             f["gmma"] * 0.5)

    if mtf_alignment < 0:
        score *= 0.5
    if f["trend"] * f["momentum"] < 0:
        score *= 0.6
    if regime == "TREND" and f["trend"] > 0.7:
        score *= 1.2

    # Filter tambahan: jika sideways dan skor < 5, HOLD (untuk mengurangi whipsaw)
    if regime == "SIDEWAYS" and abs(score) < 5:
        return "HOLD", score, "LOW"

    score = max(-10, min(10, score))
    if score >= threshold_strong:
        return "STRONG BUY", score, "HIGH"
    elif score >= threshold_buy:
        return "BUY", score, "MEDIUM"
    elif score <= -threshold_strong:
        return "STRONG SELL", score, "HIGH"
    elif score <= -threshold_buy:
        return "SELL", score, "LOW"
    else:
        return "HOLD", score, "LOW"

# ====================== ML MODEL & HYBRID ======================
def build_ml_dataset(df):
    X, y = [], []
    for i in range(60, len(df)-5):
        slice_df = df.iloc[:i+1]
        ind = get_latest_indicators(slice_df)
        if not ind: continue
        f = build_feature_vector(ind)
        features = [f["trend"], f["momentum"], f["volume"], f["smart"], f["structure"], f["gmma"]]
        future_return = df["Close"].iloc[i+5] - df["Close"].iloc[i]
        label = 1 if future_return > 0 else 0
        X.append(features)
        y.append(label)
    return np.array(X), np.array(y)

def train_ml_model(df, use_xgb=True):
    X, y = build_ml_dataset(df)
    if len(X) < 50: return None
    if use_xgb and XGB_AVAILABLE:
        model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              eval_metric="logloss", random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    return model

def ml_predict(model, indicators):
    if model is None: return 0.5
    f = build_feature_vector(indicators)
    X = [[f["trend"], f["momentum"], f["volume"], f["smart"], f["structure"], f["gmma"]]]
    prob = model.predict_proba(X)[0][1]
    return prob

def final_decision_ultra(indicators, mtf, regime, weights, model, threshold_buy=3.0, threshold_strong=6.0):
    signal, score, _ = decision_engine_pro(indicators, mtf, regime, weights, threshold_buy, threshold_strong)
    ml_prob = ml_predict(model, indicators)
    ml_weight = 2.0 if abs(score) < 3 else 1.0
    score += (ml_prob - 0.5) * 10 * ml_weight
    if ml_prob > 0.8: score += 1.5
    elif ml_prob < 0.2: score -= 1.5
    score = max(-10, min(10, score))
    if score >= threshold_strong:
        return "STRONG BUY", score, ml_prob
    elif score >= threshold_buy:
        return "BUY", score, ml_prob
    elif score <= -threshold_strong:
        return "STRONG SELL", score, ml_prob
    elif score <= -threshold_buy:
        return "SELL", score, ml_prob
    else:
        return "HOLD", score, ml_prob

# ====================== BACKTEST DENGAN METRICS LENGKAP ======================
def run_backtest_advanced(df, weights, regime_func, period_days=180, risk_per_trade=0.02, model=None,
                          threshold_buy=3.0, threshold_strong=6.0):
    if df.empty or len(df) < period_days: return None
    df_test = df.tail(period_days).copy()
    balance = 100_000_000
    positions = []
    equity = [balance]
    trades = []
    for i in range(60, len(df_test)):
        slice_df = df_test.iloc[:i+1]
        ind = get_latest_indicators(slice_df)
        if not ind:
            equity.append(balance); continue
        regime = regime_func(slice_df)
        # Gunakan hybrid decision dengan threshold yang bisa diatur
        if model is not None:
            signal, score, _ = final_decision_ultra(ind, 0.0, regime, weights, model, threshold_buy, threshold_strong)
        else:
            signal, score, _ = decision_engine_pro(ind, 0.0, regime, weights, threshold_buy, threshold_strong)
        # Entry filter tambahan (RSI>70 atau CMF<0)
        if signal in ["BUY", "STRONG BUY"] and len(positions)==0:
            if ind["rsi"] > 70 or ind["cmf"] < 0:
                continue
            entry_price = ind["price"] * (1 + SLIPPAGE)
            stop_loss = ind["swing_low"] - ind["atr"] * 0.5
            if stop_loss >= entry_price:
                stop_loss = entry_price - 2 * ind["atr"]
            target = ind["resistance"]
            if stop_loss < entry_price:
                risk_amount = balance * risk_per_trade
                price_risk = entry_price - stop_loss
                if price_risk > 0:
                    shares = int(risk_amount / price_risk)
                    shares = (shares // 100) * 100
                    if shares > 0:
                        positions.append({"entry": entry_price, "stop": stop_loss, "target": target, "shares": shares})
        elif len(positions) > 0:
            pos = positions[0]
            current_price = ind["price"]
            if current_price <= pos["stop"] or current_price >= pos["target"]:
                exit_price = current_price * (1 - SLIPPAGE) if current_price < pos["entry"] else current_price * (1 + SLIPPAGE)
                gross_pnl = (exit_price - pos["entry"]) * pos["shares"]
                fee = (pos["entry"] * pos["shares"] + exit_price * pos["shares"]) * TRADING_FEE
                pnl = gross_pnl - fee
                balance += pnl
                trades.append({"entry": pos["entry"], "exit": exit_price, "pnl": pnl, "shares": pos["shares"]})
                positions.pop()
            equity.append(balance)
        else:
            equity.append(balance)
    if len(trades) == 0: return None
    win_trades = [t for t in trades if t["pnl"] > 0]
    winrate = len(win_trades)/len(trades)*100
    total_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    total_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    profit_factor = total_profit/total_loss if total_loss != 0 else float("inf")
    peak = equity[0]
    max_dd = 0.0
    for val in equity:
        if val > peak: peak = val
        if peak != 0: dd = (peak-val)/peak*100; max_dd = max(max_dd, dd)
    # Metrics tambahan
    avg_win = np.mean([t["pnl"] for t in trades if t["pnl"] > 0]) if win_trades else 0
    avg_loss = np.mean([t["pnl"] for t in trades if t["pnl"] < 0]) if len(trades)-len(win_trades) > 0 else 0
    rr_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    return {
        "trades": len(trades),
        "winrate": winrate,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
        "final_balance": balance,
        "equity": equity,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "rr_ratio": rr_ratio
    }

# ====================== AUTO OPTIMIZE WEIGHTS (SEDERHANA) ======================
def run_backtest_simple(df, weights, period_days=180, risk_per_trade=0.02, threshold_buy=3.0):
    if df.empty or len(df) < period_days: return None
    df_test = df.tail(period_days).copy()
    balance = 100_000_000
    positions = []
    trades = []
    for i in range(60, len(df_test)):
        slice_df = df_test.iloc[:i+1]
        ind = get_latest_indicators(slice_df)
        if not ind: continue
        regime = detect_market_regime(slice_df)
        signal, score, _ = decision_engine_pro(ind, 0.0, regime, weights, threshold_buy, 6.0)
        if signal in ["BUY", "STRONG BUY"] and len(positions)==0:
            if ind["rsi"] > 70 or ind["cmf"] < 0: continue
            entry_price = ind["price"] * (1 + SLIPPAGE)
            stop_loss = ind["swing_low"] - ind["atr"] * 0.5
            if stop_loss >= entry_price:
                stop_loss = entry_price - 2 * ind["atr"]
            if stop_loss < entry_price:
                risk_amount = balance * risk_per_trade
                price_risk = entry_price - stop_loss
                if price_risk > 0:
                    shares = int(risk_amount / price_risk)
                    shares = (shares // 100) * 100
                    if shares > 0:
                        positions.append({"entry": entry_price, "stop": stop_loss, "shares": shares})
        elif len(positions) > 0:
            pos = positions[0]
            current_price = ind["price"]
            if current_price <= pos["stop"]:
                exit_price = current_price * (1 - SLIPPAGE)
                gross_pnl = (exit_price - pos["entry"]) * pos["shares"]
                fee = (pos["entry"] * pos["shares"] + exit_price * pos["shares"]) * TRADING_FEE
                pnl = gross_pnl - fee
                balance += pnl
                trades.append(pnl)
                positions.pop()
    if len(trades) == 0: return None
    winrate = sum(1 for p in trades if p>0)/len(trades)*100
    profit_factor = sum(p for p in trades if p>0) / abs(sum(p for p in trades if p<0)) if sum(p for p in trades if p<0) != 0 else 999
    return {"winrate": winrate, "profit_factor": profit_factor}

def optimize_weights(df, trials=20):
    best_score = -999
    best_weights = None
    for _ in range(trials):
        w = {k: random.uniform(0.5, 3.0) for k in ["trend","momentum","volume","smart_money","structure"]}
        bt = run_backtest_simple(df, w, 180, 0.02, 3.0)
        if bt:
            score = bt["profit_factor"] - (bt.get("max_drawdown", 0)/100)
            if score > best_score:
                best_score = score
                best_weights = w
    return best_weights

# ====================== AUTO PARAMETER TUNING (RISK & THRESHOLD) ======================
def tune_parameters(df, weights, model=None, period_days=180):
    best_score = -999
    best_params = None
    for risk in [0.01, 0.015, 0.02]:
        for thresh in [2.5, 3.0, 3.5, 4.0]:
            bt = run_backtest_advanced(df, weights, detect_market_regime, period_days, risk, model, threshold_buy=thresh, threshold_strong=thresh+2)
            if bt:
                score = bt["profit_factor"] - (bt["max_drawdown"] / 100)
                if score > best_score:
                    best_score = score
                    best_params = (risk, thresh)
    return best_params, best_score

# ====================== FUNGSI LAIN (IHSG, dll) ======================
def get_market_context():
    try:
        ihsg = yf.download("^JKSE", period="5d", progress=False, auto_adjust=False)
        if ihsg.empty or "Close" not in ihsg.columns: return "⚠️ IHSG tidak tersedia", 0
        ihsg_close = ihsg["Close"]
        if len(ihsg_close) >= 3:
            last_3 = ihsg_close.iloc[-3:].pct_change().dropna()
            if (last_3 < 0).all(): return "⚠️ IHSG turun 3 hari", -2
            elif (last_3 > 0).all(): return "✅ IHSG naik 3 hari", 2
        return "IHSG sideways", 0
    except Exception:
        return "⚠️ IHSG tidak tersedia", 0

def log_signal(ticker, signal, score, price):
    record = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "ticker": ticker, "signal": signal, "score": score, "price": price, "status": "pending"}
    st.session_state.signal_history.insert(0, record)
    if len(st.session_state.signal_history)>100: st.session_state.signal_history.pop()

def show_toast_if_changed(new_signal, old_signal):
    if old_signal and new_signal != old_signal:
        if new_signal in ["STRONG BUY","BUY"] and old_signal in ["SELL","STRONG SELL","HOLD"]:
            st.toast(f"🟢 Sinyal berubah menjadi {new_signal}!", icon="🔔")
        elif new_signal in ["STRONG SELL","SELL"] and old_signal in ["BUY","STRONG BUY","HOLD"]:
            st.toast(f"🔴 Sinyal berubah menjadi {new_signal}!", icon="⚠️")
        elif new_signal=="HOLD" and old_signal in ["BUY","STRONG BUY","SELL","STRONG SELL"]:
            st.toast("⚪ Sinyal menjadi HOLD, waspadai perubahan", icon="ℹ️")

def show_breakout_alert(breakout, last_time):
    if breakout["is_breakout"]:
        now = datetime.now()
        if last_time is None or (now-last_time).total_seconds() > BREAKOUT_COOLDOWN_HOURS*3600:
            st.toast(f"🚀 {breakout['message']}", icon="📈")
            st.session_state.last_breakout_notify_time = now
            return True
    return False

def get_signal_interpretation(signal):
    interpretations = {
        "STRONG BUY": "📌 **Peluang naik sangat kuat.** Hybrid AI + rule engine mendukung.",
        "BUY": "📌 **Kondisi cukup baik.** Masih ada ruang naik, tetapi perlu konfirmasi.",
        "HOLD": "📌 **Tidak ada sinyal jelas.** Tunggu breakout atau breakdown.",
        "SELL": "📌 **Tekanan turun mulai terlihat.** Pertimbangkan exit bertahap.",
        "STRONG SELL": "📌 **Kondisi lemah.** Risiko besar, hindari entry.",
    }
    return interpretations.get(signal, "Netral.")

def calculate_probabilities(df):
    try:
        bull=0
        if df["RSI"].iloc[-1] < 35: bull+=1
        if df["Close"].iloc[-1] > df["SMA20"].iloc[-1]: bull+=1
        if df["MACD_Hist"].iloc[-1] > 0: bull+=1
        bear=3-bull
        total=bull+bear
        return (bull/total*100) if total>0 else 50, (bear/total*100) if total>0 else 50
    except: return 50,50

def position_sizing(capital, risk_percent, entry_price, stop_loss):
    if entry_price and stop_loss and capital>0:
        risk_amount = capital * (risk_percent/100)
        price_risk = abs(entry_price - stop_loss)
        if price_risk>0:
            shares = int(risk_amount / price_risk)
            shares = (shares//100)*100
            pos_value = shares * entry_price
            risk_amount_real = shares * price_risk
            return shares, pos_value, risk_amount_real
    return 0,0,0

def detect_true_breakout(df, atr):
    if len(df) < 20:
        return {"is_breakout": False, "strength": 0, "message": "Data tidak cukup"}
    current_price = df["Close"].iloc[-1]
    resistance = df["High"].rolling(20).max().iloc[-1]
    volume = df["Volume"].iloc[-1] if "Volume" in df.columns else 0
    vol_ma = df["Volume"].rolling(20).mean().iloc[-1] if "Volume" in df.columns else 1
    volume_confirm = volume > vol_ma * 1.5
    breakout_threshold = resistance + (atr * 0.5)
    if current_price > breakout_threshold:
        strength = (current_price - resistance) / atr
        if volume_confirm:
            return {"is_breakout": True, "strength": round(strength,2), "message": f"✅ TRUE BREAKOUT with volume! Strength: {strength:.1f}x ATR"}
        else:
            return {"is_breakout": False, "strength": round(strength,2), "message": "⚠️ Breakout tapi volume rendah, rawan false"}
    elif current_price > resistance:
        return {"is_breakout": False, "strength": 0, "message": "⚠️ False breakout (belum melewati ATR buffer)"}
    else:
        return {"is_breakout": False, "strength": 0, "message": "Tidak ada breakout"}

def get_entry_levels_advanced(df, indicators, regime):
    price = indicators["price"]
    atr = indicators["atr"]
    swing_low = indicators["swing_low"]
    swing_high = indicators["swing_high"]
    ema20 = indicators["sma20"]
    support = indicators["support"]
    resistance = indicators["resistance"]
    if regime == "TREND":
        if price > ema20 and price > swing_high:
            buy_entry = price + atr * 0.2
            stop_loss = swing_low - atr * 0.5
            target = resistance if resistance > price else price + atr * 3
        else:
            buy_entry = ema20
            stop_loss = swing_low - atr * 0.5
            target = swing_high + atr
    else:
        buy_entry = support + atr * 0.3
        stop_loss = swing_low - atr * 0.5
        target = resistance - atr * 0.5
    if stop_loss >= buy_entry:
        stop_loss = buy_entry - 2 * atr
        if stop_loss < 0: stop_loss = buy_entry * 0.95
    return {"buy_entry": buy_entry, "stop_loss": stop_loss, "target": target}

# ====================== WALK-FORWARD & MULTI-STOCK ======================
def walk_forward_validation(df, train_size=200, test_size=50, use_xgb=True):
    results = []
    for start in range(0, len(df) - train_size - test_size, test_size):
        train_df = df.iloc[start:start+train_size]
        test_df = df.iloc[start+train_size:start+train_size+test_size]
        model = train_ml_model(train_df, use_xgb)
        if model is None: continue
        correct, total = 0, 0
        for i in range(60, len(test_df)-5):
            slice_df = test_df.iloc[:i+1]
            ind = get_latest_indicators(slice_df)
            if not ind: continue
            prob = ml_predict(model, ind)
            future = test_df["Close"].iloc[i+5] > test_df["Close"].iloc[i]
            pred = prob > 0.5
            if pred == future: correct += 1
            total += 1
        if total > 0: results.append(correct/total)
    return np.mean(results) if results else 0

def train_global_model(ticker_list, period="1y", use_xgb=True):
    X_all, y_all = [], []
    for t in ticker_list:
        try:
            df_raw = yf.download(t, period=period, progress=False, auto_adjust=False)
            if df_raw.empty: continue
            df = calculate_all_indicators(df_raw, 10, 3.0, "Swing (Daily)")
            X, y = build_ml_dataset(df)
            if len(X) > 0:
                X_all.extend(X); y_all.extend(y)
        except: continue
    if len(X_all) < 200: return None
    X_all = np.array(X_all); y_all = np.array(y_all)
    if use_xgb and XGB_AVAILABLE:
        model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              eval_metric="logloss", random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    model.fit(X_all, y_all)
    return model

# ====================== MAIN APP ======================
def main():
    st.sidebar.markdown("# 🤖 Robot Saham v6 Pro Tuned")
    ticker_input = st.sidebar.text_input("Kode Saham", DEFAULT_TICKER)
    ticker = ticker_input.upper().strip()
    if not ticker.endswith(".JK") and not ticker.startswith("^"): ticker += ".JK"
    timeframe = st.sidebar.selectbox("Timeframe", ["1d","1wk","1mo"], index=0)
    trading_mode = st.sidebar.selectbox("Mode Trading", ["Scalping (5-15 menit)","Intraday (30 menit - 4 jam)","Swing (Daily)","Position (Weekly - Monthly)"], index=2)
    st.session_state.trading_mode = trading_mode
    if trading_mode in ["Scalping (5-15 menit)","Intraday (30 menit - 4 jam)"]:
        st.sidebar.warning("⚠️ Mode ini menggunakan data **daily** karena yfinance tidak menyediakan data menit yang stabil.")
    if trading_mode == "Scalping (5-15 menit)": st_period, st_mult = 7, 2.5
    elif trading_mode == "Intraday (30 menit - 4 jam)": st_period, st_mult = 8, 2.7
    elif trading_mode == "Swing (Daily)": st_period, st_mult = 10, 3.0
    else: st_period, st_mult = 20, 3.0
    ui_mode = st.sidebar.selectbox("Mode Tampilan", ["Simple (Recommended)","Advanced (Full Indicators)"], index=0)

    with st.sidebar.expander("⚙️ Bobot Indikator (mempengaruhi keputusan)"):
        weights = {}
        weights["trend"] = st.slider("Trend",0.0,3.0, st.session_state.custom_weights.get("trend",2.0),0.1)
        weights["momentum"] = st.slider("Momentum",0.0,3.0, st.session_state.custom_weights.get("momentum",1.5),0.1)
        weights["volume"] = st.slider("Volume",0.0,3.0, st.session_state.custom_weights.get("volume",1.5),0.1)
        weights["smart_money"] = st.slider("Smart Money",0.0,3.0, st.session_state.custom_weights.get("smart_money",1.5),0.1)
        weights["structure"] = st.slider("Structure",0.0,3.0, st.session_state.custom_weights.get("structure",1.0),0.1)
        if st.button("Simpan Bobot"): st.session_state.custom_weights = weights
    weights = st.session_state.custom_weights.copy()

    # Auto retrain
    def should_retrain(): return (datetime.now() - st.session_state.last_retrain) > timedelta(days=7)
    if should_retrain():
        st.cache_resource.clear()
        st.session_state.last_retrain = datetime.now()
        st.sidebar.info("Auto retrain triggered (7 days passed).")

    use_xgb = st.sidebar.checkbox("Gunakan XGBoost (lebih akurat)", value=XGB_AVAILABLE, disabled=not XGB_AVAILABLE)
    if not XGB_AVAILABLE and use_xgb:
        st.sidebar.warning("XGBoost tidak terinstal, gunakan RandomForest.")
        use_xgb = False

    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    with st.spinner(f"Memuat {ticker}..."):
        period = "2y" if trading_mode == "Position (Weekly - Monthly)" else "1y"
        df_raw = yf.download(ticker, period=period, interval=timeframe, progress=False, auto_adjust=False)
    if df_raw.empty:
        st.error(f"Data {ticker} tidak tersedia")
        st.stop()
    df = calculate_all_indicators(df_raw, st_period, st_mult, trading_mode)
    indicators = get_latest_indicators(df)
    if not indicators:
        st.error("Gagal menghitung indikator")
        st.stop()
    price = indicators["price"]
    regime = detect_market_regime(df)

    # MTF
    def safe_load(ticker, period, interval):
        try:
            d = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
            if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
            if d.empty or len(d)<30: return pd.DataFrame()
            return calculate_all_indicators(d, st_period, st_mult, trading_mode)
        except: return pd.DataFrame()
    df_daily_mtf = safe_load(ticker, "1y", "1d")
    df_weekly_mtf = safe_load(ticker, "2y", "1wk")
    df_monthly_mtf = safe_load(ticker, "3y", "1mo")
    mtf_alignment = get_mtf_alignment(df_daily_mtf, df_weekly_mtf, df_monthly_mtf)

    @st.cache_resource
    def load_model_for_ticker(_df, _use_xgb):
        return train_ml_model(_df, use_xgb=_use_xgb)
    ml_model = load_model_for_ticker(df, use_xgb)

    # Parameter threshold dinamis (bisa di-tuning)
    st.sidebar.subheader("🎚️ Parameter Entry")
    threshold_buy = st.sidebar.slider("Threshold BUY", 1.0, 5.0, 3.0, 0.5)
    threshold_strong = st.sidebar.slider("Threshold STRONG BUY", 4.0, 8.0, 6.0, 0.5)

    signal, score, ml_prob = final_decision_ultra(indicators, mtf_alignment, regime, weights, ml_model, threshold_buy, threshold_strong)
    show_toast_if_changed(signal, st.session_state.last_signal)
    st.session_state.last_signal = signal
    log_signal(ticker, signal, score, price)
    market_text, _ = get_market_context()
    breakout = detect_true_breakout(df, indicators["atr"])
    show_breakout_alert(breakout, st.session_state.last_breakout_notify_time)
    entry_levels = get_entry_levels_advanced(df, indicators, regime)
    entry_eligible = score >= threshold_buy
    entry_warning = (threshold_buy-1.5) <= score < threshold_buy
    bull_prob, _ = calculate_probabilities(df)

    st.title(f"🤖 {ticker} (v6 Pro Tuned)")
    st.caption(f"Mode: {trading_mode} | Regime: {regime} | MTF: {mtf_alignment:.2f} | ML Prob: {ml_prob*100:.1f}% | Model: {'XGBoost' if use_xgb else 'RandomForest'}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Harga", f"Rp {price:,.0f}")
    col2.metric("📊 RSI", f"{indicators['rsi']:.1f}")
    sup_dir = "🟢 UPTREND" if indicators["supertrend_dir"]==1 else "🔴 DOWNTREND"
    col3.metric("📈 Supertrend", sup_dir, delta=f"Rp {indicators['supertrend_value']:,.0f}")
    col4.metric("📊 Volume", indicators["volume_status"])
    col1, col2, col3 = st.columns(3)
    col1.metric("📈 High 30d", f"Rp {indicators['high_30d']:,.0f}")
    col2.metric("📉 Low 30d", f"Rp {indicators['low_30d']:,.0f}")
    col3.metric("📊 ATR", f"Rp {indicators['atr']:,.0f}")
    col1, col2, col3 = st.columns(3)
    if "STRONG BUY" in signal: col1.success(f"### 🟢 {signal}")
    elif "STRONG SELL" in signal: col1.error(f"### 🔴 {signal}")
    elif "BUY" in signal: col1.info(f"### 📈 {signal}")
    elif "SELL" in signal: col1.warning(f"### 📉 {signal}")
    else: col1.warning(f"### 🟡 {signal}")
    col2.metric("🎯 Skor", f"{score:.1f}")
    col3.metric("🤖 AI Confidence", f"{ml_prob*100:.1f}%")
    col1, col2, col3 = st.columns(3)
    cmf_val = indicators["cmf"]
    cmf_text = "🟢 Akumulasi" if cmf_val>0.15 else ("🔴 Distribusi" if cmf_val<-0.15 else "⚪ Netral")
    col1.metric("💰 Smart Money (CMF)", cmf_text, delta=f"{cmf_val:.2f}")
    col2.metric("📈 Swing Low", f"Rp {indicators['swing_low']:,.0f}")
    col3.metric("📉 Swing High", f"Rp {indicators['swing_high']:,.0f}")
    col1, col2, col3 = st.columns(3)
    col1.metric("🎯 Buy Entry", f"Rp {entry_levels['buy_entry']:,.0f}")
    col2.metric("🛑 Stop Loss", f"Rp {entry_levels['stop_loss']:,.0f}")
    col3.metric("🎯 Target", f"Rp {entry_levels['target']:,.0f}")
    col1, col2 = st.columns(2)
    col1.metric("🌍 IHSG", market_text[:20], help=market_text)
    col2.metric("📈 Bullish Prob.", f"{bull_prob:.0f}%")

    if ui_mode == "Advanced (Full Indicators)":
        st.markdown("---")
        st.caption("📊 Indikator Lanjutan (semua masih ada)")
        adv_cols = st.columns(5)
        adv_cols[0].metric("Stochastic K", f"{indicators['stoch_k']:.1f}")
        adv_cols[1].metric("Williams %R", f"{indicators['williams_r']:.1f}")
        adv_cols[2].metric("CCI", f"{indicators['cci']:.1f}")
        adv_cols[3].metric("MFI", f"{indicators['mfi']:.1f}")
        adv_cols[4].metric("KST", f"{indicators['kst']:.1f}")
        adv_cols2 = st.columns(4)
        adv_cols2[0].metric("Aroon Up", f"{indicators['aroon_up']:.0f}")
        adv_cols2[1].metric("Aroon Down", f"{indicators['aroon_down']:.0f}")
        adv_cols2[2].metric("Elder Bull", f"{indicators['elder_bull']:.0f}")
        adv_cols2[3].metric("GMMA Spread", f"{indicators['gmma_spread']:.0f}")

    if entry_eligible: st.success(f"🟢 **ENTRY LAYAK** (Skor: {score:.1f} | Regime: {regime} | ML: {ml_prob*100:.0f}%)")
    elif entry_warning: st.warning(f"🟡 **ENTRY HATI-HATI** (Skor: {score:.1f})")
    else: st.error(f"🔴 **TIDAK LAYAK ENTRY** (Skor: {score:.1f})")

    with st.expander("📖 Arti Sinyal"): st.markdown(get_signal_interpretation(signal))
    with st.expander("📋 Detail Analisis (Hybrid AI Ultra)"):
        fv = build_feature_vector(indicators)
        st.write(f"- Regime: {regime} (dynamic weights applied)")
        st.write(f"- Trend group: {fv['trend']:.2f}")
        st.write(f"- Momentum group: {fv['momentum']:.2f}")
        st.write(f"- Volume group: {fv['volume']:.2f}")
        st.write(f"- Smart money: {fv['smart']:.2f}")
        st.write(f"- Structure: {fv['structure']:.2f}")
        st.write(f"- GMMA: {fv['gmma']:.2f}")
        st.write(f"- MTF alignment: {mtf_alignment:.2f} (gate factor)")
        st.write(f"- ML probability: {ml_prob*100:.1f}%")
        st.write(f"- Final Score: {score:.2f}")
        st.write(f"- Threshold BUY: {threshold_buy}, STRONG: {threshold_strong}")

    # ====================== BACKTEST SECTION DENGAN METRICS LENGKAP ======================
    with st.expander("📊 Backtest & Auto Tuning"):
        col1, col2 = st.columns(2)
        with col1:
            risk_bt = st.number_input("Risk per trade (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.5) / 100.0
            if st.button("Jalankan Backtest"):
                with st.spinner("Menjalankan backtest..."):
                    bt = run_backtest_advanced(df, weights, detect_market_regime, 180, risk_bt, ml_model, threshold_buy, threshold_strong)
                    if bt:
                        st.write(f"Jumlah trade: {bt['trades']}")
                        st.write(f"Winrate: {bt['winrate']:.1f}%")
                        st.write(f"Profit Factor: {bt['profit_factor']:.2f}")
                        st.write(f"Max Drawdown: {bt['max_drawdown']:.1f}%")
                        st.write(f"Final Balance: Rp {bt['final_balance']:,.0f}")
                        st.write(f"Avg Win: Rp {bt['avg_win']:,.0f}")
                        st.write(f"Avg Loss: Rp {bt['avg_loss']:,.0f}")
                        st.write(f"Risk-Reward Ratio: {bt['rr_ratio']:.2f}")
                        # Insight otomatis
                        if bt["rr_ratio"] < 1:
                            st.warning("⚠️ Risk-Reward buruk (profit kecil, loss besar). Pertimbangkan memperlebar target.")
                        if bt["winrate"] < 50:
                            st.warning("⚠️ Winrate rendah, filter perlu diperketat (misal RSI>70 or CMF<0).")
                        if bt["profit_factor"] < 1.3:
                            st.error("❌ Sistem belum profitable. Coba tuning parameter.")
                        if bt["max_drawdown"] > 30:
                            st.error("❌ Drawdown terlalu besar, kurangi risk per trade.")
                        # Plot equity curve
                        fig, ax = plt.subplots(figsize=(10,4))
                        ax.plot(bt["equity"], color='green')
                        ax.set_title("Equity Curve (Backtest)")
                        ax.set_xlabel("Iteration")
                        ax.set_ylabel("Balance (Rp)")
                        st.pyplot(fig)
                    else:
                        st.warning("Data tidak cukup atau tidak ada sinyal")
        with col2:
            if st.button("Auto Tune Parameters (Risk & Threshold)"):
                with st.spinner("Mencari parameter terbaik..."):
                    best_params, best_score = tune_parameters(df, weights, ml_model, 180)
                    if best_params:
                        st.success(f"Parameter optimal: Risk = {best_params[0]*100:.1f}%, Threshold BUY = {best_params[1]:.1f}")
                        st.write(f"Score (PF - DD/100): {best_score:.2f}")
                        # Rekomendasi set parameter
                        st.info(f"Set risk ke {best_params[0]*100:.1f}% dan threshold BUY ke {best_params[1]:.1f} untuk hasil optimal.")
                    else:
                        st.warning("Tuning gagal, coba dengan data yang lebih panjang.")

    # Tombol-tombol lain (optimize weights, walk-forward, multi-stock, riwayat) masih sama seperti sebelumnya
    with st.expander("🔧 Advanced Tools"):
        if st.button("Optimize Weights (20 trials)"):
            with st.spinner("Mengoptimalkan bobot..."):
                best_w = optimize_weights(df, trials=20)
                if best_w:
                    st.session_state.optimized_weights = best_w
                    st.success("Bobot optimal ditemukan! Silakan refresh.")
                else:
                    st.warning("Optimasi gagal")
        if st.session_state.optimized_weights:
            st.write("Bobot teroptimasi:", st.session_state.optimized_weights)
            if st.button("Gunakan Bobot Optimasi"):
                st.session_state.custom_weights = st.session_state.optimized_weights
                st.rerun()
        if st.button("Jalankan Walk-Forward Validation"):
            with st.spinner("Walk-forward validation..."):
                acc = walk_forward_validation(df, train_size=200, test_size=50, use_xgb=use_xgb)
                st.session_state.walk_forward_result = acc
                st.success(f"Walk-forward accuracy: {acc*100:.2f}%")
        if st.session_state.walk_forward_result is not None:
            st.metric("Walk-Forward Accuracy", f"{st.session_state.walk_forward_result*100:.2f}%")
        st.markdown("---")
        st.subheader("🌍 Multi-Stock Learning")
        default_tickers = ["BBCA.JK", "BBRI.JK", "TLKM.JK", "ASII.JK", "UNVR.JK"]
        tickers_input = st.text_input("Daftar kode saham (pisah koma)", value=",".join(default_tickers))
        if st.button("Train Global Model"):
            ticker_list = [t.strip() for t in tickers_input.split(",") if t.strip()]
            with st.spinner("Training global model..."):
                global_m = train_global_model(ticker_list, period="1y", use_xgb=use_xgb)
                if global_m:
                    st.session_state.global_model = global_m
                    st.success("Global model siap!")
                else:
                    st.error("Gagal training global model.")
        if st.session_state.global_model is not None:
            if st.button("Gunakan Global Model"):
                st.session_state.ml_model_override = st.session_state.global_model
                st.rerun()

    with st.expander("📜 Riwayat Sinyal & Ekspor"):
        if st.session_state.signal_history:
            st.dataframe(pd.DataFrame(st.session_state.signal_history).head(5))
            csv = pd.DataFrame(st.session_state.signal_history).to_csv(index=False).encode()
            st.download_button("📥 Ekspor CSV", csv, "signal_history.csv", "text/csv")
        else:
            st.info("Belum ada sinyal")

    with st.expander("📐 Position Sizing"):
        capital = st.number_input("Modal (Rp)", value=100_000_000.0, step=10_000_000.0)
        risk_percent = st.number_input("Risiko per trade (%)", value=2.0, step=0.5)
        shares, pos_value, risk_amt = position_sizing(capital, risk_percent, entry_levels["buy_entry"], entry_levels["stop_loss"])
        if shares>0:
            st.write(f"Jumlah saham: {shares:,} lembar")
            st.write(f"Nilai posisi: Rp {pos_value:,.0f} ({pos_value/capital*100:.1f}% dari modal)")
            st.write(f"Risiko stop loss: Rp {risk_amt:,.0f} ({risk_percent:.1f}%)")

    st.caption("⚠️ Edukasi saja, bukan rekomendasi investasi. Backtest menggunakan risk-based, fee 0.15%, slippage 0.1%. Hybrid AI = XGBoost + Rule Engine + Walk-Forward + Auto Tuning.")
    gc.collect()

if __name__ == "__main__":
    main()
