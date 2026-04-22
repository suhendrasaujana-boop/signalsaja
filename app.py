import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import numpy as np
from datetime import datetime, date, timedelta
import os
import gc

# ========== KONSTANTA ==========
DATA_PERIOD = "3mo"  # Hanya 3 bulan data (hemat memory)
CACHE_TTL = 300
DEFAULT_TICKER = "^JKSE"

TIMEFRAMES = {"1d": "1d", "1wk": "1wk", "1mo": "1mo"}
VOLUME_SPIKE_THRESHOLD = 1.8
BREAKOUT_COOLDOWN_HOURS = 24

st.set_page_config(
    layout="wide", 
    page_title="AI Signal Lite", 
    page_icon="🤖",
    menu_items={'Get Help': None, 'Report a bug': None, 'About': None}
)

# ========== SESSION STATE ==========
if 'last_resistance' not in st.session_state:
    st.session_state.last_resistance = None
if 'last_breakout_notify_time' not in st.session_state:
    st.session_state.last_breakout_notify_time = None
if 'user_volume_spike_threshold' not in st.session_state:
    st.session_state.user_volume_spike_threshold = VOLUME_SPIKE_THRESHOLD
if 'user_breakout_cooldown_hours' not in st.session_state:
    st.session_state.user_breakout_cooldown_hours = BREAKOUT_COOLDOWN_HOURS

# ========== FUNGSI BANTU ==========
def fix_ticker(ticker: str) -> str:
    ticker = ticker.strip().upper()
    if ticker.startswith('^') or ticker.endswith('.JK'):
        return ticker
    return ticker + '.JK'

def show_notification(message: str, icon: str = "ℹ️"):
    if hasattr(st, 'toast'):
        st.toast(message, icon=icon)
    else:
        st.info(message)

@st.cache_data(ttl=CACHE_TTL)
def load_data(ticker: str, timeframe: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=DATA_PERIOD, interval=timeframe, progress=False, auto_adjust=False)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(how='all')
        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].fillna(0)
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return pd.DataFrame()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    close = df['Close']
    
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        volume = df['Volume']
    else:
        volume = pd.Series(0, index=df.index)
        df['Volume'] = 0

    df['SMA20'] = close.rolling(20, min_periods=1).mean()
    df['SMA50'] = close.rolling(50, min_periods=1).mean()

    if len(df) >= 14:
        df['RSI'] = ta.momentum.rsi(close, window=14)
    else:
        df['RSI'] = 50.0

    if len(df) >= 26:
        macd = ta.trend.MACD(close)
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
    else:
        df['MACD'] = 0.0
        df['MACD_signal'] = 0.0

    if volume.sum() > 0:
        df['Volume_MA'] = volume.rolling(20, min_periods=1).mean()
    else:
        df['Volume_MA'] = 0.0

    df['support'] = df['Low'].rolling(20, min_periods=1).min()
    df['resistance'] = df['High'].rolling(20, min_periods=1).max()

    df = df.ffill().fillna(0)
    return df

def calculate_ai_score(df: pd.DataFrame, volume: pd.Series) -> int:
    if df.empty:
        return 0
    conditions = [
        df['RSI'].iloc[-1] < 35,
        df['Close'].iloc[-1] > df['SMA20'].iloc[-1],
        df['SMA20'].iloc[-1] > df['SMA50'].iloc[-1],
        df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1],
        volume.iloc[-1] > df['Volume_MA'].iloc[-1] if volume.sum() > 0 else False
    ]
    return sum(conditions)

def get_multi_timeframe_trend(ticker):
    daily = load_data(ticker, "1d")
    weekly = load_data(ticker, "1wk")
    monthly = load_data(ticker, "1mo")

    def trend(df):
        if df.empty or len(df) < 20:
            return "Neutral"
        close = df["Close"]
        sma20 = close.rolling(20).mean()
        if close.iloc[-1] > sma20.iloc[-1]:
            return "Bullish"
        elif close.iloc[-1] < sma20.iloc[-1]:
            return "Bearish"
        return "Neutral"

    return {
        "Daily": trend(daily),
        "Weekly": trend(weekly),
        "Monthly": trend(monthly)
    }

def calculate_smart_money(df):
    if df.empty or len(df) < 20:
        return 0, "Neutral"

    score = 0
    if df['CMF'].iloc[-1] > 0:
        score += 1
    if df['AD'].iloc[-1] > df['AD'].iloc[-5]:
        score += 1
    if df['Volume'].iloc[-1] > df['Volume_MA'].iloc[-1] * 1.5 and df['Volume'].iloc[-1] > 0:
        score += 1
    if df['Close'].iloc[-1] > df['SMA20'].iloc[-1]:
        score += 1

    if score >= 3:
        status = "Accumulation"
    elif score == 2:
        status = "Neutral"
    else:
        status = "Distribution"

    return score, status

def get_macro_signal():
    try:
        ihsg = yf.download("^JKSE", period="5d", progress=False)['Close']
        usd = yf.download("USDIDR=X", period="5d", progress=False)['Close']
        nasdaq = yf.download("^IXIC", period="5d", progress=False)['Close']

        score = 0
        if ihsg.iloc[-1] > ihsg.mean():
            score += 1
        if usd.iloc[-1] < usd.mean():
            score += 1
        if nasdaq.iloc[-1] > nasdaq.mean():
            score += 1

        if score >= 2:
            return "Risk ON", score
        elif score == 1:
            return "Neutral", score
        else:
            return "Risk OFF", score
    except:
        return "Neutral", 1

def get_all_signals(df, volume, ticker):
    ai_score = calculate_ai_score(df, volume)
    smart_score, smart_status = calculate_smart_money(df)
    macro_status, macro_score = get_macro_signal()
    
    # Hitung confidence
    confidence = ai_score * 10
    if smart_status == "Accumulation":
        confidence += 15
    elif smart_status == "Distribution":
        confidence -= 15
    if macro_status == "Risk ON":
        confidence += 10
    elif macro_status == "Risk OFF":
        confidence -= 10
    confidence = max(0, min(100, confidence))

    return {
        'ai_score': ai_score,
        'smart_score': smart_score,
        'smart_status': smart_status,
        'macro_status': macro_status,
        'macro_score': macro_score,
        'confidence': confidence
    }

def weighted_decision_engine(df, volume, ticker):
    signals = get_all_signals(df, volume, ticker)
    tech_score = signals['ai_score'] / 5

    smart_map = {"Accumulation": 1, "Neutral": 0.5, "Distribution": 0}
    smart_score = smart_map.get(signals['smart_status'], 0.5)

    macro_map = {"Risk ON": 1, "Neutral": 0.5, "Risk OFF": 0}
    macro_score = macro_map.get(signals['macro_status'], 0.5)

    returns = df['Close'].pct_change().dropna()
    vol = returns.std() * np.sqrt(252)
    if vol < 0.15:
        risk_score = 1
    elif vol < 0.30:
        risk_score = 0.6
    else:
        risk_score = 0.2

    momentum = df['Close'].pct_change(5).iloc[-1]
    if momentum > 0.05:
        momentum_score = 1
    elif momentum > 0:
        momentum_score = 0.6
    else:
        momentum_score = 0.2

    final_score = (0.35 * tech_score + 0.20 * smart_score + 0.15 * macro_score + 
                   0.15 * risk_score + 0.15 * momentum_score)

    return round(final_score * 100, 2)

def get_price_range(df: pd.DataFrame, days: int) -> dict:
    if df.empty or len(df) < days:
        return {'high': None, 'low': None, 'current': None}
    recent_data = df.tail(days)
    return {
        'high': recent_data['High'].max(),
        'low': recent_data['Low'].min(),
        'current': df['Close'].iloc[-1]
    }

def get_recommended_entry_stoploss(df: pd.DataFrame) -> dict:
    if df.empty:
        return {'entry': None, 'stoploss': None, 'target': None}
    price = df['Close'].iloc[-1]
    support = df['support'].iloc[-1] if pd.notna(df['support'].iloc[-1]) else price * 0.95
    resistance = df['resistance'].iloc[-1] if pd.notna(df['resistance'].iloc[-1]) else price * 1.05
    entry = (support + price) / 2
    stoploss = support * 0.97
    target = resistance
    return {'entry': entry, 'stoploss': stoploss, 'target': target}

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("# 🤖 AI Signal Lite")
    ticker_input = st.text_input("Ticker", DEFAULT_TICKER, help="Contoh: BBCA, BBRI, ASII")
    ticker = fix_ticker(ticker_input)
    timeframe = st.selectbox("Timeframe", list(TIMEFRAMES.keys()))
    
    st.markdown("### ⚙️ Pengaturan")
    user_volume_spike = st.slider("Volume Spike", 1.0, 5.0, st.session_state.user_volume_spike_threshold, 0.1)
    st.session_state.user_volume_spike_threshold = user_volume_spike
    
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    st.caption("⚠️ Hanya untuk edukasi")
    st.caption("Bukan rekomendasi beli/jual")

# ========== LOAD DATA ==========
with st.spinner("Memuat data..."):
    data = load_data(ticker, TIMEFRAMES[timeframe])

if data.empty or len(data) < 10:
    st.warning(f"Data tidak cukup untuk {ticker}")
    st.stop()

data = add_indicators(data)

# Ambil data terkini
last_close = data['Close'].iloc[-1]
prev_close = data['Close'].iloc[-2] if len(data) > 1 else last_close
change_pct = ((last_close - prev_close) / prev_close) * 100 if prev_close != 0 else 0

volume = data['Volume'] if 'Volume' in data.columns else pd.Series(0, index=data.index)

# ========== HEADER ==========
st.title(f"🤖 {ticker}")

col1, col2, col3 = st.columns(3)
col1.metric("Harga", f"{last_close:,.2f}", f"{change_pct:+.2f}%")
col2.metric("Tertinggi", f"{data['High'].iloc[-1]:,.2f}")
col3.metric("Terendah", f"{data['Low'].iloc[-1]:,.2f}")

# ========== RANGE HARGA ==========
st.markdown("### 📊 Range Harga (High/Low)")
periods = [1, 5, 7, 14, 30]
labels = ["Hari Ini", "5 Hari", "1 Minggu", "2 Minggu", "30 Hari"]
cols = st.columns(len(periods))

for idx, period in enumerate(periods):
    range_data = get_price_range(data, period)
    with cols[idx]:
        if range_data['high']:
            st.metric(labels[idx], f"{range_data['current']:,.0f}", 
                     delta=f"H:{range_data['high']:,.0f} L:{range_data['low']:,.0f}")
        else:
            st.metric(labels[idx], "N/A")

# ========== REKOMENDASI ENTRY & STOPLOSS ==========
entry_rec = get_recommended_entry_stoploss(data)
st.markdown("### 🎯 Rekomendasi Entry & Stop Loss")
col_e, col_s, col_t = st.columns(3)
col_e.metric("Entry", f"{entry_rec['entry']:,.2f}" if entry_rec['entry'] else "N/A")
col_s.metric("Stop Loss", f"{entry_rec['stoploss']:,.2f}" if entry_rec['stoploss'] else "N/A", delta_color="inverse")
col_t.metric("Target", f"{entry_rec['target']:,.2f}" if entry_rec['target'] else "N/A")

if entry_rec['entry'] and entry_rec['stoploss'] and entry_rec['target']:
    risk = entry_rec['entry'] - entry_rec['stoploss']
    reward = entry_rec['target'] - entry_rec['entry']
    rr = reward / risk if risk > 0 else 0
    st.info(f"📐 Risk Reward: 1 : {rr:.2f}")

st.markdown("---")

# ========== AI SIGNAL (UTAMA) ==========
st.header("🎯 AI SIGNAL - KEPUTUSAN BELI/JUAL")

signals = get_all_signals(data, volume, ticker)
score = signals['ai_score']

# TAMPILAN UTAMA - REKOMENDASI
col_main1, col_main2 = st.columns([2, 1])

with col_main1:
    if score >= 4:
        st.success("## 🚀 STRONG BUY")
        st.progress(100, text="Confidence: SANGAT TINGGI")
    elif score == 3:
        st.info("## 🟡 HOLD / WAIT")
        st.progress(60, text="Confidence: SEDANG")
    else:
        st.error("## 🔻 SELL / AVOID")
        st.progress(20, text="Confidence: RENDAH")

with col_main2:
    st.metric("AI SCORE", f"{score}/5")
    st.metric("Final Confidence", f"{signals['confidence']}%")

st.markdown("---")

# ========== DETAIL SIGNAL ==========
st.subheader("📊 Detail Analisis")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🔥 Momentum**")
    momentum = data['Close'].pct_change(5).iloc[-1] * 100 if len(data) >= 6 else 0
    if momentum > 3:
        st.success(f"Up {momentum:.1f}%")
    elif momentum < -3:
        st.error(f"Down {momentum:.1f}%")
    else:
        st.warning("Sideways")
    
    st.markdown("**🚀 Breakout**")
    if len(data) > 1 and data['Close'].iloc[-1] > data['resistance'].iloc[-2]:
        st.success("BREAKOUT!")
    else:
        st.info("Tidak ada breakout")

with col2:
    st.markdown("**📈 Trend**")
    mtf = get_multi_timeframe_trend(ticker)
    st.write(f"Daily: {mtf['Daily']}")
    st.write(f"Weekly: {mtf['Weekly']}")
    st.write(f"Monthly: {mtf['Monthly']}")

with col3:
    st.markdown("**🏦 Smart Money**")
    st.write(f"Status: {signals['smart_status']}")
    st.write(f"Score: {signals['smart_score']}/4")
    
    st.markdown("**🌍 Macro**")
    st.write(f"Kondisi: {signals['macro_status']}")

st.markdown("---")

# ========== PROBABILITAS ==========
st.subheader("📈 Probabilitas Pergerakan")

bull = bear = 0
if data['RSI'].iloc[-1] < 35:
    bull += 1
else:
    bear += 1
if data['Close'].iloc[-1] > data['SMA20'].iloc[-1]:
    bull += 1
else:
    bear += 1
if data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1]:
    bull += 1
else:
    bear += 1

total = bull + bear
bull_prob = (bull/total)*100 if total > 0 else 50
bear_prob = (bear/total)*100 if total > 0 else 50

col_p1, col_p2 = st.columns(2)
col_p1.metric("📈 BULLISH", f"{bull_prob:.0f}%")
col_p2.metric("📉 BEARISH", f"{bear_prob:.0f}%")

st.progress(bull_prob/100)

# ========== WEIGHTED DECISION ==========
st.markdown("---")
st.subheader("🧠 Weighted Decision Engine")

weighted_score = weighted_decision_engine(data, volume, ticker)

col_w1, col_w2, col_w3 = st.columns(3)
col_w2.metric("Final Score", f"{weighted_score}%", 
             delta="BUY" if weighted_score >= 60 else "SELL" if weighted_score < 45 else "HOLD")

if weighted_score >= 70:
    st.success("### ✅ REKOMENDASI: BELI")
    st.progress(weighted_score/100)
elif weighted_score >= 50:
    st.warning("### ⚠️ REKOMENDASI: HOLD / TUNGGU")
    st.progress(weighted_score/100)
else:
    st.error("### ❌ REKOMENDASI: HINDARI / JUAL")
    st.progress(weighted_score/100)

# ========== POSITION SIZING ==========
st.markdown("---")
st.subheader("📐 Position Sizing")

col_cap, col_risk = st.columns(2)
with col_cap:
    capital = st.number_input("Modal (Rp)", min_value=0.0, value=100_000_000.0, step=10_000_000.0, key="capital")
with col_risk:
    risk_percent = st.number_input("Risiko per Trade (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.5, key="risk")

if entry_rec['stoploss'] and capital > 0 and risk_percent > 0:
    risk_amount = capital * (risk_percent / 100)
    price_risk = entry_rec['entry'] - entry_rec['stoploss']
    if price_risk > 0:
        suggested_shares = int(risk_amount / price_risk)
        position_value = suggested_shares * entry_rec['entry']
        st.write(f"**Jumlah saham:** {suggested_shares:,} lembar")
        st.write(f"**Nilai posisi:** Rp {position_value:,.2f} ({position_value/capital*100:.1f}% dari modal)")
    else:
        st.warning("Stop loss terlalu dekat")
else:
    st.info("Masukkan modal dan risiko")

# ========== INDIKATOR TEKNIKAL ==========
with st.expander("📊 Indikator Teknikal Lengkap"):
    col_i1, col_i2 = st.columns(2)
    with col_i1:
        st.metric("RSI (14)", f"{data['RSI'].iloc[-1]:.1f}")
        st.metric("SMA20", f"{data['SMA20'].iloc[-1]:.2f}")
        st.metric("Support", f"{data['support'].iloc[-1]:.2f}")
    with col_i2:
        st.metric("MACD", f"{data['MACD'].iloc[-1]:.4f}")
        st.metric("SMA50", f"{data['SMA50'].iloc[-1]:.2f}")
        st.metric("Resistance", f"{data['resistance'].iloc[-1]:.2f}")

# ========== FOOTER ==========
st.markdown("---")
st.caption("⚠️ **DISCLAIMER:** Hanya untuk edukasi. Bukan rekomendasi beli/jual.")

gc.collect()
