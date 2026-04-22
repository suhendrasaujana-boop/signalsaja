import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from datetime import datetime, date, timedelta
import os
import gc
import warningsimport streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from datetime import datetime, date, timedelta
import os
import gc
import warnings
warnings.filterwarnings('ignore')

# ========== KONSTANTA ==========
DATA_PERIOD = "3mo"
CACHE_TTL = 300
DEFAULT_TICKER = "BBCA.JK"

TIMEFRAMES = {"1d": "1d", "1wk": "1wk", "1mo": "1mo"}
VOLUME_SPIKE_THRESHOLD = 1.8
BREAKOUT_COOLDOWN_HOURS = 24

st.set_page_config(
    layout="wide", 
    page_title="Robot Saham - AI Signal Lite", 
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

# ========== FUNGSI BANTU ==========
def fix_ticker(ticker: str) -> str:
    ticker = ticker.strip().upper()
    if ticker.startswith('^'):
        return ticker
    if ticker.endswith('.JK'):
        return ticker
    return ticker + '.JK'

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
        
        if len(df) >= 20:
            df['SMA20'] = df['Close'].rolling(20).mean()
        if len(df) >= 50:
            df['SMA50'] = df['Close'].rolling(50).mean()
        if len(df) >= 14:
            df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        
        return df
    except Exception as e:
        st.error(f"Error: {str(e)[:100]}")
        return pd.DataFrame()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    
    try:
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

        try:
            if volume.sum() > 0:
                df['AD'] = ta.volume.acc_dist_index(df['High'], df['Low'], df['Close'], df['Volume'], fillna=True)
            else:
                df['AD'] = 0.0
        except:
            df['AD'] = 0.0
            
        try:
            if volume.sum() > 0:
                df['CMF'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'], window=20, fillna=True)
            else:
                df['CMF'] = 0.0
        except:
            df['CMF'] = 0.0

        df = df.ffill().fillna(0)
        return df
        
    except Exception as e:
        return df

# ========== FUNGSI UNTUK ROBOT SAHAM ==========
def calculate_robot_indicators(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 30:
        return {}
    
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume'] if 'Volume' in df.columns else pd.Series(0, index=df.index)
    
    rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns and pd.notna(df['RSI'].iloc[-1]) else 50
    
    if len(close) >= 26:
        macd = ta.trend.MACD(close)
        macd_line = macd.macd().iloc[-1]
        macd_signal = macd.macd_signal().iloc[-1]
        macd_hist = macd_line - macd_signal
    else:
        macd_line = 0
        macd_signal = 0
        macd_hist = 0
    
    if len(close) >= 14:
        stoch_k = ta.momentum.stoch(high, low, close, window=14, smooth_window=3).iloc[-1]
        stoch_d = ta.momentum.stoch_signal(high, low, close, window=14, smooth_window=3).iloc[-1]
    else:
        stoch_k = 50
        stoch_d = 50
    
    volume_avg = volume.rolling(20).mean().iloc[-1] if volume.sum() > 0 else 1
    volume_current = volume.iloc[-1] if volume.sum() > 0 else 0
    if volume_current < volume_avg:
        volume_status = "Rendah"
    elif volume_current > volume_avg * 1.5:
        volume_status = "Tinggi"
    else:
        volume_status = "Normal"
    
    sma20 = df['SMA20'].iloc[-1] if 'SMA20' in df.columns and pd.notna(df['SMA20'].iloc[-1]) else close.iloc[-1]
    sma50 = df['SMA50'].iloc[-1] if 'SMA50' in df.columns and pd.notna(df['SMA50'].iloc[-1]) else close.iloc[-1]
    
    support = df['Low'].rolling(20).min().iloc[-1]
    resistance = df['High'].rolling(20).max().iloc[-1]
    
    high_30d = df['High'].tail(30).max()
    low_30d = df['Low'].tail(30).min()
    mid_30d = (high_30d + low_30d) / 2
    
    return {
        'rsi': round(rsi, 1),
        'macd': round(macd_line, 2),
        'macd_signal': round(macd_signal, 2),
        'macd_hist': round(macd_hist, 2),
        'stoch_k': round(stoch_k, 1),
        'stoch_d': round(stoch_d, 1),
        'volume_current': volume_current,
        'volume_avg': volume_avg,
        'volume_status': volume_status,
        'sma20': round(sma20, 2),
        'sma50': round(sma50, 2),
        'support': round(support, 2),
        'resistance': round(resistance, 2),
        'high_30d': round(high_30d, 2),
        'low_30d': round(low_30d, 2),
        'mid_30d': round(mid_30d, 2),
        'current_price': round(close.iloc[-1], 2)
    }

def calculate_multi_timeframe_score(df: pd.DataFrame) -> dict:
    if df.empty:
        return {'3m': 0, '12m': 0, '30m': 0}
    
    df_3m = df.tail(60)
    score_3m = 0
    if len(df_3m) > 0 and df_3m['Close'].iloc[-1] > df_3m['Close'].iloc[0]:
        score_3m += 1
    if 'SMA20' in df_3m.columns and len(df_3m) >= 20:
        if pd.notna(df_3m['SMA20'].iloc[-1]) and df_3m['Close'].iloc[-1] > df_3m['SMA20'].iloc[-1]:
            score_3m += 1
    if 'RSI' in df_3m.columns:
        if pd.notna(df_3m['RSI'].iloc[-1]) and df_3m['RSI'].iloc[-1] > 50:
            score_3m += 1
    score_3m = int(score_3m * 100 / 3) if score_3m > 0 else 0
    
    df_12m = df.tail(250) if len(df) >= 250 else df
    score_12m = 0
    if len(df_12m) > 0 and df_12m['Close'].iloc[-1] > df_12m['Close'].iloc[0]:
        score_12m += 1
    if 'SMA50' in df_12m.columns and len(df_12m) >= 50:
        if pd.notna(df_12m['SMA50'].iloc[-1]) and df_12m['Close'].iloc[-1] > df_12m['SMA50'].iloc[-1]:
            score_12m += 1
    if 'RSI' in df_12m.columns:
        if pd.notna(df_12m['RSI'].iloc[-1]) and df_12m['RSI'].iloc[-1] > 50:
            score_12m += 1
    score_12m = int(score_12m * 100 / 3) if score_12m > 0 else 0
    
    df_30m = df.tail(630) if len(df) >= 630 else df
    score_30m = 0
    if len(df_30m) > 0 and df_30m['Close'].iloc[-1] > df_30m['Close'].iloc[0]:
        score_30m += 1
    if 'SMA50' in df_30m.columns and len(df_30m) >= 50:
        if pd.notna(df_30m['SMA50'].iloc[-1]) and df_30m['Close'].iloc[-1] > df_30m['SMA50'].iloc[-1]:
            score_30m += 1
    if 'RSI' in df_30m.columns:
        if pd.notna(df_30m['RSI'].iloc[-1]) and df_30m['RSI'].iloc[-1] > 50:
            score_30m += 1
    score_30m = int(score_30m * 100 / 3) if score_30m > 0 else 0
    
    return {'3m': score_3m, '12m': score_12m, '30m': score_30m}

def get_robot_signal(indicators: dict, mtf_score: dict) -> dict:
    score = 0
    reasons = []
    
    rsi_val = indicators.get('rsi', 50)
    if rsi_val < 35:
        score += 2
        reasons.append(f"RSI Oversold ({rsi_val}) -> Signal BELI")
    elif rsi_val > 70:
        score -= 2
        reasons.append(f"RSI Overbought ({rsi_val}) -> Signal JUAL")
    else:
        reasons.append(f"RSI Normal ({rsi_val})")
    
    macd_hist = indicators.get('macd_hist', 0)
    if macd_hist > 0:
        score += 1
        reasons.append(f"MACD Positif ({macd_hist:.2f}) -> Bullish")
    else:
        score -= 1
        reasons.append(f"MACD Negatif ({macd_hist:.2f}) -> Bearish")
    
    stoch_k = indicators.get('stoch_k', 50)
    if stoch_k < 20:
        score += 2
        reasons.append(f"Stochastic Oversold ({stoch_k}) -> Signal BELI")
    elif stoch_k > 80:
        score -= 2
        reasons.append(f"Stochastic Overbought ({stoch_k}) -> Signal JUAL")
    else:
        reasons.append(f"Stochastic Normal ({stoch_k})")
    
    mtf_avg = (mtf_score.get('3m', 0) + mtf_score.get('12m', 0) + mtf_score.get('30m', 0)) / 3
    if mtf_avg > 60:
        score += 2
        reasons.append(f"Multi Timeframe Bullish (avg {mtf_avg:.0f})")
    elif mtf_avg < 40:
        score -= 2
        reasons.append(f"Multi Timeframe Bearish (avg {mtf_avg:.0f})")
    else:
        reasons.append(f"Multi Timeframe Netral (avg {mtf_avg:.0f})")
    
    current_price = indicators.get('current_price', 0)
    sma20 = indicators.get('sma20', 0)
    if current_price > sma20:
        score += 1
        reasons.append(f"Harga > SMA20 -> Uptrend")
    else:
        score -= 1
        reasons.append(f"Harga < SMA20 -> Downtrend")
    
    volume_status = indicators.get('volume_status', 'Normal')
    if volume_status == "Tinggi" and score > 0:
        score += 1
        reasons.append("Volume Tinggi mendukung pergerakan")
    elif volume_status == "Rendah":
        reasons.append("Volume Rendah (potensi sideways)")
    
    if score >= 4:
        signal = "STRONG BUY"
        color = "success"
        confidence = "HIGH CONFIDENCE"
        confidence_score = 85
    elif score >= 2:
        signal = "BUY"
        color = "info"
        confidence = "MEDIUM CONFIDENCE"
        confidence_score = 65
    elif score <= -4:
        signal = "STRONG SELL"
        color = "error"
        confidence = "HIGH CONFIDENCE"
        confidence_score = 85
    elif score <= -2:
        signal = "SELL"
        color = "warning"
        confidence = "LOW CONFIDENCE"
        confidence_score = 35
    else:
        signal = "WAIT / HOLD"
        color = "warning"
        confidence = "LOW CONFIDENCE"
        confidence_score = 50
    
    return {
        'signal': signal,
        'color': color,
        'score': score,
        'confidence': confidence,
        'confidence_score': confidence_score,
        'reasons': reasons
    }

def get_trend_strength(df: pd.DataFrame) -> str:
    if df.empty or len(df) < 50:
        return "WEAK"
    
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    try:
        adx = ta.trend.ADXIndicator(high, low, close, window=14)
        adx_value = adx.adx().iloc[-1] if len(close) >= 14 else 25
        
        if adx_value > 40:
            return "STRONG TREND"
        elif adx_value > 25:
            return "MODERATE TREND"
        elif adx_value > 20:
            return "WEAK TREND"
        else:
            return "SIDEWAYS"
    except:
        return "NORMAL"

# ========== FUNGSI YANG SUDAH ADA ==========
def calculate_ai_score(df: pd.DataFrame, volume: pd.Series) -> int:
    if df.empty:
        return 0
    try:
        conditions = [
            df['RSI'].iloc[-1] < 35,
            df['Close'].iloc[-1] > df['SMA20'].iloc[-1],
            df['SMA20'].iloc[-1] > df['SMA50'].iloc[-1],
            df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1],
            volume.iloc[-1] > df['Volume_MA'].iloc[-1] if volume.sum() > 0 else False
        ]
        return sum(conditions)
    except:
        return 2

def get_multi_timeframe_trend(ticker):
    try:
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
    except:
        return {"Daily": "Neutral", "Weekly": "Neutral", "Monthly": "Neutral"}

def calculate_smart_money(df):
    if df.empty or len(df) < 20:
        return 0, "Neutral"
    try:
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
    except:
        return 2, "Neutral"

def get_macro_signal():
    try:
        ihsg = yf.download("^JKSE", period="5d", progress=False)['Close']
        usd = yf.download("USDIDR=X", period="5d", progress=False)['Close']
        nasdaq = yf.download("^IXIC", period="5d", progress=False)['Close']
        score = 0
        if len(ihsg) > 0 and ihsg.iloc[-1] > ihsg.mean():
            score += 1
        if len(usd) > 0 and usd.iloc[-1] < usd.mean():
            score += 1
        if len(nasdaq) > 0 and nasdaq.iloc[-1] > nasdaq.mean():
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
    try:
        ai_score = calculate_ai_score(df, volume)
        smart_score, smart_status = calculate_smart_money(df)
        macro_status, macro_score = get_macro_signal()
        
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
    except:
        return {'ai_score': 2, 'smart_score': 2, 'smart_status': 'Neutral', 'macro_status': 'Neutral', 'macro_score': 1, 'confidence': 50}

def weighted_decision_engine(df, volume, ticker):
    try:
        signals = get_all_signals(df, volume, ticker)
        tech_score = signals['ai_score'] / 5
        smart_map = {"Accumulation": 1, "Neutral": 0.5, "Distribution": 0}
        smart_score = smart_map.get(signals['smart_status'], 0.5)
        macro_map = {"Risk ON": 1, "Neutral": 0.5, "Risk OFF": 0}
        macro_score = macro_map.get(signals['macro_status'], 0.5)
        returns = df['Close'].pct_change().dropna()
        if len(returns) > 0:
            vol = returns.std() * np.sqrt(252)
            if vol < 0.15:
                risk_score = 1
            elif vol < 0.30:
                risk_score = 0.6
            else:
                risk_score = 0.2
        else:
            risk_score = 0.5
        momentum = df['Close'].pct_change(5).iloc[-1] if len(df) >= 6 else 0
        if momentum > 0.05:
            momentum_score = 1
        elif momentum > 0:
            momentum_score = 0.6
        else:
            momentum_score = 0.2
        final_score = (0.35 * tech_score + 0.20 * smart_score + 0.15 * macro_score + 0.15 * risk_score + 0.15 * momentum_score)
        return round(final_score * 100, 2)
    except:
        return 50

def get_price_range(df: pd.DataFrame, days: int) -> dict:
    if df.empty or len(df) < days:
        return {'high': None, 'low': None, 'current': None}
    try:
        recent_data = df.tail(days)
        return {'high': recent_data['High'].max(), 'low': recent_data['Low'].min(), 'current': df['Close'].iloc[-1]}
    except:
        return {'high': None, 'low': None, 'current': None}

def get_recommended_entry_stoploss(df: pd.DataFrame) -> dict:
    if df.empty:
        return {'entry': None, 'stoploss': None, 'target': None}
    try:
        price = df['Close'].iloc[-1]
        support = df['support'].iloc[-1] if pd.notna(df['support'].iloc[-1]) else price * 0.95
        resistance = df['resistance'].iloc[-1] if pd.notna(df['resistance'].iloc[-1]) else price * 1.05
        entry = (support + price) / 2
        stoploss = support * 0.97
        target = resistance
        return {'entry': entry, 'stoploss': stoploss, 'target': target}
    except:
        return {'entry': None, 'stoploss': None, 'target': None}

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("# 🤖 Robot Saham - AI Signal")
    ticker_input = st.text_input("Kode Saham", DEFAULT_TICKER, help="Contoh: BBCA.JK, BBRI.JK, ASII.JK")
    ticker = fix_ticker(ticker_input)
    if ticker != ticker_input:
        st.info(f"Format: {ticker}")
    timeframe = st.selectbox("Timeframe", list(TIMEFRAMES.keys()))
    
    st.markdown("### ⚙️ Pengaturan")
    user_volume_spike = st.slider("Volume Spike Threshold", 1.0, 5.0, st.session_state.user_volume_spike_threshold, 0.1)
    st.session_state.user_volume_spike_threshold = user_volume_spike
    
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    st.caption("⚠️ Hanya untuk edukasi | Bukan rekomendasi beli/jual")

# ========== LOAD DATA ==========
with st.spinner(f"Memuat data {ticker}..."):
    data = load_data(ticker, TIMEFRAMES[timeframe])

if data.empty or len(data) < 10:
    st.error(f"""
    ❌ **Tidak bisa memuat data untuk {ticker}**
    
    **Coba ticker berikut:**
    - BBCA.JK (Bank Central Asia)
    - BBRI.JK (Bank Rakyat Indonesia)
    - ASII.JK (Astra International)
    - TLKM.JK (Telkom)
    - ^JKSE (IHSG)
    """)
    st.stop()

data = add_indicators(data)

if data.empty:
    st.error("Gagal memproses data")
    st.stop()

try:
    last_close = data['Close'].iloc[-1]
    prev_close = data['Close'].iloc[-2] if len(data) > 1 else last_close
    change_pct = ((last_close - prev_close) / prev_close) * 100 if prev_close != 0 else 0
    volume = data['Volume'] if 'Volume' in data.columns else pd.Series(0, index=data.index)
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

robot_indicators = calculate_robot_indicators(data)
mtf_scores = calculate_multi_timeframe_score(data)
robot_signal = get_robot_signal(robot_indicators, mtf_scores)
trend_strength = get_trend_strength(data)

ai_signals = get_all_signals(data, volume, ticker)
ai_score = ai_signals['ai_score']

# ========== HEADER ==========
st.title(f"🤖 {ticker}")
st.caption(f"Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

col1, col2, col3 = st.columns(3)
col1.metric("Harga Sekarang", f"Rp {last_close:,.2f}", f"{change_pct:+.2f}%")
col2.metric("Tertinggi", f"Rp {data['High'].iloc[-1]:,.2f}")
col3.metric("Terendah", f"Rp {data['Low'].iloc[-1]:,.2f}")

st.markdown("---")

# ========== BARIS 1: INDICATOR ==========
st.markdown("## 📊 INDICATOR")

col_i1, col_i2, col_i3, col_i4 = st.columns(4)

with col_i1:
    rsi_val = robot_indicators.get('rsi', 'N/A')
    st.metric("RSI", f"{rsi_val}")

with col_i2:
    st.metric("MACD", f"{robot_indicators.get('macd', 0):.2f}",
              delta=f"Hist: {robot_indicators.get('macd_hist', 0):.2f}")

with col_i3:
    st.metric("Stochastic", f"{robot_indicators.get('stoch_k', 50):.1f}",
              delta=f"D: {robot_indicators.get('stoch_d', 50):.1f}")

with col_i4:
    vol_status = robot_indicators.get('volume_status', 'Normal')
    vol_current = robot_indicators.get('volume_current', 0)
    st.metric("Volume", f"{vol_current:,.0f}", delta=f"Status: {vol_status}")

st.markdown("---")

# ========== BARIS 2: MULTI TIMEFRAME ==========
st.markdown("## ⏰ MULTI TIMEFRAME")

col_m1, col_m2, col_m3 = st.columns(3)

with col_m1:
    score_3m = mtf_scores.get('3m', 0)
    st.metric("3 Bulan", f"{score_3m}")
    st.progress(score_3m/100)

with col_m2:
    score_12m = mtf_scores.get('12m', 0)
    st.metric("12 Bulan", f"{score_12m}")
    st.progress(score_12m/100)

with col_m3:
    score_30m = mtf_scores.get('30m', 0)
    st.metric("30 Bulan", f"{score_30m}")
    st.progress(score_30m/100)

st.markdown("---")

# ========== BARIS 3: SIGNAL & KEYAKINAN ==========
st.markdown("## 🎯 SIGNAL")

col_s1, col_s2, col_s3 = st.columns([1, 2, 1])

with col_s2:
    if robot_signal['color'] == "success":
        st.success(f"## {robot_signal['signal']}")
    elif robot_signal['color'] == "error":
        st.error(f"## {robot_signal['signal']}")
    else:
        st.warning(f"## {robot_signal['signal']}")
    
    st.metric("Keyakinan", robot_signal['confidence'])
    st.progress(robot_signal['confidence_score']/100)

# <==== TAMBAHAN BARU: INTERPRETASI SIGNAL ROBOT ====>
st.markdown("### 🧠 Arti Signal (Untuk Keputusan)")

if robot_signal['signal'] == "STRONG BUY":
    st.success("📌 **Kondisi sangat kuat** → Momentum + trend + volume mendukung. Cocok untuk **entry sekarang** (bukan menunggu).")
    st.caption("💡 Saran: Entry di harga support atau saat breakout konfirmasi.")
elif robot_signal['signal'] == "BUY":
    st.info("📌 **Kondisi cukup kuat** → Masuk boleh, tapi idealnya **tunggu pullback kecil** agar entry lebih aman.")
    st.caption("💡 Saran: Gunakan entry bertahap (scale in), jangan langsung full.")
elif robot_signal['signal'] == "WAIT / HOLD":
    st.warning("📌 **Pasar tidak jelas** → Lebih baik **tunggu konfirmasi arah** (jangan entry dulu).")
    st.caption("💡 Saran: Amati breakout atau breakdown support/resistance.")
elif robot_signal['signal'] == "SELL":
    st.warning("📌 **Tekanan turun mulai dominan** → Kalau sudah pegang posisi, mulai waspada / pertimbangkan keluar bertahap.")
    st.caption("💡 Saran: Cut loss jika sudah breakdown support.")
elif robot_signal['signal'] == "STRONG SELL":
    st.error("📌 **Tekanan jual kuat** → Kondisi tidak sehat untuk beli, **hindari entry**.")
    st.caption("💡 Saran: Lebih baik hold cash atau cari saham lain.")

st.markdown("---")

# ========== BARIS 4: REKOMENDASI DETAIL ==========
st.markdown("## 📝 REKOMENDASI")

col_r1, col_r2 = st.columns(2)

with col_r1:
    current_price = robot_indicators.get('current_price', 0)
    support = robot_indicators.get('support', 0)
    resistance = robot_indicators.get('resistance', 0)
    
    st.markdown(f"""
    ### Harga Saat Ini
    **Rp {current_price:,.0f}**
    
    ### Level Penting
    - **Support:** Rp {support:,.0f}
    - **Resistance:** Rp {resistance:,.0f}
    """)

with col_r2:
    st.markdown(f"""
    ### Range Harga (30 Hari)
    - **Tertinggi:** Rp {robot_indicators.get('high_30d', 0):,.0f}
    - **Terendah:** Rp {robot_indicators.get('low_30d', 0):,.0f}
    - **Rata-rata:** Rp {robot_indicators.get('mid_30d', 0):,.0f}
    
    ### Moving Average
    - **SMA20:** Rp {robot_indicators.get('sma20', 0):,.0f}
    - **SMA50:** Rp {robot_indicators.get('sma50', 0):,.0f}
    """)

st.markdown("---")

# ========== BARIS 5: SUPPORT & RESISTANCE ==========
st.markdown("## 📐 SUPPORT & RESISTANCE")

col_sup, col_res = st.columns(2)

with col_sup:
    support_val = robot_indicators.get('support', 0)
    current_val = robot_indicators.get('current_price', 1)
    if support_val > 0:
        pct_from_support = (current_val - support_val) / support_val * 100
        st.metric("SUPPORT", f"Rp {support_val:,.0f}",
                  delta=f"{pct_from_support:.1f}% dari harga")
        if pct_from_support < 2:
            st.caption("⚠️ Harga sudah mendekati support")
    else:
        st.metric("SUPPORT", "N/A")

with col_res:
    resistance_val = robot_indicators.get('resistance', 0)
    if resistance_val > 0:
        pct_to_resistance = (resistance_val - current_val) / current_val * 100
        st.metric("RESISTANCE", f"Rp {resistance_val:,.0f}",
                  delta=f"+{pct_to_resistance:.1f}% dari harga")
        if pct_to_resistance < 2:
            st.caption("⚠️ Harga sudah mendekati resistance")
    else:
        st.metric("RESISTANCE", "N/A")

st.markdown("---")

# ========== BARIS 6: TREND STRENGTH ==========
st.markdown("## 📈 TREND STRENGTH")

if trend_strength == "STRONG TREND":
    st.success(f"### {trend_strength}")
    st.progress(100)
    st.caption("📌 Trend sangat kuat, momentum tinggi")
elif trend_strength == "MODERATE TREND":
    st.info(f"### {trend_strength}")
    st.progress(65)
    st.caption("📌 Trend sedang, masih ada peluang")
elif trend_strength == "WEAK TREND":
    st.warning(f"### {trend_strength}")
    st.progress(35)
    st.caption("📌 Trend lemah, potensi sideways")
else:
    st.warning(f"### {trend_strength}")
    st.progress(20)
    st.caption("📌 Market bergerak sideways, tunggu arah")

st.markdown("---")

# ========== BARIS 7: FINAL SIGNAL ==========
st.markdown("## 🏁 FINAL SIGNAL")

if robot_signal['signal'] in ["STRONG BUY", "BUY"]:
    st.success(f"# ✅ {robot_signal['signal']}")
    st.balloons()
elif robot_signal['signal'] in ["STRONG SELL", "SELL"]:
    st.error(f"# ❌ {robot_signal['signal']}")
else:
    st.warning(f"# ⏳ {robot_signal['signal']}")

with st.expander("📋 Detail Analisis & Alasan"):
    for reason in robot_signal['reasons']:
        st.write(f"- {reason}")
    st.markdown("---")
    st.markdown(f"**Total Score:** {robot_signal['score']}")

st.markdown("---")

# ========== AI SIGNAL LITE ==========
st.header("🎯 AI SIGNAL LITE - KEPUTUSAN BELI/JUAL")

col_main1, col_main2 = st.columns([2, 1])

with col_main1:
    if ai_score >= 4:
        st.success("## 🚀 STRONG BUY")
        st.progress(100)
    elif ai_score == 3:
        st.info("## 🟡 HOLD / WAIT")
        st.progress(60)
    else:
        st.error("## 🔻 SELL / AVOID")
        st.progress(20)

with col_main2:
    st.metric("AI SCORE", f"{ai_score}/5")
    st.metric("Confidence", f"{ai_signals['confidence']}%")

# <==== TAMBAHAN BARU: INTERPRETASI AI SCORE ====>
st.markdown("### 🧠 Arti AI Score")

if ai_score >= 4:
    st.success("📌 **Mayoritas indikator teknikal bullish** → peluang naik lebih besar dari turun.")
    st.caption("💡 Indikator yang mendukung: RSI oversold, harga > SMA, MACD positif, volume tinggi")
elif ai_score == 3:
    st.info("📌 **Kondisi seimbang** → market belum memilih arah kuat.")
    st.caption("💡 Sebagian indikator bullish, sebagian bearish. Tunggu konfirmasi.")
else:
    st.error("📌 **Tekanan bearish dominan** → risiko turun lebih tinggi dari peluang naik.")
    st.caption("💡 Sebagian besar indikator menunjukkan sinyal jual.")

st.markdown("---")

# ========== DETAIL SIGNAL LITE ==========
st.subheader("📊 Detail Analisis")

col_d1, col_d2, col_d3 = st.columns(3)

with col_d1:
    st.markdown("**🔥 Momentum**")
    try:
        momentum = data['Close'].pct_change(5).iloc[-1] * 100 if len(data) >= 6 else 0
        if momentum > 3:
            st.success(f"Up {momentum:.1f}%")
            st.caption("📌 Momentum positif kuat")
        elif momentum < -3:
            st.error(f"Down {momentum:.1f}%")
            st.caption("📌 Momentum negatif kuat")
        else:
            st.warning("Sideways")
            st.caption("📌 Momentum netral")
    except:
        st.info("N/A")
    
    st.markdown("**🚀 Breakout**")
    try:
        if len(data) > 1 and data['Close'].iloc[-1] > data['resistance'].iloc[-2]:
            st.success("BREAKOUT DETECTED!")
            st.caption("📌 Harga menembus resistance, potensi naik")
        else:
            st.info("Tidak ada breakout")
            st.caption("📌 Harga masih dalam range")
    except:
        st.info("N/A")

with col_d2:
    st.markdown("**📈 Multi Timeframe**")
    mtf = get_multi_timeframe_trend(ticker)
    st.write(f"Daily: {mtf['Daily']}")
    st.write(f"Weekly: {mtf['Weekly']}")
    st.write(f"Monthly: {mtf['Monthly']}")
    
    bullish_count = list(mtf.values()).count("Bullish")
    if bullish_count >= 2:
        st.success("📌 Mayoritas timeframe bullish")
    elif bullish_count == 1:
        st.warning("📌 Timeframe mixed signal")
    else:
        st.error("📌 Mayoritas timeframe bearish")

with col_d3:
    st.markdown("**🏦 Smart Money**")
    st.write(f"Status: {ai_signals['smart_status']}")
    st.write(f"Score: {ai_signals['smart_score']}/4")
    
    if ai_signals['smart_status'] == "Accumulation":
        st.success("📌 Smart money sedang mengakumulasi (tekanan beli)")
    elif ai_signals['smart_status'] == "Distribution":
        st.error("📌 Smart money mendistribusikan (tekanan jual)")
    else:
        st.info("📌 Smart money netral")
    
    st.markdown("**🌍 Macro**")
    st.write(f"Kondisi: {ai_signals['macro_status']}")
    
    if ai_signals['macro_status'] == "Risk ON":
        st.success("📌 Kondisi global mendukung risk-on")
    elif ai_signals['macro_status'] == "Risk OFF":
        st.error("📌 Kondisi global risk-off, hati-hati")
    else:
        st.info("📌 Kondisi global netral")

st.markdown("---")

# ========== PROBABILITAS ==========
st.subheader("📈 Probabilitas")

try:
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
except:
    bull_prob = 50
    bear_prob = 50

col_p1, col_p2 = st.columns(2)
col_p1.metric("📈 BULLISH", f"{bull_prob:.0f}%")
col_p2.metric("📉 BEARISH", f"{bear_prob:.0f}%")
st.progress(bull_prob/100)

if bull_prob > 60:
    st.caption("📌 Probabilitas bullish lebih tinggi")
elif bear_prob > 60:
    st.caption("📌 Probabilitas bearish lebih tinggi")
else:
    st.caption("📌 Probabilitas seimbang")

st.markdown("---")

# ========== WEIGHTED DECISION ==========
st.subheader("🧠 Weighted Decision")

weighted_score = weighted_decision_engine(data, volume, ticker)
st.metric("Final Score", f"{weighted_score}%")

if weighted_score >= 70:
    st.success("### ✅ REKOMENDASI: BELI")
    st.progress(weighted_score/100)
elif weighted_score >= 50:
    st.warning("### ⚠️ REKOMENDASI: HOLD")
    st.progress(weighted_score/100)
else:
    st.error("### ❌ REKOMENDASI: JUAL")
    st.progress(weighted_score/100)

# <==== TAMBAHAN BARU: INTERPRETASI WEIGHTED SCORE ====>
st.markdown("### 🧠 Arti Weighted Score")

if weighted_score >= 75:
    st.success("📌 **Sangat Bullish** → Semua faktor (teknikal + smart money + makro) mendukung kenaikan.")
    st.caption("💡 Peluang terbaik untuk entry. Gunakan ukuran posisi normal.")
elif weighted_score >= 70:
    st.success("📌 **Bullish** → Sebagian besar faktor mendukung kenaikan.")
    st.caption("💡 Cocok untuk entry, tapi gunakan ukuran posisi lebih kecil.")
elif weighted_score >= 60:
    st.info("📌 **Cenderung Bullish** → Ada peluang naik, tapi belum terlalu kuat.")
    st.caption("💡 Entry bertahap (scale in) lebih aman.")
elif weighted_score >= 50:
    st.warning("📌 **Netral** → Kondisi seimbang, market belum memilih arah.")
    st.caption("💡 Lebih baik tunggu konfirmasi tambahan sebelum entry.")
elif weighted_score >= 40:
    st.warning("📌 **Cenderung Bearish** → Risiko turun lebih tinggi.")
    st.caption("💡 Hindari entry baru, pertimbangkan cut loss jika sudah pegang posisi.")
elif weighted_score >= 30:
    st.error("📌 **Bearish** → Tekanan jual mulai dominan.")
    st.caption("💡 Lebih baik hold cash atau cari saham lain.")
else:
    st.error("📌 **Sangat Bearish** → Kondisi lemah, tidak sehat untuk beli.")
    st.caption("💡 Hindari entry sampai kondisi membaik.")

st.markdown("---")

# ========== REKOMENDASI ENTRY & STOPLOSS ==========
entry_rec = get_recommended_entry_stoploss(data)
st.markdown("### 🎯 Rekomendasi Entry & Stop Loss")
col_e, col_s, col_t = st.columns(3)
col_e.metric("Entry", f"Rp {entry_rec['entry']:,.2f}" if entry_rec['entry'] else "N/A")
col_s.metric("Stop Loss", f"Rp {entry_rec['stoploss']:,.2f}" if entry_rec['stoploss'] else "N/A", delta_color="inverse")
col_t.metric("Target", f"Rp {entry_rec['target']:,.2f}" if entry_rec['target'] else "N/A")

if entry_rec['entry'] and entry_rec['stoploss'] and entry_rec['target']:
    risk = entry_rec['entry'] - entry_rec['stoploss']
    reward = entry_rec['target'] - entry_rec['entry']
    rr = reward / risk if risk > 0 else 0
    st.info(f"📐 **Risk Reward:** 1 : {rr:.2f}")
    
    if rr >= 2:
        st.success("✅ Risk reward bagus! (>1:2)")
    elif rr >= 1.5:
        st.info("👍 Risk reward cukup (1:1.5)")
    elif rr >= 1:
        st.warning("⚠️ Risk reward minimal (1:1)")
    else:
        st.error("❌ Risk reward jelek, lebih baik hindari")

st.markdown("---")

# ========== POSITION SIZING ==========
st.subheader("📐 Position Sizing")

col_cap, col_risk = st.columns(2)
with col_cap:
    capital = st.number_input("Modal (Rp)", min_value=0.0, value=100_000_000.0, step=10_000_000.0)
with col_risk:
    risk_percent = st.number_input("Risiko per Trade (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.5)

if entry_rec['stoploss'] and capital > 0 and risk_percent > 0 and entry_rec['entry']:
    risk_amount = capital * (risk_percent / 100)
    price_risk = entry_rec['entry'] - entry_rec['stoploss']
    if price_risk > 0:
        suggested_shares = int(risk_amount / price_risk)
        position_value = suggested_shares * entry_rec['entry']
        st.write(f"**📊 Jumlah saham:** {suggested_shares:,} lembar")
        st.write(f"**💰 Nilai posisi:** Rp {position_value:,.2f} ({position_value/capital*100:.1f}% dari modal)")
        st.write(f"**🛑 Stop loss total:** Rp {risk_amount:,.2f} ({risk_percent:.1f}% dari modal)")
        
        if position_value > capital:
            st.warning("⚠️ Nilai posisi melebihi modal! Turunkan jumlah saham atau perbesar stop loss.")
    else:
        st.warning("⚠️ Stop loss terlalu dekat dengan entry")

st.markdown("---")

# ========== INDIKATOR TEKNIKAL LENGKAP ==========
with st.expander("📊 Indikator Teknikal Lengkap"):
    try:
        col_i1, col_i2 = st.columns(2)
        with col_i1:
            st.metric("RSI (14)", f"{data['RSI'].iloc[-1]:.1f}")
            st.metric("SMA20", f"Rp {data['SMA20'].iloc[-1]:.2f}")
            st.metric("Support", f"Rp {data['support'].iloc[-1]:.2f}")
            st.metric("Volume", f"{volume.iloc[-1]:,.0f}")
        with col_i2:
            st.metric("MACD", f"{data['MACD'].iloc[-1]:.4f}")
            st.metric("MACD Signal", f"{data['MACD_signal'].iloc[-1]:.4f}")
            st.metric("Resistance", f"Rp {data['resistance'].iloc[-1]:.2f}")
            st.metric("Volume MA20", f"{data['Volume_MA'].iloc[-1]:,.0f}")
    except Exception as e:
        st.info("Data indikator tidak tersedia")

# ========== KESIMPULAN AKHIR (PALING PENTING) ==========
st.markdown("---")
st.markdown("## 🧭 KESIMPULAN AKHIR (UNTUK USER)")

# Hitung final decision score dari robot signal dan weighted score
final_decision_score = (robot_signal['confidence_score'] + weighted_score) / 2

if final_decision_score >= 75:
    st.success("🟢 **KESIMPULAN: MARKET LAYAK DIBELI (BUY ZONE)**")
    st.write("📌 **Aksi yang disarankan:** Entry sekarang dengan ukuran posisi normal.")
    st.write("📌 **Strategi:** Beli di harga support atau saat breakout konfirmasi.")
    st.write("📌 **Risk management:** Stop loss di bawah support terdekat.")
    
elif final_decision_score >= 65:
    st.success("🟢 **KESIMPULAN: MARKET CENDERUNG BULLISH**")
    st.write("📌 **Aksi yang disarankan:** Entry bertahap (scale in), jangan langsung full.")
    st.write("📌 **Strategi:** Tunggu pullback kecil untuk entry lebih aman.")
    st.write("📌 **Risk management:** Gunakan ukuran posisi lebih kecil dari biasanya.")
    
elif final_decision_score >= 50:
    st.warning("🟡 **KESIMPULAN: MARKET NETRAL**")
    st.write("📌 **Aksi yang disarankan:** HOLD / TUNGGU, jangan entry dulu.")
    st.write("📌 **Strategi:** Tunggu konfirmasi breakout atau breakdown.")
    st.write("📌 **Risk management:** Jika sudah pegang posisi, jangan tambah posisi.")
    
elif final_decision_score >= 40:
    st.warning("🟡 **KESIMPULAN: MARKET CENDERUNG BEARISH**")
    st.write("📌 **Aksi yang disarankan:** Hindari entry baru.")
    st.write("📌 **Strategi:** Jika sudah pegang posisi, pertimbangkan cut loss bertahap.")
    st.write("📌 **Risk management:** Lebih aman hold cash.")
    
else:
    st.error("🔴 **KESIMPULAN: MARKET BERISIKO TINGGI**")
    st.write("📌 **Aksi yang disarankan:** JANGAN ENTRY, lebih baik hindari.")
    st.write("📌 **Strategi:** Tunggu sampai kondisi membaik.")
    st.write("📌 **Risk management:** Jika sudah pegang posisi, segera evaluasi cut loss.")

# Tampilkan ringkasan final
st.markdown("---")
st.markdown("### 📋 Ringkasan Final")

col_sum1, col_sum2, col_sum3 = st.columns(3)

with col_sum1:
    st.metric("Robot Signal", robot_signal['signal'])
    st.caption(f"Keyakinan: {robot_signal['confidence']}")

with col_sum2:
    st.metric("AI Score", f"{ai_score}/5")
    st.caption(f"Confidence: {ai_signals['confidence']}%")

with col_sum3:
    st.metric("Weighted Score", f"{weighted_score}%")
    st.caption(f"Final Decision: {final_decision_score:.0f}%")

# ========== FOOTER ==========
st.markdown("---")
st.caption("⚠️ **DISCLAIMER:** Dashboard ini hanya untuk edukasi dan analisis. Bukan rekomendasi beli/jual. Keputusan investasi sepenuhnya risiko Anda.")

gc.collect()
warnings.filterwarnings('ignore')

# ========== KONSTANTA ==========
DATA_PERIOD = "3mo"
CACHE_TTL = 300
DEFAULT_TICKER = "BBCA.JK"

TIMEFRAMES = {"1d": "1d", "1wk": "1wk", "1mo": "1mo"}
VOLUME_SPIKE_THRESHOLD = 1.8
BREAKOUT_COOLDOWN_HOURS = 24

st.set_page_config(
    layout="wide", 
    page_title="Robot Saham - AI Signal Lite", 
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

# ========== FUNGSI BANTU ==========
def fix_ticker(ticker: str) -> str:
    ticker = ticker.strip().upper()
    if ticker.startswith('^'):
        return ticker
    if ticker.endswith('.JK'):
        return ticker
    return ticker + '.JK'

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
        
        # Tambahkan indikator dasar
        if len(df) >= 20:
            df['SMA20'] = df['Close'].rolling(20).mean()
        if len(df) >= 50:
            df['SMA50'] = df['Close'].rolling(50).mean()
        if len(df) >= 14:
            df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        
        return df
    except Exception as e:
        st.error(f"Error: {str(e)[:100]}")
        return pd.DataFrame()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    
    try:
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

        try:
            if volume.sum() > 0:
                df['AD'] = ta.volume.acc_dist_index(df['High'], df['Low'], df['Close'], df['Volume'], fillna=True)
            else:
                df['AD'] = 0.0
        except:
            df['AD'] = 0.0
            
        try:
            if volume.sum() > 0:
                df['CMF'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'], window=20, fillna=True)
            else:
                df['CMF'] = 0.0
        except:
            df['CMF'] = 0.0

        df = df.ffill().fillna(0)
        return df
        
    except Exception as e:
        return df

# ========== FUNGSI UNTUK ROBOT SAHAM (FITUR BARU) ==========
def calculate_robot_indicators(df: pd.DataFrame) -> dict:
    """Hitung semua indikator seperti di gambar Robot Saham"""
    if df.empty or len(df) < 30:
        return {}
    
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume'] if 'Volume' in df.columns else pd.Series(0, index=df.index)
    
    # RSI
    rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns and pd.notna(df['RSI'].iloc[-1]) else 50
    
    # MACD
    if len(close) >= 26:
        macd = ta.trend.MACD(close)
        macd_line = macd.macd().iloc[-1]
        macd_signal = macd.macd_signal().iloc[-1]
        macd_hist = macd_line - macd_signal
    else:
        macd_line = 0
        macd_signal = 0
        macd_hist = 0
    
    # Stochastic
    if len(close) >= 14:
        stoch_k = ta.momentum.stoch(high, low, close, window=14, smooth_window=3).iloc[-1]
        stoch_d = ta.momentum.stoch_signal(high, low, close, window=14, smooth_window=3).iloc[-1]
    else:
        stoch_k = 50
        stoch_d = 50
    
    # Volume
    volume_avg = volume.rolling(20).mean().iloc[-1] if volume.sum() > 0 else 1
    volume_current = volume.iloc[-1] if volume.sum() > 0 else 0
    if volume_current < volume_avg:
        volume_status = "Rendah"
    elif volume_current > volume_avg * 1.5:
        volume_status = "Tinggi"
    else:
        volume_status = "Normal"
    
    # SMA
    sma20 = df['SMA20'].iloc[-1] if 'SMA20' in df.columns and pd.notna(df['SMA20'].iloc[-1]) else close.iloc[-1]
    sma50 = df['SMA50'].iloc[-1] if 'SMA50' in df.columns and pd.notna(df['SMA50'].iloc[-1]) else close.iloc[-1]
    
    # Support & Resistance
    support = df['Low'].rolling(20).min().iloc[-1]
    resistance = df['High'].rolling(20).max().iloc[-1]
    
    # Harga tertinggi, terendah, sedang
    high_30d = df['High'].tail(30).max()
    low_30d = df['Low'].tail(30).min()
    mid_30d = (high_30d + low_30d) / 2
    
    return {
        'rsi': round(rsi, 1),
        'macd': round(macd_line, 2),
        'macd_signal': round(macd_signal, 2),
        'macd_hist': round(macd_hist, 2),
        'stoch_k': round(stoch_k, 1),
        'stoch_d': round(stoch_d, 1),
        'volume_current': volume_current,
        'volume_avg': volume_avg,
        'volume_status': volume_status,
        'sma20': round(sma20, 2),
        'sma50': round(sma50, 2),
        'support': round(support, 2),
        'resistance': round(resistance, 2),
        'high_30d': round(high_30d, 2),
        'low_30d': round(low_30d, 2),
        'mid_30d': round(mid_30d, 2),
        'current_price': round(close.iloc[-1], 2)
    }

def calculate_multi_timeframe_score(df: pd.DataFrame) -> dict:
    """Hitung score untuk 3m, 12m, 30m"""
    if df.empty:
        return {'3m': 0, '12m': 0, '30m': 0}
    
    # 3 bulan (approx 60 hari trading)
    df_3m = df.tail(60)
    score_3m = 0
    if len(df_3m) > 0 and df_3m['Close'].iloc[-1] > df_3m['Close'].iloc[0]:
        score_3m += 1
    if 'SMA20' in df_3m.columns and len(df_3m) >= 20:
        if pd.notna(df_3m['SMA20'].iloc[-1]) and df_3m['Close'].iloc[-1] > df_3m['SMA20'].iloc[-1]:
            score_3m += 1
    if 'RSI' in df_3m.columns:
        if pd.notna(df_3m['RSI'].iloc[-1]) and df_3m['RSI'].iloc[-1] > 50:
            score_3m += 1
    score_3m = int(score_3m * 100 / 3) if score_3m > 0 else 0
    
    # 12 bulan (approx 250 hari trading)
    df_12m = df.tail(250) if len(df) >= 250 else df
    score_12m = 0
    if len(df_12m) > 0 and df_12m['Close'].iloc[-1] > df_12m['Close'].iloc[0]:
        score_12m += 1
    if 'SMA50' in df_12m.columns and len(df_12m) >= 50:
        if pd.notna(df_12m['SMA50'].iloc[-1]) and df_12m['Close'].iloc[-1] > df_12m['SMA50'].iloc[-1]:
            score_12m += 1
    if 'RSI' in df_12m.columns:
        if pd.notna(df_12m['RSI'].iloc[-1]) and df_12m['RSI'].iloc[-1] > 50:
            score_12m += 1
    score_12m = int(score_12m * 100 / 3) if score_12m > 0 else 0
    
    # 30 bulan (approx 630 hari trading)
    df_30m = df.tail(630) if len(df) >= 630 else df
    score_30m = 0
    if len(df_30m) > 0 and df_30m['Close'].iloc[-1] > df_30m['Close'].iloc[0]:
        score_30m += 1
    if 'SMA50' in df_30m.columns and len(df_30m) >= 50:
        if pd.notna(df_30m['SMA50'].iloc[-1]) and df_30m['Close'].iloc[-1] > df_30m['SMA50'].iloc[-1]:
            score_30m += 1
    if 'RSI' in df_30m.columns:
        if pd.notna(df_30m['RSI'].iloc[-1]) and df_30m['RSI'].iloc[-1] > 50:
            score_30m += 1
    score_30m = int(score_30m * 100 / 3) if score_30m > 0 else 0
    
    return {
        '3m': score_3m,
        '12m': score_12m,
        '30m': score_30m
    }

def get_robot_signal(indicators: dict, mtf_score: dict) -> dict:
    """Generate signal seperti di gambar Robot Saham"""
    
    score = 0
    reasons = []
    
    # RSI Signal
    rsi_val = indicators.get('rsi', 50)
    if rsi_val < 35:
        score += 2
        reasons.append(f"RSI Oversold ({rsi_val}) -> Signal BELI")
    elif rsi_val > 70:
        score -= 2
        reasons.append(f"RSI Overbought ({rsi_val}) -> Signal JUAL")
    else:
        reasons.append(f"RSI Normal ({rsi_val})")
    
    # MACD Signal
    macd_hist = indicators.get('macd_hist', 0)
    if macd_hist > 0:
        score += 1
        reasons.append(f"MACD Positif ({macd_hist:.2f}) -> Bullish")
    else:
        score -= 1
        reasons.append(f"MACD Negatif ({macd_hist:.2f}) -> Bearish")
    
    # Stochastic Signal
    stoch_k = indicators.get('stoch_k', 50)
    if stoch_k < 20:
        score += 2
        reasons.append(f"Stochastic Oversold ({stoch_k}) -> Signal BELI")
    elif stoch_k > 80:
        score -= 2
        reasons.append(f"Stochastic Overbought ({stoch_k}) -> Signal JUAL")
    else:
        reasons.append(f"Stochastic Normal ({stoch_k})")
    
    # Multi Timeframe Signal
    mtf_avg = (mtf_score.get('3m', 0) + mtf_score.get('12m', 0) + mtf_score.get('30m', 0)) / 3
    if mtf_avg > 60:
        score += 2
        reasons.append(f"Multi Timeframe Bullish (avg {mtf_avg:.0f})")
    elif mtf_avg < 40:
        score -= 2
        reasons.append(f"Multi Timeframe Bearish (avg {mtf_avg:.0f})")
    else:
        reasons.append(f"Multi Timeframe Netral (avg {mtf_avg:.0f})")
    
    # Trend Signal
    current_price = indicators.get('current_price', 0)
    sma20 = indicators.get('sma20', 0)
    if current_price > sma20:
        score += 1
        reasons.append(f"Harga > SMA20 -> Uptrend")
    else:
        score -= 1
        reasons.append(f"Harga < SMA20 -> Downtrend")
    
    # Volume Signal
    volume_status = indicators.get('volume_status', 'Normal')
    if volume_status == "Tinggi" and score > 0:
        score += 1
        reasons.append("Volume Tinggi mendukung pergerakan")
    elif volume_status == "Rendah":
        reasons.append("Volume Rendah (potensi sideways)")
    
    # Tentukan signal
    if score >= 4:
        signal = "STRONG BUY"
        color = "success"
        confidence = "HIGH CONFIDENCE"
        confidence_score = 85
    elif score >= 2:
        signal = "BUY"
        color = "info"
        confidence = "MEDIUM CONFIDENCE"
        confidence_score = 65
    elif score <= -4:
        signal = "STRONG SELL"
        color = "error"
        confidence = "HIGH CONFIDENCE"
        confidence_score = 85
    elif score <= -2:
        signal = "SELL"
        color = "warning"
        confidence = "LOW CONFIDENCE"
        confidence_score = 35
    else:
        signal = "WAIT / HOLD"
        color = "warning"
        confidence = "LOW CONFIDENCE"
        confidence_score = 50
    
    return {
        'signal': signal,
        'color': color,
        'score': score,
        'confidence': confidence,
        'confidence_score': confidence_score,
        'reasons': reasons
    }

def get_trend_strength(df: pd.DataFrame) -> str:
    """Hitung kekuatan trend"""
    if df.empty or len(df) < 50:
        return "WEAK"
    
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    try:
        adx = ta.trend.ADXIndicator(high, low, close, window=14)
        adx_value = adx.adx().iloc[-1] if len(close) >= 14 else 25
        
        if adx_value > 40:
            return "STRONG TREND"
        elif adx_value > 25:
            return "MODERATE TREND"
        elif adx_value > 20:
            return "WEAK TREND"
        else:
            return "SIDEWAYS"
    except:
        return "NORMAL"

# ========== FUNGSI YANG SUDAH ADA SEBELUMNYA ==========
def calculate_ai_score(df: pd.DataFrame, volume: pd.Series) -> int:
    if df.empty:
        return 0
    try:
        conditions = [
            df['RSI'].iloc[-1] < 35,
            df['Close'].iloc[-1] > df['SMA20'].iloc[-1],
            df['SMA20'].iloc[-1] > df['SMA50'].iloc[-1],
            df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1],
            volume.iloc[-1] > df['Volume_MA'].iloc[-1] if volume.sum() > 0 else False
        ]
        return sum(conditions)
    except:
        return 2

def get_multi_timeframe_trend(ticker):
    try:
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
    except:
        return {"Daily": "Neutral", "Weekly": "Neutral", "Monthly": "Neutral"}

def calculate_smart_money(df):
    if df.empty or len(df) < 20:
        return 0, "Neutral"
    try:
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
    except:
        return 2, "Neutral"

def get_macro_signal():
    try:
        ihsg = yf.download("^JKSE", period="5d", progress=False)['Close']
        usd = yf.download("USDIDR=X", period="5d", progress=False)['Close']
        nasdaq = yf.download("^IXIC", period="5d", progress=False)['Close']
        score = 0
        if len(ihsg) > 0 and ihsg.iloc[-1] > ihsg.mean():
            score += 1
        if len(usd) > 0 and usd.iloc[-1] < usd.mean():
            score += 1
        if len(nasdaq) > 0 and nasdaq.iloc[-1] > nasdaq.mean():
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
    try:
        ai_score = calculate_ai_score(df, volume)
        smart_score, smart_status = calculate_smart_money(df)
        macro_status, macro_score = get_macro_signal()
        
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
    except:
        return {'ai_score': 2, 'smart_score': 2, 'smart_status': 'Neutral', 'macro_status': 'Neutral', 'macro_score': 1, 'confidence': 50}

def weighted_decision_engine(df, volume, ticker):
    try:
        signals = get_all_signals(df, volume, ticker)
        tech_score = signals['ai_score'] / 5
        smart_map = {"Accumulation": 1, "Neutral": 0.5, "Distribution": 0}
        smart_score = smart_map.get(signals['smart_status'], 0.5)
        macro_map = {"Risk ON": 1, "Neutral": 0.5, "Risk OFF": 0}
        macro_score = macro_map.get(signals['macro_status'], 0.5)
        returns = df['Close'].pct_change().dropna()
        if len(returns) > 0:
            vol = returns.std() * np.sqrt(252)
            if vol < 0.15:
                risk_score = 1
            elif vol < 0.30:
                risk_score = 0.6
            else:
                risk_score = 0.2
        else:
            risk_score = 0.5
        momentum = df['Close'].pct_change(5).iloc[-1] if len(df) >= 6 else 0
        if momentum > 0.05:
            momentum_score = 1
        elif momentum > 0:
            momentum_score = 0.6
        else:
            momentum_score = 0.2
        final_score = (0.35 * tech_score + 0.20 * smart_score + 0.15 * macro_score + 0.15 * risk_score + 0.15 * momentum_score)
        return round(final_score * 100, 2)
    except:
        return 50

def get_price_range(df: pd.DataFrame, days: int) -> dict:
    if df.empty or len(df) < days:
        return {'high': None, 'low': None, 'current': None}
    try:
        recent_data = df.tail(days)
        return {'high': recent_data['High'].max(), 'low': recent_data['Low'].min(), 'current': df['Close'].iloc[-1]}
    except:
        return {'high': None, 'low': None, 'current': None}

def get_recommended_entry_stoploss(df: pd.DataFrame) -> dict:
    if df.empty:
        return {'entry': None, 'stoploss': None, 'target': None}
    try:
        price = df['Close'].iloc[-1]
        support = df['support'].iloc[-1] if pd.notna(df['support'].iloc[-1]) else price * 0.95
        resistance = df['resistance'].iloc[-1] if pd.notna(df['resistance'].iloc[-1]) else price * 1.05
        entry = (support + price) / 2
        stoploss = support * 0.97
        target = resistance
        return {'entry': entry, 'stoploss': stoploss, 'target': target}
    except:
        return {'entry': None, 'stoploss': None, 'target': None}

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("# 🤖 Robot Saham - AI Signal")
    ticker_input = st.text_input("Kode Saham", DEFAULT_TICKER, help="Contoh: BBCA.JK, BBRI.JK, ASII.JK")
    ticker = fix_ticker(ticker_input)
    if ticker != ticker_input:
        st.info(f"Format: {ticker}")
    timeframe = st.selectbox("Timeframe", list(TIMEFRAMES.keys()))
    
    st.markdown("### ⚙️ Pengaturan")
    user_volume_spike = st.slider("Volume Spike Threshold", 1.0, 5.0, st.session_state.user_volume_spike_threshold, 0.1)
    st.session_state.user_volume_spike_threshold = user_volume_spike
    
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    st.caption("⚠️ Hanya untuk edukasi | Bukan rekomendasi beli/jual")

# ========== LOAD DATA ==========
with st.spinner(f"Memuat data {ticker}..."):
    data = load_data(ticker, TIMEFRAMES[timeframe])

if data.empty or len(data) < 10:
    st.error(f"""
    ❌ **Tidak bisa memuat data untuk {ticker}**
    
    **Coba ticker berikut:**
    - BBCA.JK (Bank Central Asia)
    - BBRI.JK (Bank Rakyat Indonesia)
    - ASII.JK (Astra International)
    - TLKM.JK (Telkom)
    - ^JKSE (IHSG)
    """)
    st.stop()

data = add_indicators(data)

if data.empty:
    st.error("Gagal memproses data")
    st.stop()

# Ambil data terkini
try:
    last_close = data['Close'].iloc[-1]
    prev_close = data['Close'].iloc[-2] if len(data) > 1 else last_close
    change_pct = ((last_close - prev_close) / prev_close) * 100 if prev_close != 0 else 0
    volume = data['Volume'] if 'Volume' in data.columns else pd.Series(0, index=data.index)
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# Hitung semua indikator untuk Robot Saham
robot_indicators = calculate_robot_indicators(data)
mtf_scores = calculate_multi_timeframe_score(data)
robot_signal = get_robot_signal(robot_indicators, mtf_scores)
trend_strength = get_trend_strength(data)

# Hitung sinyal AI yang sudah ada
ai_signals = get_all_signals(data, volume, ticker)
ai_score = ai_signals['ai_score']

# ========== HEADER ==========
st.title(f"🤖 {ticker}")
st.caption(f"Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

col1, col2, col3 = st.columns(3)
col1.metric("Harga Sekarang", f"Rp {last_close:,.2f}", f"{change_pct:+.2f}%")
col2.metric("Tertinggi", f"Rp {data['High'].iloc[-1]:,.2f}")
col3.metric("Terendah", f"Rp {data['Low'].iloc[-1]:,.2f}")

st.markdown("---")

# ========== BARIS 1: INDICATOR ==========
st.markdown("## 📊 INDICATOR")

col_i1, col_i2, col_i3, col_i4 = st.columns(4)

with col_i1:
    rsi_val = robot_indicators.get('rsi', 'N/A')
    st.metric("RSI", f"{rsi_val}")

with col_i2:
    st.metric("MACD", f"{robot_indicators.get('macd', 0):.2f}",
              delta=f"Hist: {robot_indicators.get('macd_hist', 0):.2f}")

with col_i3:
    st.metric("Stochastic", f"{robot_indicators.get('stoch_k', 50):.1f}",
              delta=f"D: {robot_indicators.get('stoch_d', 50):.1f}")

with col_i4:
    vol_status = robot_indicators.get('volume_status', 'Normal')
    vol_current = robot_indicators.get('volume_current', 0)
    st.metric("Volume", f"{vol_current:,.0f}", delta=f"Status: {vol_status}")

st.markdown("---")

# ========== BARIS 2: MULTI TIMEFRAME ==========
st.markdown("## ⏰ MULTI TIMEFRAME")

col_m1, col_m2, col_m3 = st.columns(3)

with col_m1:
    score_3m = mtf_scores.get('3m', 0)
    st.metric("3 Bulan", f"{score_3m}")
    st.progress(score_3m/100)

with col_m2:
    score_12m = mtf_scores.get('12m', 0)
    st.metric("12 Bulan", f"{score_12m}")
    st.progress(score_12m/100)

with col_m3:
    score_30m = mtf_scores.get('30m', 0)
    st.metric("30 Bulan", f"{score_30m}")
    st.progress(score_30m/100)

st.markdown("---")

# ========== BARIS 3: SIGNAL & KEYAKINAN ==========
st.markdown("## 🎯 SIGNAL")

col_s1, col_s2, col_s3 = st.columns([1, 2, 1])

with col_s2:
    if robot_signal['color'] == "success":
        st.success(f"## {robot_signal['signal']}")
    elif robot_signal['color'] == "error":
        st.error(f"## {robot_signal['signal']}")
    else:
        st.warning(f"## {robot_signal['signal']}")
    
    st.metric("Keyakinan", robot_signal['confidence'])
    st.progress(robot_signal['confidence_score']/100)

st.markdown("---")

# ========== BARIS 4: REKOMENDASI DETAIL ==========
st.markdown("## 📝 REKOMENDASI")

col_r1, col_r2 = st.columns(2)

with col_r1:
    current_price = robot_indicators.get('current_price', 0)
    support = robot_indicators.get('support', 0)
    resistance = robot_indicators.get('resistance', 0)
    
    st.markdown(f"""
    ### Harga Saat Ini
    **Rp {current_price:,.0f}**
    
    ### Level Penting
    - **Support:** Rp {support:,.0f}
    - **Resistance:** Rp {resistance:,.0f}
    """)

with col_r2:
    st.markdown(f"""
    ### Range Harga (30 Hari)
    - **Tertinggi:** Rp {robot_indicators.get('high_30d', 0):,.0f}
    - **Terendah:** Rp {robot_indicators.get('low_30d', 0):,.0f}
    - **Rata-rata:** Rp {robot_indicators.get('mid_30d', 0):,.0f}
    
    ### Moving Average
    - **SMA20:** Rp {robot_indicators.get('sma20', 0):,.0f}
    - **SMA50:** Rp {robot_indicators.get('sma50', 0):,.0f}
    """)

st.markdown("---")

# ========== BARIS 5: SUPPORT & RESISTANCE ==========
st.markdown("## 📐 SUPPORT & RESISTANCE")

col_sup, col_res = st.columns(2)

with col_sup:
    support_val = robot_indicators.get('support', 0)
    current_val = robot_indicators.get('current_price', 1)
    if support_val > 0:
        pct_from_support = (current_val - support_val) / support_val * 100
        st.metric("SUPPORT", f"Rp {support_val:,.0f}",
                  delta=f"{pct_from_support:.1f}% dari harga")
    else:
        st.metric("SUPPORT", "N/A")

with col_res:
    resistance_val = robot_indicators.get('resistance', 0)
    if resistance_val > 0:
        pct_to_resistance = (resistance_val - current_val) / current_val * 100
        st.metric("RESISTANCE", f"Rp {resistance_val:,.0f}",
                  delta=f"+{pct_to_resistance:.1f}% dari harga")
    else:
        st.metric("RESISTANCE", "N/A")

st.markdown("---")

# ========== BARIS 6: TREND STRENGTH ==========
st.markdown("## 📈 TREND STRENGTH")

if trend_strength == "STRONG TREND":
    st.success(f"### {trend_strength}")
    st.progress(100)
elif trend_strength == "MODERATE TREND":
    st.info(f"### {trend_strength}")
    st.progress(65)
elif trend_strength == "WEAK TREND":
    st.warning(f"### {trend_strength}")
    st.progress(35)
else:
    st.warning(f"### {trend_strength}")
    st.progress(20)

st.markdown("---")

# ========== BARIS 7: FINAL SIGNAL ==========
st.markdown("## 🏁 FINAL SIGNAL")

if robot_signal['signal'] in ["STRONG BUY", "BUY"]:
    st.success(f"# ✅ {robot_signal['signal']}")
    st.balloons()
elif robot_signal['signal'] in ["STRONG SELL", "SELL"]:
    st.error(f"# ❌ {robot_signal['signal']}")
else:
    st.warning(f"# ⏳ {robot_signal['signal']}")

# Tampilkan alasan
with st.expander("📋 Detail Analisis & Alasan"):
    for reason in robot_signal['reasons']:
        st.write(f"- {reason}")
    st.markdown("---")
    st.markdown(f"**Total Score:** {robot_signal['score']}")

st.markdown("---")

# ========== AI SIGNAL LITE (YANG SUDAH ADA) ==========
st.header("🎯 AI SIGNAL LITE - KEPUTUSAN BELI/JUAL")

col_main1, col_main2 = st.columns([2, 1])

with col_main1:
    if ai_score >= 4:
        st.success("## 🚀 STRONG BUY")
        st.progress(100)
    elif ai_score == 3:
        st.info("## 🟡 HOLD / WAIT")
        st.progress(60)
    else:
        st.error("## 🔻 SELL / AVOID")
        st.progress(20)

with col_main2:
    st.metric("AI SCORE", f"{ai_score}/5")
    st.metric("Confidence", f"{ai_signals['confidence']}%")

st.markdown("---")

# ========== DETAIL SIGNAL LITE ==========
st.subheader("📊 Detail Analisis")

col_d1, col_d2, col_d3 = st.columns(3)

with col_d1:
    st.markdown("**🔥 Momentum**")
    try:
        momentum = data['Close'].pct_change(5).iloc[-1] * 100 if len(data) >= 6 else 0
        if momentum > 3:
            st.success(f"Up {momentum:.1f}%")
        elif momentum < -3:
            st.error(f"Down {momentum:.1f}%")
        else:
            st.warning("Sideways")
    except:
        st.info("N/A")
    
    st.markdown("**🚀 Breakout**")
    try:
        if len(data) > 1 and data['Close'].iloc[-1] > data['resistance'].iloc[-2]:
            st.success("BREAKOUT!")
        else:
            st.info("Tidak ada breakout")
    except:
        st.info("N/A")

with col_d2:
    st.markdown("**📈 Multi Timeframe**")
    mtf = get_multi_timeframe_trend(ticker)
    st.write(f"Daily: {mtf['Daily']}")
    st.write(f"Weekly: {mtf['Weekly']}")
    st.write(f"Monthly: {mtf['Monthly']}")

with col_d3:
    st.markdown("**🏦 Smart Money**")
    st.write(f"Status: {ai_signals['smart_status']}")
    st.write(f"Score: {ai_signals['smart_score']}/4")
    st.markdown("**🌍 Macro**")
    st.write(f"Kondisi: {ai_signals['macro_status']}")

st.markdown("---")

# ========== PROBABILITAS ==========
st.subheader("📈 Probabilitas")

try:
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
except:
    bull_prob = 50
    bear_prob = 50

col_p1, col_p2 = st.columns(2)
col_p1.metric("📈 BULLISH", f"{bull_prob:.0f}%")
col_p2.metric("📉 BEARISH", f"{bear_prob:.0f}%")
st.progress(bull_prob/100)

# ========== WEIGHTED DECISION ==========
st.markdown("---")
st.subheader("🧠 Weighted Decision")

weighted_score = weighted_decision_engine(data, volume, ticker)
st.metric("Final Score", f"{weighted_score}%")

if weighted_score >= 70:
    st.success("### ✅ REKOMENDASI: BELI")
    st.progress(weighted_score/100)
elif weighted_score >= 50:
    st.warning("### ⚠️ REKOMENDASI: HOLD")
    st.progress(weighted_score/100)
else:
    st.error("### ❌ REKOMENDASI: JUAL")
    st.progress(weighted_score/100)

# ========== REKOMENDASI ENTRY & STOPLOSS ==========
st.markdown("---")
entry_rec = get_recommended_entry_stoploss(data)
st.markdown("### 🎯 Rekomendasi Entry & Stop Loss")
col_e, col_s, col_t = st.columns(3)
col_e.metric("Entry", f"Rp {entry_rec['entry']:,.2f}" if entry_rec['entry'] else "N/A")
col_s.metric("Stop Loss", f"Rp {entry_rec['stoploss']:,.2f}" if entry_rec['stoploss'] else "N/A", delta_color="inverse")
col_t.metric("Target", f"Rp {entry_rec['target']:,.2f}" if entry_rec['target'] else "N/A")

if entry_rec['entry'] and entry_rec['stoploss'] and entry_rec['target']:
    risk = entry_rec['entry'] - entry_rec['stoploss']
    reward = entry_rec['target'] - entry_rec['entry']
    rr = reward / risk if risk > 0 else 0
    st.info(f"📐 **Risk Reward:** 1 : {rr:.2f}")

# ========== POSITION SIZING ==========
st.markdown("---")
st.subheader("📐 Position Sizing")

col_cap, col_risk = st.columns(2)
with col_cap:
    capital = st.number_input("Modal (Rp)", min_value=0.0, value=100_000_000.0, step=10_000_000.0)
with col_risk:
    risk_percent = st.number_input("Risiko per Trade (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.5)

if entry_rec['stoploss'] and capital > 0 and risk_percent > 0 and entry_rec['entry']:
    risk_amount = capital * (risk_percent / 100)
    price_risk = entry_rec['entry'] - entry_rec['stoploss']
    if price_risk > 0:
        suggested_shares = int(risk_amount / price_risk)
        position_value = suggested_shares * entry_rec['entry']
        st.write(f"**📊 Jumlah saham:** {suggested_shares:,} lembar")
        st.write(f"**💰 Nilai posisi:** Rp {position_value:,.2f} ({position_value/capital*100:.1f}% dari modal)")

# ========== INDIKATOR TEKNIKAL LENGKAP ==========
with st.expander("📊 Indikator Teknikal Lengkap"):
    try:
        col_i1, col_i2 = st.columns(2)
        with col_i1:
            st.metric("RSI (14)", f"{data['RSI'].iloc[-1]:.1f}")
            st.metric("SMA20", f"Rp {data['SMA20'].iloc[-1]:.2f}")
            st.metric("Support", f"Rp {data['support'].iloc[-1]:.2f}")
            st.metric("Volume", f"{volume.iloc[-1]:,.0f}")
        with col_i2:
            st.metric("MACD", f"{data['MACD'].iloc[-1]:.4f}")
            st.metric("MACD Signal", f"{data['MACD_signal'].iloc[-1]:.4f}")
            st.metric("Resistance", f"Rp {data['resistance'].iloc[-1]:.2f}")
            st.metric("Volume MA20", f"{data['Volume_MA'].iloc[-1]:,.0f}")
    except Exception as e:
        st.info("Data indikator tidak tersedia")

# ========== FOOTER ==========
st.markdown("---")
st.caption("⚠️ **DISCLAIMER:** Dashboard ini hanya untuk edukasi dan analisis. Bukan rekomendasi beli/jual. Keputusan investasi sepenuhnya risiko Anda.")

gc.collect()
