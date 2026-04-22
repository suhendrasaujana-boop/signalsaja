import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ========== KONFIGURASI HALAMAN ==========
st.set_page_config(
    layout="wide",
    page_title="Robot Saham - AI Trading Signal",
    page_icon="🤖",
    menu_items={'Get Help': None, 'Report a bug': None, 'About': None}
)

# ========== KONSTANTA ==========
DATA_PERIOD = "3mo"
CACHE_TTL = 300
DEFAULT_TICKER = "BBCA.JK"

# ========== SESSION STATE ==========
if 'last_ticker' not in st.session_state:
    st.session_state.last_ticker = DEFAULT_TICKER

# ========== FUNGSI BANTU ==========
def fix_ticker(ticker: str) -> str:
    ticker = ticker.strip().upper()
    if ticker.startswith('^'):
        return ticker
    if ticker.endswith('.JK'):
        return ticker
    return ticker + '.JK'

@st.cache_data(ttl=CACHE_TTL)
def load_data(ticker: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=DATA_PERIOD, interval="1d", progress=False, auto_adjust=False)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(how='all')
        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].fillna(0)
        return df
    except Exception as e:
        st.error(f"Error: {str(e)[:100]}")
        return pd.DataFrame()

def calculate_indicators(df: pd.DataFrame) -> dict:
    """Hitung semua indikator seperti di gambar"""
    if df.empty or len(df) < 30:
        return {}
    
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume'] if 'Volume' in df.columns else pd.Series(0, index=df.index)
    
    # RSI
    rsi = ta.momentum.rsi(close, window=14).iloc[-1] if len(close) >= 14 else 50
    
    # MACD
    macd_line = ta.trend.MACD(close).macd().iloc[-1] if len(close) >= 26 else 0
    macd_signal = ta.trend.MACD(close).macd_signal().iloc[-1] if len(close) >= 26 else 0
    macd_hist = macd_line - macd_signal
    
    # Stochastic
    stoch_k = ta.momentum.stoch(high, low, close, window=14, smooth_window=3).iloc[-1] if len(close) >= 14 else 50
    stoch_d = ta.momentum.stoch_signal(high, low, close, window=14, smooth_window=3).iloc[-1] if len(close) >= 14 else 50
    
    # Volume
    volume_avg = volume.rolling(20).mean().iloc[-1] if volume.sum() > 0 else 1
    volume_current = volume.iloc[-1] if volume.sum() > 0 else 0
    volume_status = "Rendah" if volume_current < volume_avg else "Tinggi" if volume_current > volume_avg * 1.5 else "Normal"
    
    # SMA
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    
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
    """Hitung score untuk 3m, 12m, 30m seperti di gambar"""
    if df.empty:
        return {'3m': 0, '12m': 0, '30m': 0}
    
    close = df['Close']
    
    # 3 bulan (approx 60 hari trading)
    df_3m = df.tail(60)
    score_3m = 1 if df_3m['Close'].iloc[-1] > df_3m['Close'].iloc[0] else 0
    score_3m += 1 if df_3m['Close'].iloc[-1] > df_3m['SMA20'].iloc[-1] if len(df_3m) >= 20 else 0 else 0
    score_3m += 1 if df_3m['RSI'].iloc[-1] > 50 if 'RSI' in df_3m.columns else 0 else 0
    score_3m = int(score_3m * 100 / 3) if score_3m > 0 else 0
    
    # 12 bulan (approx 250 hari trading)
    df_12m = df.tail(250) if len(df) >= 250 else df
    score_12m = 1 if df_12m['Close'].iloc[-1] > df_12m['Close'].iloc[0] else 0
    score_12m += 1 if df_12m['Close'].iloc[-1] > df_12m['SMA50'].iloc[-1] if len(df_12m) >= 50 else 0 else 0
    score_12m += 1 if df_12m['RSI'].iloc[-1] > 50 if 'RSI' in df_12m.columns else 0 else 0
    score_12m = int(score_12m * 100 / 3) if score_12m > 0 else 0
    
    # 30 bulan (approx 630 hari trading)
    df_30m = df.tail(630) if len(df) >= 630 else df
    score_30m = 1 if df_30m['Close'].iloc[-1] > df_30m['Close'].iloc[0] else 0
    score_30m += 1 if df_30m['Close'].iloc[-1] > df_30m['SMA50'].iloc[-1] if len(df_30m) >= 50 else 0 else 0
    score_30m += 1 if df_30m['RSI'].iloc[-1] > 50 if 'RSI' in df_30m.columns else 0 else 0
    score_30m = int(score_30m * 100 / 3) if score_30m > 0 else 0
    
    return {
        '3m': score_3m,
        '12m': score_12m,
        '30m': score_30m
    }

def get_signal(indicators: dict, mtf_score: dict) -> dict:
    """Generate signal berdasarkan indikator"""
    
    # Hitung skor
    score = 0
    reasons = []
    
    # RSI Signal
    if indicators.get('rsi', 50) < 35:
        score += 2
        reasons.append("RSI Oversold (<35)")
    elif indicators.get('rsi', 50) > 70:
        score -= 2
        reasons.append("RSI Overbought (>70)")
    else:
        reasons.append(f"RSI Normal ({indicators.get('rsi', 50)})")
    
    # MACD Signal
    macd_hist = indicators.get('macd_hist', 0)
    if macd_hist > 0:
        score += 1
        reasons.append("MACD Positive (Bullish)")
    else:
        score -= 1
        reasons.append("MACD Negative (Bearish)")
    
    # Stochastic Signal
    stoch_k = indicators.get('stoch_k', 50)
    if stoch_k < 20:
        score += 2
        reasons.append("Stochastic Oversold")
    elif stoch_k > 80:
        score -= 2
        reasons.append("Stochastic Overbought")
    
    # Multi Timeframe Signal
    mtf_avg = (mtf_score.get('3m', 0) + mtf_score.get('12m', 0) + mtf_score.get('30m', 0)) / 3
    if mtf_avg > 60:
        score += 2
        reasons.append("Multi Timeframe Bullish")
    elif mtf_avg < 40:
        score -= 2
        reasons.append("Multi Timeframe Bearish")
    
    # Trend Signal (Harga vs SMA)
    current_price = indicators.get('current_price', 0)
    sma20 = indicators.get('sma20', 0)
    if current_price > sma20:
        score += 1
        reasons.append("Harga > SMA20 (Uptrend)")
    else:
        score -= 1
        reasons.append("Harga < SMA20 (Downtrend)")
    
    # Volume Signal
    if indicators.get('volume_status') == "Tinggi":
        if score > 0:
            score += 1
            reasons.append("Volume Tinggi mendukung")
    elif indicators.get('volume_status') == "Rendah":
        reasons.append("Volume Rendah (sideways)")
    
    # Tentukan signal
    if score >= 3:
        signal = "STRONG BUY"
        color = "success"
        confidence = "HIGH CONFIDENCE"
        confidence_score = 80 + (score * 5)
    elif score >= 1:
        signal = "BUY"
        color = "info"
        confidence = "MEDIUM CONFIDENCE"
        confidence_score = 60 + (score * 5)
    elif score <= -3:
        signal = "STRONG SELL"
        color = "error"
        confidence = "LOW CONFIDENCE"
        confidence_score = 20 + (abs(score) * 5)
    elif score <= -1:
        signal = "SELL"
        color = "warning"
        confidence = "LOW CONFIDENCE"
        confidence_score = 30 + (abs(score) * 5)
    else:
        signal = "WAIT / HOLD"
        color = "warning"
        confidence = "NEUTRAL"
        confidence_score = 50
    
    confidence_score = min(100, max(0, confidence_score))
    
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
    adx = ta.trend.ADXIndicator(df['High'], df['Low'], close, window=14)
    adx_value = adx.adx().iloc[-1] if len(close) >= 14 else 25
    
    if adx_value > 40:
        return "STRONG TREND"
    elif adx_value > 25:
        return "MODERATE TREND"
    elif adx_value > 20:
        return "WEAK TREND"
    else:
        return "SIDEWAYS"

# ========== TAMPILAN UTAMA ==========
st.title("🤖 Robot Saham - AI Trading Signal")
st.caption("Sistem Analisis Otomatis untuk Keputusan Trading")

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("## ⚙️ Pengaturan")
    ticker_input = st.text_input("Kode Saham", DEFAULT_TICKER, 
                                 help="Contoh: BBCA.JK, BBRI.JK, ASII.JK")
    ticker = fix_ticker(ticker_input)
    
    if ticker != ticker_input:
        st.info(f"Format: {ticker}")
    
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📊 Indikator yang digunakan")
    st.markdown("""
    - RSI (14)
    - MACD
    - Stochastic
    - Volume Analysis
    - Multi Timeframe
    - Support & Resistance
    """)
    st.markdown("---")
    st.caption("⚠️ Hanya untuk edukasi")

# ========== LOAD DATA ==========
with st.spinner(f"Memuat data {ticker}..."):
    df = load_data(ticker)

if df.empty or len(df) < 20:
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

# Tambahkan RSI ke df untuk multi timeframe
if len(df) >= 14:
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

# Hitung semua indikator
indicators = calculate_indicators(df)
mtf_scores = calculate_multi_timeframe_score(df)
signal = get_signal(indicators, mtf_scores)
trend_strength = get_trend_strength(df)

# ========== HEADER: ROBOT SAHAM ==========
st.markdown("---")

# ========== BARIS 1: INDICATOR ==========
st.markdown("## 📊 INDICATOR")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("RSI", f"{indicators.get('rsi', 'N/A')}", 
              delta="Oversold" if indicators.get('rsi', 50) < 35 else "Overbought" if indicators.get('rsi', 50) > 70 else "Normal")

with col2:
    st.metric("MACD", f"{indicators.get('macd', 0):.2f}",
              delta=f"Hist: {indicators.get('macd_hist', 0):.2f}")

with col3:
    st.metric("Stochastic", f"{indicators.get('stoch_k', 50):.1f}",
              delta=f"D: {indicators.get('stoch_d', 50):.1f}")

with col4:
    vol_status = indicators.get('volume_status', 'Normal')
    vol_delta = f"Avg: {indicators.get('volume_avg', 0):,.0f}"
    if vol_status == "Tinggi":
        st.metric("Volume", f"{indicators.get('volume_current', 0):,.0f}", delta=vol_delta, delta_color="normal")
    elif vol_status == "Rendah":
        st.metric("Volume", f"{indicators.get('volume_current', 0):,.0f}", delta=vol_delta, delta_color="inverse")
    else:
        st.metric("Volume", f"{indicators.get('volume_current', 0):,.0f}", delta=vol_delta)

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

# Tampilkan signal besar di tengah
col_s1, col_s2, col_s3 = st.columns([1, 2, 1])

with col_s2:
    if signal['color'] == "success":
        st.success(f"## {signal['signal']}")
    elif signal['color'] == "error":
        st.error(f"## {signal['signal']}")
    else:
        st.warning(f"## {signal['signal']}")
    
    st.metric("Keyakinan", signal['confidence'])
    st.progress(signal['confidence_score']/100)

st.markdown("---")

# ========== BARIS 4: REKOMENDASI DETAIL ==========
st.markdown("## 📝 REKOMENDASI")

col_r1, col_r2 = st.columns(2)

with col_r1:
    st.markdown(f"""
    ### Harga Saat Ini
    **Rp {indicators.get('current_price', 0):,.0f}**
    
    ### Trend
    - **Trend Strength:** {trend_strength}
    - **Support:** Rp {indicators.get('support', 0):,.0f}
    - **Resistance:** Rp {indicators.get('resistance', 0):,.0f}
    """)

with col_r2:
    st.markdown(f"""
    ### Range Harga (30 Hari)
    - **Tertinggi:** Rp {indicators.get('high_30d', 0):,.0f}
    - **Terendah:** Rp {indicators.get('low_30d', 0):,.0f}
    - **Rata-rata:** Rp {indicators.get('mid_30d', 0):,.0f}
    
    ### Moving Average
    - **SMA20:** Rp {indicators.get('sma20', 0):,.0f}
    - **SMA50:** Rp {indicators.get('sma50', 0):,.0f}
    """)

st.markdown("---")

# ========== BARIS 5: SUPPORT & RESISTANCE ==========
st.markdown("## 📐 SUPPORT & RESISTANCE")

col_sup, col_res = st.columns(2)

with col_sup:
    st.metric("SUPPORT", f"Rp {indicators.get('support', 0):,.0f}",
              delta=f"-{(indicators.get('current_price', 0) - indicators.get('support', 0)) / indicators.get('support', 1) * 100:.1f}% dari harga")

with col_res:
    st.metric("RESISTANCE", f"Rp {indicators.get('resistance', 0):,.0f}",
              delta=f"+{(indicators.get('resistance', 0) - indicators.get('current_price', 0)) / indicators.get('current_price', 1) * 100:.1f}% dari harga")

st.markdown("---")

# ========== BARIS 6: TREND STRENGTH ==========
st.markdown("## 📈 TREND STRENGTH")

# Visualisasi trend strength
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

if signal['signal'] in ["STRONG BUY", "BUY"]:
    st.success(f"# ✅ {signal['signal']}")
    st.balloons()
elif signal['signal'] in ["STRONG SELL", "SELL"]:
    st.error(f"# ❌ {signal['signal']}")
else:
    st.warning(f"# ⏳ {signal['signal']}")

# Tampilkan alasan
with st.expander("📋 Detail Analisis & Alasan"):
    for reason in signal['reasons']:
        st.write(f"- {reason}")

# ========== FOOTER ==========
st.markdown("---")
st.caption("⚠️ **DISCLAIMER:** Robot ini hanya untuk edukasi dan analisis otomatis. Bukan rekomendasi beli/jual. Keputusan investasi sepenuhnya risiko Anda.")

# ========== SCRIPT UNTUK AUTO REFRESH (OPSIONAL) ==========
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)
