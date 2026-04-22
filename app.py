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
        
        # Tambahkan SMA20 dan SMA50
        if len(df) >= 20:
            df['SMA20'] = df['Close'].rolling(20).mean()
        if len(df) >= 50:
            df['SMA50'] = df['Close'].rolling(50).mean()
        
        # Tambahkan RSI
        if len(df) >= 14:
            df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        
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
    """Hitung score untuk 3m, 12m, 30m seperti di gambar"""
    if df.empty:
        return {'3m': 0, '12m': 0, '30m': 0}
    
    close = df['Close']
    
    # 3 bulan (approx 60 hari trading)
    df_3m = df.tail(60)
    score_3m = 0
    # Kondisi 1: Harga naik dari awal
    if df_3m['Close'].iloc[-1] > df_3m['Close'].iloc[0]:
        score_3m += 1
    # Kondisi 2: Harga > SMA20
    if 'SMA20' in df_3m.columns and len(df_3m) >= 20:
        if pd.notna(df_3m['SMA20'].iloc[-1]) and df_3m['Close'].iloc[-1] > df_3m['SMA20'].iloc[-1]:
            score_3m += 1
    # Kondisi 3: RSI > 50
    if 'RSI' in df_3m.columns:
        if pd.notna(df_3m['RSI'].iloc[-1]) and df_3m['RSI'].iloc[-1] > 50:
            score_3m += 1
    score_3m = int(score_3m * 100 / 3) if score_3m > 0 else 0
    
    # 12 bulan (approx 250 hari trading)
    df_12m = df.tail(250) if len(df) >= 250 else df
    score_12m = 0
    if df_12m['Close'].iloc[-1] > df_12m['Close'].iloc[0]:
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
    if df_30m['Close'].iloc[-1] > df_30m['Close'].iloc[0]:
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

def get_signal(indicators: dict, mtf_score: dict) -> dict:
    """Generate signal berdasarkan indikator"""
    
    # Hitung skor
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
    
    # Trend Signal (Harga vs SMA)
    current_price = indicators.get('current_price', 0)
    sma20 = indicators.get('sma20', 0)
    if current_price > sma20:
        score += 1
        reasons.append(f"Harga ({current_price:,.0f}) > SMA20 ({sma20:,.0f}) -> Uptrend")
    else:
        score -= 1
        reasons.append(f"Harga ({current_price:,.0f}) < SMA20 ({sma20:,.0f}) -> Downtrend")
    
    # Volume Signal
    volume_status = indicators.get('volume_status', 'Normal')
    if volume_status == "Tinggi" and score > 0:
        score += 1
        reasons.append("Volume Tinggi mendukung pergerakan")
    elif volume_status == "Rendah":
        reasons.append("Volume Rendah (potensi sideways)")
    else:
        reasons.append(f"Volume {volume_status}")
    
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
    rsi_val = indicators.get('rsi', 'N/A')
    if isinstance(rsi_val, (int, float)):
        if rsi_val < 35:
            st.metric("RSI", f"{rsi_val}", delta="Oversold")
        elif rsi_val > 70:
            st.metric("RSI", f"{rsi_val}", delta="Overbought")
        else:
            st.metric("RSI", f"{rsi_val}", delta="Normal")
    else:
        st.metric("RSI", "N/A")

with col2:
    st.metric("MACD", f"{indicators.get('macd', 0):.2f}",
              delta=f"Hist: {indicators.get('macd_hist', 0):.2f}")

with col3:
    st.metric("Stochastic", f"{indicators.get('stoch_k', 50):.1f}",
              delta=f"D: {indicators.get('stoch_d', 50):.1f}")

with col4:
    vol_status = indicators.get('volume_status', 'Normal')
    vol_current = indicators.get('volume_current', 0)
    vol_avg = indicators.get('volume_avg', 0)
    if vol_status == "Tinggi":
        st.metric("Volume", f"{vol_current:,.0f}", delta=f"Avg: {vol_avg:,.0f}", delta_color="normal")
    elif vol_status == "Rendah":
        st.metric("Volume", f"{vol_current:,.0f}", delta=f"Avg: {vol_avg:,.0f}", delta_color="inverse")
    else:
        st.metric("Volume", f"{vol_current:,.0f}", delta=f"Avg: {vol_avg:,.0f}")

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
    current_price = indicators.get('current_price', 0)
    support = indicators.get('support', 0)
    resistance = indicators.get('resistance', 0)
    
    st.markdown(f"""
    ### Harga Saat Ini
    **Rp {current_price:,.0f}**
    
    ### Level Penting
    - **Support:** Rp {support:,.0f}
    - **Resistance:** Rp {resistance:,.0f}
    """)
    
    # Posisi harga terhadap support/resistance
    if support > 0:
        dist_to_support = ((current_price - support) / support) * 100
        dist_to_resistance = ((resistance - current_price) / current_price) * 100
        st.markdown(f"""
        - Jarak ke Support: {dist_to_support:.1f}%
        - Jarak ke Resistance: {dist_to_resistance:.1f}%
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
    support_val = indicators.get('support', 0)
    current_val = indicators.get('current_price', 1)
    if support_val > 0:
        pct_from_support = (current_val - support_val) / support_val * 100
        st.metric("SUPPORT", f"Rp {support_val:,.0f}",
                  delta=f"{pct_from_support:.1f}% dari harga")
    else:
        st.metric("SUPPORT", "N/A")

with col_res:
    resistance_val = indicators.get('resistance', 0)
    current_val = indicators.get('current_price', 1)
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
    
    st.markdown("---")
    st.markdown(f"**Total Score:** {signal['score']}")

# ========== FOOTER ==========
st.markdown("---")
st.caption("⚠️ **DISCLAIMER:** Robot ini hanya untuk edukasi dan analisis otomatis. Bukan rekomendasi beli/jual. Keputusan investasi sepenuhnya risiko Anda.")

# ========== CSS CUSTOM ==========
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    .stProgress > div > div {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)
