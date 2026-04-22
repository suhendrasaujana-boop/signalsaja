import streamlit as st
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
if 'trading_mode' not in st.session_state:
    st.session_state.trading_mode = "Swing (Daily)"

# ========== FUNGSI BANTU ==========
def fix_ticker(ticker: str) -> str:
    ticker = ticker.strip().upper()
    if ticker.startswith('^'):
        return ticker
    if ticker.endswith('.JK'):
        return ticker
    return ticker + '.JK'

def has_valid_volume(df: pd.DataFrame) -> bool:
    """Cek apakah data volume valid"""
    if 'Volume' not in df.columns:
        return False
    if df['Volume'].sum() == 0:
        return False
    if df['Volume'].iloc[-1] == 0:
        return False
    return True

def get_data_period_by_mode(trading_mode: str) -> str:
    """Data period berdasarkan mode trading"""
    if trading_mode in ["Position (Weekly - Monthly)", "Swing (Daily)"]:
        return "2y"
    else:
        return "6mo"

@st.cache_data(ttl=CACHE_TTL)
def load_data(ticker: str, timeframe: str, trading_mode: str) -> pd.DataFrame:
    try:
        period = get_data_period_by_mode(trading_mode)
        df = yf.download(ticker, period=period, interval=timeframe, progress=False, auto_adjust=False)
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

# ========== SINGLE SOURCE OF TRUTH - SEMUA INDIKATOR DIHITUNG SEKALI ==========
@st.cache_data(ttl=CACHE_TTL)
def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Single source of truth untuk semua indicator"""
    if df.empty or len(df) < 30:
        return df
    
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume'] if has_valid_volume(df) else pd.Series(0, index=df.index)
    has_volume = has_valid_volume(df)
    
    # Moving Averages
    df['SMA20'] = close.rolling(20, min_periods=1).mean()
    df['SMA50'] = close.rolling(50, min_periods=1).mean()
    df['EMA200'] = close.rolling(200).mean() if len(close) >= 200 else close
    
    # RSI (hitung sekali)
    if len(close) >= 14:
        df['RSI'] = ta.momentum.rsi(close, window=14)
    else:
        df['RSI'] = 50.0
    
    # MACD (hitung sekali)
    if len(close) >= 26:
        macd = ta.trend.MACD(close)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    else:
        df['MACD'] = 0.0
        df['MACD_Signal'] = 0.0
        df['MACD_Hist'] = 0.0
    
    # Stochastic (hitung sekali)
    if len(close) >= 14:
        df['Stoch_K'] = ta.momentum.stoch(high, low, close, window=14, smooth_window=3)
        df['Stoch_D'] = ta.momentum.stoch_signal(high, low, close, window=14, smooth_window=3)
    else:
        df['Stoch_K'] = 50.0
        df['Stoch_D'] = 50.0
    
    # ATR untuk validasi breakout
    if len(close) >= 14:
        df['ATR'] = ta.volatility.average_true_range(high, low, close, window=14)
    else:
        df['ATR'] = close * 0.02
    
    # Volume MA
    if has_volume:
        df['Volume_MA'] = volume.rolling(20, min_periods=1).mean()
    else:
        df['Volume_MA'] = 0.0
    
    # Support & Resistance
    df['support'] = df['Low'].rolling(20, min_periods=1).min()
    df['resistance'] = df['High'].rolling(20, min_periods=1).max()
    
    # AD dan CMF (hanya jika volume valid)
    if has_volume:
        try:
            df['AD'] = ta.volume.acc_dist_index(df['High'], df['Low'], df['Close'], df['Volume'], fillna=True)
        except:
            df['AD'] = 0.0
        try:
            df['CMF'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'], window=20, fillna=True)
        except:
            df['CMF'] = 0.0
    else:
        df['AD'] = 0.0
        df['CMF'] = 0.0
    
    df = df.ffill().fillna(0)
    return df

# ========== FUNGSI UNTUK ROBOT SAHAM ==========
def calculate_robot_indicators(df: pd.DataFrame) -> dict:
    """Hitung semua indikator dari single source"""
    if df.empty or len(df) < 30:
        return {}
    
    close = df['Close']
    volume = df['Volume'] if has_valid_volume(df) else pd.Series(0, index=df.index)
    has_volume = has_valid_volume(df)
    
    # Volume status
    if has_volume and df['Volume_MA'].iloc[-1] > 0:
        volume_current = volume.iloc[-1]
        volume_avg = df['Volume_MA'].iloc[-1]
        if volume_current < volume_avg:
            volume_status = "Rendah"
        elif volume_current > volume_avg * 1.5:
            volume_status = "Tinggi"
        else:
            volume_status = "Normal"
    else:
        volume_status = "No Data"
        volume_current = 0
        volume_avg = 0
    
    # Harga tertinggi, terendah, sedang
    high_30d = df['High'].tail(30).max()
    low_30d = df['Low'].tail(30).min()
    mid_30d = (high_30d + low_30d) / 2
    
    return {
        'rsi': round(df['RSI'].iloc[-1], 1) if 'RSI' in df.columns else 50,
        'macd': round(df['MACD'].iloc[-1], 2) if 'MACD' in df.columns else 0,
        'macd_signal': round(df['MACD_Signal'].iloc[-1], 2) if 'MACD_Signal' in df.columns else 0,
        'macd_hist': round(df['MACD_Hist'].iloc[-1], 2) if 'MACD_Hist' in df.columns else 0,
        'stoch_k': round(df['Stoch_K'].iloc[-1], 1) if 'Stoch_K' in df.columns else 50,
        'stoch_d': round(df['Stoch_D'].iloc[-1], 1) if 'Stoch_D' in df.columns else 50,
        'volume_current': volume_current,
        'volume_avg': volume_avg,
        'volume_status': volume_status,
        'sma20': round(df['SMA20'].iloc[-1], 2),
        'sma50': round(df['SMA50'].iloc[-1], 2),
        'ema200': round(df['EMA200'].iloc[-1], 2) if 'EMA200' in df.columns else close.iloc[-1],
        'atr': round(df['ATR'].iloc[-1], 2) if 'ATR' in df.columns else close.iloc[-1] * 0.02,
        'support': round(df['support'].iloc[-1], 2),
        'resistance': round(df['resistance'].iloc[-1], 2),
        'high_30d': round(high_30d, 2),
        'low_30d': round(low_30d, 2),
        'mid_30d': round(mid_30d, 2),
        'current_price': round(close.iloc[-1], 2)
    }

def calculate_true_mtf_score(df_daily: pd.DataFrame, df_weekly: pd.DataFrame, df_monthly: pd.DataFrame) -> dict:
    """True Multi-Timeframe Analysis dengan trend alignment"""
    
    def get_trend_direction(df, period):
        if df.empty or len(df) < 20:
            return 0
        close = df['Close']
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        
        if close.iloc[-1] > sma20.iloc[-1] and sma20.iloc[-1] > sma50.iloc[-1]:
            return 1  # Bullish alignment
        elif close.iloc[-1] < sma20.iloc[-1] and sma20.iloc[-1] < sma50.iloc[-1]:
            return -1  # Bearish alignment
        return 0  # Mixed
    
    daily_trend = get_trend_direction(df_daily, "Daily")
    weekly_trend = get_trend_direction(df_weekly, "Weekly")
    monthly_trend = get_trend_direction(df_monthly, "Monthly")
    
    # Trend alignment score
    alignment_score = (daily_trend + weekly_trend + monthly_trend) / 3
    
    if alignment_score >= 0.66:
        mtf_signal = "STRONG BULLISH ALIGNMENT"
        mtf_score = 80
    elif alignment_score >= 0.33:
        mtf_signal = "BULLISH"
        mtf_score = 60
    elif alignment_score <= -0.66:
        mtf_signal = "STRONG BEARISH ALIGNMENT"
        mtf_score = 20
    elif alignment_score <= -0.33:
        mtf_signal = "BEARISH"
        mtf_score = 40
    else:
        mtf_signal = "MIXED (Wait)"
        mtf_score = 50
    
    return {
        'daily': daily_trend,
        'weekly': weekly_trend,
        'monthly': monthly_trend,
        'alignment': alignment_score,
        'signal': mtf_signal,
        'score': mtf_score
    }

def detect_true_breakout(df: pd.DataFrame, atr: float) -> dict:
    """Deteksi breakout dengan validasi ATR - mengurangi false positive"""
    if len(df) < 20:
        return {'is_breakout': False, 'strength': 0, 'message': "Data tidak cukup"}
    
    current_price = df['Close'].iloc[-1]
    resistance = df['resistance'].iloc[-1]
    
    # True breakout: harga > resistance + (0.5 * ATR)
    breakout_threshold = resistance + (atr * 0.5)
    
    if current_price > breakout_threshold:
        strength = (current_price - resistance) / atr
        return {
            'is_breakout': True,
            'strength': round(strength, 2),
            'message': f"✅ TRUE BREAKOUT! Strength: {strength:.1f}x ATR"
        }
    elif current_price > resistance:
        return {
            'is_breakout': False,
            'strength': 0,
            'message': "⚠️ False breakout (belum melewati ATR buffer)"
        }
    else:
        return {
            'is_breakout': False,
            'strength': 0,
            'message': "Tidak ada breakout"
        }

def get_robot_signal_weighted(indicators: dict, mtf_score: dict, trend_strength: str) -> dict:
    """Weighted signal dengan prioritas: Trend > Volume > Momentum > Oscillator"""
    
    # Inisialisasi komponen skor
    trend_score = 0
    volume_score = 0
    momentum_score = 0
    oscillator_score = 0
    
    reasons = []
    
    # 1. TREND (Bobot tertinggi - 40%)
    current_price = indicators.get('current_price', 0)
    sma20 = indicators.get('sma20', 0)
    sma50 = indicators.get('sma50', 0)
    ema200 = indicators.get('ema200', current_price)
    
    if current_price > ema200:
        trend_score += 2
        reasons.append("✅ Harga > EMA200 (Big Trend Bullish)")
    elif current_price < ema200:
        trend_score -= 2
        reasons.append("❌ Harga < EMA200 (Big Trend Bearish)")
    
    if current_price > sma20 and sma20 > sma50:
        trend_score += 1
        reasons.append("✅ Golden Cross (Uptrend)")
    elif current_price < sma20 and sma20 < sma50:
        trend_score -= 1
        reasons.append("❌ Death Cross (Downtrend)")
    
    if trend_strength == "STRONG TREND":
        trend_score += 1
        reasons.append("✅ Trend Strength: STRONG")
    elif trend_strength == "WEAK TREND":
        trend_score -= 1
        reasons.append("⚠️ Trend Strength: WEAK")
    
    # MTF Alignment
    mtf_align = mtf_score.get('alignment', 0)
    if mtf_align > 0.5:
        trend_score += 1
        reasons.append("✅ MTF Bullish Alignment")
    elif mtf_align < -0.5:
        trend_score -= 1
        reasons.append("❌ MTF Bearish Alignment")
    
    # 2. VOLUME (Bobot kedua - 25%)
    volume_status = indicators.get('volume_status', 'Normal')
    if volume_status == "Tinggi":
        volume_score += 1
        reasons.append("✅ Volume Tinggi (Konfirmasi Pergerakan)")
    elif volume_status == "Rendah":
        volume_score -= 1
        reasons.append("⚠️ Volume Rendah (Potensi Sideways)")
    elif volume_status == "No Data":
        volume_score = 0
        reasons.append("ℹ️ Tidak ada data volume")
    
    # 3. MOMENTUM (Bobot ketiga - 20%)
    rsi_val = indicators.get('rsi', 50)
    if 30 <= rsi_val <= 50:
        momentum_score += 1
        reasons.append(f"✅ RSI {rsi_val} (Oversold Recovery Zone)")
    elif 50 < rsi_val <= 65:
        momentum_score += 0.5
        reasons.append(f"✅ RSI {rsi_val} (Bullish Momentum)")
    elif rsi_val > 70:
        momentum_score -= 1
        reasons.append(f"⚠️ RSI {rsi_val} (Overbought - Hati-hati)")
    elif rsi_val < 30:
        momentum_score -= 0.5
        reasons.append(f"⚠️ RSI {rsi_val} (Extreme Oversold)")
    else:
        reasons.append(f"RSI {rsi_val} (Netral)")
    
    # 4. OSCILLATOR (Bobot terendah - 15%)
    macd_hist = indicators.get('macd_hist', 0)
    if macd_hist > 0:
        oscillator_score += 1
        reasons.append(f"✅ MACD Positif ({macd_hist:.2f})")
    else:
        oscillator_score -= 1
        reasons.append(f"❌ MACD Negatif ({macd_hist:.2f})")
    
    stoch_k = indicators.get('stoch_k', 50)
    if stoch_k < 20:
        oscillator_score += 1
        reasons.append(f"✅ Stochastic Oversold ({stoch_k:.0f})")
    elif stoch_k > 80:
        oscillator_score -= 1
        reasons.append(f"⚠️ Stochastic Overbought ({stoch_k:.0f})")
    
    # Hitung final score dengan bobot yang benar
    # Normalisasi setiap komponen ke skala -10 sampai +10
    normalized_trend = trend_score * 2.5  # Max 10
    normalized_volume = volume_score * 2.5  # Max 10
    normalized_momentum = momentum_score * 2.5  # Max 10
    normalized_oscillator = oscillator_score * 2.5  # Max 10
    
    final_score = (normalized_trend * 0.40 + normalized_volume * 0.25 + 
                   normalized_momentum * 0.20 + normalized_oscillator * 0.15)
    
    # Clip ke range -10 sampai +10
    final_score = max(-10, min(10, final_score))
    
    # Tentukan signal
    if final_score >= 6:
        signal = "STRONG BUY"
        confidence = "HIGH CONFIDENCE"
        confidence_score = 85
        color = "success"
    elif final_score >= 3:
        signal = "BUY"
        confidence = "MEDIUM CONFIDENCE"
        confidence_score = 65
        color = "info"
    elif final_score <= -6:
        signal = "STRONG SELL"
        confidence = "HIGH CONFIDENCE"
        confidence_score = 85
        color = "error"
    elif final_score <= -3:
        signal = "SELL"
        confidence = "LOW CONFIDENCE"
        confidence_score = 35
        color = "warning"
    else:
        signal = "WAIT / HOLD"
        confidence = "LOW CONFIDENCE"
        confidence_score = 50
        color = "warning"
    
    return {
        'signal': signal,
        'color': color,
        'score': round(final_score, 1),
        'trend_component': round(normalized_trend, 1),
        'volume_component': round(normalized_volume, 1),
        'momentum_component': round(normalized_momentum, 1),
        'oscillator_component': round(normalized_oscillator, 1),
        'confidence': confidence,
        'confidence_score': confidence_score,
        'reasons': reasons
    }

def get_trend_strength(df: pd.DataFrame) -> str:
    """Hitung kekuatan trend dengan ADX"""
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
def calculate_ai_score(df: pd.DataFrame) -> int:
    """AI Score berdasarkan kondisi teknikal (tanpa double count)"""
    if df.empty:
        return 0
    try:
        conditions = [
            df['RSI'].iloc[-1] < 35,
            df['Close'].iloc[-1] > df['SMA20'].iloc[-1],
            df['SMA20'].iloc[-1] > df['SMA50'].iloc[-1],
            df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1],
            df['Volume'].iloc[-1] > df['Volume_MA'].iloc[-1] if has_valid_volume(df) else False
        ]
        return sum(conditions)
    except:
        return 2

def get_multi_timeframe_trend(ticker, trading_mode):
    """Multi timeframe trend untuk display"""
    try:
        daily = load_data(ticker, "1d", trading_mode)
        weekly = load_data(ticker, "1wk", trading_mode)
        monthly = load_data(ticker, "1mo", trading_mode)

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
    """Smart money detection - hanya jika volume valid"""
    if not has_valid_volume(df) or len(df) < 20:
        return 0, "NO_VOLUME_DATA"
    
    try:
        score = 0
        if df['CMF'].iloc[-1] > 0:
            score += 1
        if df['AD'].iloc[-1] > df['AD'].iloc[-5]:
            score += 1
        if df['Volume'].iloc[-1] > df['Volume_MA'].iloc[-1] * 1.5:
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

def get_all_signals(df, ticker):
    """Gabungan semua sinyal"""
    try:
        ai_score = calculate_ai_score(df)
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

def weighted_decision_engine(df, ticker):
    """Weighted decision dengan raw technical score"""
    try:
        # Raw technical score (bukan dari AI Score)
        tech_score = 0
        if df['RSI'].iloc[-1] < 35:
            tech_score += 1
        elif df['RSI'].iloc[-1] > 70:
            tech_score -= 1
        
        if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]:
            tech_score += 1
        else:
            tech_score -= 1
        
        if df['Close'].iloc[-1] > df['SMA20'].iloc[-1]:
            tech_score += 1
        else:
            tech_score -= 1
        
        if has_valid_volume(df) and df['Volume'].iloc[-1] > df['Volume_MA'].iloc[-1]:
            tech_score += 1
        
        # Normalize tech_score ke 0-1
        tech_score = (tech_score + 3) / 6
        
        signals = get_all_signals(df, ticker)
        smart_map = {"Accumulation": 1, "Neutral": 0.5, "Distribution": 0, "NO_VOLUME_DATA": 0.5}
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
        
        final_score = (0.30 * tech_score + 0.20 * smart_score + 0.15 * macro_score + 
                       0.15 * risk_score + 0.20 * momentum_score)
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

def get_recommended_entry_stoploss(df: pd.DataFrame, atr: float) -> dict:
    if df.empty:
        return {'entry': None, 'stoploss': None, 'target': None}
    try:
        price = df['Close'].iloc[-1]
        support = df['support'].iloc[-1] if pd.notna(df['support'].iloc[-1]) else price * 0.95
        resistance = df['resistance'].iloc[-1] if pd.notna(df['resistance'].iloc[-1]) else price * 1.05
        
        # Gunakan ATR untuk stop loss yang lebih realistis
        atr_stop = atr * 1.5 if atr > 0 else price * 0.03
        
        entry = (support + price) / 2
        stoploss = min(support * 0.97, entry - atr_stop)
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
    timeframe = st.selectbox("Timeframe Chart", list(TIMEFRAMES.keys()))
    
    st.markdown("---")
    st.markdown("### 🧭 Mode Trading (PENTING!)")
    
    trading_mode = st.selectbox(
        "Pilih gaya trading Anda",
        ["Scalping (5-15 menit)", "Intraday (30 menit - 4 jam)", "Swing (Daily)", "Position (Weekly - Monthly)"],
        index=2,
        help="Pilih sesuai gaya trading Anda. Signal akan disesuaikan dengan mode ini."
    )
    st.session_state.trading_mode = trading_mode
    
    # Tampilkan penjelasan singkat per mode
    if trading_mode == "Scalping (5-15 menit)":
        st.caption("⚡ Fokus: Volume + Momentum cepat")
        st.caption("⚠️ Noise tinggi, butuh eksekusi cepat")
    elif trading_mode == "Intraday (30 menit - 4 jam)":
        st.caption("📊 Fokus: Breakout + Trend harian")
        st.caption("🎯 Cocok untuk trader aktif")
    elif trading_mode == "Swing (Daily)":
        st.caption("📈 Fokus: SMA + RSI + Trend stabil")
        st.caption("🕐 Hold 1-5 hari")
    else:
        st.caption("🧭 Fokus: Trend besar + Akumulasi")
        st.caption("📅 Hold mingguan - bulanan")
    
    st.markdown("---")
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
    data_raw = load_data(ticker, TIMEFRAMES[timeframe], trading_mode)

if data_raw.empty or len(data_raw) < 10:
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

# Hitung semua indikator SEKALI (single source of truth)
data = calculate_all_indicators(data_raw)

if data.empty:
    st.error("Gagal memproses data")
    st.stop()

# Ambil data terkini
try:
    last_close = data['Close'].iloc[-1]
    prev_close = data['Close'].iloc[-2] if len(data) > 1 else last_close
    change_pct = ((last_close - prev_close) / prev_close) * 100 if prev_close != 0 else 0
    has_vol = has_valid_volume(data)
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# Hitung semua komponen
robot_indicators = calculate_robot_indicators(data)
trend_strength = get_trend_strength(data)

# Load data untuk MTF
df_daily = load_data(ticker, "1d", trading_mode)
df_weekly = load_data(ticker, "1wk", trading_mode)
df_monthly = load_data(ticker, "1mo", trading_mode)

# Hitung MTF dengan fungsi yang sudah diperbaiki
mtf_analysis = calculate_true_mtf_score(df_daily, df_weekly, df_monthly)

# Hitung sinyal dengan bobot yang benar
robot_signal = get_robot_signal_weighted(robot_indicators, mtf_analysis, trend_strength)

# Hitung breakout dengan validasi ATR
breakout_analysis = detect_true_breakout(data, robot_indicators.get('atr', last_close * 0.02))

# Hitung sinyal AI
ai_signals = get_all_signals(data, ticker)
ai_score = ai_signals['ai_score']

# ========== HEADER ==========
st.title(f"🤖 {ticker}")
st.caption(f"Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Mode: {trading_mode}")

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
    st.metric("Daily Trend", "Bullish" if mtf_analysis['daily'] > 0 else "Bearish" if mtf_analysis['daily'] < 0 else "Mixed")
with col_m2:
    st.metric("Weekly Trend", "Bullish" if mtf_analysis['weekly'] > 0 else "Bearish" if mtf_analysis['weekly'] < 0 else "Mixed")
with col_m3:
    st.metric("Monthly Trend", "Bullish" if mtf_analysis['monthly'] > 0 else "Bearish" if mtf_analysis['monthly'] < 0 else "Mixed")

st.caption(f"📊 MTF Alignment: {mtf_analysis['signal']} (Score: {mtf_analysis['score']})")

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

# ========== BREAKOUT DETECTOR (DENGAN VALIDASI ATR) ==========
st.markdown("## 🚀 BREAKOUT DETECTOR")

if breakout_analysis['is_breakout']:
    st.success(breakout_analysis['message'])
else:
    st.info(breakout_analysis['message'])

st.markdown("---")

# ========== REKOMENDASI DETAIL ==========
st.markdown("## 📝 REKOMENDASI")

col_r1, col_r2 = st.columns(2)

with col_r1:
    current_price = robot_indicators.get('current_price', 0)
    support = robot_indicators.get('support', 0)
    resistance = robot_indicators.get('resistance', 0)
    ema200 = robot_indicators.get('ema200', 0)
    
    st.markdown(f"""
    ### Harga Saat Ini
    **Rp {current_price:,.0f}**
    
    ### Level Penting
    - **Support:** Rp {support:,.0f}
    - **Resistance:** Rp {resistance:,.0f}
    - **EMA200:** Rp {ema200:,.0f}
    """)
    
    if current_price > ema200:
        st.success("✅ Harga di atas EMA200 (Bullish Structure)")
    else:
        st.error("❌ Harga di bawah EMA200 (Bearish Structure)")

with col_r2:
    st.markdown(f"""
    ### Range Harga (30 Hari)
    - **Tertinggi:** Rp {robot_indicators.get('high_30d', 0):,.0f}
    - **Terendah:** Rp {robot_indicators.get('low_30d', 0):,.0f}
    - **Rata-rata:** Rp {robot_indicators.get('mid_30d', 0):,.0f}
    
    ### Moving Average
    - **SMA20:** Rp {robot_indicators.get('sma20', 0):,.0f}
    - **SMA50:** Rp {robot_indicators.get('sma50', 0):,.0f}
    - **ATR:** Rp {robot_indicators.get('atr', 0):,.0f}
    """)

st.markdown("---")

# ========== SUPPORT & RESISTANCE ==========
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
            st.warning("⚠️ Harga sudah mendekati support")
    else:
        st.metric("SUPPORT", "N/A")

with col_res:
    resistance_val = robot_indicators.get('resistance', 0)
    if resistance_val > 0:
        pct_to_resistance = (resistance_val - current_val) / current_val * 100
        st.metric("RESISTANCE", f"Rp {resistance_val:,.0f}",
                  delta=f"+{pct_to_resistance:.1f}% dari harga")
        if pct_to_resistance < 2:
            st.warning("⚠️ Harga sudah mendekati resistance")
    else:
        st.metric("RESISTANCE", "N/A")

st.markdown("---")

# ========== TREND STRENGTH ==========
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

# ========== FINAL SIGNAL ==========
st.markdown("## 🏁 FINAL SIGNAL")

if robot_signal['signal'] in ["STRONG BUY", "BUY"]:
    st.success(f"# ✅ {robot_signal['signal']}")
    st.balloons()
elif robot_signal['signal'] in ["STRONG SELL", "SELL"]:
    st.error(f"# ❌ {robot_signal['signal']}")
else:
    st.warning(f"# ⏳ {robot_signal['signal']}")

# Tampilkan komponen skor
st.caption(f"📊 Komponen Signal:")
col_c1, col_c2, col_c3, col_c4 = st.columns(4)
col_c1.metric("Trend", f"{robot_signal['trend_component']:.1f}")
col_c2.metric("Volume", f"{robot_signal['volume_component']:.1f}")
col_c3.metric("Momentum", f"{robot_signal['momentum_component']:.1f}")
col_c4.metric("Oscillator", f"{robot_signal['oscillator_component']:.1f}")

with st.expander("📋 Detail Analisis & Alasan"):
    for reason in robot_signal['reasons']:
        st.write(f"- {reason}")
    st.markdown("---")
    st.markdown(f"**Total Score:** {robot_signal['score']}")

st.markdown("---")

# ========== INTERPRETASI SESUAI MODE TRADING ==========
st.markdown("## 🧠 INTERPRETASI SESUAI MODE")

# Tentukan konteks berdasarkan mode trading
if trading_mode == "Scalping (5-15 menit)":
    context = "⚡ **Scalping** - Fokus: volume + momentum cepat (noise tinggi)"
    context_detail = "Untuk scalping, sinyal ini perlu dikonfirmasi dengan order flow. Entry cepat, exit lebih cepat."
    weight_factor = 0.7
elif trading_mode == "Intraday (30 menit - 4 jam)":
    context = "📊 **Intraday** - Fokus: breakout & trend harian"
    context_detail = "Untuk intraday, fokus pada breakout level dan volume spike. Hold beberapa jam hingga sesi tutup."
    weight_factor = 1.0
elif trading_mode == "Swing (Daily)":
    context = "📈 **Swing** - Fokus: SMA + RSI + trend stabil"
    context_detail = "Untuk swing, konfirmasi dengan multi-timeframe. Hold 1-5 hari."
    weight_factor = 1.2
else:
    context = "🧭 **Position** - Fokus: trend besar & akumulasi"
    context_detail = "Untuk posisi trading, fokus pada akumulasi smart money dan trend makro. Hold mingguan-bulanan."
    weight_factor = 1.3

st.info(f"📌 **Mode aktif:** {trading_mode}")
st.caption(context)
st.caption(context_detail)

st.markdown("---")

# ========== KEPUTUSAN FINAL BERDASARKAN MODE ==========
st.markdown("## 🧭 KEPUTUSAN FINAL (MODE-AWARE)")

signal_score = robot_signal['score']
adjusted_score = signal_score * weight_factor

st.caption(f"Raw Score: {signal_score} | Adjusted Score (x{weight_factor}): {adjusted_score:.1f}")

if adjusted_score >= 5:
    st.success("🟢 **ENTRY LAYAK** (mode mendukung)")
    if trading_mode == "Scalping (5-15 menit)":
        st.write("📌 Saran: Entry dengan stop loss ketat, target kecil TP cepat")
    elif trading_mode == "Intraday (30 menit - 4 jam)":
        st.write("📌 Saran: Entry di breakout, target support/resistance berikutnya")
    elif trading_mode == "Swing (Daily)":
        st.write("📌 Saran: Entry bertahap, hold 1-5 hari")
    else:
        st.write("📌 Saran: Akumulasi bertahap, hold mingguan")
        
elif adjusted_score >= 2:
    st.warning("🟡 **ENTRY BOLEH** (tapi hati-hati)")
    st.write("📌 Saran: Gunakan ukuran posisi lebih kecil dari biasanya (50% posisi normal)")
    
else:
    st.error("🔴 **TIDAK LAYAK ENTRY**")
    st.write("📌 Saran: Lebih baik tunggu atau cari saham lain")

st.markdown("---")

# ========== ARTI SIGNAL GLOBAL ==========
st.markdown("## 🧠 ARTI KEPUTUSAN SAAT INI")

signal = robot_signal['signal']

if signal == "STRONG BUY":
    st.success("📌 **Sistem melihat peluang naik sangat kuat.** Momentum + trend + volume mendukung. Cocok untuk **entry sekarang** atau akumulasi bertahap.")
    st.caption("💡 Saran: Entry di harga support atau saat breakout konfirmasi.")
elif signal == "BUY":
    st.info("📌 **Kondisi cukup bagus, tapi belum sempurna.** Masuk boleh, tapi lebih aman jika **tunggu koreksi kecil** dulu.")
    st.caption("💡 Saran: Gunakan entry bertahap (scale in), jangan langsung full.")
elif signal == "WAIT / HOLD":
    st.warning("📌 **Pasar tidak jelas.** Tidak ada keunggulan arah. Lebih baik **tunggu konfirmasi** breakout atau breakdown.")
    st.caption("💡 Saran: Amati pergerakan harga, jangan entry dulu.")
elif signal == "SELL":
    st.warning("📌 **Tekanan turun mulai dominan.** Jika sudah punya posisi, mulai **pertimbangkan exit bertahap**.")
    st.caption("💡 Saran: Cut loss jika sudah breakdown support.")
elif signal == "STRONG SELL":
    st.error("📌 **Kondisi lemah.** Risiko turun lebih besar daripada peluang naik. **Hindari entry.**")
    st.caption("💡 Saran: Lebih baik hold cash atau cari saham lain.")

st.markdown("---")

# ========== KESIMPULAN CEPAT ==========
st.markdown("## 🧾 KESIMPULAN CEPAT")

final_signal = robot_signal['signal']
weighted = weighted_decision_engine(data, ticker)

if final_signal in ["STRONG BUY", "BUY"] and weighted >= 60:
    st.success("👉 **CENDERUNG BELI** (entry bertahap lebih aman)")
    st.write("📌 **Aksi:** Entry di harga support atau tunggu pullback kecil.")
    
elif final_signal in ["STRONG BUY", "BUY"] and weighted < 60:
    st.info("👉 **CENDERUNG BELI TAPI HATI-HATI**")
    st.write("📌 **Aksi:** Entry dengan ukuran posisi lebih kecil dari biasanya.")
    
elif final_signal == "WAIT / HOLD":
    st.warning("👉 **TUNGGU** (tidak ada keunggulan saat ini)")
    st.write("📌 **Aksi:** Jangan entry baru. Jika sudah pegang posisi: tahan dulu.")
    
elif final_signal in ["SELL", "STRONG SELL"]:
    st.error("👉 **HINDARI BELI / EXIT** jika sudah punya posisi")
    st.write("📌 **Aksi:** Jangan entry baru, pertimbangkan cut loss.")
    
else:
    st.info("👉 **NETRAL** (market belum jelas)")
    st.write("📌 **Aksi:** Tunggu konfirmasi arah yang lebih jelas.")

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

with col_d2:
    st.markdown("**📈 Multi Timeframe**")
    mtf = get_multi_timeframe_trend(ticker, trading_mode)
    st.write(f"Daily: {mtf['Daily']}")
    st.write(f"Weekly: {mtf['Weekly']}")
    st.write(f"Monthly: {mtf['Monthly']}")

with col_d3:
    st.markdown("**🏦 Smart Money**")
    smart_status = ai_signals['smart_status']
    if smart_status == "NO_VOLUME_DATA":
        st.info("Tidak ada data volume")
    else:
        st.write(f"Status: {smart_status}")
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
    if data['MACD'].iloc[-1] > data['MACD_Signal'].iloc[-1]:
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

st.markdown("---")

# ========== WEIGHTED DECISION ==========
st.subheader("🧠 Weighted Decision")

weighted_score = weighted_decision_engine(data, ticker)
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

st.markdown("---")

# ========== REKOMENDASI ENTRY & STOPLOSS ==========
entry_rec = get_recommended_entry_stoploss(data, robot_indicators.get('atr', last_close * 0.02))
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

# ========== INDIKATOR TEKNIKAL LENGKAP ==========
with st.expander("📊 Indikator Teknikal Lengkap"):
    try:
        col_i1, col_i2 = st.columns(2)
        with col_i1:
            st.metric("RSI (14)", f"{data['RSI'].iloc[-1]:.1f}")
            st.metric("SMA20", f"Rp {data['SMA20'].iloc[-1]:.2f}")
            st.metric("EMA200", f"Rp {data['EMA200'].iloc[-1]:.2f}")
            st.metric("Support", f"Rp {data['support'].iloc[-1]:.2f}")
            st.metric("ATR", f"Rp {data['ATR'].iloc[-1]:.2f}")
        with col_i2:
            st.metric("MACD", f"{data['MACD'].iloc[-1]:.4f}")
            st.metric("MACD Signal", f"{data['MACD_Signal'].iloc[-1]:.4f}")
            st.metric("Stoch K", f"{data['Stoch_K'].iloc[-1]:.1f}")
            st.metric("Stoch D", f"{data['Stoch_D'].iloc[-1]:.1f}")
            st.metric("Resistance", f"Rp {data['resistance'].iloc[-1]:.2f}")
            if has_valid_volume(data):
                st.metric("Volume", f"{data['Volume'].iloc[-1]:,.0f}")
                st.metric("Volume MA20", f"{data['Volume_MA'].iloc[-1]:,.0f}")
    except Exception as e:
        st.info("Data indikator tidak tersedia")

# ========== FOOTER ==========
st.markdown("---")
st.caption("⚠️ **DISCLAIMER:** Dashboard ini hanya untuk edukasi dan analisis. Bukan rekomendasi beli/jual. Keputusan investasi sepenuhnya risiko Anda.")

gc.collect()
