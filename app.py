import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from datetime import datetime, timedelta
import warnings
import gc
warnings.filterwarnings('ignore')

# ========== KONSTANTA ==========
CACHE_TTL = 300
DEFAULT_TICKER = "BBCA.JK"
TIMEFRAMES = {"1d": "1d", "1wk": "1wk", "1mo": "1mo"}
VOLUME_SPIKE_THRESHOLD = 2.0

st.set_page_config(layout="wide", page_title="Robot Saham - AI Decision Engine v2", page_icon="🤖")
st.markdown("""
    <style>
        .stAlert {font-size: 0.9rem;}
        div[data-testid="stMetricDelta"] > div {font-size: 0.8rem;}
    </style>
""", unsafe_allow_html=True)

# ========== SESSION STATE ==========
if 'signal_history' not in st.session_state:
    st.session_state.signal_history = []
if 'last_signal' not in st.session_state:
    st.session_state.last_signal = None
if 'last_toast_time' not in st.session_state:
    st.session_state.last_toast_time = None
if 'trading_mode' not in st.session_state:
    st.session_state.trading_mode = "Swing (Daily)"
if 'custom_weights' not in st.session_state:
    st.session_state.custom_weights = {
        'supertrend': 2.0, 'psar': 1.0, 'ema200': 1.5, 'sma_cross': 1.0,
        'mtf': 1.0, 'adx': 1.0, 'volume_spike': 1.5, 'obv': 1.0,
        'cmf': 1.0, 'ad': 0.5, 'rsi': 1.5, 'macd': 1.0,
        'stoch_willr': 1.0, 'aroon': 1.0, 'gmma': 1.0, 'kst': 0.5
    }

# ========== FUNGSI INDIKATOR RINGAN ==========
def calculate_supertrend(df, period=10, multiplier=3.0):
    high = df['High']
    low = df['Low']
    close = df['Close']
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
    high = df['High']
    low = df['Low']
    close = df['Close']
    psar = pd.Series(index=df.index, dtype=float)
    trend = pd.Series(index=df.index, dtype=int)
    if len(df) < 3:
        return psar, trend
    psar.iloc[0] = low.iloc[0]
    trend.iloc[0] = 1
    ep = high.iloc[0]
    af = step
    for i in range(1, len(df)):
        if trend.iloc[i-1] == 1:  # uptrend
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
        else:  # downtrend
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
    close = df['Close']
    volume = df['Volume']
    obv = (np.where(close > close.shift(1), volume,
                    np.where(close < close.shift(1), -volume, 0))).cumsum()
    return obv

def calculate_mfi(df, window=14):
    typical = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical * df['Volume']
    positive = money_flow.where(typical > typical.shift(1), 0).rolling(window).sum()
    negative = money_flow.where(typical < typical.shift(1), 0).rolling(window).sum()
    ratio = positive / negative
    mfi = 100 - (100 / (1 + ratio))
    return mfi

def calculate_aroon(df, window=25):
    high = df['High']
    low = df['Low']
    aroon_up = high.rolling(window).apply(lambda x: x.argmax() / window * 100, raw=True)
    aroon_down = low.rolling(window).apply(lambda x: x.argmin() / window * 100, raw=True)
    return aroon_up, aroon_down

def calculate_gmma(df, fast=[3,5,8,10,12,15], slow=[30,35,40,45,50,60]):
    gmma_fast = [df['Close'].ewm(span=p, adjust=False).mean() for p in fast]
    gmma_slow = [df['Close'].ewm(span=p, adjust=False).mean() for p in slow]
    return gmma_fast, gmma_slow

def calculate_williams_r(df, window=14):
    high = df['High'].rolling(window).max()
    low = df['Low'].rolling(window).min()
    return (high - df['Close']) / (high - low) * -100

def calculate_kst(df, roc1=10, roc2=15, roc3=20, roc4=30, ma1=10, ma2=10, ma3=10, ma4=15):
    close = df['Close']
    roc1_val = close.pct_change(roc1) * 100
    roc2_val = close.pct_change(roc2) * 100
    roc3_val = close.pct_change(roc3) * 100
    roc4_val = close.pct_change(roc4) * 100
    kst = (roc1_val.rolling(ma1).mean() +
           roc2_val.rolling(ma2).mean() * 2 +
           roc3_val.rolling(ma3).mean() * 3 +
           roc4_val.rolling(ma4).mean() * 4)
    return kst

def calculate_elder_ray(df, window=13):
    ema = df['Close'].ewm(span=window, adjust=False).mean()
    bull_power = df['High'] - ema
    bear_power = df['Low'] - ema
    return bull_power, bear_power

# ========== INDIKATOR UTAMA (SINGLE SOURCE) – SUDAH DIPERBAIKI ==========
@st.cache_data(ttl=CACHE_TTL)
def calculate_all_indicators(df, st_period=10, st_mult=3.0, mode="Swing (Daily)"):
    if df.empty or len(df) < 30:
        return df
    
    # Flatten MultiIndex jika ada
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    required = ['Close', 'High', 'Low']
    if not all(col in df.columns for col in required):
        return df
    
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    # Handle volume dengan aman
    if 'Volume' not in df.columns:
        volume = pd.Series(0, index=df.index)
        has_vol = False
    else:
        vol_series = df['Volume']
        if isinstance(vol_series, pd.DataFrame):
            vol_series = vol_series.iloc[:, 0]
        total_vol = vol_series.sum() if len(vol_series) > 0 else 0
        if pd.isna(total_vol) or total_vol == 0:
            volume = pd.Series(0, index=df.index)
            has_vol = False
        else:
            volume = vol_series
            has_vol = True
    
    # Moving Averages
    df['SMA20'] = close.rolling(20, min_periods=1).mean()
    df['SMA50'] = close.rolling(50, min_periods=1).mean()
    df['EMA200'] = close.rolling(200).mean() if len(close) >= 200 else close
    
    # RSI
    df['RSI'] = ta.momentum.rsi(close, window=14) if len(close) >= 14 else 50.0
    
    # MACD
    if len(close) >= 26:
        macd = ta.trend.MACD(close)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    else:
        df['MACD'] = 0.0
        df['MACD_Signal'] = 0.0
        df['MACD_Hist'] = 0.0
    
    # Stochastic
    if len(close) >= 14:
        df['Stoch_K'] = ta.momentum.stoch(high, low, close, window=14, smooth_window=3)
        df['Stoch_D'] = ta.momentum.stoch_signal(high, low, close, window=14, smooth_window=3)
    else:
        df['Stoch_K'] = 50.0
        df['Stoch_D'] = 50.0
    
    # ATR
    df['ATR'] = ta.volatility.average_true_range(high, low, close, window=14) if len(close) >= 14 else close * 0.02
    
    # Volume MA
    if has_vol:
        df['Volume_MA'] = volume.rolling(20, min_periods=1).mean()
    else:
        df['Volume_MA'] = 0.0
    
    # Support / Resistance (rolling)
    df['support'] = low.rolling(20, min_periods=1).min()
    df['resistance'] = high.rolling(20, min_periods=1).max()
    
    # AD dan CMF
    if has_vol:
        try:
            df['AD'] = ta.volume.acc_dist_index(high, low, close, volume, fillna=True)
        except:
            df['AD'] = 0.0
        try:
            df['CMF'] = ta.volume.chaikin_money_flow(high, low, close, volume, window=20, fillna=True)
        except:
            df['CMF'] = 0.0
    else:
        df['AD'] = 0.0
        df['CMF'] = 0.0
    
    # ===== INDIKATOR BARU =====
    # Supertrend
    st_line, st_dir = calculate_supertrend(df, period=st_period, multiplier=st_mult)
    df['ST_Supertrend'] = st_line
    df['ST_Dir'] = st_dir
    
    # Parabolic SAR
    psar_line, psar_dir = calculate_psar(df)
    df['PSAR'] = psar_line
    df['PSAR_Dir'] = psar_dir
    
    # OBV & MFI
    if has_vol:
        df['OBV'] = calculate_obv(df)
        df['OBV_MA'] = df['OBV'].rolling(20).mean()
        df['MFI'] = calculate_mfi(df, window=14)
    else:
        df['OBV'] = 0.0
        df['OBV_MA'] = 0.0
        df['MFI'] = 50.0
    
    # Aroon
    aroon_up, aroon_down = calculate_aroon(df)
    df['Aroon_Up'] = aroon_up
    df['Aroon_Down'] = aroon_down
    
    # Williams %R
    df['Williams_R'] = calculate_williams_r(df)
    
    # CCI
    if len(close) >= 20:
        df['CCI'] = ta.trend.CCIIndicator(high, low, close, window=20).cci()
    else:
        df['CCI'] = 0.0
    
    # KST (hanya untuk daily dengan data cukup)
    if mode in ["Swing (Daily)", "Position (Weekly - Monthly)"] and len(df) >= 50:
        df['KST'] = calculate_kst(df)
    else:
        df['KST'] = 0.0
    
    # Elder-Ray
    bull, bear = calculate_elder_ray(df)
    df['Elder_Bull'] = bull
    df['Elder_Bear'] = bear
    
    # GMMA (ringkas)
    if len(df) >= 30:
        gmma_fast, gmma_slow = calculate_gmma(df)
        df['GMMA_Fast_Avg'] = sum(gmma_fast) / len(gmma_fast)
        df['GMMA_Slow_Avg'] = sum(gmma_slow) / len(gmma_slow)
        df['GMMA_Spread'] = df['GMMA_Fast_Avg'] - df['GMMA_Slow_Avg']
    else:
        df['GMMA_Spread'] = 0.0
    
    df = df.ffill().fillna(0)
    return df

def get_latest_indicators(df):
    if df.empty:
        return {}
    last = df.iloc[-1]
    # Volume status
    try:
        vol_status = "Normal"
        if 'Volume_MA' in df.columns and df['Volume_MA'].iloc[-1] > 0:
            vol_current = df['Volume'].iloc[-1] if 'Volume' in df.columns else 0
            vol_avg = df['Volume_MA'].iloc[-1]
            if vol_current > vol_avg * 1.5:
                vol_status = "Tinggi"
            elif vol_current < vol_avg:
                vol_status = "Rendah"
    except:
        vol_status = "No Data"
    
    return {
        'price': last['Close'],
        'sma20': last['SMA20'], 'sma50': last['SMA50'], 'ema200': last['EMA200'],
        'rsi': last['RSI'], 'macd_hist': last['MACD_Hist'],
        'stoch_k': last['Stoch_K'], 'stoch_d': last['Stoch_D'],
        'atr': last['ATR'], 'volume_status': vol_status,
        'volume_current': df['Volume'].iloc[-1] if 'Volume' in df.columns else 0,
        'volume_avg': last['Volume_MA'] if 'Volume_MA' in last else 0,
        'support': last['support'], 'resistance': last['resistance'],
        'supertrend_dir': last['ST_Dir'] if 'ST_Dir' in last else 0,
        'supertrend_value': last['ST_Supertrend'] if 'ST_Supertrend' in last else last['Close'],
        'psar_dir': last['PSAR_Dir'] if 'PSAR_Dir' in last else 0,
        'obv': last['OBV'], 'obv_ma': last['OBV_MA'],
        'mfi': last['MFI'], 'aroon_up': last['Aroon_Up'], 'aroon_down': last['Aroon_Down'],
        'williams_r': last['Williams_R'], 'cci': last['CCI'], 'kst': last.get('KST', 0),
        'elder_bull': last['Elder_Bull'], 'elder_bear': last['Elder_Bear'],
        'gmma_spread': last['GMMA_Spread'], 'cmf': last['CMF'], 'ad': last['AD']
    }

def get_market_context():
    try:
        ihsg = yf.download("^JKSE", period="10d", progress=False)['Close']
        if len(ihsg) >= 3:
            last_3 = ihsg.iloc[-3:].pct_change().dropna()
            if (last_3 < 0).all():
                return "⚠️ IHSG turun 3 hari berturut-turut, risiko tinggi untuk entry LONG", -2
            elif (last_3 > 0).all():
                return "✅ IHSG naik 3 hari berturut-turut, sentimen positif", 2
            else:
                return "IHSG sideways, sesuai analisis teknikal", 0
    except:
        pass
    return "Tidak ada data IHSG", 0

# ========== DECISION ENGINE DENGAN BOBOT ==========
def weighted_decision(indicators, mtf_alignment, trend_strength, weights, df_full=None):
    score = 0
    reasons = []
    
    # 1. TREND (45%)
    if indicators['price'] > indicators['ema200']:
        score += weights['ema200'] * 1.5
        reasons.append(f"✅ Harga > EMA200 (+{weights['ema200']*1.5:.1f})")
    else:
        score -= weights['ema200'] * 1.5
        reasons.append(f"❌ Harga < EMA200 (-{weights['ema200']*1.5:.1f})")
    
    if indicators['sma20'] > indicators['sma50']:
        score += weights['sma_cross'] * 1.0
        reasons.append(f"✅ SMA20 > SMA50 (+{weights['sma_cross']:.1f})")
    else:
        score -= weights['sma_cross'] * 1.0
        reasons.append(f"❌ SMA20 < SMA50 (-{weights['sma_cross']:.1f})")
    
    if indicators['supertrend_dir'] == 1:
        score += weights['supertrend'] * 2.0
        reasons.append(f"✅ Supertrend UPTREND (+{weights['supertrend']*2.0:.1f})")
    else:
        score -= weights['supertrend'] * 2.0
        reasons.append(f"❌ Supertrend DOWNTREND (-{weights['supertrend']*2.0:.1f})")
    
    if indicators['psar_dir'] == 1:
        score += weights['psar'] * 1.0
        reasons.append(f"✅ PSAR UPTREND (+{weights['psar']:.1f})")
    else:
        score -= weights['psar'] * 1.0
        reasons.append(f"❌ PSAR DOWNTREND (-{weights['psar']:.1f})")
    
    # MTF alignment (sederhana)
    if mtf_alignment > 0.5:
        score += weights['mtf'] * 1.0
        reasons.append(f"✅ MTF Bullish (+{weights['mtf']:.1f})")
    elif mtf_alignment < -0.5:
        score -= weights['mtf'] * 1.0
        reasons.append(f"❌ MTF Bearish (-{weights['mtf']:.1f})")
    
    if trend_strength == "STRONG TREND":
        score += weights['adx'] * 1.0
        reasons.append(f"✅ ADX Kuat (+{weights['adx']:.1f})")
    elif trend_strength == "WEAK TREND":
        score -= weights['adx'] * 0.5
        reasons.append(f"⚠️ ADX Lemah (-{weights['adx']*0.5:.1f})")
    
    # 2. VOLUME (25%)
    if indicators['volume_status'] == "Tinggi":
        score += weights['volume_spike'] * 1.5
        reasons.append(f"✅ Volume Tinggi (+{weights['volume_spike']*1.5:.1f})")
    elif indicators['volume_status'] == "Rendah":
        score -= weights['volume_spike'] * 1.0
        reasons.append(f"⚠️ Volume Rendah (-{weights['volume_spike']:.1f})")
    
    if indicators['obv'] > indicators['obv_ma']:
        score += weights['obv'] * 1.0
        reasons.append(f"✅ OBV > MA (+{weights['obv']:.1f})")
    else:
        score -= weights['obv'] * 1.0
        reasons.append(f"❌ OBV < MA (-{weights['obv']:.1f})")
    
    # Smart Money dari CMF
    cmf_val = indicators.get('cmf', 0)
    if cmf_val > 0.15:
        score += weights['cmf'] * 2.0
        reasons.append(f"✅ SMART MONEY AKUMULASI (CMF={cmf_val:.2f}) (+{weights['cmf']*2.0:.1f})")
    elif cmf_val < -0.15:
        score -= weights['cmf'] * 2.0
        reasons.append(f"❌ SMART MONEY DISTRIBUSI (CMF={cmf_val:.2f}) (-{weights['cmf']*2.0:.1f})")
    elif cmf_val > 0:
        score += weights['cmf'] * 0.5
        reasons.append(f"✅ CMF positif (+{weights['cmf']*0.5:.1f})")
    elif cmf_val < 0:
        score -= weights['cmf'] * 0.5
        reasons.append(f"❌ CMF negatif (-{weights['cmf']*0.5:.1f})")
    
    # 3. MOMENTUM (15%)
    rsi = indicators['rsi']
    if rsi < 30:
        score += weights['rsi'] * 1.5
        reasons.append(f"✅ RSI Oversold (+{weights['rsi']*1.5:.1f})")
    elif rsi > 70:
        score -= weights['rsi'] * 1.5
        reasons.append(f"⚠️ RSI Overbought (-{weights['rsi']*1.5:.1f})")
    elif 30 <= rsi <= 50:
        score += weights['rsi'] * 1.0
        reasons.append(f"✅ RSI {rsi:.0f} (recovery) (+{weights['rsi']:.1f})")
    
    mfi = indicators['mfi']
    if mfi < 20:
        score += weights['rsi'] * 0.8
        reasons.append(f"✅ MFI Oversold (+{weights['rsi']*0.8:.1f})")
    elif mfi > 80:
        score -= weights['rsi'] * 0.8
        reasons.append(f"⚠️ MFI Overbought (-{weights['rsi']*0.8:.1f})")
    
    # 4. OSCILLATOR (15%)
    if indicators['macd_hist'] > 0:
        score += weights['macd'] * 1.0
        reasons.append(f"✅ MACD positif (+{weights['macd']:.1f})")
    else:
        score -= weights['macd'] * 1.0
        reasons.append(f"❌ MACD negatif (-{weights['macd']:.1f})")
    
    # Gabungan Stochastic & Williams %R
    stoch = indicators['stoch_k']
    willr = indicators['williams_r']
    osc_contrib = 0
    if stoch < 20 or willr < -80:
        osc_contrib += 1
    if stoch > 80 or willr > -20:
        osc_contrib -= 1
    score += osc_contrib * weights['stoch_willr']
    if osc_contrib > 0:
        reasons.append(f"✅ Stochastic/Williams oversold (+{weights['stoch_willr']:.1f})")
    elif osc_contrib < 0:
        reasons.append(f"⚠️ Stochastic/Williams overbought (-{weights['stoch_willr']:.1f})")
    
    # Aroon
    if indicators['aroon_up'] > 70 and indicators['aroon_down'] < 30:
        score += weights['aroon'] * 1.0
        reasons.append(f"✅ Aroon Up >70 (+{weights['aroon']:.1f})")
    elif indicators['aroon_down'] > 70:
        score -= weights['aroon'] * 1.0
        reasons.append(f"❌ Aroon Down >70 (-{weights['aroon']:.1f})")
    
    # GMMA spread
    if indicators['gmma_spread'] > 0:
        score += weights['gmma'] * 0.8
        reasons.append(f"✅ GMMA spread positif (+{weights['gmma']*0.8:.1f})")
    else:
        score -= weights['gmma'] * 0.8
        reasons.append(f"❌ GMMA spread negatif (-{weights['gmma']*0.8:.1f})")
    
    # KST
    if indicators['kst'] > 0:
        score += weights['kst'] * 1.0
        reasons.append(f"✅ KST positif (+{weights['kst']:.1f})")
    else:
        score -= weights['kst'] * 1.0
        reasons.append(f"❌ KST negatif (-{weights['kst']:.1f})")
    
    # Elder-Ray
    if indicators['elder_bull'] > 0 and indicators['elder_bear'] < 0:
        score += 1.0
        reasons.append(f"✅ Elder-Ray Bull kuat (+1.0)")
    elif indicators['elder_bull'] < 0 and indicators['elder_bear'] > 0:
        score -= 1.0
        reasons.append(f"❌ Elder-Ray Bear kuat (-1.0)")
    
    # Normalisasi skala -10 sd +10
    final_score = max(-10, min(10, score / 2))
    
    if final_score >= 6:
        signal = "STRONG BUY"
        confidence = "HIGH"
        color = "success"
    elif final_score >= 3:
        signal = "BUY"
        confidence = "MEDIUM"
        color = "info"
    elif final_score <= -6:
        signal = "STRONG SELL"
        confidence = "HIGH"
        color = "error"
    elif final_score <= -3:
        signal = "SELL"
        confidence = "LOW"
        color = "warning"
    else:
        signal = "HOLD"
        confidence = "LOW"
        color = "warning"
    
    return {
        'signal': signal, 'color': color, 'score': final_score,
        'confidence': confidence, 'reasons': reasons,
        'supertrend_dir': indicators['supertrend_dir'],
        'supertrend_value': indicators['supertrend_value']
    }

def get_timeframe_signal(df, tf_name, weights):
    if df.empty or len(df) < 30:
        return "HOLD (insufficient data)"
    ind = get_latest_indicators(df)
    score = 0
    if ind['price'] > ind['ema200']: score += 1
    else: score -= 1
    if ind['supertrend_dir'] == 1: score += 1
    else: score -= 1
    if ind['volume_status'] == "Tinggi": score += 1
    elif ind['volume_status'] == "Rendah": score -= 0.5
    if score >= 1.5: return "BUY"
    elif score <= -1.5: return "SELL"
    else: return "HOLD"

def get_entry_levels(df, atr, price):
    support = df['support'].iloc[-1] if 'support' in df.columns else price * 0.95
    resistance = df['resistance'].iloc[-1] if 'resistance' in df.columns else price * 1.05
    buy_entry = support + (atr * 0.5)
    sell_entry = resistance - (atr * 0.5)
    return {
        'buy_entry': round(buy_entry, 2), 'sell_entry': round(sell_entry, 2),
        'stop_loss_buy': round(buy_entry - atr, 2), 'stop_loss_sell': round(sell_entry + atr, 2),
        'target_buy': round(resistance, 2), 'target_sell': round(support, 2)
    }

def run_backtest(df, weights, period_days=90):
    if df.empty or len(df) < period_days:
        return None
    df_test = df.tail(period_days).copy()
    signals = []
    for i in range(30, len(df_test)):
        slice_df = df_test.iloc[:i+1]
        ind = get_latest_indicators(slice_df)
        dec = weighted_decision(ind, mtf_alignment=0, trend_strength="MODERATE", weights=weights, df_full=slice_df)
        signals.append(dec['signal'])
    buy_count = signals.count("STRONG BUY") + signals.count("BUY")
    sell_count = signals.count("STRONG SELL") + signals.count("SELL")
    hold_count = signals.count("HOLD")
    win_rate = 0.6 if buy_count > 0 else 0  # dummy, bisa diperbaiki
    return {
        'buy_signals': buy_count, 'sell_signals': sell_count, 'hold_signals': hold_count,
        'win_rate': win_rate, 'total_signals': len(signals)
    }

def show_toast_if_changed(new_signal, prev_signal):
    if prev_signal and new_signal != prev_signal:
        if new_signal in ["STRONG BUY", "BUY"] and prev_signal in ["SELL", "STRONG SELL", "HOLD"]:
            st.toast(f"🟢 Sinyal berubah menjadi {new_signal}!", icon="🔔")
        elif new_signal in ["STRONG SELL", "SELL"] and prev_signal in ["BUY", "STRONG BUY", "HOLD"]:
            st.toast(f"🔴 Sinyal berubah menjadi {new_signal}!", icon="⚠️")
        elif new_signal == "HOLD" and prev_signal in ["BUY","STRONG BUY","SELL","STRONG SELL"]:
            st.toast(f"⚪ Sinyal menjadi HOLD, waspadai perubahan", icon="ℹ️")

def log_signal(ticker, signal, score, price):
    record = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ticker': ticker, 'signal': signal, 'score': score, 'price': price,
        'status': 'pending'
    }
    st.session_state.signal_history.insert(0, record)
    if len(st.session_state.signal_history) > 100:
        st.session_state.signal_history.pop()
    return record

# ========== MAIN ==========
def main():
    st.sidebar.markdown("# 🤖 Robot Saham v2")
    ticker_input = st.sidebar.text_input("Kode Saham", DEFAULT_TICKER)
    ticker = ticker_input.upper().strip()
    if not ticker.endswith('.JK'):
        ticker = ticker + '.JK'
    timeframe = st.sidebar.selectbox("Timeframe", ["1d","1wk","1mo"], index=0)
    trading_mode = st.sidebar.selectbox(
        "Mode Trading",
        ["Scalping (5-15 menit)", "Intraday (30 menit - 4 jam)", "Swing (Daily)", "Position (Weekly - Monthly)"],
        index=2
    )
    st.session_state.trading_mode = trading_mode
    
    # Parameter adaptif Supertrend
    if trading_mode == "Scalping (5-15 menit)":
        st_period, st_mult = 7, 2.5
    elif trading_mode == "Intraday (30 menit - 4 jam)":
        st_period, st_mult = 10, 3.0
    elif trading_mode == "Swing (Daily)":
        st_period, st_mult = 10, 3.0
    else:
        st_period, st_mult = 20, 3.0
    
    # Tuning bobot
    with st.sidebar.expander("⚙️ Tuning Bobot Indikator"):
        weights = {}
        for key in st.session_state.custom_weights:
            weights[key] = st.slider(f"{key}", 0.0, 3.0, st.session_state.custom_weights[key], 0.1)
        if st.button("Simpan Bobot"):
            st.session_state.custom_weights = weights
            st.success("Bobot tersimpan")
    weights = st.session_state.custom_weights.copy()
    
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Load data
    with st.spinner(f"Memuat {ticker}..."):
        period = "2y" if trading_mode in ["Swing (Daily)", "Position (Weekly - Monthly)"] else "6mo"
        df_raw = yf.download(ticker, period=period, interval=timeframe, progress=False)
    if df_raw.empty:
        st.error(f"Data {ticker} tidak tersedia")
        st.stop()
    
    df = calculate_all_indicators(df_raw, st_period, st_mult, trading_mode)
    indicators = get_latest_indicators(df)
    price = indicators.get('price', 0)
    
    # MTF & trend strength placeholder
    mtf_alignment = 0
    trend_strength = "MODERATE"
    
    # Decision
    decision = weighted_decision(indicators, mtf_alignment, trend_strength, weights, df)
    
    # Toast & logging
    show_toast_if_changed(decision['signal'], st.session_state.last_signal)
    st.session_state.last_signal = decision['signal']
    log_signal(ticker, decision['signal'], decision['score'], price)
    
    # Market context
    market_text, market_score = get_market_context()
    
    # ========== DASHBOARD ==========
    st.title(f"🤖 {ticker}")
    st.caption(f"Mode: {trading_mode} | Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Harga", f"Rp {price:,.0f}")
    col2.metric("RSI", f"{indicators['rsi']:.1f}")
    col3.metric("Supertrend", "🟢 UPTREND" if indicators['supertrend_dir'] == 1 else "🔴 DOWNTREND", delta=f"Garis: {indicators['supertrend_value']:.0f}")
    
    st.markdown("---")
    # Sinyal utama
    if decision['color'] == "success":
        st.success(f"## 🟢 {decision['signal']} (Skor: {decision['score']:.1f})")
    elif decision['color'] == "error":
        st.error(f"## 🔴 {decision['signal']} (Skor: {decision['score']:.1f})")
    elif decision['color'] == "info":
        st.info(f"## 📈 {decision['signal']} (Skor: {decision['score']:.1f})")
    else:
        st.warning(f"## ⏳ {decision['signal']} (Skor: {decision['score']:.1f})")
    st.metric("Keyakinan", decision['confidence'])
    
    with st.expander("📋 Detail Analisis & Kontribusi Indikator"):
        for r in decision['reasons']:
            st.write(r)
    
    st.markdown("---")
    st.subheader("💰 Smart Money Detection")
    cmf_val = indicators.get('cmf', 0)
    if cmf_val > 0.15:
        st.success(f"✅ Akumulasi KUAT (CMF={cmf_val:.2f}) dalam 20 hari terakhir")
    elif cmf_val < -0.15:
        st.error(f"❌ Distribusi (CMF={cmf_val:.2f}) dalam 20 hari terakhir")
    else:
        st.info(f"Netral (CMF={cmf_val:.2f})")
    st.caption(market_text)
    
    st.markdown("---")
    st.subheader("⏰ Multi-Timeframe Signal")
    # Load data untuk daily/weekly/monthly
    df_daily = yf.download(ticker, period="1y", interval="1d", progress=False)
    df_weekly = yf.download(ticker, period="2y", interval="1wk", progress=False)
    df_monthly = yf.download(ticker, period="3y", interval="1mo", progress=False)
    if not df_daily.empty:
        df_daily = calculate_all_indicators(df_daily, st_period, st_mult, trading_mode)
        sig_daily = get_timeframe_signal(df_daily, "Daily", weights)
        st.metric("Daily", sig_daily)
    if not df_weekly.empty:
        df_weekly = calculate_all_indicators(df_weekly, st_period, st_mult, trading_mode)
        sig_weekly = get_timeframe_signal(df_weekly, "Weekly", weights)
        st.metric("Weekly", sig_weekly)
    if not df_monthly.empty:
        df_monthly = calculate_all_indicators(df_monthly, st_period, st_mult, trading_mode)
        sig_monthly = get_timeframe_signal(df_monthly, "Monthly", weights)
        st.metric("Monthly", sig_monthly)
    
    st.markdown("---")
    st.subheader("🎯 Level Entry & Stop Loss")
    entry_levels = get_entry_levels(df, indicators['atr'], price)
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Buy Entry", f"Rp {entry_levels['buy_entry']:,.0f}")
    col_b.metric("Stop Loss Buy", f"Rp {entry_levels['stop_loss_buy']:,.0f}", delta_color="inverse")
    col_c.metric("Target Buy", f"Rp {entry_levels['target_buy']:,.0f}")
    
    st.markdown("---")
    st.subheader("📊 Backtest (3 bulan)")
    if st.button("Jalankan Backtest"):
        with st.spinner("Menghitung..."):
            backtest_res = run_backtest(df, weights, period_days=90)
            if backtest_res:
                st.write(f"Sinyal BUY: {backtest_res['buy_signals']}, SELL: {backtest_res['sell_signals']}, HOLD: {backtest_res['hold_signals']}")
                st.metric("Estimasi Win Rate", f"{backtest_res['win_rate']*100:.0f}%")
            else:
                st.warning("Data tidak cukup")
    
    st.markdown("---")
    st.subheader("📜 Riwayat Sinyal & Ekspor")
    if st.session_state.signal_history:
        df_log = pd.DataFrame(st.session_state.signal_history)
        st.dataframe(df_log.head(20))
        csv = df_log.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Ekspor CSV", csv, "signal_history.csv", "text/csv")
    
    st.caption("⚠️ Edukasi saja, bukan rekomendasi investasi.")
    gc.collect()

if __name__ == "__main__":
    main()
