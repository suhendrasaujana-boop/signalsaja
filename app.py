import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from datetime import datetime
import warnings
import gc
warnings.filterwarnings('ignore')

# ========== KONSTANTA & KONFIGURASI ==========
st.set_page_config(layout="wide", page_title="Robot Saham - AI Decision Engine", page_icon="🤖")
CACHE_TTL = 300
DEFAULT_TICKER = "BBCA.JK"

# CSS minimal agar margin/padding lebih rapat (tapi ukuran font default)
st.markdown("""
<style>
    /* Kurangi margin antar elemen */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.2rem;
    }
    /* Metric tetap default, tidak diubah ukuran */
</style>
""", unsafe_allow_html=True)

# Session state untuk logging & bobot
if 'signal_history' not in st.session_state:
    st.session_state.signal_history = []
if 'last_signal' not in st.session_state:
    st.session_state.last_signal = None
if 'trading_mode' not in st.session_state:
    st.session_state.trading_mode = "Swing (Daily)"
if 'custom_weights' not in st.session_state:
    st.session_state.custom_weights = {
        'supertrend': 2.0, 'psar': 1.0, 'ema200': 1.5, 'sma_cross': 1.0,
        'mtf': 1.0, 'adx': 1.0, 'volume_spike': 1.5, 'obv': 1.0,
        'cmf': 1.0, 'ad': 0.5, 'rsi': 1.5, 'macd': 1.0,
        'stoch_willr': 1.0, 'aroon': 1.0, 'gmma': 1.0, 'kst': 0.5
    }

# ========== FUNGSI INDIKATOR (SUPERTREND, PSAR, OBV, MFI, DLL) ==========
# --- Fungsi-fungsi ini sama dengan yang sudah kita sepakati ---
def calculate_supertrend(df, period=10, multiplier=3.0):
    high, low, close = df['High'], df['Low'], df['Close']
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
    high, low, close = df['High'], df['Low'], df['Close']
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
    # df harus memiliki kolom 'Close' dan 'Volume'
    close = df['Close']
    volume = df['Volume']
    return (np.where(close > close.shift(1), volume,
                     np.where(close < close.shift(1), -volume, 0))).cumsum()

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
    bull = df['High'] - ema
    bear = df['Low'] - ema
    return bull, bear

# ========== SINGLE SOURCE OF TRUTH ==========
@st.cache_data(ttl=CACHE_TTL)
def calculate_all_indicators(df, st_period=10, st_mult=3.0, mode="Swing (Daily)"):
    if df.empty or len(df) < 30:
        return df
    # Flatten MultiIndex jika ada
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    close = df['Close']; high = df['High']; low = df['Low']
    # Volume handling
    has_vol = 'Volume' in df.columns and df['Volume'].sum() != 0
    volume = df['Volume'] if has_vol else pd.Series(0, index=df.index)
    
    # Indikator dasar
    df['SMA20'] = close.rolling(20, min_periods=1).mean()
    df['SMA50'] = close.rolling(50, min_periods=1).mean()
    df['EMA200'] = close.rolling(200).mean() if len(close) >= 200 else close
    df['RSI'] = ta.momentum.rsi(close, window=14) if len(close) >= 14 else 50.0
    if len(close) >= 26:
        macd = ta.trend.MACD(close)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    else:
        df['MACD'] = df['MACD_Signal'] = df['MACD_Hist'] = 0.0
    if len(close) >= 14:
        df['Stoch_K'] = ta.momentum.stoch(high, low, close, window=14, smooth_window=3)
        df['Stoch_D'] = ta.momentum.stoch_signal(high, low, close, window=14, smooth_window=3)
    else:
        df['Stoch_K'] = df['Stoch_D'] = 50.0
    df['ATR'] = ta.volatility.average_true_range(high, low, close, window=14) if len(close) >= 14 else close * 0.02
    if has_vol:
        df['Volume_MA'] = volume.rolling(20, min_periods=1).mean()
    else:
        df['Volume_MA'] = 0.0
    df['support'] = low.rolling(20, min_periods=1).min()
    df['resistance'] = high.rolling(20, min_periods=1).max()
    # AD dan CMF
    if has_vol:
        try:
            df['AD'] = ta.volume.acc_dist_index(high, low, close, volume, fillna=True)
            df['CMF'] = ta.volume.chaikin_money_flow(high, low, close, volume, window=20, fillna=True)
        except:
            df['AD'] = df['CMF'] = 0.0
    else:
        df['AD'] = df['CMF'] = 0.0
    
    # Indikator tambahan
    st_line, st_dir = calculate_supertrend(df, period=st_period, multiplier=st_mult)
    df['ST_Supertrend'] = st_line
    df['ST_Dir'] = st_dir
    psar_line, psar_dir = calculate_psar(df)
    df['PSAR'] = psar_line; df['PSAR_Dir'] = psar_dir
    if has_vol:
        df['OBV'] = calculate_obv(df)
        df['OBV_MA'] = df['OBV'].rolling(20).mean()
        df['MFI'] = calculate_mfi(df, window=14)
    else:
        df['OBV'] = df['OBV_MA'] = 0.0; df['MFI'] = 50.0
    aroon_up, aroon_down = calculate_aroon(df)
    df['Aroon_Up'] = aroon_up; df['Aroon_Down'] = aroon_down
    df['Williams_R'] = calculate_williams_r(df)
    if len(close) >= 20:
        df['CCI'] = ta.trend.CCIIndicator(high, low, close, window=20).cci()
    else:
        df['CCI'] = 0.0
    if mode in ["Swing (Daily)", "Position (Weekly - Monthly)"] and len(df) >= 50:
        df['KST'] = calculate_kst(df)
    else:
        df['KST'] = 0.0
    bull, bear = calculate_elder_ray(df)
    df['Elder_Bull'] = bull; df['Elder_Bear'] = bear
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
        if 'Volume_MA' in df.columns and df['Volume_MA'].iloc[-1] > 0:
            vol = df['Volume'].iloc[-1]
            vol_ma = df['Volume_MA'].iloc[-1]
            if vol > vol_ma * 1.5:
                vol_status = "Tinggi"
            elif vol < vol_ma:
                vol_status = "Rendah"
            else:
                vol_status = "Normal"
        else:
            vol_status = "No Data"
    except:
        vol_status = "Normal"
    return {
        'price': last['Close'], 'sma20': last['SMA20'], 'sma50': last['SMA50'], 'ema200': last['EMA200'],
        'rsi': last['RSI'], 'macd_hist': last['MACD_Hist'], 'stoch_k': last['Stoch_K'], 'stoch_d': last['Stoch_D'],
        'atr': last['ATR'], 'volume_status': vol_status, 'support': last['support'], 'resistance': last['resistance'],
        'supertrend_dir': last['ST_Dir'], 'supertrend_value': last['ST_Supertrend'], 'psar_dir': last['PSAR_Dir'],
        'obv': last['OBV'], 'obv_ma': last['OBV_MA'], 'mfi': last['MFI'], 'aroon_up': last['Aroon_Up'], 'aroon_down': last['Aroon_Down'],
        'williams_r': last['Williams_R'], 'cci': last['CCI'], 'kst': last.get('KST', 0),
        'elder_bull': last['Elder_Bull'], 'elder_bear': last['Elder_Bear'], 'gmma_spread': last['GMMA_Spread'],
        'cmf': last['CMF'], 'ad': last['AD']
    }

# ========== DECISION ENGINE ==========
def weighted_decision(indicators, mtf_alignment, trend_strength, weights, df_full=None):
    score = 0
    reasons = []
    # Trend
    if indicators['price'] > indicators['ema200']:
        score += weights['ema200']*1.5
        reasons.append(f"Harga > EMA200 (+{weights['ema200']*1.5:.1f})")
    else:
        score -= weights['ema200']*1.5
        reasons.append(f"Harga < EMA200 (-{weights['ema200']*1.5:.1f})")
    if indicators['sma20'] > indicators['sma50']:
        score += weights['sma_cross']*1.0
        reasons.append(f"SMA20 > SMA50 (+{weights['sma_cross']:.1f})")
    else:
        score -= weights['sma_cross']*1.0
        reasons.append(f"SMA20 < SMA50 (-{weights['sma_cross']:.1f})")
    if indicators['supertrend_dir'] == 1:
        score += weights['supertrend']*2.0
        reasons.append(f"Supertrend UPTREND (+{weights['supertrend']*2.0:.1f})")
    else:
        score -= weights['supertrend']*2.0
        reasons.append(f"Supertrend DOWNTREND (-{weights['supertrend']*2.0:.1f})")
    if indicators['psar_dir'] == 1:
        score += weights['psar']*1.0
        reasons.append(f"PSAR UPTREND (+{weights['psar']:.1f})")
    else:
        score -= weights['psar']*1.0
        reasons.append(f"PSAR DOWNTREND (-{weights['psar']:.1f})")
    if mtf_alignment > 0.5:
        score += weights['mtf']*1.0
        reasons.append(f"MTF Bullish (+{weights['mtf']:.1f})")
    elif mtf_alignment < -0.5:
        score -= weights['mtf']*1.0
        reasons.append(f"MTF Bearish (-{weights['mtf']:.1f})")
    if trend_strength == "STRONG TREND":
        score += weights['adx']*1.0
        reasons.append(f"ADX Kuat (+{weights['adx']:.1f})")
    elif trend_strength == "WEAK TREND":
        score -= weights['adx']*0.5
        reasons.append(f"ADX Lemah (-{weights['adx']*0.5:.1f})")
    # Volume
    if indicators['volume_status'] == "Tinggi":
        score += weights['volume_spike']*1.5
        reasons.append(f"Volume Tinggi (+{weights['volume_spike']*1.5:.1f})")
    elif indicators['volume_status'] == "Rendah":
        score -= weights['volume_spike']*1.0
        reasons.append(f"Volume Rendah (-{weights['volume_spike']:.1f})")
    if indicators['obv'] > indicators['obv_ma']:
        score += weights['obv']*1.0
        reasons.append(f"OBV > MA (+{weights['obv']:.1f})")
    else:
        score -= weights['obv']*1.0
        reasons.append(f"OBV < MA (-{weights['obv']:.1f})")
    cmf = indicators['cmf']
    if cmf > 0.15:
        score += weights['cmf']*2.0
        reasons.append(f"SMART MONEY AKUMULASI (CMF={cmf:.2f}) (+{weights['cmf']*2.0:.1f})")
    elif cmf < -0.15:
        score -= weights['cmf']*2.0
        reasons.append(f"SMART MONEY DISTRIBUSI (CMF={cmf:.2f}) (-{weights['cmf']*2.0:.1f})")
    elif cmf > 0:
        score += weights['cmf']*0.5
        reasons.append(f"CMF positif (+{weights['cmf']*0.5:.1f})")
    elif cmf < 0:
        score -= weights['cmf']*0.5
        reasons.append(f"CMF negatif (-{weights['cmf']*0.5:.1f})")
    # Momentum & Oscillator
    rsi = indicators['rsi']
    if rsi < 30:
        score += weights['rsi']*1.5
        reasons.append(f"RSI Oversold (+{weights['rsi']*1.5:.1f})")
    elif rsi > 70:
        score -= weights['rsi']*1.5
        reasons.append(f"RSI Overbought (-{weights['rsi']*1.5:.1f})")
    elif 30 <= rsi <= 50:
        score += weights['rsi']*1.0
        reasons.append(f"RSI {rsi:.0f} recovery (+{weights['rsi']:.1f})")
    mfi = indicators['mfi']
    if mfi < 20:
        score += weights['rsi']*0.8
        reasons.append(f"MFI Oversold (+{weights['rsi']*0.8:.1f})")
    elif mfi > 80:
        score -= weights['rsi']*0.8
        reasons.append(f"MFI Overbought (-{weights['rsi']*0.8:.1f})")
    if indicators['macd_hist'] > 0:
        score += weights['macd']*1.0
        reasons.append(f"MACD positif (+{weights['macd']:.1f})")
    else:
        score -= weights['macd']*1.0
        reasons.append(f"MACD negatif (-{weights['macd']:.1f})")
    # Stochastic + Williams
    stoch = indicators['stoch_k']
    willr = indicators['williams_r']
    osc_contrib = 0
    if stoch < 20 or willr < -80:
        osc_contrib += 1
    if stoch > 80 or willr > -20:
        osc_contrib -= 1
    score += osc_contrib * weights['stoch_willr']
    if osc_contrib > 0:
        reasons.append(f"Stoch/Williams oversold (+{weights['stoch_willr']:.1f})")
    elif osc_contrib < 0:
        reasons.append(f"Stoch/Williams overbought (-{weights['stoch_willr']:.1f})")
    # Aroon, GMMA, KST, Elder
    if indicators['aroon_up'] > 70 and indicators['aroon_down'] < 30:
        score += weights['aroon']*1.0
        reasons.append(f"Aroon Up >70 (+{weights['aroon']:.1f})")
    elif indicators['aroon_down'] > 70:
        score -= weights['aroon']*1.0
        reasons.append(f"Aroon Down >70 (-{weights['aroon']:.1f})")
    if indicators['gmma_spread'] > 0:
        score += weights['gmma']*0.8
        reasons.append(f"GMMA spread positif (+{weights['gmma']*0.8:.1f})")
    else:
        score -= weights['gmma']*0.8
        reasons.append(f"GMMA spread negatif (-{weights['gmma']*0.8:.1f})")
    if indicators['kst'] > 0:
        score += weights['kst']*1.0
        reasons.append(f"KST positif (+{weights['kst']:.1f})")
    else:
        score -= weights['kst']*1.0
        reasons.append(f"KST negatif (-{weights['kst']:.1f})")
    if indicators['elder_bull'] > 0 and indicators['elder_bear'] < 0:
        score += 1.0
        reasons.append("Elder-Ray Bull kuat (+1.0)")
    elif indicators['elder_bull'] < 0 and indicators['elder_bear'] > 0:
        score -= 1.0
        reasons.append("Elder-Ray Bear kuat (-1.0)")
    
    final_score = max(-10, min(10, score / 2.0))
    if final_score >= 6:
        signal, confidence, color = "STRONG BUY", "HIGH", "success"
    elif final_score >= 3:
        signal, confidence, color = "BUY", "MEDIUM", "info"
    elif final_score <= -6:
        signal, confidence, color = "STRONG SELL", "HIGH", "error"
    elif final_score <= -3:
        signal, confidence, color = "SELL", "LOW", "warning"
    else:
        signal, confidence, color = "HOLD", "LOW", "warning"
    return {'signal': signal, 'color': color, 'score': final_score, 'confidence': confidence, 'reasons': reasons}

# ========== FUNGSI PENDUKUNG LAIN (MTF, BACKTEST, LOGGING, DLL) ==========
def get_market_context():
    try:
        ihsg = yf.download("^JKSE", period="10d", progress=False)['Close']
        if len(ihsg) >= 3:
            last_3 = ihsg.iloc[-3:].pct_change().dropna()
            if (last_3 < 0).all():
                return "⚠️ IHSG turun 3 hari berturut-turut", -2
            elif (last_3 > 0).all():
                return "✅ IHSG naik 3 hari berturut-turut", 2
        return "IHSG sideways", 0
    except:
        return "Tidak ada data IHSG", 0

def get_timeframe_signal(df):
    if df.empty or len(df) < 30:
        return "HOLD"
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
    return {
        'buy_entry': buy_entry, 'stop_loss': buy_entry - atr, 'target': resistance,
        'sell_entry': resistance - (atr * 0.5), 'stop_loss_sell': resistance + atr, 'target_sell': support
    }

def run_backtest(df, weights, period_days=90):
    if df.empty or len(df) < period_days:
        return None
    df_test = df.tail(period_days).copy()
    signals = []
    for i in range(30, len(df_test)):
        slice_df = df_test.iloc[:i+1]
        ind = get_latest_indicators(slice_df)
        dec = weighted_decision(ind, 0, "MODERATE", weights, slice_df)
        signals.append(dec['signal'])
    buy = signals.count("STRONG BUY") + signals.count("BUY")
    sell = signals.count("STRONG SELL") + signals.count("SELL")
    hold = signals.count("HOLD")
    return {'buy_signals': buy, 'sell_signals': sell, 'hold_signals': hold, 'total': len(signals)}

def log_signal(ticker, signal, score, price):
    st.session_state.signal_history.insert(0, {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ticker': ticker, 'signal': signal, 'score': score, 'price': price, 'status': 'pending'
    })
    if len(st.session_state.signal_history) > 100:
        st.session_state.signal_history.pop()

def show_toast_if_changed(new, old):
    if old and new != old:
        if new in ["STRONG BUY","BUY"] and old in ["SELL","STRONG SELL","HOLD"]:
            st.toast(f"🟢 Sinyal berubah menjadi {new}!", icon="🔔")
        elif new in ["STRONG SELL","SELL"] and old in ["BUY","STRONG BUY","HOLD"]:
            st.toast(f"🔴 Sinyal berubah menjadi {new}!", icon="⚠️")

# ========== MAIN APP ==========
def main():
    st.sidebar.markdown("# 🤖 Robot Saham")
    ticker_input = st.sidebar.text_input("Kode Saham", DEFAULT_TICKER)
    ticker = ticker_input.upper().strip()
    if not ticker.endswith('.JK') and not ticker.startswith('^'):
        ticker += '.JK'
    timeframe = st.sidebar.selectbox("Timeframe", ["1d","1wk","1mo"], index=0)
    trading_mode = st.sidebar.selectbox(
        "Mode Trading", ["Scalping (5-15 menit)","Intraday (30 menit - 4 jam)","Swing (Daily)","Position (Weekly - Monthly)"], index=2
    )
    st.session_state.trading_mode = trading_mode
    # Tuning bobot
    with st.sidebar.expander("⚙️ Tuning Bobot"):
        weights = {}
        for k in st.session_state.custom_weights:
            weights[k] = st.slider(k, 0.0, 3.0, st.session_state.custom_weights[k], 0.1)
        if st.button("Simpan Bobot"):
            st.session_state.custom_weights = weights
    weights = st.session_state.custom_weights.copy()
    if st.sidebar.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()
    
    # Parameter adaptif
    if trading_mode == "Scalping (5-15 menit)":
        st_period, st_mult = 7, 2.5
    elif trading_mode == "Intraday (30 menit - 4 jam)":
        st_period, st_mult = 10, 3.0
    elif trading_mode == "Swing (Daily)":
        st_period, st_mult = 10, 3.0
    else:
        st_period, st_mult = 20, 3.0
    
    # Load data
    with st.spinner("Memuat data..."):
        period = "2y" if trading_mode in ["Swing (Daily)","Position (Weekly - Monthly)"] else "6mo"
        df_raw = yf.download(ticker, period=period, interval=timeframe, progress=False)
    if df_raw.empty:
        st.error("Data tidak tersedia")
        st.stop()
    df = calculate_all_indicators(df_raw, st_period, st_mult, trading_mode)
    ind = get_latest_indicators(df)
    price = ind['price']
    
    # MTF analysis sederhana untuk alignment (opsional)
    mtf_alignment = 0  # bisa diperbaiki nanti
    trend_strength = "MODERATE"
    decision = weighted_decision(ind, mtf_alignment, trend_strength, weights, df)
    show_toast_if_changed(decision['signal'], st.session_state.last_signal)
    st.session_state.last_signal = decision['signal']
    log_signal(ticker, decision['signal'], decision['score'], price)
    market_text, _ = get_market_context()
    
    # ---- LAYOUT UTAMA (tanpa scroll) ----
    st.title(f"🤖 {ticker}")
    st.caption(f"Mode: {trading_mode} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Baris 1: Harga, RSI, Supertrend, Volume Status
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Harga", f"Rp {price:,.0f}")
    c2.metric("📊 RSI", f"{ind['rsi']:.1f}", help="<30 Oversold, >70 Overbought")
    sup_dir = "🟢 UPTREND" if ind['supertrend_dir'] == 1 else "🔴 DOWNTREND"
    c3.metric("📈 Supertrend", sup_dir, delta=f"Garis: Rp {ind['supertrend_value']:,.0f}")
    c4.metric("📊 Volume", ind['volume_status'])
    
    # Baris 2: Signal, Skor, Keyakinan
    c1, c2, c3 = st.columns(3)
    if decision['color'] == "success":
        c1.success(f"### {decision['signal']}")
    elif decision['color'] == "error":
        c1.error(f"### {decision['signal']}")
    else:
        c1.warning(f"### {decision['signal']}")
    c2.metric("🎯 Skor", f"{decision['score']:.1f}")
    c3.metric("🔒 Keyakinan", decision['confidence'])
    
    # Baris 3: CMF, A/D, ATR
    c1, c2, c3 = st.columns(3)
    cmf_val = ind['cmf']
    cmf_text = f"{'🟢 Akumulasi' if cmf_val>0.15 else '🔴 Distribusi' if cmf_val<-0.15 else '⚪ Netral'}"
    c1.metric("💰 Smart Money (CMF)", cmf_text, delta=f"{cmf_val:.2f}")
    # A/D Line : hitung perubahan 5 hari
    ad_change = 0
    if len(df) >= 6 and 'AD' in df.columns:
        ad_prev = df['AD'].iloc[-5]; ad_curr = df['AD'].iloc[-1]
        ad_change = ((ad_curr - ad_prev) / max(abs(ad_prev),1)) * 100 if ad_prev != 0 else 0
    ad_trend = "🟢 Menguat" if ad_change > 3 else "🔴 Melemah" if ad_change < -3 else "⚪ Stabil"
    c2.metric("📈 A/D Line", ad_trend, delta=f"{ad_change:+.1f}% (5h)")
    atr_val = ind['atr']
    c3.metric("📊 ATR", f"Rp {atr_val:,.0f}", delta=f"SL rekom: Rp {atr_val*1.5:,.0f}")
    
    # Baris 4: Multi-timeframe signal (daily, weekly, monthly)
    # Muat data MTF secara sederhana (gunakan fungsi yang sudah ada)
    with st.spinner("Memuat MTF..."):
        df_d = yf.download(ticker, period="1y", interval="1d", progress=False)
        df_w = yf.download(ticker, period="2y", interval="1wk", progress=False)
        df_m = yf.download(ticker, period="3y", interval="1mo", progress=False)
        sig_d = get_timeframe_signal(calculate_all_indicators(df_d, st_period, st_mult, trading_mode)) if not df_d.empty else "N/A"
        sig_w = get_timeframe_signal(calculate_all_indicators(df_w, st_period, st_mult, trading_mode)) if not df_w.empty else "N/A"
        sig_m = get_timeframe_signal(calculate_all_indicators(df_m, st_period, st_mult, trading_mode)) if not df_m.empty else "N/A"
    c1, c2, c3 = st.columns(3)
    c1.metric("Daily", sig_d)
    c2.metric("Weekly", sig_w)
    c3.metric("Monthly", sig_m)
    
    # Baris 5: Level Entry & Stop Loss & Target + Konteks IHSG
    entry = get_entry_levels(df, atr_val, price)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 Buy Entry", f"Rp {entry['buy_entry']:,.0f}")
    c2.metric("🛑 Stop Loss", f"Rp {entry['stop_loss']:,.0f}", delta_color="inverse")
    c3.metric("🎯 Target", f"Rp {entry['target']:,.0f}")
    c4.metric("🌍 IHSG", market_text[:20], help=market_text)
    
    # ========== KONTEN BAGIAN BAWAH (scroll jika perlu, tapi tidak menghilangkan info utama) ==========
    st.markdown("---")
    with st.expander("📋 Detail Analisis & Kontribusi Semua Indikator"):
        for r in decision['reasons']:
            st.write(f"- {r}")
        st.markdown(f"**Total Skor: {decision['score']}**")
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("📊 Backtest (3 bulan)"):
            if st.button("Jalankan Backtest"):
                res = run_backtest(df, weights, 90)
                if res:
                    st.write(f"BUY: {res['buy_signals']} | SELL: {res['sell_signals']} | HOLD: {res['hold_signals']}")
                else:
                    st.warning("Data tidak cukup")
    with col2:
        with st.expander("📜 Riwayat Sinyal & Ekspor"):
            if st.session_state.signal_history:
                st.dataframe(pd.DataFrame(st.session_state.signal_history).head(5))
                csv = pd.DataFrame(st.session_state.signal_history).to_csv(index=False).encode()
                st.download_button("📥 Ekspor CSV", csv, "signals.csv", "text/csv")
            else:
                st.info("Belum ada sinyal")
    st.caption("⚠️ Edukasi saja, bukan rekomendasi investasi.")
    gc.collect()

if __name__ == "__main__":
    main()
