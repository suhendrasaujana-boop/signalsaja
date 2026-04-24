import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from datetime import datetime
import warnings
import gc
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="Robot Saham - AI Decision Engine", page_icon="🤖")

# CSS untuk margin lebih rapat (tapi ukuran font tetap nyaman)
st.markdown("""
<style>
    .block-container { padding-top: 0.8rem; padding-bottom: 0rem; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.3rem; }
    div[data-testid="stMetricValue"] { font-size: 1.1rem !important; }
    div[data-testid="stMetricLabel"] { font-size: 0.7rem !important; }
</style>
""", unsafe_allow_html=True)

# Konstanta
CACHE_TTL = 300
DEFAULT_TICKER = "BBCA.JK"
BREAKOUT_COOLDOWN_HOURS = 24

# Session state
if 'signal_history' not in st.session_state:
    st.session_state.signal_history = []
if 'last_signal' not in st.session_state:
    st.session_state.last_signal = None
if 'last_breakout_notify_time' not in st.session_state:
    st.session_state.last_breakout_notify_time = None
if 'trading_mode' not in st.session_state:
    st.session_state.trading_mode = "Swing (Daily)"
if 'custom_weights' not in st.session_state:
    st.session_state.custom_weights = {
        'supertrend': 2.0, 'psar': 1.0, 'ema200': 1.5, 'sma_cross': 1.0,
        'mtf': 1.0, 'adx': 1.0, 'volume_spike': 1.5, 'obv': 1.0,
        'cmf': 1.0, 'ad': 0.5, 'rsi': 1.5, 'macd': 1.0,
        'stoch_willr': 1.0, 'aroon': 1.0, 'gmma': 1.0, 'kst': 0.5
    }

# ========== FUNGSI INDIKATOR (semua sudah ada) ==========
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
                    direction.iloc[i] = -1; st_line.iloc[i] = upper.iloc[i]
                else:
                    direction.iloc[i] = 1; st_line.iloc[i] = max(lower.iloc[i], st_line.iloc[i-1])
            else:
                if close.iloc[i] > upper.iloc[i]:
                    direction.iloc[i] = 1; st_line.iloc[i] = lower.iloc[i]
                else:
                    direction.iloc[i] = -1; st_line.iloc[i] = min(upper.iloc[i], st_line.iloc[i-1])
    return st_line, direction

def calculate_psar(df, step=0.02, max_step=0.2):
    high, low, close = df['High'], df['Low'], df['Close']
    psar = pd.Series(index=df.index, dtype=float)
    trend = pd.Series(index=df.index, dtype=int)
    if len(df) < 3:
        return psar, trend
    psar.iloc[0] = low.iloc[0]; trend.iloc[0] = 1; ep = high.iloc[0]; af = step
    for i in range(1, len(df)):
        if trend.iloc[i-1] == 1:
            psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
            if low.iloc[i] <= psar.iloc[i]:
                trend.iloc[i] = -1; psar.iloc[i] = ep; af = step; ep = low.iloc[i]
            else:
                trend.iloc[i] = 1
                if high.iloc[i] > ep: ep = high.iloc[i]; af = min(af + step, max_step)
        else:
            psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
            if high.iloc[i] >= psar.iloc[i]:
                trend.iloc[i] = 1; psar.iloc[i] = ep; af = step; ep = high.iloc[i]
            else:
                trend.iloc[i] = -1
                if low.iloc[i] < ep: ep = low.iloc[i]; af = min(af + step, max_step)
    return psar, trend

def calculate_obv(df):
    close = df['Close']; volume = df['Volume']
    return (np.where(close > close.shift(1), volume, np.where(close < close.shift(1), -volume, 0))).cumsum()

def calculate_mfi(df, window=14):
    typical = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical * df['Volume']
    pos = money_flow.where(typical > typical.shift(1), 0).rolling(window).sum()
    neg = money_flow.where(typical < typical.shift(1), 0).rolling(window).sum()
    ratio = pos / neg
    return 100 - (100 / (1 + ratio))

def calculate_aroon(df, window=25):
    high = df['High']; low = df['Low']
    aroon_up = high.rolling(window).apply(lambda x: (x.argmax() / window) * 100, raw=True)
    aroon_down = low.rolling(window).apply(lambda x: (x.argmin() / window) * 100, raw=True)
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
    r1 = close.pct_change(roc1)*100; r2 = close.pct_change(roc2)*100; r3 = close.pct_change(roc3)*100; r4 = close.pct_change(roc4)*100
    return (r1.rolling(ma1).mean() + r2.rolling(ma2).mean()*2 + r3.rolling(ma3).mean()*3 + r4.rolling(ma4).mean()*4)

def calculate_elder_ray(df, window=13):
    ema = df['Close'].ewm(span=window, adjust=False).mean()
    return df['High'] - ema, df['Low'] - ema

# ========== INDIKATOR UTAMA (SINGLE SOURCE) ==========
@st.cache_data(ttl=CACHE_TTL)
def calculate_all_indicators(df, st_period=10, st_mult=3.0, mode="Swing (Daily)"):
    if df.empty or len(df) < 30:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    close = df['Close']; high = df['High']; low = df['Low']
    has_vol = 'Volume' in df.columns and df['Volume'].sum() != 0
    volume = df['Volume'] if has_vol else pd.Series(0, index=df.index)
    
    # Dasar
    df['SMA20'] = close.rolling(20, min_periods=1).mean()
    df['SMA50'] = close.rolling(50, min_periods=1).mean()
    df['EMA200'] = close.rolling(200).mean() if len(close) >= 200 else close
    df['RSI'] = ta.momentum.rsi(close, window=14) if len(close) >= 14 else 50.0
    if len(close) >= 26:
        macd = ta.trend.MACD(close)
        df['MACD'] = macd.macd(); df['MACD_Signal'] = macd.macd_signal(); df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
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
    if has_vol:
        try:
            df['AD'] = ta.volume.acc_dist_index(high, low, close, volume, fillna=True)
            df['CMF'] = ta.volume.chaikin_money_flow(high, low, close, volume, window=20, fillna=True)
        except:
            df['AD'] = df['CMF'] = 0.0
    else:
        df['AD'] = df['CMF'] = 0.0
    
    # Tambahan
    st_line, st_dir = calculate_supertrend(df, period=st_period, multiplier=st_mult)
    df['ST_Supertrend'] = st_line; df['ST_Dir'] = st_dir
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
            vol = df['Volume'].iloc[-1]; vol_ma = df['Volume_MA'].iloc[-1]
            if vol > vol_ma * 1.5: vol_status = "Tinggi"
            elif vol < vol_ma: vol_status = "Rendah"
            else: vol_status = "Normal"
        else: vol_status = "Normal"
    except: vol_status = "Normal"
    high_30d = df['High'].tail(30).max()
    low_30d = df['Low'].tail(30).min()
    return {
        'price': last['Close'], 'sma20': last['SMA20'], 'sma50': last['SMA50'], 'ema200': last['EMA200'],
        'rsi': last['RSI'], 'macd_hist': last['MACD_Hist'], 'stoch_k': last['Stoch_K'], 'stoch_d': last['Stoch_D'],
        'atr': last['ATR'], 'volume_status': vol_status, 'support': last['support'], 'resistance': last['resistance'],
        'supertrend_dir': last['ST_Dir'], 'supertrend_value': last['ST_Supertrend'], 'psar_dir': last['PSAR_Dir'],
        'obv': last['OBV'], 'obv_ma': last['OBV_MA'], 'mfi': last['MFI'], 'aroon_up': last['Aroon_Up'], 'aroon_down': last['Aroon_Down'],
        'williams_r': last['Williams_R'], 'cci': last['CCI'], 'kst': last.get('KST', 0),
        'elder_bull': last['Elder_Bull'], 'elder_bear': last['Elder_Bear'], 'gmma_spread': last['GMMA_Spread'],
        'cmf': last['CMF'], 'ad': last['AD'], 'high_30d': high_30d, 'low_30d': low_30d
    }

def detect_true_breakout(df, atr):
    if len(df) < 20:
        return {'is_breakout': False, 'strength': 0, 'message': "Data tidak cukup"}
    current_price = df['Close'].iloc[-1]
    resistance = df['resistance'].iloc[-1]
    breakout_threshold = resistance + (atr * 0.5)
    if current_price > breakout_threshold:
        strength = (current_price - resistance) / atr
        return {'is_breakout': True, 'strength': round(strength, 2), 'message': f"✅ TRUE BREAKOUT! Strength: {strength:.1f}x ATR"}
    elif current_price > resistance:
        return {'is_breakout': False, 'strength': 0, 'message': "⚠️ False breakout (belum melewati ATR buffer)"}
    else:
        return {'is_breakout': False, 'strength': 0, 'message': "Tidak ada breakout"}

# ========== DECISION ENGINE (bobot lengkap) ==========
def weighted_decision(indicators, mtf_alignment, trend_strength, weights):
    score = 0
    reasons = []
    # Trend
    if indicators['price'] > indicators['ema200']:
        score += weights['ema200']*1.5; reasons.append(f"Harga > EMA200 (+{weights['ema200']*1.5:.1f})")
    else:
        score -= weights['ema200']*1.5; reasons.append(f"Harga < EMA200 (-{weights['ema200']*1.5:.1f})")
    if indicators['sma20'] > indicators['sma50']:
        score += weights['sma_cross']*1.0; reasons.append(f"SMA20 > SMA50 (+{weights['sma_cross']:.1f})")
    else:
        score -= weights['sma_cross']*1.0; reasons.append(f"SMA20 < SMA50 (-{weights['sma_cross']:.1f})")
    if indicators['supertrend_dir'] == 1:
        score += weights['supertrend']*2.0; reasons.append(f"Supertrend UPTREND (+{weights['supertrend']*2.0:.1f})")
    else:
        score -= weights['supertrend']*2.0; reasons.append(f"Supertrend DOWNTREND (-{weights['supertrend']*2.0:.1f})")
    if indicators['psar_dir'] == 1:
        score += weights['psar']*1.0; reasons.append(f"PSAR UPTREND (+{weights['psar']:.1f})")
    else:
        score -= weights['psar']*1.0; reasons.append(f"PSAR DOWNTREND (-{weights['psar']:.1f})")
    if mtf_alignment > 0.5:
        score += weights['mtf']*1.0; reasons.append(f"MTF Bullish (+{weights['mtf']:.1f})")
    elif mtf_alignment < -0.5:
        score -= weights['mtf']*1.0; reasons.append(f"MTF Bearish (-{weights['mtf']:.1f})")
    if trend_strength == "STRONG TREND":
        score += weights['adx']*1.0; reasons.append(f"ADX Kuat (+{weights['adx']:.1f})")
    elif trend_strength == "WEAK TREND":
        score -= weights['adx']*0.5; reasons.append(f"ADX Lemah (-{weights['adx']*0.5:.1f})")
    # Volume
    if indicators['volume_status'] == "Tinggi":
        score += weights['volume_spike']*1.5; reasons.append(f"Volume Tinggi (+{weights['volume_spike']*1.5:.1f})")
    elif indicators['volume_status'] == "Rendah":
        score -= weights['volume_spike']*1.0; reasons.append(f"Volume Rendah (-{weights['volume_spike']:.1f})")
    if indicators['obv'] > indicators['obv_ma']:
        score += weights['obv']*1.0; reasons.append(f"OBV > MA (+{weights['obv']:.1f})")
    else:
        score -= weights['obv']*1.0; reasons.append(f"OBV < MA (-{weights['obv']:.1f})")
    cmf = indicators['cmf']
    if cmf > 0.15:
        score += weights['cmf']*2.0; reasons.append(f"SMART MONEY AKUMULASI (CMF={cmf:.2f}) (+{weights['cmf']*2.0:.1f})")
    elif cmf < -0.15:
        score -= weights['cmf']*2.0; reasons.append(f"SMART MONEY DISTRIBUSI (CMF={cmf:.2f}) (-{weights['cmf']*2.0:.1f})")
    elif cmf > 0:
        score += weights['cmf']*0.5; reasons.append(f"CMF positif (+{weights['cmf']*0.5:.1f})")
    elif cmf < 0:
        score -= weights['cmf']*0.5; reasons.append(f"CMF negatif (-{weights['cmf']*0.5:.1f})")
    # Momentum
    rsi = indicators['rsi']
    if rsi < 30:
        score += weights['rsi']*1.5; reasons.append(f"RSI Oversold (+{weights['rsi']*1.5:.1f})")
    elif rsi > 70:
        score -= weights['rsi']*1.5; reasons.append(f"RSI Overbought (-{weights['rsi']*1.5:.1f})")
    elif 30 <= rsi <= 50:
        score += weights['rsi']*1.0; reasons.append(f"RSI {rsi:.0f} recovery (+{weights['rsi']:.1f})")
    mfi = indicators['mfi']
    if mfi < 20:
        score += weights['rsi']*0.8; reasons.append(f"MFI Oversold (+{weights['rsi']*0.8:.1f})")
    elif mfi > 80:
        score -= weights['rsi']*0.8; reasons.append(f"MFI Overbought (-{weights['rsi']*0.8:.1f})")
    # Oscillator
    if indicators['macd_hist'] > 0:
        score += weights['macd']*1.0; reasons.append(f"MACD positif (+{weights['macd']:.1f})")
    else:
        score -= weights['macd']*1.0; reasons.append(f"MACD negatif (-{weights['macd']:.1f})")
    stoch = indicators['stoch_k']; willr = indicators['williams_r']
    osc_contrib = (1 if stoch<20 or willr<-80 else 0) - (1 if stoch>80 or willr>-20 else 0)
    score += osc_contrib * weights['stoch_willr']
    if osc_contrib>0: reasons.append(f"Stoch/Williams oversold (+{weights['stoch_willr']:.1f})")
    elif osc_contrib<0: reasons.append(f"Stoch/Williams overbought (-{weights['stoch_willr']:.1f})")
    if indicators['aroon_up'] > 70 and indicators['aroon_down'] < 30:
        score += weights['aroon']*1.0; reasons.append(f"Aroon Up >70 (+{weights['aroon']:.1f})")
    elif indicators['aroon_down'] > 70:
        score -= weights['aroon']*1.0; reasons.append(f"Aroon Down >70 (-{weights['aroon']:.1f})")
    if indicators['gmma_spread'] > 0:
        score += weights['gmma']*0.8; reasons.append(f"GMMA spread positif (+{weights['gmma']*0.8:.1f})")
    else:
        score -= weights['gmma']*0.8; reasons.append(f"GMMA spread negatif (-{weights['gmma']*0.8:.1f})")
    if indicators['kst'] > 0:
        score += weights['kst']*1.0; reasons.append(f"KST positif (+{weights['kst']:.1f})")
    else:
        score -= weights['kst']*1.0; reasons.append(f"KST negatif (-{weights['kst']:.1f})")
    if indicators['elder_bull'] > 0 and indicators['elder_bear'] < 0:
        score += 1.0; reasons.append("Elder-Ray Bull kuat (+1.0)")
    elif indicators['elder_bull'] < 0 and indicators['elder_bear'] > 0:
        score -= 1.0; reasons.append("Elder-Ray Bear kuat (-1.0)")
    final_score = max(-10, min(10, score / 2.0))
    if final_score >= 6: signal, confidence, color = "STRONG BUY", "HIGH", "success"
    elif final_score >= 3: signal, confidence, color = "BUY", "MEDIUM", "info"
    elif final_score <= -6: signal, confidence, color = "STRONG SELL", "HIGH", "error"
    elif final_score <= -3: signal, confidence, color = "SELL", "LOW", "warning"
    else: signal, confidence, color = "HOLD", "LOW", "warning"
    return {'signal': signal, 'color': color, 'score': final_score, 'confidence': confidence, 'reasons': reasons}

# ========== FUNGSI PENDUKUNG ==========
def get_market_context():
    try:
        ihsg = yf.download("^JKSE", period="5d", progress=False, auto_adjust=False)['Close']
        if len(ihsg) >= 3:
            last_3 = ihsg.iloc[-3:].pct_change().dropna()
            if (last_3 < 0).all(): return "⚠️ IHSG turun 3 hari", -2
            elif (last_3 > 0).all(): return "✅ IHSG naik 3 hari", 2
        return "IHSG sideways", 0
    except: return "⚠️ IHSG tidak tersedia", 0

def get_timeframe_signal(df):
    if df.empty or len(df) < 30: return "HOLD"
    ind = get_latest_indicators(df)
    score = (1 if ind['price'] > ind['ema200'] else -1) + (1 if ind['supertrend_dir']==1 else -1)
    if ind['volume_status'] == "Tinggi": score+=1
    elif ind['volume_status'] == "Rendah": score-=0.5
    if score >= 1.5: return "BUY"
    elif score <= -1.5: return "SELL"
    else: return "HOLD"

def get_entry_levels(df, atr, price):
    support = df['support'].iloc[-1] if 'support' in df.columns else price*0.95
    resistance = df['resistance'].iloc[-1] if 'resistance' in df.columns else price*1.05
    buy_entry = support + atr*0.5
    return {'buy_entry': buy_entry, 'stop_loss': buy_entry - atr, 'target': resistance}

def run_backtest(df, weights, period_days=90):
    if df.empty or len(df) < period_days: return None
    df_test = df.tail(period_days).copy()
    signals = []
    for i in range(30, len(df_test)):
        slice_df = df_test.iloc[:i+1]
        ind = get_latest_indicators(slice_df)
        dec = weighted_decision(ind, 0, "MODERATE", weights)
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

def show_breakout_alert(breakout, last_time):
    if breakout['is_breakout']:
        now = datetime.now()
        if last_time is None or (now - last_time).total_seconds() > BREAKOUT_COOLDOWN_HOURS*3600:
            st.toast(f"🚀 {breakout['message']}", icon="📈")
            st.session_state.last_breakout_notify_time = now
            return True
    return False

def get_signal_interpretation(signal):
    interpretations = {
        "STRONG BUY": "📌 **Sistem melihat peluang naik sangat kuat.** Momentum + trend + volume mendukung. Cocok untuk **entry sekarang** atau akumulasi bertahap.",
        "BUY": "📌 **Kondisi cukup bagus, tapi belum sempurna.** Masuk boleh, tapi lebih aman jika **tunggu koreksi kecil** dulu.",
        "HOLD": "📌 **Pasar tidak jelas.** Tidak ada keunggulan arah. Lebih baik **tunggu konfirmasi** breakout atau breakdown.",
        "SELL": "📌 **Tekanan turun mulai dominan.** Jika sudah punya posisi, mulai **pertimbangkan exit bertahap**.",
        "STRONG SELL": "📌 **Kondisi lemah.** Risiko turun lebih besar daripada peluang naik. **Hindari entry.**"
    }
    return interpretations.get(signal, "Netral.")

def calculate_probabilities(df):
    try:
        bull = sum([df['RSI'].iloc[-1] < 35, df['Close'].iloc[-1] > df['SMA20'].iloc[-1], df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]])
        bear = 3 - bull
        total = bull + bear
        return (bull/total*100) if total>0 else 50, (bear/total*100) if total>0 else 50
    except:
        return 50, 50

def position_sizing(capital, risk_percent, entry_price, stop_loss):
    if entry_price and stop_loss and capital>0:
        risk_amount = capital * (risk_percent/100)
        price_risk = abs(entry_price - stop_loss)
        if price_risk > 0:
            shares = int(risk_amount / price_risk)
            pos_value = shares * entry_price
            return shares, pos_value, risk_amount
    return 0, 0, 0

# ========== MAIN APP ==========
def main():
    st.sidebar.markdown("# 🤖 Robot Saham v2")
    ticker_input = st.sidebar.text_input("Kode Saham", DEFAULT_TICKER)
    ticker = ticker_input.upper().strip()
    if not ticker.endswith('.JK') and not ticker.startswith('^'): ticker += '.JK'
    timeframe = st.sidebar.selectbox("Timeframe", ["1d","1wk","1mo"], index=0)
    trading_mode = st.sidebar.selectbox(
        "Mode Trading", ["Scalping (5-15 menit)","Intraday (30 menit - 4 jam)","Swing (Daily)","Position (Weekly - Monthly)"], index=2
    )
    st.session_state.trading_mode = trading_mode
    # Penjelasan mode (opsional)
    mode_hints = {
        "Scalping (5-15 menit)": "⚡ Fokus: Volume + Momentum cepat (noise tinggi)",
        "Intraday (30 menit - 4 jam)": "📊 Fokus: Breakout + Trend harian",
        "Swing (Daily)": "📈 Fokus: SMA + RSI + trend stabil, hold 1-5 hari",
        "Position (Weekly - Monthly)": "🧭 Fokus: Trend besar + akumulasi, hold mingguan-bulanan"
    }
    st.sidebar.caption(mode_hints[trading_mode])
    
    with st.sidebar.expander("⚙️ Tuning Bobot Indikator"):
        weights = {}
        for k in st.session_state.custom_weights:
            weights[k] = st.slider(k, 0.0, 3.0, st.session_state.custom_weights[k], 0.1)
        if st.button("Simpan Bobot"):
            st.session_state.custom_weights = weights
    weights = st.session_state.custom_weights.copy()
    
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Parameter adaptif
    if trading_mode == "Scalping (5-15 menit)": st_period, st_mult = 7, 2.5
    elif trading_mode == "Intraday (30 menit - 4 jam)": st_period, st_mult = 10, 3.0
    elif trading_mode == "Swing (Daily)": st_period, st_mult = 10, 3.0
    else: st_period, st_mult = 20, 3.0
    
    # Load data
    with st.spinner("Memuat data..."):
        period = "2y" if trading_mode in ["Swing (Daily)","Position (Weekly - Monthly)"] else "6mo"
        df_raw = yf.download(ticker, period=period, interval=timeframe, progress=False, auto_adjust=False)
    if df_raw.empty:
        st.error("Data tidak tersedia")
        st.stop()
    df = calculate_all_indicators(df_raw, st_period, st_mult, trading_mode)
    ind = get_latest_indicators(df)
    price = ind['price']
    
    # MTF alignment dummy
    mtf_alignment = 0
    trend_strength = "MODERATE"
    decision = weighted_decision(ind, mtf_alignment, trend_strength, weights)
    show_toast_if_changed(decision['signal'], st.session_state.last_signal)
    st.session_state.last_signal = decision['signal']
    log_signal(ticker, decision['signal'], decision['score'], price)
    market_text, _ = get_market_context()
    
    # Breakout detection
    breakout = detect_true_breakout(df, ind['atr'])
    show_breakout_alert(breakout, st.session_state.last_breakout_notify_time)
    
    # Weight factor based on mode
    weight_factors = {"Scalping":0.7, "Intraday":1.0, "Swing":1.2, "Position":1.3}
    wf = weight_factors.get(trading_mode.split()[0], 1.0)
    adjusted_score = decision['score'] * wf
    entry_eligible = adjusted_score >= 5
    entry_warning = adjusted_score >= 2 and not entry_eligible
    
    # Probabilitas
    bull_prob, bear_prob = calculate_probabilities(df)
    
    # ========== DASHBOARD UTAMA (6 baris, sedikit scroll) ==========
    st.title(f"🤖 {ticker}")
    st.caption(f"Mode: {trading_mode} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Baris 1: Harga, RSI, Supertrend, Volume Status
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("💰 Harga", f"Rp {price:,.0f}")
    c2.metric("📊 RSI", f"{ind['rsi']:.1f}")
    sup_dir = "🟢 UPTREND" if ind['supertrend_dir']==1 else "🔴 DOWNTREND"
    c3.metric("📈 Supertrend", sup_dir, delta=f"Rp {ind['supertrend_value']:,.0f}")
    c4.metric("📊 Volume", ind['volume_status'])
    
    # Baris 2: High/Low 30d, ATR
    c1,c2,c3 = st.columns(3)
    c1.metric("📈 High 30d", f"Rp {ind['high_30d']:,.0f}")
    c2.metric("📉 Low 30d", f"Rp {ind['low_30d']:,.0f}")
    c3.metric("📊 ATR", f"Rp {ind['atr']:,.0f}", delta=f"SL: Rp {ind['atr']*1.5:,.0f}")
    
    # Baris 3: Signal & Skor & Keyakinan
    c1,c2,c3 = st.columns(3)
    if decision['signal'] in ["STRONG BUY","BUY"]:
        c1.success(f"### 🟢 {decision['signal']}")
    elif decision['signal'] in ["STRONG SELL","SELL"]:
        c1.error(f"### 🔴 {decision['signal']}")
    else:
        c1.warning(f"### 🟡 {decision['signal']}")
    c2.metric("🎯 Skor", f"{decision['score']:.1f}")
    c3.metric("🔒 Keyakinan", decision['confidence'])
    
    # Baris 4: Smart Money (CMF + AD) dan Breakout
    c1,c2,c3 = st.columns(3)
    cmf_val = ind['cmf']
    cmf_text = "🟢 Akumulasi" if cmf_val>0.15 else "🔴 Distribusi" if cmf_val<-0.15 else "⚪ Netral"
    c1.metric("💰 Smart Money (CMF)", cmf_text, delta=f"{cmf_val:.2f}")
    ad_change = 0
    if len(df)>=6 and 'AD' in df.columns:
        ad_prev = df['AD'].iloc[-5]; ad_curr = df['AD'].iloc[-1]
        ad_change = ((ad_curr - ad_prev) / max(abs(ad_prev),1))*100
    ad_trend = "🟢 Menguat" if ad_change>3 else "🔴 Melemah" if ad_change<-3 else "⚪ Stabil"
    c2.metric("📈 A/D Line", ad_trend, delta=f"{ad_change:+.1f}% (5h)")
    c3.metric("🚀 Breakout", breakout['message'][:30], help=breakout['message'])
    
    # Baris 5: Multi-timeframe
    with st.spinner("MTF..."):
        df_d = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=False)
        df_w = yf.download(ticker, period="2y", interval="1wk", progress=False, auto_adjust=False)
        df_m = yf.download(ticker, period="3y", interval="1mo", progress=False, auto_adjust=False)
        sig_d = get_timeframe_signal(calculate_all_indicators(df_d, st_period, st_mult, trading_mode)) if not df_d.empty else "N/A"
        sig_w = get_timeframe_signal(calculate_all_indicators(df_w, st_period, st_mult, trading_mode)) if not df_w.empty else "N/A"
        sig_m = get_timeframe_signal(calculate_all_indicators(df_m, st_period, st_mult, trading_mode)) if not df_m.empty else "N/A"
    c1,c2,c3 = st.columns(3)
    c1.metric("Daily", sig_d); c2.metric("Weekly", sig_w); c3.metric("Monthly", sig_m)
    
    # Baris 6: Entry & Stop Loss + IHSG + Probabilitas
    entry = get_entry_levels(df, ind['atr'], price)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("🎯 Buy Entry", f"Rp {entry['buy_entry']:,.0f}")
    c2.metric("🛑 Stop Loss", f"Rp {entry['stop_loss']:,.0f}", delta_color="inverse")
    c3.metric("🎯 Target", f"Rp {entry['target']:,.0f}")
    c4.metric("🌍 IHSG", market_text[:20], help=market_text)
    c5,c6 = st.columns(2)
    c5.metric("📈 Bullish Prob", f"{bull_prob:.0f}%")
    c6.metric("📉 Bearish Prob", f"{bear_prob:.0f}%")
    
    # Baris 7: Keputusan Final Entry Layak/Tidak
    if entry_eligible:
        st.success(f"🟢 **ENTRY LAYAK** (Adjusted Score: {adjusted_score:.1f} | Mode: {trading_mode})")
        st.write("📌 Saran: Entry sesuai level, gunakan stop loss.")
    elif entry_warning:
        st.warning(f"🟡 **ENTRY BOLEH TAPI HATI-HATI** (Adjusted Score: {adjusted_score:.1f})")
        st.write("📌 Saran: Gunakan ukuran posisi 50% dari normal.")
    else:
        st.error(f"🔴 **TIDAK LAYAK ENTRY** (Adjusted Score: {adjusted_score:.1f})")
        st.write("📌 Saran: Tunggu atau cari saham lain.")
    
    # Interpretasi sinyal (opsional, bisa di expander)
    with st.expander("📖 Arti Sinyal Saat Ini"):
        st.markdown(get_signal_interpretation(decision['signal']))
    
    # ========== EXPANDER UNTUK DETAIL LAINNYA ==========
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
    
    with st.expander("📐 Position Sizing"):
        capital = st.number_input("Modal (Rp)", value=100_000_000.0, step=10_000_000.0)
        risk_percent = st.number_input("Risiko per trade (%)", value=2.0, step=0.5)
        shares, pos_value, risk_amount = position_sizing(capital, risk_percent, entry['buy_entry'], entry['stop_loss'])
        if shares > 0:
            st.write(f"Jumlah saham: {shares:,} lembar")
            st.write(f"Nilai posisi: Rp {pos_value:,.0f} ({pos_value/capital*100:.1f}% dari modal)")
            st.write(f"Risiko stop loss: Rp {risk_amount:,.0f} ({risk_percent:.1f}%)")
    
    st.caption("⚠️ Edukasi saja, bukan rekomendasi investasi.")
    gc.collect()

if __name__ == "__main__":
    main()
