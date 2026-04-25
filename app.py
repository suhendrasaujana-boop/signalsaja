import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from datetime import datetime
import warnings
import gc
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="Robot Saham PRO - Decision Engine", page_icon="🤖")

# ========== CUSTOM CSS PRO (clean, card-style) ==========
st.markdown("""
<style>
    /* Global */
    .main {
        background-color: #f8f9fa;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        max-width: 1200px;
    }
    /* Custom card */
    .card {
        background-color: white;
        border-radius: 16px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 600;
        color: #212529;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #6c757d;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .signal-box {
        text-align: center;
        padding: 0.5rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.6rem;
    }
    .signal-badge {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
    }
    hr {
        margin: 0.5rem 0;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e9ecef;
        border-radius: 12px;
    }
    /* sidebar lebih rapi */
    [data-testid="stSidebar"] {
        background-color: #f1f3f5;
        border-right: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# ========== KONSTAN & SESSION STATE ==========
CACHE_TTL = 300
DEFAULT_TICKER = "BBCA.JK"
BREAKOUT_COOLDOWN_HOURS = 24

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
        'trend': 2.0, 'momentum': 1.5, 'volume': 1.5, 'smart_money': 1.5, 'structure': 1.0
    }

# ========== FUNGSI INDIKATOR (TIDAK BERUBAH) ==========
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
    close = df['Close']
    volume = df['Volume']
    return (np.where(close > close.shift(1), volume,
                     np.where(close < close.shift(1), -volume, 0))).cumsum()

def calculate_aroon(df, window=25):
    high = df['High']
    low = df['Low']
    aroon_up = high.rolling(window).apply(lambda x: (x.argmax() / window) * 100, raw=True)
    aroon_down = low.rolling(window).apply(lambda x: (x.argmin() / window) * 100, raw=True)
    return aroon_up, aroon_down

def calculate_gmma(df, fast=[3,5,8,10,12,15], slow=[30,35,40,45,50,60]):
    close = df['Close']
    gmma_fast = [close.ewm(span=p, adjust=False).mean() for p in fast]
    gmma_slow = [close.ewm(span=p, adjust=False).mean() for p in slow]
    return gmma_fast, gmma_slow

def calculate_williams_r(df, window=14):
    high = df['High'].rolling(window).max()
    low = df['Low'].rolling(window).min()
    return (high - df['Close']) / (high - low) * -100

def calculate_kst(df, roc1=10, roc2=15, roc3=20, roc4=30, ma1=10, ma2=10, ma3=10, ma4=15):
    close = df['Close']
    r1 = close.pct_change(roc1)*100
    r2 = close.pct_change(roc2)*100
    r3 = close.pct_change(roc3)*100
    r4 = close.pct_change(roc4)*100
    return (r1.rolling(ma1).mean() + r2.rolling(ma2).mean()*2 +
            r3.rolling(ma3).mean()*3 + r4.rolling(ma4).mean()*4)

def calculate_elder_ray(df, window=13):
    ema = df['Close'].ewm(span=window, adjust=False).mean()
    return df['High'] - ema, df['Low'] - ema

# ========== MARKET REGIME & STRUCTURE ==========
def detect_market_regime(df, adx_threshold=25):
    if len(df) < 30:
        return "SIDEWAYS"
    adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14).adx()
    current_adx = adx.iloc[-1]
    ema20 = df['Close'].ewm(span=20).mean()
    ema_slope = ema20.iloc[-1] - ema20.iloc[-5]
    if current_adx > adx_threshold and abs(ema_slope / ema20.iloc[-1]) > 0.002:
        return "TREND"
    else:
        return "SIDEWAYS"

def get_swing_points(df, lookback=5):
    if len(df) < lookback*2:
        return df['Low'].iloc[-1], df['High'].iloc[-1]
    low = df['Low'].iloc[-lookback*2:]
    high = df['High'].iloc[-lookback*2:]
    swing_low = low.rolling(lookback, center=True).min().iloc[-1]
    swing_high = high.rolling(lookback, center=True).max().iloc[-1]
    return swing_low, swing_high

def get_mtf_alignment(df_daily, df_weekly, df_monthly):
    def sig_to_num(df):
        if df.empty or len(df) < 30:
            return 0
        ind = get_latest_indicators(df)
        score = 0
        if ind['price'] > ind['ema200']:
            score += 1
        else:
            score -= 1
        if ind['supertrend_dir'] == 1:
            score += 1
        else:
            score -= 1
        return 1 if score >= 1 else -1 if score <= -1 else 0
    d = sig_to_num(df_daily)
    w = sig_to_num(df_weekly)
    m = sig_to_num(df_monthly)
    return d*0.5 + w*0.3 + m*0.2

# ========== INDIKATOR UTAMA (SAMA PERSIS) ==========
@st.cache_data(ttl=CACHE_TTL)
def calculate_all_indicators(df, st_period=10, st_mult=3.0, mode="Swing (Daily)"):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty or len(df) < 30:
        idx = df.index if not df.empty else pd.date_range(end=datetime.today(), periods=1)
        dummy_df = pd.DataFrame(index=idx)
        required_cols = [
            'Close','High','Low','Volume','SMA20','SMA50','EMA200','RSI',
            'MACD','MACD_Signal','MACD_Hist','Stoch_K','Stoch_D','ATR',
            'Volume_MA','support','resistance','AD','CMF','ST_Supertrend','ST_Dir',
            'PSAR','PSAR_Dir','OBV','OBV_MA','MFI','Aroon_Up','Aroon_Down',
            'Williams_R','CCI','KST','Elder_Bull','Elder_Bear','GMMA_Spread'
        ]
        for col in required_cols:
            dummy_df[col] = 0.0
        if not df.empty and 'Close' in df.columns:
            dummy_df['Close'] = df['Close'].values
        return dummy_df

    close = df['Close']
    high = df['High']
    low = df['Low']
    has_vol = 'Volume' in df.columns and df['Volume'].sum() != 0
    volume = df['Volume'] if has_vol else pd.Series(0, index=df.index)

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
    if has_vol:
        try:
            df['AD'] = ta.volume.acc_dist_index(high, low, close, volume, fillna=True)
            df['CMF'] = ta.volume.chaikin_money_flow(high, low, close, volume, window=20, fillna=True)
        except:
            df['AD'] = df['CMF'] = 0.0
    else:
        df['AD'] = df['CMF'] = 0.0

    st_line, st_dir = calculate_supertrend(df, period=st_period, multiplier=st_mult)
    df['ST_Supertrend'] = st_line
    df['ST_Dir'] = st_dir
    psar_line, psar_dir = calculate_psar(df)
    df['PSAR'] = psar_line
    df['PSAR_Dir'] = psar_dir
    if has_vol:
        df['OBV'] = calculate_obv(df)
        df['OBV_MA'] = df['OBV'].rolling(20).mean()
        df['MFI'] = ta.volume.money_flow_index(high, low, close, volume, window=14, fillna=True)
    else:
        df['OBV'] = df['OBV_MA'] = 0.0
        df['MFI'] = 50.0
    aroon_up, aroon_down = calculate_aroon(df)
    df['Aroon_Up'] = aroon_up
    df['Aroon_Down'] = aroon_down
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
    df['Elder_Bull'] = bull
    df['Elder_Bear'] = bear
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
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    last_dict = df.iloc[-1].to_dict()
    def get_val(key, default=0.0):
        return last_dict.get(key, default)
    price = get_val('Close', 0.0)
    vol_status = "Normal"
    if 'Volume_MA' in df.columns and 'Volume' in df.columns:
        try:
            vol_ma = df['Volume_MA'].iloc[-1]
            if vol_ma > 0:
                vol = df['Volume'].iloc[-1]
                if vol > vol_ma * 1.5:
                    vol_status = "Tinggi"
                elif vol < vol_ma:
                    vol_status = "Rendah"
        except:
            pass
    high_30d = df['High'].tail(30).max() if ('High' in df and len(df) >= 30) else price
    low_30d = df['Low'].tail(30).min() if ('Low' in df and len(df) >= 30) else price
    swing_low, swing_high = get_swing_points(df)
    return {
        'price': price,
        'sma20': get_val('SMA20', price),
        'sma50': get_val('SMA50', price),
        'ema200': get_val('EMA200', price),
        'rsi': get_val('RSI', 50.0),
        'macd_hist': get_val('MACD_Hist', 0.0),
        'stoch_k': get_val('Stoch_K', 50.0),
        'stoch_d': get_val('Stoch_D', 50.0),
        'atr': get_val('ATR', price * 0.02),
        'volume_status': vol_status,
        'support': get_val('support', price * 0.95),
        'resistance': get_val('resistance', price * 1.05),
        'supertrend_dir': get_val('ST_Dir', 0),
        'supertrend_value': get_val('ST_Supertrend', price),
        'psar_dir': get_val('PSAR_Dir', 0),
        'obv': get_val('OBV', 0.0),
        'obv_ma': get_val('OBV_MA', 0.0),
        'mfi': get_val('MFI', 50.0),
        'aroon_up': get_val('Aroon_Up', 50.0),
        'aroon_down': get_val('Aroon_Down', 50.0),
        'williams_r': get_val('Williams_R', -50.0),
        'cci': get_val('CCI', 0.0),
        'kst': get_val('KST', 0.0),
        'elder_bull': get_val('Elder_Bull', 0.0),
        'elder_bear': get_val('Elder_Bear', 0.0),
        'gmma_spread': get_val('GMMA_Spread', 0.0),
        'cmf': get_val('CMF', 0.0),
        'ad': get_val('AD', 0.0),
        'high_30d': high_30d,
        'low_30d': low_30d,
        'swing_low': swing_low,
        'swing_high': swing_high
    }

# ========== DECISION ENGINE FINAL (CONDITIONAL SCORING) ==========
def weighted_decision_final(indicators, mtf_alignment, regime, weights):
    score = 0
    reasons = []

    # Trend group
    trend_score = 0
    if indicators['price'] > indicators['ema200']:
        trend_score += 1.5
    else:
        trend_score -= 1.5
    if indicators['supertrend_dir'] == 1:
        trend_score += 2.0
    else:
        trend_score -= 2.0
    if indicators['psar_dir'] == 1:
        trend_score += 1.0
    else:
        trend_score -= 1.0
    if indicators['sma20'] > indicators['sma50']:
        trend_score += 1.0
    else:
        trend_score -= 1.0
    trend_score = trend_score / 4.0

    # Momentum (hanya RSI)
    rsi = indicators['rsi']
    mom_score = 0.0
    if rsi < 30:
        mom_score = 1.0
    elif rsi > 70:
        mom_score = -1.0
    elif rsi < 45:
        mom_score = 0.3
    elif rsi > 55:
        mom_score = -0.3

    # Volume
    vol_score = 0
    if indicators['volume_status'] == "Tinggi":
        vol_score += 1.0
    elif indicators['volume_status'] == "Rendah":
        vol_score -= 1.0
    if indicators['obv'] > indicators['obv_ma']:
        vol_score += 0.5
    else:
        vol_score -= 0.5
    vol_score = vol_score / 2.0

    # Smart Money (CMF only)
    smart_score = 0
    cmf = indicators['cmf']
    if cmf > 0.15:
        smart_score = 1.5
    elif cmf < -0.15:
        smart_score = -1.5
    elif cmf > 0:
        smart_score = 0.5
    elif cmf < 0:
        smart_score = -0.5

    # Structure
    struct_score = 0
    price = indicators['price']
    swing_low = indicators['swing_low']
    swing_high = indicators['swing_high']
    if price > swing_high:
        struct_score = 1.0
    elif price < swing_low:
        struct_score = -1.0

    # Conditional scoring
    if regime == "TREND":
        score = (trend_score * 2.5) + (struct_score * 1.5) + (vol_score * 1.0) + (smart_score * 0.8)
        if mom_score < -0.5:
            score -= 0.5
        elif mom_score > 0.5:
            score += 0.3
        reasons.append(f"TREND MODE: trend={trend_score:.2f}, struct={struct_score:.2f}")
    else:
        score = (mom_score * 2.0) + (smart_score * 1.0) + (vol_score * 0.5) + (struct_score * 0.5)
        if price <= indicators['support'] and mom_score > 0:
            score += 1.0
        elif price >= indicators['resistance'] and mom_score < 0:
            score -= 1.0
        reasons.append(f"SIDEWAYS MODE: mom={mom_score:.2f}, smart={smart_score:.2f}")

    score += mtf_alignment * 1.0
    reasons.append(f"MTF Alignment: {mtf_alignment:.2f}")

    final_score = max(-10, min(10, score * 2.0))

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

# ========== ENTRY & STOP LOSS ==========
def get_entry_levels_advanced(df, indicators, regime):
    price = indicators['price']
    atr = indicators['atr']
    swing_low = indicators['swing_low']
    swing_high = indicators['swing_high']
    ema20 = df['Close'].ewm(span=20).mean().iloc[-1]
    support = indicators['support']
    resistance = indicators['resistance']
    
    if regime == "TREND":
        if price > ema20 and price > swing_high:
            buy_entry = price + atr*0.2
            stop_loss = swing_low - atr*0.5
            target = resistance if resistance > price else price + atr*3
        else:
            buy_entry = ema20
            stop_loss = swing_low - atr*0.5
            target = swing_high + atr
    else:
        buy_entry = support + atr*0.3
        stop_loss = swing_low - atr*0.5
        target = resistance - atr*0.5
    return {
        'buy_entry': buy_entry,
        'stop_loss': stop_loss,
        'target': target
    }

# ========== BACKTEST ==========
def run_backtest_advanced(df, weights, regime_func, period_days=180):
    if df.empty or len(df) < period_days:
        return None
    df_test = df.tail(period_days).copy()
    balance = 100_000_000
    positions = []
    equity = [balance]
    trades = []
    
    for i in range(60, len(df_test)):
        slice_df = df_test.iloc[:i+1]
        ind = get_latest_indicators(slice_df)
        regime = regime_func(slice_df)
        dec = weighted_decision_final(ind, 0, regime, weights)
        if dec['signal'] in ['BUY', 'STRONG BUY'] and len(positions) == 0:
            entry_price = ind['price']
            stop_loss = ind['swing_low'] - ind['atr']*0.5
            target = ind['resistance']
            positions.append({'entry': entry_price, 'stop': stop_loss, 'target': target})
        elif len(positions) > 0:
            pos = positions[0]
            current_price = ind['price']
            if current_price <= pos['stop'] or current_price >= pos['target']:
                exit_price = current_price
                pnl = (exit_price - pos['entry']) / pos['entry'] * balance
                balance += pnl
                trades.append({'entry': pos['entry'], 'exit': exit_price, 'pnl': pnl})
                positions.pop()
            equity.append(balance)
        else:
            equity.append(balance)
    if len(trades) == 0:
        return None
    win_trades = [t for t in trades if t['pnl'] > 0]
    winrate = len(win_trades)/len(trades)*100 if trades else 0
    profit_factor = sum(t['pnl'] for t in trades if t['pnl']>0) / abs(sum(t['pnl'] for t in trades if t['pnl']<0)) if sum(t['pnl'] for t in trades if t['pnl']<0) != 0 else 0
    max_drawdown = (max(equity) - min(equity)) / max(equity) * 100 if max(equity) > 0 else 0
    return {
        'trades': len(trades),
        'winrate': winrate,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'final_balance': balance,
        'equity': equity
    }

# ========== FUNGSI PENDUKUNG UI ==========
def get_market_context():
    try:
        ihsg = yf.download("^JKSE", period="5d", progress=False, auto_adjust=False)['Close']
        if len(ihsg) >= 3:
            last_3 = ihsg.iloc[-3:].pct_change().dropna()
            if (last_3 < 0).all():
                return "⚠️ IHSG turun 3 hari", -2
            elif (last_3 > 0).all():
                return "✅ IHSG naik 3 hari", 2
        return "IHSG sideways", 0
    except:
        return "⚠️ IHSG tidak tersedia", 0

def get_timeframe_signal(df):
    if df.empty or len(df) < 30:
        return "HOLD"
    ind = get_latest_indicators(df)
    score = 0
    if ind['price'] > ind['ema200']:
        score += 1
    else:
        score -= 1
    if ind['supertrend_dir'] == 1:
        score += 1
    else:
        score -= 1
    if ind['volume_status'] == "Tinggi":
        score += 1
    elif ind['volume_status'] == "Rendah":
        score -= 0.5
    if score >= 1.5:
        return "BUY"
    elif score <= -1.5:
        return "SELL"
    else:
        return "HOLD"

def log_signal(ticker, signal, score, price):
    record = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ticker': ticker,
        'signal': signal,
        'score': score,
        'price': price,
        'status': 'pending'
    }
    st.session_state.signal_history.insert(0, record)
    if len(st.session_state.signal_history) > 100:
        st.session_state.signal_history.pop()

def show_toast_if_changed(new_signal, old_signal):
    if old_signal and new_signal != old_signal:
        if new_signal in ["STRONG BUY", "BUY"] and old_signal in ["SELL", "STRONG SELL", "HOLD"]:
            st.toast(f"🟢 Sinyal berubah menjadi {new_signal}!", icon="🔔")
        elif new_signal in ["STRONG SELL", "SELL"] and old_signal in ["BUY", "STRONG BUY", "HOLD"]:
            st.toast(f"🔴 Sinyal berubah menjadi {new_signal}!", icon="⚠️")
        elif new_signal == "HOLD" and old_signal in ["BUY","STRONG BUY","SELL","STRONG SELL"]:
            st.toast(f"⚪ Sinyal menjadi HOLD, waspadai perubahan", icon="ℹ️")

def show_breakout_alert(breakout, last_time):
    if breakout['is_breakout']:
        now = datetime.now()
        if last_time is None or (now - last_time).total_seconds() > BREAKOUT_COOLDOWN_HOURS * 3600:
            st.toast(f"🚀 {breakout['message']}", icon="📈")
            st.session_state.last_breakout_notify_time = now
            return True
    return False

def get_signal_interpretation(signal):
    interpretations = {
        "STRONG BUY": "Momentum beli sangat kuat. Tren dan struktur mendukung. Cocok untuk entry.",
        "BUY": "Kondisi cukup baik. Ada peluang naik, namun tetap gunakan stop loss.",
        "HOLD": "Tidak ada sinyal jelas. Tunggu breakout atau konfirmasi arah.",
        "SELL": "Tekanan jual mulai terlihat. Pertimbangkan exit bertahap.",
        "STRONG SELL": "Kondisi lemah. Risiko turun tinggi, hindari entry."
    }
    return interpretations.get(signal, "Netral.")

def calculate_probabilities(df):
    try:
        bull = 0
        if df['RSI'].iloc[-1] < 35:
            bull += 1
        if df['Close'].iloc[-1] > df['SMA20'].iloc[-1]:
            bull += 1
        if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]:
            bull += 1
        bear = 3 - bull
        total = bull + bear
        return (bull/total*100) if total>0 else 50, (bear/total*100) if total>0 else 50
    except:
        return 50, 50

def position_sizing(capital, risk_percent, entry_price, stop_loss):
    if entry_price and stop_loss and capital > 0:
        risk_amount = capital * (risk_percent / 100)
        price_risk = abs(entry_price - stop_loss)
        if price_risk > 0:
            shares = int(risk_amount / price_risk)
            pos_value = shares * entry_price
            return shares, pos_value, risk_amount
    return 0, 0, 0

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

# ========== MAIN APP (UI PRO) ==========
def main():
    # ---------------- SIDEBAR (input & kontrol) ----------------
    st.sidebar.markdown("# 🤖 Robot Saham PRO")
    ticker_input = st.sidebar.text_input("Kode Saham", DEFAULT_TICKER)
    ticker = ticker_input.upper().strip()
    if not ticker.endswith('.JK') and not ticker.startswith('^'):
        ticker += '.JK'
    timeframe = st.sidebar.selectbox("Timeframe", ["1d", "1wk", "1mo"], index=0)
    
    trading_mode = st.sidebar.selectbox(
        "Mode Trading",
        ["Scalping (5-15 menit)", "Intraday (30 menit - 4 jam)", "Swing (Daily)", "Position (Weekly - Monthly)"],
        index=2
    )
    st.session_state.trading_mode = trading_mode
    if trading_mode in ["Scalping (5-15 menit)", "Intraday (30 menit - 4 jam)"]:
        st.sidebar.warning("⚠️ Data daily → sinyal cocok untuk trading harian.")
    
    # Parameter Supertrend
    if trading_mode == "Scalping (5-15 menit)":
        st_period, st_mult = 7, 2.5
    elif trading_mode == "Intraday (30 menit - 4 jam)":
        st_period, st_mult = 8, 2.7
    elif trading_mode == "Swing (Daily)":
        st_period, st_mult = 10, 3.0
    else:
        st_period, st_mult = 20, 3.0

    # Toggle tampilan advanced (hanya untuk indikator lanjutan)
    show_advanced = st.sidebar.checkbox("Tampilkan indikator lanjutan", value=False)
    
    # Bobot (hanya untuk referensi, tidak digunakan langsung di decision final, tapi disimpan)
    with st.sidebar.expander("⚙️ Bobot (referensi)"):
        weights = {}
        weights['trend'] = st.slider("Trend", 0.0, 3.0, st.session_state.custom_weights.get('trend', 2.0), 0.1)
        weights['momentum'] = st.slider("Momentum", 0.0, 3.0, st.session_state.custom_weights.get('momentum', 1.5), 0.1)
        weights['volume'] = st.slider("Volume", 0.0, 3.0, st.session_state.custom_weights.get('volume', 1.5), 0.1)
        weights['smart_money'] = st.slider("Smart Money", 0.0, 3.0, st.session_state.custom_weights.get('smart_money', 1.5), 0.1)
        weights['structure'] = st.slider("Structure", 0.0, 3.0, st.session_state.custom_weights.get('structure', 1.0), 0.1)
        if st.button("Simpan Bobot"):
            st.session_state.custom_weights = weights
    weights = st.session_state.custom_weights.copy()

    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    # ---------------- LOAD DATA & ENGINE ----------------
    with st.spinner(f"Memuat {ticker}..."):
        period = "2y" if trading_mode == "Position (Weekly - Monthly)" else "1y"
        df_raw = yf.download(ticker, period=period, interval=timeframe, progress=False, auto_adjust=False)
    if df_raw.empty:
        st.error(f"Data {ticker} tidak tersedia")
        st.stop()

    df = calculate_all_indicators(df_raw, st_period, st_mult, trading_mode)
    indicators = get_latest_indicators(df)
    price = indicators['price']

    regime = detect_market_regime(df)

    # MTF alignment (safe)
    def safe_load(ticker, period, interval):
        try:
            d = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            if d.empty or len(d) < 30:
                return pd.DataFrame()
            return calculate_all_indicators(d, st_period, st_mult, trading_mode)
        except:
            return pd.DataFrame()
    df_daily_mtf = safe_load(ticker, "1y", "1d")
    df_weekly_mtf = safe_load(ticker, "2y", "1wk")
    df_monthly_mtf = safe_load(ticker, "3y", "1mo")
    mtf_alignment = get_mtf_alignment(df_daily_mtf, df_weekly_mtf, df_monthly_mtf)

    decision = weighted_decision_final(indicators, mtf_alignment, regime, weights)
    show_toast_if_changed(decision['signal'], st.session_state.last_signal)
    st.session_state.last_signal = decision['signal']
    log_signal(ticker, decision['signal'], decision['score'], price)
    market_text, _ = get_market_context()

    breakout = detect_true_breakout(df, indicators['atr'])
    show_breakout_alert(breakout, st.session_state.last_breakout_notify_time)

    entry_levels = get_entry_levels_advanced(df, indicators, regime)
    bull_prob, bear_prob = calculate_probabilities(df)

    # ---------------- TAMPILAN UTAMA (PRO) ----------------
    st.markdown(f"## 🤖 {ticker}")
    st.caption(f"Mode: {trading_mode} | Timeframe: {timeframe} | Regime: {regime} | MTF Alignment: {mtf_alignment:.2f}")

    # ------------------------------------------------------------
    # SECTION 1: HEADER CARD (harga & perubahan simulasi)
    # ------------------------------------------------------------
    change_pct = 0
    if len(df) >= 2:
        change_pct = (price / df['Close'].iloc[-2] - 1) * 100
    arrow = "▲" if change_pct > 0 else "▼" if change_pct < 0 else "■"
    color_change = "green" if change_pct > 0 else "red" if change_pct < 0 else "gray"
    st.markdown(f"""
    <div style="background: white; border-radius: 16px; padding: 1rem; margin-bottom: 1.2rem; border: 1px solid #e9ecef;">
        <div style="display: flex; justify-content: space-between; align-items: baseline;">
            <span style="font-size: 2rem; font-weight: 700;">Rp {price:,.0f}</span>
            <span style="font-size: 1.1rem; color: {color_change};">{arrow} {abs(change_pct):.2f}%</span>
        </div>
        <div style="font-size: 0.8rem; color: #6c757d;">{datetime.now().strftime('%d %b %Y, %H:%M')}</div>
    </div>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------------
    # SECTION 2: SIGNAL CARD
    # ------------------------------------------------------------
    signal_color = {
        "STRONG BUY": "#d4edda", "BUY": "#d1e7dd",
        "HOLD": "#fff3cd",
        "SELL": "#f8d7da", "STRONG SELL": "#f5c6cb"
    }.get(decision['signal'], "#e2e3e5")
    text_color = "#155724" if "BUY" in decision['signal'] else "#721c24" if "SELL" in decision['signal'] else "#856404"
    st.markdown(f"""
    <div style="background-color: {signal_color}; border-radius: 20px; padding: 1rem; text-align: center; margin-bottom: 1.2rem;">
        <div style="font-size: 2rem; font-weight: 800; letter-spacing: 1px; color: {text_color};">{decision['signal']}</div>
        <div style="font-size: 0.9rem; margin-top: 0.2rem; color: {text_color};">Confidence: {decision['confidence']} | Skor: {decision['score']:.1f}</div>
        <div style="font-size: 0.8rem; margin-top: 0.3rem;">{get_signal_interpretation(decision['signal'])}</div>
    </div>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------------
    # SECTION 3: KEY METRICS (2 kolom)
    # ------------------------------------------------------------
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📊 Market Structure")
        trend_text = "🟢 UPTREND" if indicators['supertrend_dir'] == 1 else "🔴 DOWNTREND"
        st.metric("Trend (Supertrend)", trend_text)
        rsi_val = indicators['rsi']
        rsi_label = f"{rsi_val:.1f} "
        if rsi_val < 30:
            rsi_label += "(Oversold)"
        elif rsi_val > 70:
            rsi_label += "(Overbought)"
        st.metric("RSI", rsi_label)
        st.metric("Volume", indicators['volume_status'])
    with col2:
        st.markdown("#### 💰 Smart Money")
        cmf_val = indicators['cmf']
        cmf_label = "🟢 Akumulasi" if cmf_val > 0.15 else "🔴 Distribusi" if cmf_val < -0.15 else "⚪ Netral"
        st.metric("CMF", f"{cmf_label} ({cmf_val:.2f})")
        st.metric("Probabilitas", f"🟢 Bullish {bull_prob:.0f}% / 🔴 Bearish {bear_prob:.0f}%")
        if market_text and "tidak tersedia" not in market_text:
            st.metric("IHSG", market_text[:15])
        else:
            st.metric("IHSG", "—")

    # ------------------------------------------------------------
    # SECTION 4: TRADING PLAN & KEY LEVELS
    # ------------------------------------------------------------
    st.markdown("#### 🎯 Trading Plan")
    plan_cols = st.columns(3)
    entry_val = entry_levels['buy_entry']
    plan_cols[0].metric("🎯 Buy Entry", f"Rp {entry_val:,.0f}" if entry_val > 0 else "—")
    sl_val = entry_levels['stop_loss']
    plan_cols[1].metric("🛑 Stop Loss", f"Rp {sl_val:,.0f}" if sl_val > 0 else "—")
    tgt_val = entry_levels['target']
    plan_cols[2].metric("🎯 Target", f"Rp {tgt_val:,.0f}" if tgt_val > 0 else "—")

    st.markdown("#### 📌 Key Levels")
    level_cols = st.columns(2)
    support_val = indicators['support']
    resistance_val = indicators['resistance']
    level_cols[0].metric("Support", f"Rp {support_val:,.0f}" if support_val > 0 else "—")
    level_cols[1].metric("Resistance", f"Rp {resistance_val:,.0f}" if resistance_val > 0 else "—")

    swing_low = indicators['swing_low']
    swing_high = indicators['swing_high']
    if swing_low > 0 or swing_high > 0:
        st.markdown("##### 🔁 Swing Points")
        sw_cols = st.columns(2)
        sw_cols[0].metric("Swing Low", f"Rp {swing_low:,.0f}" if swing_low > 0 else "—")
        sw_cols[1].metric("Swing High", f"Rp {swing_high:,.0f}" if swing_high > 0 else "—")

    # ------------------------------------------------------------
    # SECTION 5: ADVANCED METRICS (opsional)
    # ------------------------------------------------------------
    if show_advanced:
        with st.expander("📈 Indikator Lanjutan"):
            adv_cols = st.columns(5)
            adv_cols[0].metric("Stochastic K", f"{indicators['stoch_k']:.1f}")
            adv_cols[1].metric("Stochastic D", f"{indicators['stoch_d']:.1f}")
            adv_cols[2].metric("Williams %R", f"{indicators['williams_r']:.1f}")
            adv_cols[3].metric("CCI", f"{indicators['cci']:.1f}")
            adv_cols[4].metric("MFI", f"{indicators['mfi']:.1f}")
            adv_cols2 = st.columns(4)
            adv_cols2[0].metric("KST", f"{indicators['kst']:.1f}")
            adv_cols2[1].metric("Elder Bull", f"{indicators['elder_bull']:.0f}")
            adv_cols2[2].metric("Elder Bear", f"{indicators['elder_bear']:.0f}")
            adv_cols2[3].metric("GMMA Spread", f"{indicators['gmma_spread']:.0f}")

    # ------------------------------------------------------------
    # SECTION 6: DETAIL ANALISIS & BACKTEST
    # ------------------------------------------------------------
    with st.expander("📋 Detail Analisis (skor & kontribusi)"):
        for reason in decision['reasons']:
            st.write(f"- {reason}")
        st.markdown(f"**Total Skor: {decision['score']}**")

    col_bt, col_hist = st.columns(2)
    with col_bt:
        with st.expander("📊 Backtest (6 bulan)"):
            if st.button("Jalankan Backtest"):
                with st.spinner("Menjalankan..."):
                    bt = run_backtest_advanced(df, weights, detect_market_regime, 180)
                    if bt:
                        st.write(f"Jumlah trade: {bt['trades']}")
                        st.write(f"Winrate: {bt['winrate']:.1f}%")
                        st.write(f"Profit Factor: {bt['profit_factor']:.2f}")
                        st.write(f"Max Drawdown: {bt['max_drawdown']:.1f}%")
                        st.write(f"Final Balance: Rp {bt['final_balance']:,.0f}")
                    else:
                        st.warning("Data tidak cukup atau tidak ada sinyal")
    with col_hist:
        with st.expander("📜 Riwayat Sinyal"):
            if st.session_state.signal_history:
                st.dataframe(pd.DataFrame(st.session_state.signal_history).head(5))
                csv = pd.DataFrame(st.session_state.signal_history).to_csv(index=False).encode()
                st.download_button("📥 Ekspor CSV", csv, "signal_history.csv", "text/csv")
            else:
                st.info("Belum ada sinyal")

    with st.expander("📐 Position Sizing"):
        capital = st.number_input("Modal (Rp)", value=100_000_000.0, step=10_000_000.0)
        risk_percent = st.number_input("Risiko per trade (%)", value=2.0, step=0.5)
        shares, pos_value, risk_amt = position_sizing(capital, risk_percent, entry_levels['buy_entry'], entry_levels['stop_loss'])
        if shares > 0:
            st.write(f"Jumlah saham: {shares:,} lembar")
            st.write(f"Nilai posisi: Rp {pos_value:,.0f} ({pos_value/capital*100:.1f}% dari modal)")
            st.write(f"Risiko stop loss: Rp {risk_amt:,.0f} ({risk_percent:.1f}%)")

    st.caption("⚠️ Edukasi & simulasi, bukan rekomendasi investasi.")
    gc.collect()

if __name__ == "__main__":
    main()
