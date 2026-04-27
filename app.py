import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import gc
import random
import traceback

# ===================== FALLBACK =====================
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Robot Saham v11 - Multi-Stock Global AI", page_icon="🤖")

# ===================== SESSION STATE =====================
if "signal_history" not in st.session_state:
    st.session_state.signal_history = []
if "last_signal" not in st.session_state:
    st.session_state.last_signal = None
if "custom_weights" not in st.session_state:
    st.session_state.custom_weights = {
        "trend": 2.0,
        "momentum": 1.5,
        "volume": 1.5,
        "smart_money": 1.5,
        "structure": 1.0,
    }
if "global_model" not in st.session_state:
    st.session_state.global_model = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None

# ===================== FUNGSI INDIKATOR MANUAL =====================
def manual_atr(high, low, close, window=14):
    tr = np.maximum(high - low, np.abs(high - close.shift()), np.abs(low - close.shift()))
    return tr.rolling(window).mean()

def manual_rsi(close, window=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def manual_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line, macd - signal_line

def calculate_supertrend(high_series, low_series, close_series, period=10, multiplier=3.0):
    """
    high_series, low_series, close_series: pandas Series 1D
    """
    if TA_AVAILABLE:
        atr = ta.volatility.average_true_range(high_series, low_series, close_series, window=period)
    else:
        atr = manual_atr(high_series, low_series, close_series, period)
    hl_avg = (high_series + low_series) / 2
    upper = hl_avg + (multiplier * atr)
    lower = hl_avg - (multiplier * atr)
    st_line = pd.Series(index=close_series.index, dtype=float)
    direction = pd.Series(index=close_series.index, dtype=int)
    for i in range(period, len(close_series)):
        if i == period:
            st_line.iloc[i] = upper.iloc[i]
            direction.iloc[i] = 1 if close_series.iloc[i] > st_line.iloc[i] else -1
        else:
            if direction.iloc[i-1] == 1:
                if close_series.iloc[i] < lower.iloc[i]:
                    direction.iloc[i] = -1
                    st_line.iloc[i] = upper.iloc[i]
                else:
                    direction.iloc[i] = 1
                    st_line.iloc[i] = max(lower.iloc[i], st_line.iloc[i-1])
            else:
                if close_series.iloc[i] > upper.iloc[i]:
                    direction.iloc[i] = 1
                    st_line.iloc[i] = lower.iloc[i]
                else:
                    direction.iloc[i] = -1
                    st_line.iloc[i] = min(upper.iloc[i], st_line.iloc[i-1])
    return st_line, direction

@st.cache_data(ttl=300)
def calculate_all_indicators(df, st_period=10, st_mult=3.0):
    """
    df adalah DataFrame dari yfinance. Kolom akan di-squeeze menjadi Series 1D.
    """
    if df.empty:
        return pd.DataFrame()
    
    # Konversi setiap kolom menjadi Series 1D
    close_series = df["Close"].squeeze()
    high_series = df["High"].squeeze()
    low_series = df["Low"].squeeze()
    
    # Volume handling: cek keberadaan dan apakah ada data volume
    if "Volume" in df.columns:
        vol_series = df["Volume"].squeeze()
        # Hitung sum sebagai scalar
        total_volume = float(vol_series.sum())
        has_volume = total_volume != 0
    else:
        vol_series = pd.Series(0, index=df.index)
        has_volume = False
    
    # DataFrame output
    df_out = pd.DataFrame(index=df.index)
    df_out["Close"] = close_series
    df_out["High"] = high_series
    df_out["Low"] = low_series
    df_out["Volume"] = vol_series
    
    df_out["SMA20"] = close_series.rolling(20).mean()
    df_out["SMA50"] = close_series.rolling(50).mean()
    df_out["EMA200"] = close_series.ewm(span=200, adjust=False).mean()
    
    if TA_AVAILABLE:
        df_out["RSI"] = ta.momentum.rsi(close_series, window=14)
        macd = ta.trend.MACD(close_series)
        df_out["MACD_Hist"] = macd.macd_diff()
        df_out["Stoch_K"] = ta.momentum.stoch(high_series, low_series, close_series, window=14, smooth_window=3)
        df_out["ATR"] = ta.volatility.average_true_range(high_series, low_series, close_series, window=14)
        if has_volume:
            df_out["CMF"] = ta.volume.chaikin_money_flow(high_series, low_series, close_series, vol_series, window=20)
        else:
            df_out["CMF"] = 0.0
    else:
        df_out["RSI"] = manual_rsi(close_series, 14)
        _, _, df_out["MACD_Hist"] = manual_macd(close_series)
        df_out["Stoch_K"] = 50.0
        df_out["ATR"] = manual_atr(high_series, low_series, close_series, 14)
        df_out["CMF"] = 0.0
    
    # Supertrend
    st_line, st_dir = calculate_supertrend(high_series, low_series, close_series, st_period, st_mult)
    df_out["ST_Supertrend"] = st_line
    df_out["ST_Dir"] = st_dir
    
    # Volume status
    if has_volume:
        vol_ma = vol_series.rolling(20).mean()
        # Boolean Series kemudian di-convert ke string via np.where
        condition_high = vol_series > vol_ma * 1.5
        condition_low = vol_series < vol_ma
        df_out["Volume_Status"] = np.where(condition_high, "Tinggi",
                                           np.where(condition_low, "Rendah", "Normal"))
    else:
        df_out["Volume_Status"] = "Normal"
    
    df_out["Swing_Low"] = low_series.rolling(5).min()
    df_out["Swing_High"] = high_series.rolling(5).max()
    
    df_out = df_out.ffill().fillna(0)
    return df_out

def get_latest_indicators(df):
    if df.empty:
        return {}
    last = df.iloc[-1]
    price = last.get("Close", 0.0)
    return {
        "price": price,
        "sma20": last.get("SMA20", price),
        "sma50": last.get("SMA50", price),
        "ema200": last.get("EMA200", price),
        "rsi": last.get("RSI", 50.0),
        "macd_hist": last.get("MACD_Hist", 0.0),
        "stoch_k": last.get("Stoch_K", 50.0),
        "atr": last.get("ATR", price * 0.02),
        "volume_status": last.get("Volume_Status", "Normal"),
        "cmf": last.get("CMF", 0.0),
        "supertrend_dir": last.get("ST_Dir", 0),
        "supertrend_value": last.get("ST_Supertrend", price),
        "swing_low": last.get("Swing_Low", price * 0.95),
        "swing_high": last.get("Swing_High", price * 1.05),
    }

def detect_market_regime(df):
    if len(df) < 30:
        return "SIDEWAYS"
    close_series = df["Close"].squeeze()
    ema20 = close_series.ewm(span=20).mean()
    slope = ema20.iloc[-1] - ema20.iloc[-5]
    if abs(slope / ema20.iloc[-1]) > 0.002:
        return "TREND"
    else:
        return "SIDEWAYS"

def build_feature_vector(ind):
    trend = np.mean([
        1 if ind["price"] > ind["ema200"] else -1,
        ind["supertrend_dir"],
        1 if ind["sma20"] > ind["sma50"] else -1,
    ])
    rsi_norm = (ind["rsi"] - 50) / 50
    momentum = np.mean([rsi_norm, np.sign(ind["macd_hist"])])
    volume = 1 if ind["volume_status"] == "Tinggi" else (-1 if ind["volume_status"] == "Rendah" else 0)
    smart = np.clip(ind["cmf"] * 3, -1, 1)
    structure = 0
    if ind["price"] > ind["swing_high"]:
        structure = 1
    elif ind["price"] < ind["swing_low"]:
        structure = -1
    return {"trend": trend, "momentum": momentum, "volume": volume, "smart": smart, "structure": structure}

def decision_engine_pro(indicators, regime, weights, threshold_buy=2.0, threshold_strong=4.0):
    f = build_feature_vector(indicators)
    w = weights
    score = (f["trend"] * w["trend"] +
             f["momentum"] * w["momentum"] +
             f["volume"] * w["volume"] +
             f["smart"] * w["smart_money"] +
             f["structure"] * w["structure"])
    score = max(-10, min(10, score))
    if score >= threshold_strong:
        return "STRONG BUY", score
    elif score >= threshold_buy:
        return "BUY", score
    elif score <= -threshold_strong:
        return "STRONG SELL", score
    elif score <= -threshold_buy:
        return "SELL", score
    else:
        return "HOLD", score

# ===================== MODEL GLOBAL =====================
def train_global_model(ticker_list, period="1y", use_xgb=True):
    X_all, y_all = [], []
    for ticker in ticker_list:
        try:
            df_raw = yf.download(ticker, period=period, progress=False, auto_adjust=False)
            if df_raw.empty:
                continue
            df = calculate_all_indicators(df_raw, 10, 3.0)
            close_series = df["Close"].squeeze()
            for i in range(60, len(df)-5):
                ret = (close_series.iloc[i+5] - close_series.iloc[i]) / close_series.iloc[i]
                if ret > 0.02:
                    label = 1
                elif ret < -0.02:
                    label = 0
                else:
                    continue
                ind = get_latest_indicators(df.iloc[:i+1])
                if not ind:
                    continue
                feat = [ind["rsi"], ind["macd_hist"], ind["cmf"], ind["atr"],
                        1 if ind["volume_status"]=="Tinggi" else (0 if ind["volume_status"]=="Normal" else -1),
                        ind["supertrend_dir"]]
                X_all.append(feat)
                y_all.append(label)
        except Exception as e:
            st.warning(f"Gagal training {ticker}: {e}")
    if len(X_all) < 100:
        return None, None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    if use_xgb and XGB_AVAILABLE:
        model = XGBClassifier(n_estimators=100, max_depth=3, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_scaled, y_all)
    return model, scaler

def ml_predict(model, scaler, indicators):
    if model is None:
        return 0.5
    feat = [[indicators["rsi"], indicators["macd_hist"], indicators["cmf"], indicators["atr"],
             1 if indicators["volume_status"]=="Tinggi" else (0 if indicators["volume_status"]=="Normal" else -1),
             indicators["supertrend_dir"]]]
    feat_scaled = scaler.transform(feat)
    prob = model.predict_proba(feat_scaled)[0][1]
    return prob

def final_decision(indicators, regime, weights, model, scaler, threshold_buy, threshold_strong):
    signal, score = decision_engine_pro(indicators, regime, weights, threshold_buy, threshold_strong)
    ml_prob = ml_predict(model, scaler, indicators) if model is not None else 0.5
    score += (ml_prob - 0.5) * 2.0
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

# ===================== MAIN APP =====================
def main():
    try:
        st.sidebar.markdown("# 🤖 Robot Saham v11 - Multi-Stock Global AI")

        ticker_input = st.sidebar.text_input("Kode Saham / Indeks", "BBCA.JK")
        ticker = ticker_input.upper().strip()
        if not ticker.endswith(".JK") and not ticker.startswith("^") and not ticker.endswith(".SA"):
            ticker += ".JK"
        timeframe = st.sidebar.selectbox("Timeframe", ["1d", "1wk", "1mo"], index=0)
        ui_mode = st.sidebar.selectbox("Mode Tampilan", ["Simple (Recommended)", "Advanced"], index=0)

        with st.sidebar.expander("⚙️ Bobot Indikator"):
            weights = {
                "trend": st.slider("Trend", 0.0, 3.0, st.session_state.custom_weights["trend"], 0.1),
                "momentum": st.slider("Momentum", 0.0, 3.0, st.session_state.custom_weights["momentum"], 0.1),
                "volume": st.slider("Volume", 0.0, 3.0, st.session_state.custom_weights["volume"], 0.1),
                "smart_money": st.slider("Smart Money", 0.0, 3.0, st.session_state.custom_weights["smart_money"], 0.1),
                "structure": st.slider("Structure", 0.0, 3.0, st.session_state.custom_weights["structure"], 0.1),
            }
            if st.button("Simpan Bobot"):
                st.session_state.custom_weights = weights
        weights = st.session_state.custom_weights

        use_xgb = st.sidebar.checkbox("Gunakan XGBoost", value=XGB_AVAILABLE, disabled=not XGB_AVAILABLE)

        if st.sidebar.button("🔥 Train Global Model"):
            default_tickers = ["BBCA.JK", "BBRI.JK", "TLKM.JK", "ASII.JK", "UNVR.JK"]
            tickers_input = st.sidebar.text_input("Daftar saham training", value=",".join(default_tickers))
            ticker_list = [t.strip() for t in tickers_input.split(",") if t.strip()]
            with st.spinner("Training global model..."):
                model, scaler = train_global_model(ticker_list, period="1y", use_xgb=use_xgb)
                if model is not None:
                    st.session_state.global_model = model
                    st.session_state.scaler = scaler
                    st.success("Global model siap!")
                else:
                    st.error("Training gagal (data tidak cukup).")

        if st.sidebar.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

        threshold_buy = st.sidebar.slider("Threshold BUY", 1.0, 5.0, 2.0, 0.5)
        threshold_strong = st.sidebar.slider("Threshold STRONG", 3.0, 8.0, 4.0, 0.5)

        with st.spinner(f"Memuat {ticker}..."):
            try:
                df_raw = yf.download(ticker, period="1y", interval=timeframe, progress=False, auto_adjust=False)
                if df_raw.empty:
                    st.error(f"Data {ticker} tidak tersedia. Periksa kode.")
                    return
            except Exception as e:
                st.error(f"Gagal download data: {e}")
                return

        df = calculate_all_indicators(df_raw, 10, 3.0)
        if df.empty:
            st.error("Indikator gagal dihitung.")
            return

        indicators = get_latest_indicators(df)
        if not indicators:
            st.error("Tidak bisa mengambil indikator terbaru.")
            return

        price = indicators["price"]
        regime = detect_market_regime(df)

        model = st.session_state.global_model
        scaler = st.session_state.scaler
        signal, score, ml_prob = final_decision(indicators, regime, weights, model, scaler,
                                                threshold_buy, threshold_strong)

        if signal != st.session_state.last_signal:
            if signal in ["STRONG BUY", "BUY"]:
                st.toast(f"🟢 Sinyal berubah menjadi {signal}!")
            elif signal in ["STRONG SELL", "SELL"]:
                st.toast(f"🔴 Sinyal berubah menjadi {signal}!")
            st.session_state.last_signal = signal

        st.session_state.signal_history.insert(0, {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ticker": ticker,
            "signal": signal,
            "score": score,
            "price": price
        })
        if len(st.session_state.signal_history) > 50:
            st.session_state.signal_history.pop()

        st.title(f"🤖 {ticker} (v11 Global AI)")
        st.caption(f"Regime: {regime} | ML Prob: {ml_prob*100:.1f}% | Model Global: {'Aktif' if model else 'Tidak (rule only)'}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💰 Harga", f"Rp {price:,.0f}")
        col2.metric("📊 RSI", f"{indicators['rsi']:.1f}")
        sup_dir = "🟢 UPTREND" if indicators["supertrend_dir"] == 1 else "🔴 DOWNTREND"
        col3.metric("📈 Supertrend", sup_dir)
        col4.metric("📊 Volume", indicators["volume_status"])

        col1, col2, col3 = st.columns(3)
        if "STRONG BUY" in signal:
            col1.success(f"### 🟢 {signal}")
        elif "STRONG SELL" in signal:
            col1.error(f"### 🔴 {signal}")
        elif "BUY" in signal:
            col1.info(f"### 📈 {signal}")
        elif "SELL" in signal:
            col1.warning(f"### 📉 {signal}")
        else:
            col1.warning(f"### 🟡 {signal}")
        col2.metric("🎯 Skor", f"{score:.1f}")
        col3.metric("🤖 AI Confidence", f"{ml_prob*100:.1f}%")

        if score >= threshold_buy:
            st.success(f"🟢 **ENTRY LAYAK** (Skor: {score:.1f})")
        else:
            st.error(f"🔴 **TIDAK LAYAK ENTRY** (Skor: {score:.1f})")

        with st.expander("📖 Arti Sinyal"):
            interpretations = {
                "STRONG BUY": "Peluang naik sangat kuat. Hybrid AI + global model.",
                "BUY": "Kondisi cukup baik. Masih ada ruang naik.",
                "HOLD": "Tidak ada sinyal jelas. Tunggu breakout.",
                "SELL": "Tekanan turun mulai terlihat. Pertimbangkan exit.",
                "STRONG SELL": "Kondisi lemah. Risiko besar, hindari entry.",
            }
            st.write(interpretations.get(signal, "Netral."))

        if ui_mode == "Advanced":
            with st.expander("📋 Detail Analisis"):
                fv = build_feature_vector(indicators)
                st.write(f"- Trend: {fv['trend']:.2f}")
                st.write(f"- Momentum: {fv['momentum']:.2f}")
                st.write(f"- Volume: {fv['volume']:.2f}")
                st.write(f"- Smart Money: {fv['smart']:.2f}")
                st.write(f"- Structure: {fv['structure']:.2f}")

        with st.expander("📜 Riwayat Sinyal"):
            if st.session_state.signal_history:
                st.dataframe(pd.DataFrame(st.session_state.signal_history).head(10))
            else:
                st.info("Belum ada sinyal")

        st.caption("⚠️ Edukasi bukan rekomendasi. Pastikan library `ta` dan `xgboost` terinstal.")
        gc.collect()

    except Exception as e:
        st.error(f"Terjadi error: {str(e)}")
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
