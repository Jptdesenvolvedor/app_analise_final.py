import math
from datetime import datetime, timedelta, timezone

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# =========================
# Configuração da página
# =========================
st.set_page_config(layout="wide")
st.title("📊 Análise Técnica Avançada de Ativos")

# =========================
# Ativos padrão (opcional)
# =========================
ATIVOS_FIXOS = {
    "Criptomoedas": {
        "Bitcoin (BTC)": "BTC-USD",
        "Ethereum (ETH)": "ETH-USD",
        "Solana (SOL)": "SOL-USD",
    },
    "Ações Americanas": {
        "Apple (AAPL)": "AAPL",
        "Nvidia (NVDA)": "NVDA",
        "Microsoft (MSFT)": "MSFT",
    },
    "Ações Brasil": {
        "Petrobras (PETR4)": "PETR4.SA",
        "Vale (VALE3)": "VALE3.SA",
        "Itaú (ITUB4)": "ITUB4.SA",
    },
}

# =========================
# Mapeamento de Períodos/Intervalos
# =========================
PERIODOS_UI = {
    "7 dias": "7d",
    "1 mês": "1mo",
    "3 meses": "3mo",
    "6 meses": "6mo",
    "1 ano": "1y",
    "5 anos": "5y",
    "10 anos": "10y",
}

# Intervalos mostrados na UI (4h será resample de 1h)
INTERVALOS_UI = {
    "15 minutos": "15m",
    "30 minutos": "30m",
    "1 hora": "1h",
    "4 horas (resample)": "4h",   # trataremos manualmente
    "1 dia": "1d",
}

# Regras do Yahoo Finance: o que é suportado nativamente
INTERVALOS_VALIDOS = {
    "15m": {"5d", "1mo"},
    "30m": {"5d", "1mo"},
    "1h": {"7d", "1mo", "3mo"},
    "1d": {"1mo", "3mo", "6mo", "1y", "5y", "10y"},
}

modo = st.radio("Modo de seleção do ativo:", ["Escolher da lista", "Digitar ticker"])

if modo == "Escolher da lista":
    categoria = st.selectbox("Categoria:", list(ATIVOS_FIXOS.keys()))
    nome_ativo = st.selectbox("Ativo:", list(ATIVOS_FIXOS[categoria].keys()))
    ticker = ATIVOS_FIXOS[categoria][nome_ativo]
else:
    ticker = st.text_input("Digite o ticker (ex: AAPL, PETR4.SA, BTC-USD):", "AAPL").strip().upper()
    nome_ativo = ticker

periodo_label = st.selectbox("Período:", list(PERIODOS_UI.keys()), index=1)
intervalo_label = st.selectbox("Intervalo:", list(INTERVALOS_UI.keys()), index=2)

periodo = PERIODOS_UI[periodo_label]
intervalo_ui = INTERVALOS_UI[intervalo_label]

def _period_to_days(period: str) -> int:
    if period.endswith("d"):
        return int(period[:-1])
    if period.endswith("mo"):
        return int(period[:-2]) * 30
    if period.endswith("y"):
        return int(period[:-1]) * 365
    return 30

def compatibilizar_periodo_intervalo(periodo: str, intervalo_ui: str):
    msg = None
    precisa_resample_4h = False

    if intervalo_ui == "4h":
        intervalo_real = "1h"
        precisa_resample_4h = True
        if periodo not in INTERVALOS_VALIDOS["1h"]:
            periodo_ajustado = "3mo"
            msg = "⚠️ Intervalo 4h requer horário. Ajustei o período para **3 meses** (Yahoo)."
            return periodo_ajustado, intervalo_real, precisa_resample_4h, msg
        return periodo, intervalo_real, precisa_resample_4h, msg

    intervalo_real = intervalo_ui
    suportados = INTERVALOS_VALIDOS.get(intervalo_real, set())
    if periodo not in suportados:
        if intervalo_real in {"15m", "30m"}:
            periodo_ajustado = "1mo"
            msg = f"⚠️ Intervalo {intervalo_label} requer período curto. Ajustei para **1 mês** (Yahoo)."
        elif intervalo_real == "1h":
            periodo_ajustado = "3mo"
            msg = "⚠️ Intervalo 1h requer até 3 meses. Ajustei para **3 meses** (Yahoo)."
        else:
            periodo_ajustado = "1y"
            msg = "⚠️ Ajustei o período para **1 ano** por compatibilidade."
        return periodo_ajustado, intervalo_real, precisa_resample_4h, msg

    return periodo, intervalo_real, precisa_resample_4h, msg

def baixar_dados_robusto(ticker: str, periodo: str, intervalo_real: str) -> pd.DataFrame:
    df = yf.download(ticker, period=periodo, interval=intervalo_real, auto_adjust=False, progress=False)
    if df is not None and not df.empty:
        return df.dropna()

    dias = _period_to_days(periodo)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=dias)
    df = yf.download(ticker, start=start, end=end, interval=intervalo_real, auto_adjust=False, progress=False)
    if df is not None and not df.empty:
        return df.dropna()

    return pd.DataFrame()

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty:
        return df
    ohlc = {"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    out = df.resample(rule).agg(ohlc).dropna()
    if "Adj Close" in df.columns:
        out["Adj Close"] = df["Adj Close"].resample(rule).last()
    return out

def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA21"] = df["Close"].rolling(21).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    df["EMA17"] = df["Close"].ewm(span=17, adjust=False).mean()
    df["EMA72"] = df["Close"].ewm(span=72, adjust=False).mean()
    df["EMA305"] = df["Close"].ewm(span=305, adjust=False).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    if "Volume" in df.columns:
        df["Volume_Financeiro"] = df["Volume"] * df["Close"]
    else:
        df["Volume"] = 0
        df["Volume_Financeiro"] = 0.0

    return df

def calcular_fibonacci(df: pd.DataFrame):
    if df.empty:
        return {}
    max_price = float(df["Close"].max())
    min_price = float(df["Close"].min())
    diff = max_price - min_price
    return {
        "0.0%": max_price,
        "23.6%": max_price - 0.236 * diff,
        "38.2%": max_price - 0.382 * diff,
        "50.0%": max_price - 0.5 * diff,
        "61.8%": max_price - 0.618 * diff,
        "100.0%": min_price,
    }

def diagnostico_rsi(valor):
    if pd.isna(valor):
        return "-"
    if valor < 30:
        return "📈 Alta provável (RSI < 30)"
    if valor > 70:
        return "📉 Baixa provável (RSI > 70)"
    return "🔍 Neutro"

if st.button("🔎 Analisar"):
    try:
        periodo_uso, intervalo_real, precisa_resample_4h, msg_info = compatibilizar_periodo_intervalo(
            periodo, intervalo_ui
        )
        if msg_info:
            st.info(msg_info)

        df_raw = baixar_dados_robusto(ticker, periodo_uso, intervalo_real)
        if df_raw.empty:
            st.warning("⚠️ Nenhum dado foi encontrado para esse ticker/período/intervalo. Tente outra combinação.")
            st.stop()

        if precisa_resample_4h:
            df = resample_ohlcv(df_raw, "4H")
            if df.empty:
                st.warning("⚠️ Não foi possível gerar o timeframe de 4 horas.")
                st.stop()
        else:
            df = df_raw

        df = calcular_indicadores(df)
        fibo = calcular_fibonacci(df)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Preço + MA + PhiCube + Fibonacci")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df.index, df["Close"], label="Preço", linewidth=1.5)
            ax.plot(df.index, df["MA21"], label="MA21")
            ax.plot(df.index, df["MA200"], label="MA200")
            ax.plot(df.index, df["EMA17"], label="EMA17 (Phi)", linestyle="--")
            ax.plot(df.index, df["EMA72"], label="EMA72 (Phi)", linestyle="--")
            ax.plot(df.index, df["EMA305"], label="EMA305 (Phi)", linestyle="--")
            for lvl, val in fibo.items():
                ax.axhline(val, linestyle="--", alpha=0.3, label=f"Fib {lvl}")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        with col2:
            st.subheader("📈 RSI, MACD e Volume Financeiro")
            fig2, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
            axs[0].plot(df.index, df["RSI"], label="RSI")
            axs[0].axhline(70, linestyle="--")
            axs[0].axhline(30, linestyle="--")
            axs[0].legend()
            axs[0].grid(True)
            axs[1].plot(df.index, df["MACD"], label="MACD")
            axs[1].plot(df.index, df["Signal"], label="Signal")
            axs[1].legend()
            axs[1].grid(True)
            axs[2].bar(df.index, df["Volume_Financeiro"], label="Volume R$")
            axs[2].legend()
            axs[2].grid(True)
            st.pyplot(fig2)

        st.subheader("📍 Diagnóstico RSI")
        st.success(f"{nome_ativo}: {diagnostico_rsi(df['RSI'].iloc[-1])}")

        with st.expander("Ver primeiras linhas dos dados"):
            st.write(df.head())

    except Exception as e:
        st.error(f"Erro ao processar: {e}")