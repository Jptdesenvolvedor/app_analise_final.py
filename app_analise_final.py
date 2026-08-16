"""
📊 Análise Técnica de Ativos — Streamlit
=========================================

Aplicação de análise técnica para ações (Brasil/EUA) e criptomoedas, usando
dados públicos do Yahoo Finance (via `yfinance`).

Este arquivo foi reorganizado para uso profissional:
- Download de dados em cache, com retentativas e tratamento de erros;
- Regras de período/intervalo alinhadas aos limites reais do Yahoo Finance
  (ver referência oficial do yfinance: intraday data cannot extend last 60 days);
- Diagnóstico técnico (RSI, cruzamento de MACD, tendência vs. médias);
- Exportação dos dados calculados em CSV;
- Painel de "Fontes e Validação": links para RI oficial das empresas,
  reguladores (CVM/SEC), bolsas e provedores de dados, e veículos
  especializados — para que o usuário sempre confira o dado bruto na fonte.

⚠️ Aviso: esta ferramenta tem finalidade educacional/informativa. Nada aqui
constitui recomendação de investimento. Sempre confirme os números nas
fontes oficiais listadas no painel "Fontes e Validação" e, se necessário,
consulte um profissional certificado (CVM/APIMEC) antes de decidir.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf

# =============================================================================
# Configuração da página
# =============================================================================
st.set_page_config(page_title="Análise Técnica de Ativos", page_icon="📊", layout="wide")
st.title("📊 Análise Técnica Avançada de Ativos")
st.caption(
    "Dados via Yahoo Finance (yfinance). Ferramenta educacional/informativa — "
    "não constitui recomendação de investimento."
)

# =============================================================================
# Ativos padrão (atalhos) + fontes oficiais correspondentes
# =============================================================================
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

# Fontes oficiais (RI das empresas / site do projeto, no caso de criptoativos).
# Verificadas manualmente — mantenha atualizado se a empresa trocar de domínio.
FONTES_OFICIAIS = {
    "AAPL": {"nome": "Apple Inc. — Investor Relations", "url": "https://investor.apple.com/investor-relations/default.aspx"},
    "NVDA": {"nome": "NVIDIA Corporation — Investor Relations", "url": "https://investor.nvidia.com/home/default.aspx"},
    "MSFT": {"nome": "Microsoft Corporation — Investor Relations", "url": "https://www.microsoft.com/en-us/investor"},
    "PETR4.SA": {"nome": "Petrobras — Relações com Investidores", "url": "https://www.investidorpetrobras.com.br/"},
    "VALE3.SA": {"nome": "Vale S.A. — Relações com Investidores", "url": "https://vale.com/investors"},
    "ITUB4.SA": {"nome": "Itaú Unibanco — Relações com Investidores", "url": "https://www.itau.com.br/relacoes-com-investidores/Home.aspx?linguagem=pt"},
    "BTC-USD": {"nome": "Bitcoin.org (referência do protocolo)", "url": "https://bitcoin.org/en/"},
    "ETH-USD": {"nome": "Ethereum Foundation", "url": "https://ethereum.org/en/foundation/"},
    "SOL-USD": {"nome": "Solana Foundation", "url": "https://solana.org/"},
}

# CoinGecko/CoinMarketCap: slugs conhecidos para os criptoativos padrão do app.
CRIPTO_SLUGS = {
    "BTC-USD": "bitcoin",
    "ETH-USD": "ethereum",
    "SOL-USD": "solana",
}

# Fontes gerais confiáveis, sempre exibidas (não dependem do ativo escolhido).
FONTES_GERAIS = {
    "Cotações e dados de mercado": [
        ("Yahoo Finance", "https://finance.yahoo.com/"),
        ("TradingView", "https://www.tradingview.com/"),
        ("Investing.com", "https://www.investing.com/"),
        ("StockAnalysis.com", "https://stockanalysis.com/"),
    ],
    "Renda variável Brasil (fundamentos e regulação)": [
        ("Fundamentus", "https://www.fundamentus.com.br/"),
        ("StatusInvest", "https://statusinvest.com.br/"),
        ("B3 — Empresas listadas", "https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/empresas-listadas.htm"),
        ("CVM — Consulta de documentos (RAD)", "https://www.rad.cvm.gov.br/ENET/frmConsultaExternaCVM.aspx"),
    ],
    "Regulação (EUA)": [
        ("SEC EDGAR — Full Text Search", "https://www.sec.gov/edgar/search/"),
    ],
    "Notícias e análises especializadas (Brasil)": [
        ("InfoMoney", "https://www.infomoney.com.br/"),
        ("Valor Investe", "https://valorinveste.globo.com/"),
        ("Money Times", "https://www.moneytimes.com.br/"),
        ("Suno Research", "https://www.suno.com.br/research/"),
    ],
}

# =============================================================================
# Períodos / Intervalos
# =============================================================================
PERIODOS_UI = {
    "7 dias": "7d",
    "1 mês": "1mo",
    "3 meses": "3mo",
    "6 meses": "6mo",
    "1 ano": "1y",
    "5 anos": "5y",
    "10 anos": "10y",
}

INTERVALOS_UI = {
    "15 minutos": "15m",
    "30 minutos": "30m",
    "1 hora": "1h",
    "4 horas (resample)": "4h",  # obtido via resample de dados de 1h
    "1 dia": "1d",
}

# Limite real documentado pelo Yahoo Finance / yfinance: dados intraday
# (intervalo < 1 dia) não podem ser solicitados para mais de 60 dias de
# histórico. Fonte: documentação oficial do yfinance
# (https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html
# — "Intraday data cannot extend last 60 days").
LIMITE_DIAS_INTRADAY = 60
INTERVALOS_INTRADAY = {"15m", "30m", "1h"}


def _period_to_days(period: str) -> int:
    """Converte um período do estilo yfinance ('7d', '1mo', '1y', ...) em dias aproximados."""
    if period.endswith("d"):
        return int(period[:-1])
    if period.endswith("mo"):
        return int(period[:-2]) * 30
    if period.endswith("y"):
        return int(period[:-1]) * 365
    return 30


def compatibilizar_periodo_intervalo(periodo: str, intervalo_ui: str, intervalo_label: str):
    """
    Garante que a combinação período × intervalo é aceita pelo Yahoo Finance.
    Retorna (periodo_ajustado, intervalo_real, precisa_resample_4h, mensagem_aviso).
    """
    precisa_resample_4h = intervalo_ui == "4h"
    intervalo_real = "1h" if precisa_resample_4h else intervalo_ui

    if intervalo_real in INTERVALOS_INTRADAY:
        dias_solicitados = _period_to_days(periodo)
        if dias_solicitados > LIMITE_DIAS_INTRADAY:
            candidatos = [
                p for p in PERIODOS_UI.values() if _period_to_days(p) <= LIMITE_DIAS_INTRADAY
            ]
            periodo_ajustado = max(candidatos, key=_period_to_days) if candidatos else "7d"
            msg = (
                f"⚠️ O intervalo **{intervalo_label}** só é suportado pelo Yahoo Finance para "
                f"até **{LIMITE_DIAS_INTRADAY} dias** de histórico. Ajustei o período para "
                f"**{periodo_ajustado}**."
            )
            return periodo_ajustado, intervalo_real, precisa_resample_4h, msg

    return periodo, intervalo_real, precisa_resample_4h, None


# =============================================================================
# Download e preparação dos dados
# =============================================================================
@st.cache_data(ttl=300, show_spinner=False)
def baixar_dados_robusto(ticker: str, periodo: str, intervalo_real: str, tentativas: int = 3) -> pd.DataFrame:
    """
    Baixa dados OHLCV do Yahoo Finance com retentativas e um fallback por
    intervalo de datas explícito. Resultado é cacheado por 5 minutos por
    (ticker, período, intervalo) para reduzir chamadas repetidas à API.
    """
    ultimo_erro: Optional[Exception] = None

    for tentativa in range(1, tentativas + 1):
        try:
            df = yf.download(
                ticker, period=periodo, interval=intervalo_real,
                auto_adjust=False, progress=False, threads=False,
            )
            df = _normalizar_colunas(df)
            if df is not None and not df.empty:
                return df.dropna(how="all")
            break  # resultado vazio sem exceção: não adianta tentar de novo
        except Exception as e:  # falhas de rede/rate-limit são transitórias
            ultimo_erro = e
            if tentativa < tentativas:
                time.sleep(1.5 * tentativa)

    # Fallback: solicitar por intervalo de datas explícito (start/end)
    try:
        dias = _period_to_days(periodo)
        fim = datetime.now(timezone.utc)
        inicio = fim - timedelta(days=dias)
        df = yf.download(
            ticker, start=inicio, end=fim, interval=intervalo_real,
            auto_adjust=False, progress=False, threads=False,
        )
        df = _normalizar_colunas(df)
        if df is not None and not df.empty:
            return df.dropna(how="all")
    except Exception as e:
        ultimo_erro = e

    if ultimo_erro is not None:
        raise RuntimeError(f"Falha ao consultar o Yahoo Finance para '{ticker}': {ultimo_erro}")
    return pd.DataFrame()


def _normalizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    """Achata colunas MultiIndex (algumas versões do yfinance retornam
    (campo, ticker) mesmo para um único ativo) para um índice simples."""
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty:
        return df
    ohlc = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ohlc = {k: v for k, v in ohlc.items() if k in df.columns}
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    out = df.resample(rule).agg(ohlc).dropna(how="all")
    if "Adj Close" in df.columns:
        out["Adj Close"] = df["Adj Close"].resample(rule).last()
    return out


def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = len(df)

    df["MA21"] = df["Close"].rolling(21).mean() if n >= 21 else pd.NA
    df["MA200"] = df["Close"].rolling(200).mean() if n >= 200 else pd.NA
    df["EMA17"] = df["Close"].ewm(span=17, adjust=False).mean()
    df["EMA72"] = df["Close"].ewm(span=72, adjust=False).mean()
    df["EMA305"] = df["Close"].ewm(span=305, adjust=False).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, pd.NA)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(100).where(loss.ne(0) | gain.eq(0), 100)

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


def calcular_fibonacci(df: pd.DataFrame) -> dict:
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


@dataclass
class Diagnostico:
    rsi_texto: str
    macd_texto: str
    tendencia_texto: str


def diagnosticar(df: pd.DataFrame) -> Diagnostico:
    rsi = df["RSI"].iloc[-1] if "RSI" in df else float("nan")
    if pd.isna(rsi):
        rsi_texto = "RSI indisponível (dados insuficientes)."
    elif rsi < 30:
        rsi_texto = f"📈 RSI em {rsi:.1f} — zona de sobrevenda (possível alta)."
    elif rsi > 70:
        rsi_texto = f"📉 RSI em {rsi:.1f} — zona de sobrecompra (possível correção)."
    else:
        rsi_texto = f"🔍 RSI em {rsi:.1f} — zona neutra."

    macd_texto = "MACD indisponível (dados insuficientes)."
    if {"MACD", "Signal"}.issubset(df.columns) and len(df) >= 2:
        macd, sig = df["MACD"].iloc[-1], df["Signal"].iloc[-1]
        macd_prev, sig_prev = df["MACD"].iloc[-2], df["Signal"].iloc[-2]
        if pd.notna(macd) and pd.notna(sig) and pd.notna(macd_prev) and pd.notna(sig_prev):
            if macd_prev <= sig_prev and macd > sig:
                macd_texto = "🟢 MACD cruzou a linha de sinal para cima (momento comprador)."
            elif macd_prev >= sig_prev and macd < sig:
                macd_texto = "🔴 MACD cruzou a linha de sinal para baixo (momento vendedor)."
            else:
                macd_texto = "MACD acima da linha de sinal." if macd > sig else "MACD abaixo da linha de sinal."

    tendencia_texto = "Tendência (vs. MA200) indisponível — histórico curto demais para MA200."
    if "MA200" in df.columns and pd.notna(df["MA200"].iloc[-1]):
        preco = df["Close"].iloc[-1]
        ma200 = df["MA200"].iloc[-1]
        tendencia_texto = (
            "📈 Preço acima da MA200 — viés de longo prazo em alta."
            if preco >= ma200 else
            "📉 Preço abaixo da MA200 — viés de longo prazo em baixa."
        )

    return Diagnostico(rsi_texto, macd_texto, tendencia_texto)


# =============================================================================
# Fontes de validação — links dinâmicos por ativo
# =============================================================================
def eh_ticker_br(ticker: str) -> bool:
    return ticker.upper().endswith(".SA")


def eh_ticker_cripto(ticker: str) -> bool:
    return ticker.upper().endswith("-USD")


def montar_links_cruzados(ticker: str) -> dict:
    """Gera links de cotação/dados para o ticker informado em provedores
    confiáveis. Como o app aceita qualquer ticker digitado pelo usuário, os
    links são construídos por padrão de URL conhecido de cada provedor —
    confira sempre se a página aberta corresponde ao ativo certo."""
    ticker = ticker.strip().upper()
    grupos: dict[str, list[tuple[str, str]]] = {}

    # TradingView exige o prefixo da bolsa no símbolo para resolver corretamente.
    if eh_ticker_br(ticker):
        tv_simbolo = f"BMFBOVESPA-{ticker.replace('.SA', '')}"
    elif eh_ticker_cripto(ticker):
        tv_simbolo = ticker.replace("-", "")
    else:
        tv_simbolo = ticker

    grupos["Cotação (confira o gráfico e o preço)"] = [
        ("Yahoo Finance", f"https://finance.yahoo.com/quote/{ticker}"),
        ("TradingView", f"https://www.tradingview.com/symbols/{tv_simbolo}/"),
        ("Investing.com (busca)", f"https://www.investing.com/search/?q={ticker}"),
    ]

    if eh_ticker_cripto(ticker):
        base = ticker.split("-")[0]
        slug = CRIPTO_SLUGS.get(ticker)
        links = []
        if slug:
            links.append(("CoinGecko", f"https://www.coingecko.com/en/coins/{slug}"))
            links.append(("CoinMarketCap", f"https://coinmarketcap.com/currencies/{slug}/"))
        else:
            links.append(("CoinGecko (busca)", f"https://www.coingecko.com/en/search?query={base}"))
            links.append(("CoinMarketCap (busca)", f"https://coinmarketcap.com/search/?q={base}"))
        grupos["Dados de criptoativos"] = links

    elif eh_ticker_br(ticker):
        codigo = ticker.replace(".SA", "")
        grupos["Fundamentos e regulação (Brasil)"] = [
            ("StatusInvest", f"https://statusinvest.com.br/acoes/{codigo.lower()}"),
            ("Fundamentus", f"https://www.fundamentus.com.br/detalhes.php?papel={codigo}"),
            ("StockAnalysis.com (B3)", f"https://stockanalysis.com/quote/bvmf/{codigo}/"),
            ("Google Finance (B3)", f"https://www.google.com/finance/quote/{codigo}:BVMF"),
            ("CVM — Consulta de documentos (RAD)", "https://www.rad.cvm.gov.br/ENET/frmConsultaExternaCVM.aspx"),
        ]
    else:
        grupos["Fundamentos e regulação (EUA/outros)"] = [
            ("StockAnalysis.com", f"https://stockanalysis.com/stocks/{ticker.lower()}/"),
            ("SEC EDGAR — Full Text Search", f"https://www.sec.gov/edgar/search/#/q={ticker}"),
        ]

    return grupos


def render_fontes_validacao(ticker: str, nome_ativo: str):
    st.subheader("🔗 Fontes e validação da informação")
    st.caption(
        "Compare sempre os números do gráfico com os dados na fonte oficial "
        "antes de tomar qualquer decisão."
    )

    oficial = FONTES_OFICIAIS.get(ticker.strip().upper())
    if oficial:
        st.markdown(f"**Fonte oficial de {nome_ativo}:** [{oficial['nome']}]({oficial['url']})")
    else:
        st.caption(
            "Ticker fora da lista padrão — não há RI oficial cadastrado neste app para ele. "
            "Procure por \"[nome da empresa] Investor Relations\" ou \"Relações com Investidores\" "
            "diretamente no site institucional da companhia."
        )

    links = montar_links_cruzados(ticker)
    cols = st.columns(len(links)) if links else []
    for col, (categoria, itens) in zip(cols, links.items()):
        with col:
            st.markdown(f"**{categoria}**")
            for label, url in itens:
                st.markdown(f"- [{label}]({url})")

    with st.expander("📚 Sites e fontes especializadas (gerais, sempre confiáveis)"):
        for categoria, itens in FONTES_GERAIS.items():
            st.markdown(f"**{categoria}**")
            for label, url in itens:
                st.markdown(f"- [{label}]({url})")


# =============================================================================
# Barra lateral — seleção do ativo e parâmetros
# =============================================================================
with st.sidebar:
    st.header("⚙️ Parâmetros da análise")

    modo = st.radio("Modo de seleção do ativo:", ["Escolher da lista", "Digitar ticker"])

    if modo == "Escolher da lista":
        categoria = st.selectbox("Categoria:", list(ATIVOS_FIXOS.keys()))
        nome_ativo = st.selectbox("Ativo:", list(ATIVOS_FIXOS[categoria].keys()))
        ticker = ATIVOS_FIXOS[categoria][nome_ativo]
    else:
        ticker_input = st.text_input(
            "Digite o ticker (ex.: AAPL, PETR4.SA, BTC-USD):", "AAPL"
        )
        ticker = ticker_input.strip().upper()
        nome_ativo = ticker

    periodo_label = st.selectbox("Período:", list(PERIODOS_UI.keys()), index=1)
    intervalo_label = st.selectbox("Intervalo:", list(INTERVALOS_UI.keys()), index=2)

    periodo = PERIODOS_UI[periodo_label]
    intervalo_ui = INTERVALOS_UI[intervalo_label]

    analisar = st.button("🔎 Analisar", use_container_width=True, type="primary")

    st.divider()
    st.caption(
        "⚠️ Ferramenta educacional. Os dados vêm do Yahoo Finance (gratuito, com "
        "possíveis atrasos e ajustes retroativos). Não é recomendação de investimento."
    )

# Validação básica do ticker digitado
if modo == "Digitar ticker" and analisar:
    if not ticker or not all(c.isalnum() or c in ".-^=" for c in ticker):
        st.error("Ticker inválido. Use apenas letras, números e os símbolos . - ^ = (ex.: PETR4.SA).")
        st.stop()

# =============================================================================
# Execução da análise
# =============================================================================
if analisar:
    try:
        periodo_uso, intervalo_real, precisa_resample_4h, msg_info = compatibilizar_periodo_intervalo(
            periodo, intervalo_ui, intervalo_label
        )
        if msg_info:
            st.info(msg_info)

        with st.spinner(f"Baixando dados de {nome_ativo} ({ticker})..."):
            df_raw = baixar_dados_robusto(ticker, periodo_uso, intervalo_real)

        if df_raw.empty:
            st.warning(
                "⚠️ Nenhum dado foi encontrado para esse ticker/período/intervalo. "
                "Verifique se o ticker está correto (ex.: ações do Brasil terminam em "
                "**.SA**, criptomoedas terminam em **-USD**) ou tente outra combinação."
            )
            st.stop()

        if precisa_resample_4h:
            df = resample_ohlcv(df_raw, "4h")
            if df.empty:
                st.warning("⚠️ Não foi possível gerar o timeframe de 4 horas com os dados retornados.")
                st.stop()
        else:
            df = df_raw

        if len(df) < 15:
            st.warning(
                f"⚠️ Apenas {len(df)} candles retornados — poucos dados para indicadores "
                "confiáveis (RSI/MACD precisam de histórico maior). Considere um período mais longo."
            )

        df = calcular_indicadores(df)
        fibo = calcular_fibonacci(df)
        diag = diagnosticar(df)

        ultimo_preco = df["Close"].iloc[-1]
        ultima_data = df.index[-1]
        st.success(
            f"**{nome_ativo}** ({ticker}) — último preço: **{ultimo_preco:,.2f}** "
            f"em {ultima_data:%d/%m/%Y %H:%M}"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Preço + Médias + Fibonacci")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df.index, df["Close"], label="Preço", linewidth=1.6, color="black")
            if df["MA21"].notna().any():
                ax.plot(df.index, df["MA21"], label="MA21")
            if df["MA200"].notna().any():
                ax.plot(df.index, df["MA200"], label="MA200")
            ax.plot(df.index, df["EMA17"], label="EMA17 (Phi)", linestyle="--")
            ax.plot(df.index, df["EMA72"], label="EMA72 (Phi)", linestyle="--")
            ax.plot(df.index, df["EMA305"], label="EMA305 (Phi)", linestyle="--")
            for lvl, val in fibo.items():
                ax.axhline(val, linestyle="--", alpha=0.3)
                ax.text(df.index[0], val, f" Fib {lvl}", fontsize=7, alpha=0.7, va="bottom")
            ax.set_title(f"{nome_ativo} — {periodo_label} / {intervalo_label}")
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.subheader("📈 RSI, MACD e Volume Financeiro")
            fig2, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
            axs[0].plot(df.index, df["RSI"], label="RSI", color="purple")
            axs[0].axhline(70, linestyle="--", color="red", alpha=0.6)
            axs[0].axhline(30, linestyle="--", color="green", alpha=0.6)
            axs[0].set_title("RSI (14)")
            axs[0].legend(fontsize=8)
            axs[0].grid(True, alpha=0.3)

            axs[1].plot(df.index, df["MACD"], label="MACD")
            axs[1].plot(df.index, df["Signal"], label="Signal")
            axs[1].set_title("MACD (12, 26, 9)")
            axs[1].legend(fontsize=8)
            axs[1].grid(True, alpha=0.3)

            axs[2].bar(df.index, df["Volume_Financeiro"], label="Volume financeiro", width=0.8)
            axs[2].set_title("Volume Financeiro (Volume × Close)")
            axs[2].legend(fontsize=8)
            axs[2].grid(True, alpha=0.3)

            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

        st.subheader("📍 Diagnóstico técnico")
        d1, d2, d3 = st.columns(3)
        d1.info(diag.rsi_texto)
        d2.info(diag.macd_texto)
        d3.info(diag.tendencia_texto)
        st.caption(
            "Diagnóstico gerado automaticamente a partir de regras técnicas simples "
            "(RSI, cruzamento de MACD e posição em relação à MA200). Não é recomendação "
            "de compra ou venda — use como um dos vários insumos da sua análise."
        )

        with st.expander("📄 Ver dados calculados"):
            st.dataframe(df.tail(50))
            csv = df.to_csv().encode("utf-8")
            st.download_button(
                "⬇️ Baixar dados completos (CSV)",
                data=csv,
                file_name=f"{ticker}_{periodo_uso}_{intervalo_real}.csv",
                mime="text/csv",
            )

        st.divider()
        render_fontes_validacao(ticker, nome_ativo)

    except Exception as e:
        st.error(f"Erro ao processar: {e}")
        st.caption(
            "Se o erro persistir, o Yahoo Finance pode estar limitando requisições "
            "temporariamente (rate limit) — aguarde alguns minutos e tente novamente."
        )
else:
    st.info("Configure os parâmetros na barra lateral e clique em **🔎 Analisar**.")
    with st.expander("📚 Sites e fontes especializadas (gerais, sempre confiáveis)", expanded=False):
        for categoria, itens in FONTES_GERAIS.items():
            st.markdown(f"**{categoria}**")
            for label, url in itens:
                st.markdown(f"- [{label}]({url})")
