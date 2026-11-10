"""
📊 Análise Técnica de Ativos — Streamlit
=========================================

Análise técnica de ações (Brasil/EUA) e criptomoedas com dados do Yahoo Finance
(via `yfinance`). A lógica fica no pacote `analise/`; este arquivo só monta a UI.

⚠️ Ferramenta educacional/informativa. Nada aqui constitui recomendação de
investimento. Confirme os números nas fontes oficiais (aba "Fontes").
"""

from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from analise import data as dados
from analise.backtest import ESTRATEGIAS, executar_backtest
from analise.charts import OpcoesGrafico, grafico_backtest, grafico_comparacao, grafico_principal
from analise.config import (
    ATIVOS_FIXOS, INTERVALOS_INTRADAY, INTERVALOS_UI, MIN_CANDLES, PERIODOS_UI,
    compatibilizar_periodo_intervalo,
)
from analise.diagnostico import detectar_eventos, diagnosticar
from analise.fundamentos import buscar_fundamentos, formatar_valor
from analise.indicators import calcular_fibonacci, calcular_indicadores, suporte_resistencia
from analise.sources import FONTES_GERAIS, FONTES_OFICIAIS, eh_ticker_cripto, montar_links_cruzados

st.set_page_config(page_title="Análise Técnica de Ativos", page_icon="📊", layout="wide")
st.title("📊 Análise Técnica Avançada de Ativos")
st.caption(
    "Dados via Yahoo Finance (yfinance). Ferramenta educacional/informativa — "
    "não constitui recomendação de investimento."
)

TICKER_RE = re.compile(r"^[A-Z0-9.\-^=]{1,15}$")
TOM_ICONE = {"alta": "🟢", "baixa": "🔴", "neutro": "⚪"}


# --------------------------------------------------------------------------- #
# Camadas com cache (o Streamlit reexecuta o script a cada interação)
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=300, show_spinner=False, max_entries=128)
def baixar(ticker: str, periodo: str, intervalo_real: str) -> pd.DataFrame:
    # Erros/vazio levantam exceção e, portanto, não ficam em cache.
    return dados.baixar_dados(ticker, periodo, intervalo_real)


@st.cache_data(ttl=3600, show_spinner=False)
def fundamentos(ticker: str) -> dict:
    return buscar_fundamentos(ticker)


def preparar(ticker: str, periodo: str, intervalo_ui: str, intervalo_label: str):
    """Baixa, reamostra (4h) e calcula indicadores. Retorna (df, params)."""
    params = compatibilizar_periodo_intervalo(periodo, intervalo_ui, intervalo_label)
    bruto = baixar(ticker, params.periodo, params.intervalo_real)
    if params.resample_4h:
        bruto = dados.resample_ohlcv(bruto, "4h", cripto=eh_ticker_cripto(ticker))
        if bruto.empty:
            raise dados.DadosIndisponiveis("Não foi possível gerar o timeframe de 4 horas.")
    intraday = params.intervalo_real in INTERVALOS_INTRADAY
    return calcular_indicadores(bruto, intraday=intraday), params


# --------------------------------------------------------------------------- #
# Barra lateral
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Parâmetros")
    modo = st.radio("Seleção do ativo:", ["Escolher da lista", "Digitar ticker"])
    if modo == "Escolher da lista":
        categoria = st.selectbox("Categoria:", list(ATIVOS_FIXOS))
        nome_ativo = st.selectbox("Ativo:", list(ATIVOS_FIXOS[categoria]))
        ticker = ATIVOS_FIXOS[categoria][nome_ativo]
    else:
        ticker = st.text_input("Ticker (ex.: AAPL, PETR4.SA, BTC-USD):", "AAPL").strip().upper()
        nome_ativo = ticker

    periodo_label = st.selectbox("Período:", list(PERIODOS_UI), index=4)
    intervalo_label = st.selectbox("Intervalo:", list(INTERVALOS_UI), index=4)
    periodo, intervalo_ui = PERIODOS_UI[periodo_label], INTERVALOS_UI[intervalo_label]

    with st.expander("🎛️ Camadas do gráfico"):
        op = OpcoesGrafico(
            candles=st.checkbox("Candles (senão, linha)", True),
            medias=st.checkbox("MA21 / MA200", True),
            emas_phi=st.checkbox("EMAs PhiCube (17/72/305)", True),
            bollinger=st.checkbox("Bandas de Bollinger", False),
            vwap=st.checkbox("VWAP (só intraday)", False),
            fibonacci=st.checkbox("Fibonacci", True),
            suporte_resistencia=st.checkbox("Suporte / resistência", True),
            estocastico=st.checkbox("Estocástico", False),
            volume_financeiro=st.checkbox("Volume financeiro (Volume × Close)", False),
        )

    st.divider()
    st.caption(
        "⚠️ Ferramenta educacional. Dados gratuitos do Yahoo Finance, com possíveis "
        "atrasos e ajustes retroativos. Não é recomendação de investimento."
    )

if not TICKER_RE.match(ticker):
    st.error("Ticker inválido. Use letras, números e os símbolos . - ^ = (ex.: PETR4.SA).")
    st.stop()

# --------------------------------------------------------------------------- #
# Dados
# --------------------------------------------------------------------------- #
try:
    with st.spinner(f"Baixando dados de {nome_ativo} ({ticker})..."):
        df, params = preparar(ticker, periodo, intervalo_ui, intervalo_label)
except dados.DadosIndisponiveis as e:
    st.warning(
        f"⚠️ {e} Verifique o ticker (ações do Brasil terminam em **.SA**, cripto em **-USD**), "
        "tente outra combinação ou aguarde alguns minutos (o Yahoo pode estar limitando requisições)."
    )
    st.stop()
except Exception as e:  # noqa: BLE001 — qualquer falha inesperada vira mensagem amigável
    st.error(f"Erro ao processar: {e}")
    st.stop()

if params.aviso:
    st.info(params.aviso)

n = len(df)
if n < 15:
    st.warning(f"⚠️ Apenas {n} candles — poucos para indicadores confiáveis. Use um período maior.")
indisponiveis = [k for k, minimo in MIN_CANDLES.items() if n < minimo]
if indisponiveis:
    st.caption(f"ℹ️ Sem candles suficientes para: {', '.join(indisponiveis)} (não são desenhados).")

ultimo = df["Close"].iloc[-1]
ts = df.index[-1]
if params.intervalo_real in INTERVALOS_INTRADAY and getattr(ts, "tzinfo", None) is not None:
    ts_txt = f"{ts.tz_convert('America/Sao_Paulo'):%d/%m/%Y %H:%M} (Brasília)"
else:
    ts_txt = f"{ts:%d/%m/%Y}"
variacao = df["Close"].pct_change().iloc[-1] if n > 1 else float("nan")
st.success(f"**{nome_ativo}** ({ticker}) — último: **{ultimo:,.2f}** "
           f"({variacao:+.2%} no último candle) em {ts_txt}")

cripto = eh_ticker_cripto(ticker)
intraday_horario = params.intervalo_real in {"15m", "30m", "1h"} and not params.resample_4h

aba_graf, aba_diag, aba_bt, aba_comp, aba_fund, aba_dados, aba_fontes = st.tabs(
    ["📈 Gráfico", "📍 Diagnóstico e alertas", "🧪 Backtest", "⚖️ Comparar ativos",
     "🏢 Fundamentos", "📄 Dados", "🔗 Fontes"]
)

# --------------------------------------------------------------------------- #
with aba_graf:
    fib = calcular_fibonacci(df) if op.fibonacci else None
    sr = suporte_resistencia(df) if op.suporte_resistencia else None
    titulo = f"{nome_ativo} — {periodo_label if not params.aviso else params.periodo} / {intervalo_label}"
    st.plotly_chart(
        grafico_principal(df, titulo, op, fib, sr, cripto=cripto, intraday_horario=intraday_horario),
        width="stretch",
    )
    if fib:
        st.caption(f"Fibonacci: retração de **{fib.tendencia}** sobre máxima/mínima do período carregado.")

# --------------------------------------------------------------------------- #
with aba_diag:
    itens = diagnosticar(df)
    cols = st.columns(3)
    for i, it in enumerate(itens):
        with cols[i % 3]:
            st.markdown(f"**{TOM_ICONE[it.tom]} {it.titulo}**")
            st.write(it.texto)
    st.caption(
        "Regras técnicas simples, geradas automaticamente. Não é recomendação de compra ou "
        "venda — use como um dos vários insumos da sua análise."
    )

    st.subheader("🔔 Eventos recentes")
    eventos = detectar_eventos(df)
    if eventos.empty:
        st.write("Nenhum cruzamento relevante no período.")
    else:
        recentes = eventos.head(15).copy()
        recentes["quando"] = recentes["data"].apply(
            lambda d: f"{d:%d/%m/%Y %H:%M}" if params.intervalo_real in INTERVALOS_INTRADAY else f"{d:%d/%m/%Y}")
        recentes["evento"] = recentes["tom"].map(TOM_ICONE) + " " + recentes["evento"]
        st.dataframe(recentes[["quando", "evento"]], hide_index=True, width="stretch")
        if (eventos["data"] == df.index[-1]).any():
            st.warning("🔔 Há evento(s) no **último candle** — veja o topo da lista.")

# --------------------------------------------------------------------------- #
with aba_bt:
    st.write(
        "Simulação **long-only** da regra escolhida sobre o período carregado, executada no "
        "candle seguinte ao sinal. Desempenho passado não garante resultado futuro."
    )
    c1, c2 = st.columns([2, 1])
    nome_est = c1.selectbox("Estratégia:", list(ESTRATEGIAS))
    custo = c2.number_input("Custo por operação (%)", 0.0, 5.0, 0.1, 0.05) / 100
    res = executar_backtest(df, ESTRATEGIAS[nome_est](df), custo)

    m = st.columns(5)
    m[0].metric("Retorno da estratégia", f"{res.retorno_estrategia:+.1%}")
    m[1].metric("Buy & hold", f"{res.retorno_buy_hold:+.1%}")
    m[2].metric("Operações", res.n_trades)
    m[3].metric("Taxa de acerto", "—" if res.taxa_acerto is None else f"{res.taxa_acerto:.0%}")
    m[4].metric("Drawdown máx.", f"{res.max_drawdown:.1%}")
    st.caption(f"Tempo posicionado: {res.tempo_posicionado:.0%}. Sem impostos, slippage ou dividendos.")
    st.plotly_chart(grafico_backtest(res, nome_est), width="stretch")
    if not res.trades.empty:
        with st.expander("Operações simuladas"):
            st.dataframe(res.trades.assign(retorno=res.trades["retorno"].map("{:+.1%}".format)),
                         hide_index=True, width="stretch")

# --------------------------------------------------------------------------- #
with aba_comp:
    st.write(f"Desempenho relativo no mesmo período/intervalo ({periodo_label} / {intervalo_label}).")
    extras = st.text_input("Tickers para comparar (separados por vírgula):", "AAPL, MSFT, NVDA")
    lista = [t.strip().upper() for t in extras.split(",") if t.strip()]
    lista = [t for t in dict.fromkeys([ticker] + lista) if TICKER_RE.match(t)][:8]
    series, falhas = {}, []
    for t in lista:
        try:
            d, _ = preparar(t, periodo, intervalo_ui, intervalo_label)
            series[t] = dados.normalizar_base100(d["Close"])
        except Exception:  # noqa: BLE001
            falhas.append(t)
    if len(series) >= 2:
        st.plotly_chart(grafico_comparacao(series), width="stretch")
        resumo = pd.DataFrame({"Retorno no período": {t: s.iloc[-1] / 100 - 1 for t, s in series.items()}})
        st.dataframe(resumo.style.format("{:+.1%}"), width="stretch")
    else:
        st.info("Informe ao menos um ticker válido além do selecionado.")
    if falhas:
        st.caption(f"Sem dados para: {', '.join(falhas)}.")

# --------------------------------------------------------------------------- #
with aba_fund:
    info = fundamentos(ticker)
    if cripto or not info:
        st.info("Fundamentos indisponíveis para este ativo no Yahoo Finance (comuns em cripto/ETFs).")
    else:
        itens_f = list(info.items())
        cols = st.columns(3)
        for i, (rotulo, valor) in enumerate(itens_f):
            cols[i % 3].metric(rotulo, formatar_valor(rotulo, valor))
        st.caption("Valores como informados pelo Yahoo Finance; confirme no RI/CVM/SEC (aba Fontes).")

# --------------------------------------------------------------------------- #
with aba_dados:
    st.dataframe(df.tail(100), width="stretch")
    st.download_button(
        "⬇️ Baixar dados completos (CSV)",
        data=df.to_csv().encode("utf-8"),
        file_name=f"{ticker}_{params.periodo}_{params.intervalo_real}.csv",
        mime="text/csv",
    )

# --------------------------------------------------------------------------- #
with aba_fontes:
    st.caption("Compare sempre os números do gráfico com a fonte oficial antes de decidir.")
    oficial = FONTES_OFICIAIS.get(ticker)
    if oficial:
        st.markdown(f"**Fonte oficial de {nome_ativo}:** [{oficial['nome']}]({oficial['url']})")
    else:
        st.caption(
            "Ticker fora da lista padrão — sem RI cadastrado. Procure por "
            "\"[empresa] Investor Relations\" / \"Relações com Investidores\" no site da companhia."
        )
    links = montar_links_cruzados(ticker)
    for col, (cat, itens_l) in zip(st.columns(len(links)), links.items()):
        with col:
            st.markdown(f"**{cat}**")
            for label, url in itens_l:
                st.markdown(f"- [{label}]({url})")
    with st.expander("📚 Sites e fontes especializadas"):
        for cat, itens_l in FONTES_GERAIS.items():
            st.markdown(f"**{cat}**")
            for label, url in itens_l:
                st.markdown(f"- [{label}]({url})")
