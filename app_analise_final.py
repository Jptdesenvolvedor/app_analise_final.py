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
from analise.backtest import ESTRATEGIAS, executar_backtest, executar_treino_teste
from analise.apresentacao import cartao_decisao, lista_motivos
from analise.charts import OpcoesGrafico, grafico_backtest, grafico_comparacao, grafico_principal
from analise.config import (
    ATIVOS_FIXOS, INTERVALOS_INTRADAY, INTERVALOS_UI, MIN_CANDLES, PERIODOS_UI,
    compatibilizar_periodo_intervalo,
)
from analise.diagnostico import detectar_eventos, diagnosticar
from analise.fundamentos import buscar_fundamentos, formatar_valor
from analise.indicators import (
    calcular_fibonacci, calcular_fibonacci_perna, calcular_indicadores, suporte_resistencia,
)
from analise.regime import (
    BANDA_PADRAO, checklist_tendencia, dimensionar_posicao, estado_defensivo, momentum_12_1,
)
from analise.sinal import calcular_decisao, tendencia_mercado
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


def benchmark_do_ativo(ticker: str) -> tuple[str, str] | None:
    """Índice de referência do 'mercado geral' para o ativo: (ticker, nome)."""
    t = ticker.upper()
    if t.endswith(".SA"):
        return "^BVSP", "Ibovespa"
    if t.endswith("-USD"):
        return None if t == "BTC-USD" else ("BTC-USD", "Bitcoin")
    return "SPY", "S&P 500"


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

    periodo_label = st.selectbox("Período:", list(PERIODOS_UI), index=5)
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

        modo_fib = st.radio(
            "Fibonacci sobre:", ["Perna de alta recente", "Máxima/mínima do período inteiro"],
            help="A perna recente usa só os últimos 120 candles e traz zona de ouro e alvos de extensão.",
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
cripto = eh_ticker_cripto(ticker)

# Painel de decisão: sinal, tendência por prazo e do mercado geral
decisao = calcular_decisao(df)
extras = []
bench = benchmark_do_ativo(ticker)
if bench:
    try:
        d_b = baixar(bench[0], "2y", "1d")
        extras.append(tendencia_mercado(d_b["Close"], bench[1]))
    except Exception:  # noqa: BLE001 — o mercado geral é informativo; sem ele o painel segue
        pass
st.markdown(
    cartao_decisao(decisao, nome_ativo, ticker, float(ultimo),
                   None if pd.isna(variacao) else float(variacao), ts_txt, extras),
    unsafe_allow_html=True,
)
if params.intervalo_real != "1d":
    st.caption("ℹ️ O sinal foi calibrado para candles **diários**. Em intervalos curtos use-o apenas como referência.")
with st.expander("🔎 Por que este sinal?"):
    st.markdown(lista_motivos(decisao))
    st.caption("Pontuação: regime de longo prazo (±40), modelo de tendência (±20), momentum (±10), médio prazo (±15) "
               "e curto prazo (±15). ≥ +50 comprar · +15 a +50 manter · −15 a +15 aguardar · −50 a −15 reduzir · "
               "≤ −50 vender. Veja a aba Backtest para o histórico das regras neste ativo.")

intraday_horario = params.intervalo_real in {"15m", "30m", "1h"} and not params.resample_4h

aba_graf, aba_diag, aba_regime, aba_bt, aba_comp, aba_fund, aba_dados, aba_fontes = st.tabs(
    ["📈 Gráfico", "📍 Diagnóstico e alertas", "🧭 Regime e risco", "🧪 Backtest",
     "⚖️ Comparar ativos", "🏢 Fundamentos", "📄 Dados", "🔗 Fontes"]
)

# --------------------------------------------------------------------------- #
with aba_graf:
    perna = modo_fib.startswith("Perna")
    fib = (calcular_fibonacci_perna(df) if perna else calcular_fibonacci(df)) if op.fibonacci else None
    sr = suporte_resistencia(df) if op.suporte_resistencia else None
    titulo = f"{nome_ativo} — {periodo_label if not params.aviso else params.periodo} / {intervalo_label}"
    st.plotly_chart(
        grafico_principal(df, titulo, op, fib, sr, cripto=cripto, intraday_horario=intraday_horario),
        width="stretch",
    )
    if fib:
        if perna:
            st.caption("Fibonacci da **perna de alta recente** (últimos 120 candles, sem olhar o futuro); "
                       "linhas 127,2% e 161,8% são alvos de extensão.")
        else:
            st.caption(f"Fibonacci: retração de **{fib.tendencia}** sobre máxima/mínima do período carregado.")
    elif op.fibonacci and perna:
        st.caption("Sem perna de alta recente relevante (mínimo de 10%) — Fibonacci não desenhado.")

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
with aba_regime:
    st.write(
        "Regras que **sobreviveram a backtests longos fora da amostra** (SPY/QQQ e milhares de ações, "
        "treino 2017–22 × teste 2023–26). Pensadas para candles **diários**."
    )
    if params.intervalo_real != "1d":
        st.info("Esta aba foi calibrada para candles diários. Selecione o intervalo **1 dia** "
                "e um período de 2 anos ou mais.")
    close = df["Close"]
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Regime (núcleo defensivo)")
        if len(close) < 200:
            st.caption("Precisa de 200 candles.")
        else:
            investido = bool(estado_defensivo(close).iloc[-1])
            ma200_v = float(close.rolling(200).mean().iloc[-1])
            st.metric("Posição sugerida pela regra", "INVESTIDO" if investido else "EM CAIXA",
                      f"{close.iloc[-1] / ma200_v - 1:+.1%} da MA200", delta_color="off")
            st.caption(f"Entra acima de +{BANDA_PADRAO:.0%} da MA200 e só sai abaixo de −{BANDA_PADRAO:.0%} "
                       "(a banda evita trocas por ruído). No SPY reduziu a queda máxima de −34% para "
                       "cerca de −21%, em troca de ~5 pontos de retorno anual.")
        mom = momentum_12_1(close)
        st.metric("Momentum 12-1", "—" if pd.isna(mom) else f"{mom:+.1%}")
    with c2:
        st.subheader("Modelo de tendência (Minervini/O'Neil)")
        crit = checklist_tendencia(close)
        if crit is None:
            st.caption("Precisa de 252 candles.")
        else:
            for item in crit:
                st.write(f"{'✅' if item.ok else '❌'} {item.nome}")
            st.caption(f"{sum(c.ok for c in crit)} de {len(crit)} critérios. Em carteira de 15 ações, "
                       "tendência + momentum foi o único sinal consistente entre treino e teste.")

    st.subheader("Tamanho da posição (risco fixo por operação)")
    k1, k2, k3, k4 = st.columns(4)
    capital = k1.number_input("Capital", min_value=0.0, value=100_000.0, step=1_000.0)
    risco = k2.number_input("Risco por operação (%)", 0.1, 5.0, 1.0, 0.1) / 100
    mult = k3.number_input("Stop = N × ATR", 0.5, 6.0, 2.0, 0.5)
    teto = k4.number_input("Máx. por ativo (%)", 1.0, 100.0, 20.0, 1.0) / 100
    atr_v = float(df["ATR"].iloc[-1]) if "ATR" in df and pd.notna(df["ATR"].iloc[-1]) else float("nan")
    plano = dimensionar_posicao(capital, float(ultimo), atr_v, risco, mult, teto, fracionario=cripto)
    if plano is None:
        if pd.isna(atr_v):
            st.warning("ATR indisponível (poucos candles) — não dá para calcular o stop.")
        else:
            st.warning(
                f"Não cabe 1 unidade com esses parâmetros: o risco de 1 unidade até o stop é de "
                f"{mult * atr_v:,.2f} e você aceita arriscar {capital * risco:,.2f}. "
                "Aumente o capital ou o risco, ou reduza o múltiplo do ATR.")
    else:
        mm = st.columns(5)
        mm[0].metric("Quantidade", f"{plano.quantidade:,.6g}" if cripto else f"{plano.quantidade:,}")
        mm[1].metric("Valor da posição", f"{plano.valor_posicao:,.0f}", f"{plano.pct_capital:.1%} do capital",
                     delta_color="off")
        mm[2].metric("Stop", f"{plano.stop:,.2f}")
        mm[3].metric("Alvo (3× o risco)", f"{plano.alvo:,.2f}")
        mm[4].metric("Risco máximo", f"{plano.risco_valor:,.0f}")
        st.caption("O stop é onde a tese se invalida: se for atingido, a perda fica limitada ao risco "
                   "definido. Ferramenta de cálculo, não recomendação.")

# --------------------------------------------------------------------------- #
with aba_bt:
    st.write(
        "Simulação **long-only** da regra escolhida sobre o período carregado, executada no "
        "candle seguinte ao sinal. Desempenho passado não garante resultado futuro."
    )
    c1, c2, c3 = st.columns([2, 1, 1])
    nome_est = c1.selectbox("Estratégia:", list(ESTRATEGIAS))
    custo = c2.number_input("Custo por operação (%)", 0.0, 5.0, 0.1, 0.05) / 100
    usar_stop = c3.checkbox("Stop móvel por ATR", False)
    mult_stop = c3.number_input("× ATR", 1.0, 6.0, 3.0, 0.5, disabled=not usar_stop)
    stop = float(mult_stop) if usar_stop else None
    sinal = ESTRATEGIAS[nome_est](df)
    res = executar_backtest(df, sinal, custo, stop)

    def _pct(x):
        return "—" if x is None or pd.isna(x) else f"{x:+.1%}"

    linha1 = st.columns(3)
    linha1[0].metric("Retorno da estratégia", f"{res.retorno_estrategia:+.1%}")
    linha1[1].metric("Buy & hold", f"{res.retorno_buy_hold:+.1%}")
    linha1[2].metric("Operações", res.n_trades)
    linha2 = st.columns(3)
    linha2[0].metric("Taxa de acerto", "—" if res.taxa_acerto is None else f"{res.taxa_acerto:.0%}")
    linha2[1].metric("Drawdown máx.", f"{res.max_drawdown:.1%}")
    linha2[2].metric("Sharpe", "—" if res.sharpe is None else f"{res.sharpe:.2f}")
    sh_bh = "—" if res.sharpe_buy_hold is None else f"{res.sharpe_buy_hold:.2f}"
    st.caption(f"Buy & hold: drawdown máx. {res.max_drawdown_buy_hold:.1%} · Sharpe {sh_bh} · "
               f"CAGR {_pct(res.cagr_buy_hold)}. Estratégia: CAGR {_pct(res.cagr)}, "
               f"tempo posicionado {res.tempo_posicionado:.0%}. Sem impostos, slippage ou dividendos.")
    if (res.retorno_estrategia < res.retorno_buy_hold and res.sharpe is not None
            and res.sharpe_buy_hold is not None and res.sharpe < res.sharpe_buy_hold):
        st.warning("Nesta janela a estratégia ficou **abaixo do buy & hold** em retorno e em Sharpe. "
                   "Regras técnicas isoladas costumam não bater o mercado depois dos custos.")
    st.plotly_chart(grafico_backtest(res, nome_est), width="stretch")

    st.subheader("Treino × teste (fora da amostra)")
    st.caption("A mesma regra em duas janelas seguidas (60% / 40% do período). Se funciona só em uma, "
               "provavelmente é sorte do período.")
    partes = executar_treino_teste(df, sinal, custo, stop)
    if len(partes) < 2:
        st.info("Período curto demais para dividir (mínimo de 30 candles em cada parte). Use mais histórico.")
    else:
        linhas = []
        for nome_p, r in partes.items():
            linhas.append({
                "Janela": nome_p.capitalize(), "Retorno": _pct(r.retorno_estrategia),
                "Buy & hold": _pct(r.retorno_buy_hold), "CAGR": _pct(r.cagr),
                "Drawdown": f"{r.max_drawdown:.1%}", "DD buy & hold": f"{r.max_drawdown_buy_hold:.1%}",
                "Sharpe": "—" if r.sharpe is None else f"{r.sharpe:.2f}", "Operações": r.n_trades,
            })
        st.dataframe(pd.DataFrame(linhas), hide_index=True, width="stretch")
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
