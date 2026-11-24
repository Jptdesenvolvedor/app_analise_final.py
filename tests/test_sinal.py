import numpy as np
import pandas as pd
import pytest

from analise.apresentacao import barra_score, cartao_decisao, chip_tendencia, lista_motivos
from analise.indicators import calcular_indicadores
from analise.sinal import (
    Tendencia, calcular_decisao, tendencia_curta, tendencia_longa, tendencia_media, tendencia_mercado,
)


def ohlc(close, start="2020-01-01"):
    close = pd.Series(close, dtype=float)
    idx = pd.bdate_range(start, periods=len(close))
    return pd.DataFrame({"Open": close.values, "High": close.values * 1.005, "Low": close.values * 0.995,
                         "Close": close.values, "Volume": 1000.0}, index=idx)


def com_ruido(base, seed=3, vol=0.004):
    r = np.random.default_rng(seed)
    return np.asarray(base) * np.exp(np.cumsum(r.normal(0, vol, len(base))) * 0.2)


def _alta_com_recuos():
    """Alta firme com recuos periódicos de ~8% (ciclo de 60 candles). A fase termina com RSI ~52."""
    i = np.arange(500)
    return (50 + 350 * i / 499) * (1 + 0.08 * np.sin(2 * np.pi * (i + 32) / 60))


ALTA_LISA = lambda: calcular_indicadores(ohlc(com_ruido(np.linspace(50, 400, 500))))   # noqa: E731
ALTA = lambda: calcular_indicadores(ohlc(_alta_com_recuos()))                         # noqa: E731
BAIXA = lambda: calcular_indicadores(ohlc(com_ruido(np.linspace(400, 60, 500))))      # noqa: E731
LATERAL = lambda: calcular_indicadores(ohlc(100 * (1 + 0.008 * np.sin(np.linspace(0, 40, 500)))))  # noqa: E731


# ----------------------------- tendências -----------------------------------
def test_tendencia_longa_alta_baixa_lateral():
    assert tendencia_longa(ALTA_LISA()["Close"]).direcao == "alta"
    assert tendencia_longa(ALTA_LISA()["Close"]).forca == "forte"
    assert tendencia_longa(ALTA()["Close"]).direcao == "alta"       # com recuos continua alta
    assert tendencia_longa(BAIXA()["Close"]).direcao == "baixa"
    assert tendencia_longa(LATERAL()["Close"]).direcao == "lateral"


def test_tendencias_com_historico_curto_sao_indefinidas():
    curto = calcular_indicadores(ohlc(np.linspace(10, 12, 40)))
    assert tendencia_longa(curto["Close"]).direcao == "indefinida"
    assert tendencia_media(curto["Close"]).direcao == "indefinida"
    assert tendencia_curta(curto).direcao == "indefinida"
    assert tendencia_longa(curto["Close"]).rotulo == "Indefinida"


def test_tendencia_media_e_curta_em_alta_e_baixa():
    assert tendencia_media(ALTA_LISA()["Close"]).direcao == "alta"
    assert tendencia_media(BAIXA()["Close"]).direcao == "baixa"
    assert tendencia_curta(ALTA_LISA()).direcao == "alta"
    assert tendencia_curta(BAIXA()).direcao == "baixa"


def test_tendencia_mercado_usa_regra_longa():
    t = tendencia_mercado(ALTA()["Close"], "SPY")
    assert t.prazo == "Mercado (SPY)" and t.direcao == "alta"


# ----------------------------- decisão --------------------------------------
def test_decisao_alta_forte_compra_ou_mantem():
    d = calcular_decisao(ALTA())
    assert d.acao in ("COMPRAR", "MANTER") and d.tom == "alta"
    assert d.score > 40 and d.confianca in ("Alta", "Média")
    assert d.stop is not None and d.alvo is not None and d.stop < d.alvo
    assert {m.titulo for m in d.motivos} >= {"Regime de longo prazo", "Modelo de tendência", "Momentum 12-1"}


def test_decisao_queda_vende_ou_reduz():
    d = calcular_decisao(BAIXA())
    assert d.acao in ("VENDER", "REDUZIR") and d.tom == "baixa"
    assert d.score < -40
    assert d.stop is None and d.alvo is None          # sem plano de compra em queda


def test_decisao_lateral_aguarda():
    d = calcular_decisao(LATERAL())
    assert d.acao == "AGUARDAR" and d.tom == "neutro"
    assert -15 < d.score < 15


def test_decisao_historico_curto_nao_quebra_e_aguarda():
    d = calcular_decisao(calcular_indicadores(ohlc(np.linspace(10, 12, 25))))
    assert d.acao == "AGUARDAR" and d.confianca == "Baixa"
    assert "insuficiente" in d.resumo.lower()


def test_decisao_historico_medio_reescala_e_perde_confianca():
    d = calcular_decisao(calcular_indicadores(ohlc(com_ruido(np.linspace(50, 120, 120)))))
    assert -100 <= d.score <= 100
    assert d.confianca == "Baixa"                      # sem MA200/252 candles, cobertura < 60%


def test_rsi_esticado_rebaixa_compra_para_manter():
    df = ALTA().copy()
    base = calcular_decisao(df)
    assert base.acao == "COMPRAR"
    df.loc[df.index[-1], "RSI"] = 90.0
    d = calcular_decisao(df)
    assert d.acao == "MANTER" and any(m.titulo == "RSI esticado" for m in d.motivos)


def test_decisao_nao_olha_o_futuro():
    df = ALTA()
    antes = calcular_decisao(df.iloc[:400])
    mexido = df.copy()
    mexido.iloc[400:, mexido.columns.get_indexer(["Close", "Open", "High", "Low"])] *= 0.2
    assert calcular_decisao(calcular_indicadores(mexido[["Open", "High", "Low", "Close", "Volume"]].iloc[:400])).score \
        == pytest.approx(antes.score)


# ----------------------------- apresentação ---------------------------------
def test_cartao_contem_selo_preco_e_aviso():
    d = calcular_decisao(ALTA())
    html = cartao_decisao(d, "Apple <Inc>", "AAPL", 123.45, 0.0123, "24/11/2026",
                          extras=[Tendencia("Mercado (SPY)", "alta", "moderada", "ok")])
    assert d.acao in html and "123.45" in html and "+1.23%" in html
    assert "Mercado (SPY)" in html and "Não é recomendação" in html
    assert "&lt;Inc&gt;" in html and "<Inc>" not in html     # escapa texto do usuário


def test_barra_score_posiciona_pino():
    assert "left:calc(0.0% - 2px)" in barra_score(-100)
    assert "left:calc(50.0% - 2px)" in barra_score(0)
    assert "left:calc(100.0% - 2px)" in barra_score(250)    # saturado em 100


def test_chip_e_lista_de_motivos():
    assert "↗ Alta forte" in chip_tendencia(Tendencia("Longo", "alta", "forte", "x"))
    assert "↘ Baixa moderada" in chip_tendencia(Tendencia("Curto", "baixa", "moderada", "x"))
    md = lista_motivos(calcular_decisao(ALTA()))
    assert md.count("\n") >= 3 and "**Regime de longo prazo**" in md


def test_zonas_da_barra_batem_com_os_limiares():
    from analise.apresentacao import ZONAS
    from analise.sinal import _acao_do_score
    assert sum(w for _, w, _ in ZONAS) == 200
    inicio = -100.0
    for nome, largura, _ in ZONAS:
        meio = inicio + largura / 2                       # centro da zona
        assert _acao_do_score(meio).capitalize() == nome, (nome, meio)
        inicio += largura


def test_sem_regime_de_longo_prazo_o_sinal_nao_e_firme():
    """Listagem recente (120 candles) em queda forte: sem MA200 não dá para dizer VENDER; aguarda e explica."""
    queda = calcular_indicadores(ohlc(com_ruido(np.linspace(100, 20, 120))))
    d = calcular_decisao(queda)
    assert d.acao == "AGUARDAR" and d.confianca == "Baixa" and "insuficiente" in d.resumo.lower()
    assert d.score < -50                                   # a pontuação continua mostrando a direção
    com_historico = calcular_decisao(calcular_indicadores(ohlc(com_ruido(np.linspace(400, 60, 500)))))
    assert com_historico.acao == "VENDER"                  # com 200+ candles o sinal firme volta
