import numpy as np
import pandas as pd
import pytest

from analise.analistas import Analistas, montar_analistas
from analise.apresentacao import cartao_decisao, linha_analistas
from analise.charts import OpcoesGrafico, grafico_alvos, grafico_principal, grafico_recomendacoes
from analise.indicators import calcular_indicadores
from analise.sinal import calcular_decisao

INFO = {"currentPrice": 100.0, "targetMeanPrice": 120.0, "targetMedianPrice": 118.0, "targetHighPrice": 150.0,
        "targetLowPrice": 80.0, "numberOfAnalystOpinions": 30, "recommendationKey": "buy", "recommendationMean": 2.1}
ALVOS = {"current": 100.0, "mean": 120.0, "median": 118.0, "high": 150.0, "low": 80.0}
RESUMO = pd.DataFrame({"period": ["0m", "-1m", "-2m", "-3m"], "strongBuy": [6, 6, 6, 5], "buy": [20, 18, 19, 19],
                       "hold": [13, 13, 13, 14], "sell": [3, 3, 3, 3], "strongSell": [3, 3, 3, 2]})


def revisoes(linhas):
    idx = pd.DatetimeIndex([l[0] for l in linhas], name="GradeDate")
    return pd.DataFrame({"Firm": [l[1] for l in linhas], "ToGrade": "Buy", "FromGrade": "Hold", "Action": "main",
                         "priceTargetAction": [l[2] for l in linhas], "currentPriceTarget": [l[3] for l in linhas],
                         "priorPriceTarget": [l[4] for l in linhas]}, index=idx)


REV = revisoes([("2026-11-10", "A", "Raises", 130.0, 120.0), ("2026-11-05", "B", "Raises", 140.0, 125.0),
                ("2026-10-20", "C", "Lowers", 100.0, 110.0), ("2026-10-01", "D", "Announces", 125.0, 0.0),
                ("2026-01-05", "E", "Lowers", 90.0, 100.0)])


# ----------------------------- montagem -------------------------------------
def test_montar_calcula_upsides_e_rotulos():
    a = montar_analistas(INFO, ALVOS, RESUMO, REV)
    assert a.media == 120.0 and a.n == 30 and a.recomendacao == "Compra" and a.nota == pytest.approx(2.1)
    assert a.upside_media == pytest.approx(0.20) and a.upside_baixo == pytest.approx(-0.20)
    assert a.upside_alto == pytest.approx(0.50) and a.upside_mediana == pytest.approx(0.18)
    assert a.dispersao == pytest.approx((150 - 80) / 120)
    assert a.distribuicao["Compra forte"] == 6 and a.distribuicao_anterior["Compra forte"] == 5
    assert a.pct_compra == pytest.approx(26 / 45)


def test_sem_nenhum_alvo_devolve_none():
    assert montar_analistas({"currentPrice": 10.0}, {}, None, None) is None
    assert montar_analistas({}, None, pd.DataFrame(), pd.DataFrame()) is None
    assert montar_analistas({"targetMeanPrice": float("nan"), "targetHighPrice": 0}, {}, None, None) is None


def test_usa_info_quando_alvos_vazios_e_ignora_lixo():
    a = montar_analistas({"currentPrice": "100", "targetMeanPrice": 110, "numberOfAnalystOpinions": None,
                          "recommendationKey": "none"}, {}, None, None)
    assert a.media == 110.0 and a.preco_atual == 100.0 and a.n is None
    assert a.recomendacao == "Sem recomendação" and a.distribuicao == {} and a.revisoes.empty
    assert a.upside_alto is None and a.dispersao is None and a.pct_compra is None


def test_revisoes_ordenadas_e_alvo_anterior_zero_vira_none():
    a = montar_analistas(INFO, ALVOS, RESUMO, REV)
    assert list(a.revisoes["casa"]) == ["A", "B", "C", "D", "E"]                  # mais recente primeiro
    assert a.revisoes["data"].is_monotonic_decreasing
    assert pd.isna(a.revisoes.loc[a.revisoes["casa"] == "D", "alvo_anterior"].iloc[0])   # 0,0 = sem alvo anterior
    assert len(montar_analistas(INFO, ALVOS, RESUMO, REV, max_revisoes=2).revisoes) == 2


# ----------------------------- leitura --------------------------------------
def test_tendencia_de_alvos_conta_so_a_janela():
    t = montar_analistas(INFO, ALVOS, RESUMO, REV).tendencia_alvos(dias=90)
    assert t == {"subiu": 2, "baixou": 1}                                         # a revisão de janeiro fica de fora
    assert montar_analistas(INFO, ALVOS, RESUMO, None).tendencia_alvos() == {"subiu": 0, "baixou": 0}


def test_leitura_descreve_consenso():
    base = montar_analistas(INFO, ALVOS, RESUMO, REV)
    assert "20% acima" in base.leitura and "revisados para cima" in base.leitura
    acima = montar_analistas({**INFO, "currentPrice": 150.0}, {**ALVOS, "current": 150.0}, RESUMO, None)
    assert "acima do alvo médio" in acima.leitura
    perto = montar_analistas({**INFO, "currentPrice": 119.0}, {**ALVOS, "current": 119.0}, RESUMO, None)
    assert "perto do alvo médio" in perto.leitura
    dividido = montar_analistas({**INFO, "targetHighPrice": 300.0}, {**ALVOS, "high": 300.0}, RESUMO, None)
    assert "muito divergentes" in dividido.leitura


# ----------------------------- gráficos e cartão ----------------------------
def test_grafico_alvos_e_recomendacoes():
    a = montar_analistas(INFO, ALVOS, RESUMO, REV)
    nomes = {t.name for t in grafico_alvos(a).data}
    assert {"Mínimo", "Média", "Mediana", "Máximo", "Preço atual"} <= nomes
    fig = grafico_recomendacoes(a)
    assert len(fig.data) == 5 and len(fig.data[0].y) == 2                         # hoje e 3 meses antes
    assert grafico_recomendacoes(montar_analistas(INFO, ALVOS, None, None)).data[0].y == ("Hoje",)


def _df_precos():
    idx = pd.bdate_range("2024-01-01", periods=300)
    c = pd.Series(100 + np.arange(300) * 0.2, index=idx)
    return calcular_indicadores(pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": 1e6}))


def test_linhas_de_alvo_no_grafico_principal():
    df = _df_precos()
    alvos = {"baixo": 80.0, "media": 120.0, "alto": 150.0}
    com = grafico_principal(df, "t", OpcoesGrafico(alvos_analistas=True, fibonacci=False, suporte_resistencia=False), alvos=alvos)
    sem = grafico_principal(df, "t", OpcoesGrafico(alvos_analistas=False, fibonacci=False, suporte_resistencia=False), alvos=alvos)
    textos = [an.text for an in com.layout.annotations]
    assert sum("Alvo analistas" in t for t in textos) == 3
    assert not any("Alvo analistas" in (an.text or "") for an in sem.layout.annotations)   # desligado: nenhuma linha


def test_cartao_mostra_faixa_de_analistas_e_some_sem_cobertura():
    a = montar_analistas(INFO, ALVOS, RESUMO, REV)
    d = calcular_decisao(_df_precos())
    html = cartao_decisao(d, "X", "X", 100.0, 0.01, "hoje", analistas=a)
    assert "Alvo dos analistas" in html and "120.00" in html and "+20.0%" in html and "30 analistas" in html
    assert "faixa 80.00–150.00" in html and "recomendação: Compra" in html
    assert "Alvo dos analistas" not in cartao_decisao(d, "X", "X", 100.0, 0.01, "hoje")
    assert linha_analistas(None) == ""
    assert "color:#dc2626" in linha_analistas(montar_analistas({**INFO, "currentPrice": 150.0}, {**ALVOS, "current": 150.0}, None, None))
