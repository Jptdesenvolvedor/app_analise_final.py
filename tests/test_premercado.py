import numpy as np
import pandas as pd
import pytest

from analise import premercado as pm
from analise.apresentacao import cartao_decisao, faixa_sessao
from analise.charts import grafico_sessoes
from analise.indicators import calcular_indicadores
from analise.sinal import calcular_decisao


# ----------------------------- suporte e estado -----------------------------
def test_suporta_so_acoes_dos_eua():
    assert pm.suporta("AAPL") and pm.suporta("spy") and pm.suporta("NVDA")
    for t in ("PETR4.SA", "BTC-USD", "BRK.B", "^GSPC", "EURUSD=X"):
        assert not pm.suporta(t), t


def test_sessao_pre_mercado_usa_campos_do_pre():
    s = pm.montar_sessao({"marketState": "PRE", "regularMarketPrice": 100.0, "regularMarketPreviousClose": 99.0,
                          "preMarketPrice": 101.0, "preMarketChangePercent": 1.0, "preMarketTime": 1795600000})
    assert s.estado == "PRE" and s.rotulo == "Pré-mercado" and s.tem_estendido
    assert s.preco_estendido == 101.0 and s.var_estendida == pytest.approx(0.01)   # 1,0 ponto percentual = 1%
    assert s.referencia == "fechamento de ontem" and str(s.horario.tz) == "America/Sao_Paulo"


def test_sessao_after_hours_e_percentual_em_pontos():
    s = pm.montar_sessao({"marketState": "POST", "regularMarketPrice": 354.07, "postMarketPrice": 353.8885,
                          "postMarketChangePercent": -0.0512663, "postMarketTime": 1795558233})
    assert s.estado == "POST" and s.var_estendida == pytest.approx(-0.000512663)
    assert s.referencia == "fechamento de hoje"


def test_variacao_calculada_quando_o_yahoo_nao_informa():
    s = pm.montar_sessao({"marketState": "PRE", "regularMarketPrice": 200.0, "preMarketPrice": 210.0})
    assert s.var_estendida == pytest.approx(0.05)


def test_mercado_aberto_fechado_e_sem_dados():
    assert pm.montar_sessao({"marketState": "REGULAR", "regularMarketPrice": 10.0}).estado == "REGULAR"
    assert not pm.montar_sessao({"marketState": "REGULAR", "regularMarketPrice": 10.0, "preMarketPrice": 11}).tem_estendido
    for estado in ("CLOSED", "POSTPOST", "", None):
        assert pm.montar_sessao({"marketState": estado}).estado == "FECHADO"
    vazio = pm.montar_sessao({})
    assert vazio.estado == "FECHADO" and not vazio.tem_estendido and vazio.horario is None


def test_pre_mercado_sem_preco_nao_inventa_estendido():
    s = pm.montar_sessao({"marketState": "PRE", "regularMarketPrice": 100.0})
    assert s.estado == "PRE" and not s.tem_estendido


# ----------------------------- sessões e histórico --------------------------
def candles(dias, gaps, retornos, base=100.0):
    """Candles de 5 min (UTC, como o Yahoo) com pré, regular e pós. `gaps` e `retornos` em fração por dia."""
    linhas, fech_ant = [], base
    for dia, gap, ret in zip(dias, gaps, retornos):
        ny = pd.date_range(f"{dia} 04:00", f"{dia} 19:55", freq="5min", tz="America/New_York")
        for t in ny:
            minutos = t.hour * 60 + t.minute
            if minutos < 570:                                   # pré: termina em fech_ant × (1+gap)
                preco = fech_ant * (1 + gap)
            elif minutos < 960:                                 # regular: sobe linearmente até fech_ant × (1+ret)
                f = (minutos - 570) / 385
                preco = fech_ant * (1 + gap) + f * (fech_ant * (1 + ret) - fech_ant * (1 + gap))
            else:
                preco = fech_ant * (1 + ret)
            linhas.append((t.tz_convert("UTC"), preco, preco, preco, preco, 1000))
        fech_ant = fech_ant * (1 + ret)
    df = pd.DataFrame(linhas, columns=["Datetime", "Open", "High", "Low", "Close", "Volume"]).set_index("Datetime")
    return df


DIAS = ["2026-11-16", "2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20"]


def test_classifica_sessoes_nos_horarios_de_nova_york():
    h = pm.classificar_sessoes(candles(DIAS[:1], [0.0], [0.0]))
    assert str(h.index.tz) == "America/New_York"
    g = h.groupby("sessao").size()
    assert g["PRE"] == 66 and g["REGULAR"] == 78 and g["POST"] == 48          # 4:00–9:30, 9:30–16:00, 16:00–20:00
    assert h[h["sessao"] == "REGULAR"].index.min().strftime("%H:%M") == "09:30"
    assert h[h["sessao"] == "POST"].index.min().strftime("%H:%M") == "16:00"


def test_classifica_aceita_indice_sem_fuso_como_utc():
    c = candles(DIAS[:1], [0.0], [0.0])
    c.index = c.index.tz_localize(None)
    assert str(pm.classificar_sessoes(c).index.tz) == "America/New_York"


def test_historico_gap_abertura_dia_e_acerto():
    h = pm.historico_premercado(candles(DIAS, [0.0, 0.01, -0.01, 0.02, 0.005], [0.0, 0.02, 0.01, 0.03, -0.01]))
    assert len(h) == 4                                          # o 1º dia só serve de fechamento anterior
    assert h["data"].is_monotonic_decreasing
    d18 = h[h["data"] == "2026-11-18"].iloc[0]                  # gap −1%, dia +1% → pré errou
    assert d18["gap_pre"] == pytest.approx(-0.01) and d18["dia"] == pytest.approx(0.01) and not d18["acertou"]
    d17 = h[h["data"] == "2026-11-17"].iloc[0]                  # gap +1%, dia +2% → acertou
    assert d17["gap_pre"] == pytest.approx(0.01) and d17["dia"] == pytest.approx(0.02) and d17["acertou"]
    assert d17["gap_aber"] == pytest.approx(0.01, abs=1e-3)     # a abertura coincide com o fim do pré
    d20 = h[h["data"] == "2026-11-20"].iloc[0]                  # gap +0,5%, dia −1% → errou
    assert not d20["acertou"]


def test_taxa_de_acerto_e_filtro_de_gap():
    h = pm.historico_premercado(candles(DIAS, [0.0, 0.01, -0.01, 0.02, 0.005], [0.0, 0.02, 0.01, 0.03, -0.01]))
    taxa, n = pm.taxa_de_acerto(h)
    assert n == 4 and taxa == pytest.approx(2 / 4)              # acertou 17 e 19; errou 18 e 20
    taxa2, n2 = pm.taxa_de_acerto(h, gap_minimo=0.0075)         # só |gap| ≥ 0,75%: dias 17, 18 e 19
    assert n2 == 3 and taxa2 == pytest.approx(2 / 3)
    assert pm.taxa_de_acerto(h, gap_minimo=0.5) == (None, 0)


def test_historico_sem_pre_mercado_ou_vazio():
    c = candles(DIAS[:2], [0.0, 0.0], [0.0, 0.0])
    so_regular = pm.classificar_sessoes(c)
    so_regular = so_regular[so_regular["sessao"] == "REGULAR"].drop(columns="sessao")
    assert pm.historico_premercado(so_regular).empty
    assert pm.historico_premercado(c.iloc[0:0]).empty if len(c) else True


def test_ultimas_sessoes_pega_so_os_n_ultimos_dias():
    u = pm.ultimas_sessoes(candles(DIAS, [0.0] * 5, [0.0] * 5), n=2)
    assert sorted(set(u.index.date.astype(str))) == ["2026-11-19", "2026-11-20"] and "sessao" in u


# ----------------------------- gráfico e cartão -----------------------------
def test_grafico_sessoes_cores_legenda_e_linha_de_referencia():
    u = pm.ultimas_sessoes(candles(DIAS, [0.01] * 5, [0.0] * 5), n=2)
    fig = grafico_sessoes(u, fechamento_anterior=100.0)
    assert {t.name for t in fig.data} == {"Pré-mercado", "Sessão regular", "After-hours"}
    assert sum(bool(t.showlegend) for t in fig.data) == 3                   # cada sessão aparece 1× na legenda
    assert len(fig.data) == 6                                              # 2 dias × 3 sessões, sem ligar um dia ao outro
    assert any("Fechamento anterior" in (a.text or "") for a in fig.layout.annotations)
    assert len(grafico_sessoes(pd.DataFrame()).data) == 0


def _decisao():
    idx = pd.bdate_range("2024-01-01", periods=300)
    c = pd.Series(100 + np.arange(300) * 0.2, index=idx)
    return calcular_decisao(calcular_indicadores(pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c, "Volume": 1e6})))


def test_faixa_e_cartao_mostram_pre_mercado():
    s = pm.montar_sessao({"marketState": "PRE", "regularMarketPrice": 100.0, "preMarketPrice": 102.0,
                          "preMarketChangePercent": 2.0, "preMarketTime": 1795600000})
    html = faixa_sessao(s)
    assert "Pré-mercado" in html and "102.00" in html and "+2.00%" in html and "fechamento de ontem" in html
    assert "(Brasília)" in html and "#16a34a" in html
    cartao = cartao_decisao(_decisao(), "X", "X", 100.0, 0.01, "hoje", sessao=s)
    assert 'class="rt-sessao"' in cartao and "Pré-mercado" in cartao
    assert 'class="rt-sessao"' not in cartao_decisao(_decisao(), "X", "X", 100.0, 0.01, "hoje")   # o CSS sempre existe
    assert faixa_sessao(None) == ""


def test_faixa_mercado_fechado_sem_preco_estendido_e_queda_em_vermelho():
    assert "Mercado fechado" in faixa_sessao(pm.montar_sessao({"marketState": "CLOSED"}))
    assert "$" not in faixa_sessao(pm.montar_sessao({"marketState": "CLOSED"}))
    queda = pm.montar_sessao({"marketState": "POST", "regularMarketPrice": 100.0, "postMarketPrice": 98.0,
                              "postMarketChangePercent": -2.0})
    assert "color:#dc2626" in faixa_sessao(queda) and "After-hours" in faixa_sessao(queda)


def test_fechamento_anterior_do_ultimo_dia():
    c = candles(DIAS[:3], [0.0, 0.01, 0.0], [0.02, 0.01, 0.0], base=100.0)
    assert pm.fechamento_anterior_do_ultimo_dia(c) == pytest.approx(100 * 1.02 * 1.01, rel=1e-6)   # fecha do dia 17
    assert pm.fechamento_anterior_do_ultimo_dia(candles(DIAS[:1], [0.0], [0.0])) is None          # só um dia
