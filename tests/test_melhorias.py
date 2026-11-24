import numpy as np
import pandas as pd
import pytest

from analise.backtest import (
    ESTRATEGIAS, aplicar_stop_atr, executar_backtest, executar_treino_teste,
)
from analise.diagnostico import diagnosticar
from analise.indicators import calcular_fibonacci_perna, calcular_indicadores
from analise.regime import (
    checklist_tendencia, dimensionar_posicao, estado_defensivo, momentum_12_1, serie_tendencia,
)


def ohlc(close, start="2020-01-01"):
    close = pd.Series(close, dtype=float)
    idx = pd.bdate_range(start, periods=len(close))
    return pd.DataFrame({"Open": close.values, "High": close.values * 1.005,
                         "Low": close.values * 0.995, "Close": close.values,
                         "Volume": 1000.0}, index=idx)


# ----------------------------- regime ---------------------------------------
def test_defensivo_alta_investido_queda_caixa():
    alta = pd.Series(np.linspace(100, 300, 400), index=pd.bdate_range("2020-01-01", periods=400))
    assert bool(estado_defensivo(alta).iloc[-1]) is True
    assert bool(estado_defensivo(alta.iloc[::-1].set_axis(alta.index)).iloc[-1]) is False
    assert not estado_defensivo(alta).iloc[:199].any()  # sem MA200, nunca investido


def test_defensivo_histerese_mantem_estado_dentro_da_banda():
    base = list(np.linspace(100, 200, 300))
    ma = np.mean((base + [201])[-200:])
    idx = pd.bdate_range("2020-01-01", periods=301)
    assert bool(estado_defensivo(pd.Series(base + [201], index=idx)).iloc[-1]) is True
    dentro = pd.Series(base + [ma * 0.99], index=idx)      # −1%: dentro da banda de ±2%
    assert bool(estado_defensivo(dentro).iloc[-1]) is True
    fora = pd.Series(base + [ma * 0.90], index=idx)        # −10%: abaixo da banda
    assert bool(estado_defensivo(fora).iloc[-1]) is False


def test_defensivo_e_causal():
    close = pd.Series(100 + np.random.default_rng(0).normal(0, 1, 500).cumsum(),
                      index=pd.bdate_range("2020-01-01", periods=500))
    pd.testing.assert_series_equal(estado_defensivo(close).iloc[:400], estado_defensivo(close.iloc[:400]))


def test_tendencia_alta_forte_atende_criterios():
    close = pd.Series(np.linspace(50, 400, 400), index=pd.bdate_range("2020-01-01", periods=400))
    assert checklist_tendencia(close.iloc[:100]) is None
    assert all(c.ok for c in checklist_tendencia(close))
    assert serie_tendencia(close).iloc[-1] == 1
    assert serie_tendencia(close).iloc[:250].sum() == 0  # precisa de 252 candles


def test_tendencia_queda_nao_atende():
    close = pd.Series(np.linspace(400, 50, 400), index=pd.bdate_range("2020-01-01", periods=400))
    assert serie_tendencia(close).iloc[-1] == 0


def test_momentum_12_1():
    close = pd.Series(np.linspace(100, 200, 300))
    assert momentum_12_1(close) == pytest.approx(close.iloc[-22] / close.iloc[-253] - 1)
    assert np.isnan(momentum_12_1(close.iloc[:100]))


def test_dimensionar_posicao_respeita_risco_e_teto():
    p = dimensionar_posicao(100_000, 100.0, 2.0)          # stop a 4 → 1% = 1000 → 250 un.; teto 20% = 200 un.
    assert p.quantidade == 200 and p.stop == 96.0 and p.alvo == 112.0
    assert p.pct_capital == pytest.approx(0.20)
    p2 = dimensionar_posicao(100_000, 100.0, 10.0)        # stop a 20 → 1% = 1000 → 50 un.
    assert p2.quantidade == 50 and p2.risco_valor == pytest.approx(1000.0)
    assert dimensionar_posicao(100, 1000.0, 2.0) is None   # não cabe 1 unidade
    assert dimensionar_posicao(100_000, 100.0, float("nan")) is None


# ----------------------------- stop e backtest ------------------------------
def _serie_com_queda():
    return np.concatenate([np.linspace(100, 150, 60), np.linspace(150, 100, 30), np.linspace(100, 160, 60)])


def test_stop_atr_sai_na_queda_e_so_reentra_com_novo_sinal():
    df = calcular_indicadores(ohlc(_serie_com_queda()))
    sinal = pd.Series(1, index=df.index)                    # sinal sempre comprado
    saida = aplicar_stop_atr(df, sinal, mult=2.0)
    assert saida.iloc[10] == 1
    assert saida.iloc[80] == 0                               # stopado durante a queda
    assert saida.iloc[-1] == 0                               # bloqueado: sinal nunca passou por 0
    sinal2 = sinal.copy()
    sinal2.iloc[85:90] = 0                                   # sinal zera e volta
    assert aplicar_stop_atr(df, sinal2, mult=2.0).iloc[-1] == 1


def test_stop_atr_e_causal():
    df = calcular_indicadores(ohlc(_serie_com_queda()))
    sinal = pd.Series(1, index=df.index)
    cheio = aplicar_stop_atr(df, sinal, 2.0)
    parcial = aplicar_stop_atr(df.iloc[:100], sinal.iloc[:100], 2.0)
    pd.testing.assert_series_equal(cheio.iloc[:100], parcial)


def test_backtest_traz_cagr_e_sharpe():
    df = calcular_indicadores(ohlc(np.linspace(100, 200, 500)))
    res = executar_backtest(df, pd.Series(1, index=df.index))
    assert res.cagr > 0 and res.sharpe > 0
    assert res.cagr_buy_hold == pytest.approx(res.cagr, rel=0.05)
    assert res.max_drawdown_buy_hold == pytest.approx(0.0, abs=1e-9)


def test_stop_reduz_perda_em_queda():
    df = calcular_indicadores(ohlc(np.concatenate([np.linspace(100, 150, 50), np.linspace(150, 60, 100)])))
    sinal = pd.Series(1, index=df.index)
    sem = executar_backtest(df, sinal)
    com = executar_backtest(df, sinal, stop_atr=2.0)
    assert com.retorno_estrategia > sem.retorno_estrategia
    assert com.max_drawdown > sem.max_drawdown   # drawdown (negativo) menos profundo


def test_treino_teste_divide_serie():
    df = calcular_indicadores(ohlc(np.linspace(100, 200, 500)))
    r = executar_treino_teste(df, pd.Series(1, index=df.index), fracao_treino=0.6)
    assert set(r) == {"treino", "teste"}
    assert len(r["treino"].curva_estrategia) == 300 and len(r["teste"].curva_estrategia) == 200
    assert executar_treino_teste(df.iloc[:40], pd.Series(1, index=df.index[:40])) == {}  # curto demais


def test_novas_estrategias_rodam():
    df = calcular_indicadores(ohlc(np.linspace(50, 400, 400)))
    for nome in ("Núcleo defensivo: MA200 com banda 2%", "Modelo de tendência (Minervini)"):
        pos = ESTRATEGIAS[nome](df)
        assert set(pos.unique()) <= {0, 1} and pos.iloc[-1] == 1
        assert executar_backtest(df, pos, 0.001).n_trades >= 1


# ----------------------------- Fibonacci perna ------------------------------
def _perna(retracao_final):
    sobe = np.linspace(100, 200, 60)
    return np.concatenate([np.full(30, 100.0), sobe, np.linspace(200, 200 - retracao_final * 100, 20)])


def test_fibonacci_perna_zona_de_ouro_e_alvos():
    df = ohlc(_perna(0.5))
    f = calcular_fibonacci_perna(df)
    topo, fundo = df["High"].max(), df["Low"].iloc[:df["High"].to_numpy().argmax()].min()
    assert f.tendencia == "alta" and f.na_zona_ouro
    assert f.retracao_atual == pytest.approx((topo - df["Close"].iloc[-1]) / (topo - fundo))
    assert f.extensoes["127.2%"] == pytest.approx(fundo + 1.272 * (topo - fundo))
    assert not calcular_fibonacci_perna(ohlc(_perna(0.1))).na_zona_ouro   # recuo raso: fora da zona


def test_fibonacci_perna_sem_perna_relevante():
    assert calcular_fibonacci_perna(ohlc(np.full(100, 50.0))) is None
    assert calcular_fibonacci_perna(ohlc(np.linspace(200, 100, 100))) is None  # só queda: topo no início
    assert calcular_fibonacci_perna(ohlc([1, 2, 3])) is None


def test_fibonacci_perna_nao_olha_o_futuro():
    df = ohlc(np.concatenate([_perna(0.5), np.linspace(150, 400, 80)]))
    n = 110
    antes = calcular_fibonacci_perna(df.iloc[:n])
    df_futuro_alterado = df.copy()
    df_futuro_alterado.iloc[n:] *= 3                       # futuro diferente não pode mudar o passado
    assert calcular_fibonacci_perna(df_futuro_alterado.iloc[:n]).niveis == antes.niveis


# ----------------------------- diagnóstico ----------------------------------
def test_diagnostico_inclui_itens_estrategicos():
    df = calcular_indicadores(ohlc(np.linspace(50, 400, 400)))
    titulos = {i.titulo: i for i in diagnosticar(df)}
    for t in ("Regime (núcleo defensivo)", "Modelo de tendência", "Momentum 12-1", "Fibonacci (perna recente)"):
        assert t in titulos
    assert titulos["Regime (núcleo defensivo)"].tom == "alta"
    assert titulos["Modelo de tendência"].tom == "alta"


def test_diagnostico_historico_curto_nao_quebra():
    df = calcular_indicadores(ohlc(np.linspace(10, 12, 40)))
    titulos = {i.titulo: i for i in diagnosticar(df)}
    assert "Precisa de 200 candles" in titulos["Regime (núcleo defensivo)"].texto


def test_dimensionar_posicao_fracionario_para_cripto():
    p = dimensionar_posicao(100_000, 60_000.0, 3_000.0, fracionario=True)   # stop a 6000 → 1000/6000 BTC
    assert p.quantidade == pytest.approx(1000 / 6000, abs=1e-6)
    assert p.risco_valor == pytest.approx(1000.0, rel=1e-5)
    assert dimensionar_posicao(100_000, 60_000.0, 3_000.0) is None          # inteiro: não cabe 1 BTC
