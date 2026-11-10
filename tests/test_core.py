import numpy as np
import pandas as pd
import pytest

from analise.backtest import ESTRATEGIAS, executar_backtest
from analise.config import compatibilizar_periodo_intervalo, periodo_em_dias
from analise.data import normalizar_base100, normalizar_colunas, resample_ohlcv
from analise.diagnostico import detectar_eventos, diagnosticar
from analise.indicators import (
    calcular_fibonacci, calcular_indicadores, encontrar_pivots, rsi_wilder, suporte_resistencia,
)
from analise.sources import montar_links_cruzados


def ohlc(close, freq="D", start="2024-01-01"):
    close = pd.Series(close, dtype=float)
    idx = pd.date_range(start, periods=len(close), freq=freq)
    return pd.DataFrame({"Open": close.values, "High": close.values * 1.01,
                         "Low": close.values * 0.99, "Close": close.values,
                         "Volume": 1000.0}, index=idx)


# ----------------------------- config --------------------------------------
def test_periodo_em_dias():
    assert periodo_em_dias("7d") == 7
    assert periodo_em_dias("3mo") == 90
    assert periodo_em_dias("2y") == 730


def test_15m_limitado_a_60_dias():
    p = compatibilizar_periodo_intervalo("1y", "15m", "15 minutos")
    assert p.periodo == "1mo" and p.aviso and p.intervalo_real == "15m"


def test_1h_aceita_ate_700_dias():
    assert compatibilizar_periodo_intervalo("1y", "1h", "1 hora").aviso is None
    p = compatibilizar_periodo_intervalo("5y", "1h", "1 hora")
    assert p.periodo == "1y" and p.aviso


def test_4h_usa_1h_com_resample():
    p = compatibilizar_periodo_intervalo("6mo", "4h", "4 horas")
    assert p.resample_4h and p.intervalo_real == "1h" and p.aviso is None


def test_diario_sem_limite():
    assert compatibilizar_periodo_intervalo("10y", "1d", "1 dia").aviso is None


# ----------------------------- RSI -----------------------------------------
def test_rsi_aquecimento_nao_e_100():
    rsi = rsi_wilder(pd.Series(np.arange(1, 11, dtype=float) + np.sin(np.arange(10))))
    assert rsi.isna().all()  # menos de 14 variações


def test_rsi_so_alta_e_100():
    rsi = rsi_wilder(pd.Series(np.arange(1, 40, dtype=float)))
    assert rsi.iloc[-1] == 100.0
    assert rsi.iloc[:14].isna().all()


def test_rsi_preco_parado_e_50():
    assert rsi_wilder(pd.Series([10.0] * 40)).iloc[-1] == 50.0


def test_rsi_so_queda_e_0():
    assert rsi_wilder(pd.Series(np.arange(40, 0, -1, dtype=float))).iloc[-1] == pytest.approx(0.0)


def test_rsi_limites():
    rng = np.random.default_rng(1)
    rsi = rsi_wilder(pd.Series(100 + rng.normal(0, 1, 300).cumsum())).dropna()
    assert ((rsi >= 0) & (rsi <= 100)).all()


# ----------------------------- indicadores ---------------------------------
def test_emas_respeitam_min_periods():
    df = calcular_indicadores(ohlc(np.linspace(10, 20, 100)))
    assert df["EMA305"].isna().all()
    assert df["EMA72"].notna().sum() == 100 - 71
    assert df["MA200"].isna().all()
    assert df["MA21"].notna().sum() == 100 - 20


def test_colunas_geradas():
    df = calcular_indicadores(ohlc(np.linspace(10, 20, 400)))
    for c in ["RSI", "MACD", "Signal", "BB_Sup", "ATR", "Stoch_K", "Volume_Financeiro", "VWAP"]:
        assert c in df.columns
    assert df["VWAP"].isna().all()  # só intraday
    assert df["EMA305"].notna().iloc[-1]


def test_vwap_intraday():
    df = calcular_indicadores(ohlc(np.linspace(10, 12, 50), freq="h"), intraday=True)
    assert df["VWAP"].notna().all()
    assert df["VWAP"].iloc[0] == pytest.approx((df["High"].iloc[0] + df["Low"].iloc[0] + df["Close"].iloc[0]) / 3)


# ----------------------------- Fibonacci -----------------------------------
def test_fibonacci_alta():
    close = [10, 12, 15, 20]
    df = ohlc(close)
    f = calcular_fibonacci(df)
    assert f.tendencia == "alta"
    topo, fundo = df["High"].max(), df["Low"].min()
    assert f.niveis["0.0%"] == pytest.approx(topo)
    assert f.niveis["100.0%"] == pytest.approx(fundo)
    assert f.niveis["50.0%"] == pytest.approx((topo + fundo) / 2)


def test_fibonacci_baixa_inverte_direcao():
    df = ohlc([20, 15, 12, 10])
    f = calcular_fibonacci(df)
    assert f.tendencia == "baixa"
    assert f.niveis["0.0%"] == pytest.approx(df["Low"].min())
    assert f.niveis["100.0%"] == pytest.approx(df["High"].max())


def test_fibonacci_vazio():
    assert calcular_fibonacci(pd.DataFrame()) is None


# ----------------------------- pivôs / S&R ---------------------------------
def test_pivots_e_suporte_resistencia():
    close = ([10, 11, 12, 13, 14, 15, 14, 13, 12, 11, 10, 11, 12, 13, 14, 15, 14, 13, 12, 11, 10] * 2)
    df = ohlc(close)
    topos = encontrar_pivots(df["High"], 3, "topo")
    assert len(topos) >= 2
    sr = suporte_resistencia(df.iloc[:-3], janela=3)
    assert sr["resistencias"] and sr["suportes"]
    assert all(n < df["Close"].iloc[-4] for n, _ in sr["suportes"])


# ----------------------------- dados ---------------------------------------
def test_normalizar_colunas_multiindex():
    df = ohlc([1, 2, 3])
    df.columns = pd.MultiIndex.from_product([df.columns, ["X"]])
    out = normalizar_colunas(df)
    assert "Close" in out.columns and not isinstance(out.columns, pd.MultiIndex)


def test_normalizar_colunas_vazio():
    assert normalizar_colunas(pd.DataFrame()).empty


def test_resample_4h_ohlcv():
    idx = pd.date_range("2024-01-02 10:00", periods=8, freq="h", tz="America/Sao_Paulo")
    df = pd.DataFrame({"Open": range(8), "High": range(1, 9), "Low": range(8),
                       "Close": range(1, 9), "Volume": 10.0}, index=idx)
    out = resample_ohlcv(df, "4h")
    assert len(out) == 2
    assert out.index[0].hour == 10 and out.index[1].hour == 14  # alinhado ao início do pregão
    assert out["Open"].iloc[0] == 0 and out["Close"].iloc[0] == 4
    assert out["High"].iloc[0] == 4 and out["Volume"].iloc[0] == 40


def test_resample_cripto_usa_grade_utc():
    idx = pd.date_range("2024-01-01 01:00", periods=8, freq="h", tz="UTC")
    df = pd.DataFrame({"Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.5, "Volume": 1.0}, index=idx)
    assert list(resample_ohlcv(df, "4h", cripto=True).index.hour) == [0, 4, 8]


def test_base100():
    s = normalizar_base100(pd.Series([50.0, 100.0, 75.0]))
    assert list(s) == [100.0, 200.0, 150.0]


# ----------------------------- diagnóstico / backtest ----------------------
def _df_longo(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return calcular_indicadores(ohlc(100 + rng.normal(0.1, 1, n).cumsum()))


def test_diagnostico_curto_nao_quebra_nem_diz_sobrecompra():
    itens = diagnosticar(calcular_indicadores(ohlc([10, 11, 12, 13, 14])))
    rsi = next(i for i in itens if i.titulo.startswith("RSI"))
    assert "Indisponível" in rsi.texto


def test_diagnostico_completo():
    itens = diagnosticar(_df_longo())
    assert {i.titulo for i in itens} >= {"RSI (14)", "MACD", "Tendência (MA200)", "EMAs PhiCube", "Volume"}
    assert all(i.tom in {"alta", "baixa", "neutro"} for i in itens)


def test_eventos_ordenados():
    ev = detectar_eventos(_df_longo())
    assert not ev.empty
    assert ev["data"].is_monotonic_decreasing


def test_backtest_sem_lookahead_e_buy_hold():
    df = _df_longo()
    res = executar_backtest(df, pd.Series(1, index=df.index))  # sempre comprado
    # Posição aplicada no candle seguinte: o 1º retorno não conta; resto = buy & hold.
    esperado = df["Close"].iloc[-1] / df["Close"].iloc[0] - 1
    assert res.retorno_buy_hold == pytest.approx(esperado)
    assert res.retorno_estrategia == pytest.approx(esperado, abs=1e-9)
    assert res.n_trades == 1


def test_backtest_nunca_comprado():
    df = _df_longo()
    res = executar_backtest(df, pd.Series(0, index=df.index))
    assert res.retorno_estrategia == 0 and res.n_trades == 0 and res.taxa_acerto is None


def test_backtest_custo_reduz_retorno():
    df = _df_longo()
    pos = ESTRATEGIAS["Cruzamento MACD × Sinal"](df)
    sem = executar_backtest(df, pos, 0.0).retorno_estrategia
    com = executar_backtest(df, pos, 0.01).retorno_estrategia
    assert com < sem


@pytest.mark.parametrize("nome", list(ESTRATEGIAS))
def test_estrategias_retornam_0_ou_1(nome):
    pos = ESTRATEGIAS[nome](_df_longo())
    assert set(pos.unique()) <= {0, 1}


# ----------------------------- fontes --------------------------------------
def test_links_por_tipo():
    assert "Dados de criptoativos" in montar_links_cruzados("BTC-USD")
    br = montar_links_cruzados("PETR4.SA")
    assert any("statusinvest.com.br/acoes/petr4" in u for _, u in br["Fundamentos e regulação (Brasil)"])
    assert "Fundamentos e regulação (EUA/outros)" in montar_links_cruzados("aapl")
