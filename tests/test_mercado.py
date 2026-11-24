import numpy as np
import pandas as pd
import pytest

from analise import mercado as m


def q(sym, nome, cap, vol=1e6, tipo="EQUITY", bolsa="NMS"):
    return {"symbol": sym, "shortName": nome, "marketCap": cap, "averageDailyVolume3Month": vol,
            "quoteType": tipo, "exchange": bolsa}


QUOTES = [
    q("AAPL", "Apple Inc.", 5e12), q("GOOGL", "Alphabet Inc.", 3.4e12, vol=30e6), q("GOOG", "Alphabet Inc.", 3.39e12, vol=20e6),
    q("BRK-A", "Berkshire Hathaway Inc.", 1.2e12, vol=5e3), q("BRK-B", "Berkshire Hathaway Inc.", 1.2e12, vol=5e6),
    q("JPM-PC", "JPMorgan Chase & Co.", 1.1e12), q("JPM", "JPMorgan Chase & Co.", 0.8e12),
    q("SPYX", "Algum ETF", 9e12, tipo="ETF"), q("SEMCAP", "Sem Valor", None), q("ZERO", "Zero", 0), q("NAN", "Nan", float("nan")),
    q("BAC", "Bank of America Corporation", 0.35e12), q("BK", "Bank of New York Mellon Corporation", 0.07e12),
]


# ----------------------------- seleção --------------------------------------
def test_chave_empresa_junta_classes_e_nao_confunde_empresas():
    assert m.chave_empresa("Alphabet Inc.") == m.chave_empresa("Alphabet Inc.") == "alphabet"
    assert m.chave_empresa("Berkshire Hathaway Inc. New") == "berkshire hathaway"
    assert m.chave_empresa("Alphabet Inc. Class A") == m.chave_empresa("Alphabet Inc. Class C")
    assert m.chave_empresa("Bank of America Corporation") != m.chave_empresa("Bank of New York Mellon Corporation")
    assert m.chave_empresa(None) == ""


def test_selecionar_top_dedup_filtra_e_ordena():
    top = m.selecionar_top(QUOTES, n=10)
    tickers = [t["ticker"] for t in top]
    assert tickers == ["AAPL", "GOOGL", "BRK-B", "JPM", "BAC", "BK"]
    assert "GOOG" not in tickers and "BRK-A" not in tickers          # fica a classe mais negociada
    assert "JPM-PC" not in tickers                                   # preferencial não é a empresa
    for lixo in ("SPYX", "SEMCAP", "ZERO", "NAN"):                   # ETF, sem valor de mercado, zero e NaN
        assert lixo not in tickers
    assert [t["cap"] for t in top] == sorted((t["cap"] for t in top), reverse=True)
    assert set(top[0]) == {"ticker", "nome", "cap", "bolsa"}


def test_selecionar_top_respeita_n_e_lista_vazia():
    assert [t["ticker"] for t in m.selecionar_top(QUOTES, n=2)] == ["AAPL", "GOOGL"]
    assert m.selecionar_top([], 50) == [] and m.selecionar_top([{"foo": 1}], 50) == []


def test_filtro_de_nao_ordinarias():
    for s in ("JPM-PC", "BAC-PL", "XYZ-WS", "ABC-U", "ABC-R", "ABC-P"):
        assert m.RE_NAO_ORDINARIA.search(s), s
    for s in ("BRK-B", "BRK-A", "AAPL", "BF-B"):
        assert not m.RE_NAO_ORDINARIA.search(s), s


# ----------------------------- resumo por ação ------------------------------
def ohlcv(n, rumo="alta", seed=1):
    r = np.random.default_rng(seed)
    base = np.linspace(50, 200, n) if rumo == "alta" else np.linspace(200, 50, n)
    c = base * np.exp(np.cumsum(r.normal(0, 0.004, n)) * 0.2)
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": 1e6}, index=idx)


def test_resumo_ticker_alta_e_queda():
    a, b = m.resumo_ticker(ohlcv(400, "alta")), m.resumo_ticker(ohlcv(400, "queda"))
    assert a["acao"] in ("COMPRAR", "MANTER") and a["score"] > 40 and a["tendencia_dir"] == "alta"
    assert b["acao"] in ("VENDER", "REDUZIR") and b["score"] < -40 and b["tendencia_dir"] == "baixa"
    assert a["dist_ma200"] > 0 > b["dist_ma200"]
    df = ohlcv(400, "alta")
    assert a["preco"] == pytest.approx(df["Close"].iloc[-1])
    assert a["var_dia"] == pytest.approx(df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1)
    assert a["var_mes"] == pytest.approx(df["Close"].iloc[-1] / df["Close"].iloc[-22] - 1)
    assert a["var_ano"] == pytest.approx(df["Close"].iloc[-1] / df["Close"].iloc[-253] - 1)


def test_resumo_ticker_historico_curto_ou_vazio():
    assert m.resumo_ticker(ohlcv(20)) is None and m.resumo_ticker(pd.DataFrame()) is None
    novo = m.resumo_ticker(ohlcv(100, "queda"))                       # listagem recente em queda forte
    assert novo["acao"] == "AGUARDAR" and novo["tendencia"] == "Indefinida" and np.isnan(novo["dist_ma200"])
    assert np.isnan(novo["var_ano"])


def test_montar_tabela_ordem_e_ausentes():
    top = [{"ticker": "AAA", "nome": "A", "cap": 3e12, "bolsa": ""}, {"ticker": "BBB", "nome": "B", "cap": 2e12, "bolsa": ""},
           {"ticker": "CCC", "nome": "C", "cap": 1e12, "bolsa": ""}, {"ticker": "DDD", "nome": "D", "cap": 5e11, "bolsa": ""}]
    dados = {"AAA": ohlcv(400, "alta"), "CCC": ohlcv(400, "queda"), "DDD": ohlcv(10)}      # BBB sem dados; DDD curto demais
    t = m.montar_tabela(top, dados)
    assert list(t["ticker"]) == ["AAA", "CCC"] and list(t["posicao"]) == [1, 3]               # posição é a do ranking original
    assert t.loc[0, "cap"] == 3e12 and t.loc[0, "empresa"] == "A"
    assert m.montar_tabela(top, {}).empty


# ----------------------------- amplitude ------------------------------------
def tabela_fake():
    return pd.DataFrame({
        "ticker": list("ABCDEFGH"),
        "acao": ["COMPRAR", "COMPRAR", "MANTER", "AGUARDAR", "REDUZIR", "VENDER", "VENDER", "VENDER"],
        "dist_ma200": [0.2, 0.1, 0.05, 0.0, -0.05, -0.2, -0.1, np.nan],
        "var_dia": [0.03, 0.01, -0.01, 0.0, 0.02, -0.04, -0.02, np.nan]})


def test_amplitude_conta_sinais_e_destaques():
    a = m.amplitude(tabela_fake(), n_destaques=2)
    assert a["n"] == 8 and a["por_sinal"] == {"COMPRAR": 2, "MANTER": 1, "AGUARDAR": 1, "REDUZIR": 1, "VENDER": 3}
    assert a["pct_acima_ma200"] == pytest.approx(3 / 7)                # dist > 0: A, B, C; NaN ignorado; D (=0) não conta
    assert a["pct_alta_dia"] == pytest.approx(3 / 7)                   # A, B, E
    assert [x["ticker"] for x in a["maiores_altas"]] == ["A", "E"]
    assert [x["ticker"] for x in a["maiores_quedas"]] == ["F", "G"]


def test_amplitude_vazia_e_leitura():
    a = m.amplitude(pd.DataFrame())
    assert a["n"] == 0 and a["pct_acima_ma200"] is None and m.leitura_mercado(a) == ""
    assert "bom fôlego" in m.leitura_mercado({"pct_acima_ma200": 0.8})
    assert "dividido" in m.leitura_mercado({"pct_acima_ma200": 0.5})
    assert "fraco" in m.leitura_mercado({"pct_acima_ma200": 0.2})
