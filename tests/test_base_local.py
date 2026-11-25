import numpy as np
import pandas as pd
import pytest

from analise import base_local as bl
from analise.data import DadosIndisponiveis


@pytest.fixture
def base(tmp_path, monkeypatch):
    """Base local falsa: AAPL com candles, notícias, comunicados e balanços; FORA sem arquivos."""
    for d in ("data_sip", "news", "edgar"):
        (tmp_path / d).mkdir()
    monkeypatch.setenv("ANALISE_BASE_LOCAL", str(tmp_path))

    idx = pd.bdate_range("2024-01-01", periods=600)
    close = 100 * np.cumprod(1 + np.random.default_rng(1).normal(0.0004, 0.01, 600))
    pd.DataFrame({"timestamp": idx.strftime("%Y-%m-%d"), "open": close, "high": close * 1.01,
                  "low": close * 0.99, "close": close, "volume": 1e6}).to_csv(tmp_path / "data_sip" / "AAPL.csv", index=False)

    pd.DataFrame({
        "id": [1, 2, 2, 3], "created_at": ["2026-05-01 10:00:00+00:00", "2026-06-01 12:00:00+00:00",
                                           "2026-06-01 12:00:00+00:00", "2026-07-01 09:00:00+00:00"],
        "headline": ["antiga", "do meio", "do meio", "recente"], "summary": ["", "r2", "r2", None],
        "source": ["a", "b", "b", "c"], "symbols": "AAPL",
    }).to_csv(tmp_path / "news" / "AAPL.csv", index=False)

    # 8-K de resultados: um ANTES do fechamento (16h NY) e outro DEPOIS
    pd.DataFrame({
        "form": ["8-K", "8-K", "10-Q", "8-K"],
        "filingDate": ["2025-01-10", "2025-04-15", "2025-04-16", "2025-05-02"],
        "acceptanceDateTime": ["2025-01-10T14:00:00.000Z", "2025-04-15T21:30:00.000Z",
                               "2025-04-16T11:00:00.000Z", "2025-05-02T14:00:00.000Z"],
        "reportDate": ["2025-01-09", "2025-04-15", "2025-03-31", "2025-05-01"],
        "items": ["2.02,9.01", "2.02", np.nan, "5.02,9.01"],
        "accessionNumber": ["a", "b", "c", "d"],
    }).to_csv(tmp_path / "edgar" / "AAPL_filings.csv", index=False)

    linhas = []
    for fim, ini, lpa, rec, filed in (
            ("2024-03-31", "2024-01-01", 1.00, 100.0, "2024-05-01"),
            ("2025-03-31", "2025-01-01", 1.50, 120.0, "2025-05-01"),
            ("2025-03-31", "2025-01-01", 9.99, 999.0, "2025-08-01")):   # reapresentação: deve ser ignorada
        linhas.append(("eps_diluted", fim, ini, lpa, filed, "10-Q"))
        linhas.append(("revenue", fim, ini, rec, filed, "10-Q"))
    linhas.append(("revenue", "2025-12-31", "2025-01-01", 480.0, "2026-02-01", "10-K"))   # anual: não é trimestre
    pd.DataFrame(linhas, columns=["metric", "end", "start", "val", "filed", "form"]).to_csv(
        tmp_path / "edgar" / "AAPL_facts.csv", index=False)

    (tmp_path / "edgar" / "ASML_filings.csv").write_text("")          # empresa estrangeira: arquivo vazio
    (tmp_path / "edgar" / "ASML_facts.csv").write_text("")
    return tmp_path


# ----------------------------- localização ----------------------------------
def test_sem_pasta_nao_ha_base(tmp_path, monkeypatch):
    monkeypatch.setenv("ANALISE_BASE_LOCAL", str(tmp_path / "nao_existe"))
    assert bl.raiz() is None and not bl.disponivel()


def test_disponibilidade_por_ativo(base):
    assert bl.disponivel()
    assert bl.tem_candles("aapl") and bl.tem_ri("AAPL") and bl.tem_noticias("AAPL")
    assert not bl.tem_candles("PETR4.SA") and not bl.tem_ri("ZZZZ") and not bl.tem_noticias("ZZZZ")


# ----------------------------- candles --------------------------------------
def test_candles_no_formato_do_app(base):
    df = bl.candles_diarios("AAPL")
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None and df.index.is_monotonic_increasing
    assert len(df) == 600


def test_baixar_local_corta_periodo_e_reamostra(base):
    df1 = bl.baixar_local("AAPL", "6mo", "1d")
    assert 100 < len(df1) < 140 and (df1.index[-1] - df1.index[0]).days <= 185
    sem, mes = bl.baixar_local("AAPL", "1y", "1wk"), bl.baixar_local("AAPL", "2y", "1mo")
    assert 45 <= len(sem) <= 54 and 20 <= len(mes) <= 26
    completo = bl.candles_diarios("AAPL")
    assert mes["High"].max() <= completo["High"].max() + 1e-9
    assert mes["Volume"].sum() <= completo["Volume"].sum() + 1e-6


def test_baixar_local_rejeita_intraday_e_ticker_ausente(base):
    with pytest.raises(DadosIndisponiveis, match="diários"):
        bl.baixar_local("AAPL", "1mo", "1h")
    with pytest.raises(DadosIndisponiveis, match="não está na base"):
        bl.baixar_local("PETR4.SA", "1y", "1d")


# ----------------------------- notícias -------------------------------------
def test_noticias_ordem_e_duplicatas(base):
    n = bl.noticias("AAPL")
    assert list(n["titulo"]) == ["recente", "do meio", "antiga"]          # mais recente primeiro, sem repetir id 2
    assert n["data"].is_monotonic_decreasing
    assert len(bl.noticias("AAPL", limite=2)) == 2
    assert bl.noticias("ZZZZ").empty and list(bl.noticias("ZZZZ").columns) == ["data", "titulo", "fonte", "resumo"]


# ----------------------------- comunicados ----------------------------------
def test_comunicados_horario_rotulos_e_resultado(base):
    c = bl.comunicados("AAPL")
    assert list(c["quando"].dt.strftime("%Y-%m-%d")) == ["2025-05-02", "2025-04-16", "2025-04-15", "2025-01-10"]
    assert str(c["quando"].dt.tz) == "America/Sao_Paulo"
    assert c.loc[2, "quando"].hour == 18 and c.loc[2, "quando"].minute == 30      # 21h30 UTC = 18h30 em Brasília
    assert list(c["resultado"]) == [False, False, True, True]
    assert c.loc[2, "assunto"] == "Resultados trimestrais"
    assert c.loc[3, "assunto"] == "Resultados trimestrais"                        # 9.01 (anexos) é escondido
    assert c.loc[0, "assunto"] == "Mudança de diretoria"
    assert c.loc[1, "tipo"] == "Balanço trimestral (10-Q)" and c.loc[1, "assunto"].startswith("Demonstrações")
    assert bl.comunicados("ASML").empty and bl.comunicados("ZZZZ").empty


# ----------------------------- resultados -----------------------------------
def test_resultados_trimestrais_crescimento_e_primeira_publicacao(base):
    r = bl.resultados_trimestrais("AAPL")
    assert len(r) == 2                                       # o anual (10-K) não entra
    atual = r.iloc[0]
    assert atual["trimestre"] == pd.Timestamp("2025-03-31")
    assert atual["lpa"] == pytest.approx(1.50) and atual["receita"] == pytest.approx(120.0)   # não a reapresentação
    assert atual["publicado"] == pd.Timestamp("2025-05-01")
    assert atual["cresc_lpa"] == pytest.approx(0.50) and atual["cresc_receita"] == pytest.approx(0.20)
    assert np.isnan(r.iloc[1]["cresc_lpa"])                  # sem trimestre do ano anterior para comparar


def test_resultados_empresa_estrangeira_ou_ausente_vazio(base):
    assert bl.resultados_trimestrais("ASML").empty and bl.resultados_trimestrais("ZZZZ").empty


# ----------------------------- reação ao resultado --------------------------
def test_reacao_apos_fechamento_cai_no_pregao_seguinte(tmp_path, monkeypatch):
    for d in ("data_sip", "edgar"):
        (tmp_path / d).mkdir()
    monkeypatch.setenv("ANALISE_BASE_LOCAL", str(tmp_path))
    idx = pd.bdate_range("2024-01-01", periods=120)
    close = np.full(120, 100.0)
    vol = np.full(120, 1e6)
    i = 80
    close[i:], vol[i] = 110.0, 4e6                           # no pregão i o preço salta 10% com 4× o volume
    pd.DataFrame({"timestamp": idx.strftime("%Y-%m-%d"), "open": close, "high": close, "low": close,
                  "close": close, "volume": vol}).to_csv(tmp_path / "data_sip" / "XYZ.csv", index=False)
    dia = idx[i - 1]                                         # divulgado no dia anterior, DEPOIS do fechamento (21h UTC)
    pd.DataFrame({"form": ["8-K"], "filingDate": [dia.strftime("%Y-%m-%d")],
                  "acceptanceDateTime": [dia.strftime("%Y-%m-%dT21:30:00.000Z")], "reportDate": [""],
                  "items": ["2.02"], "accessionNumber": ["x"]}).to_csv(tmp_path / "edgar" / "XYZ_filings.csv", index=False)
    r = bl.reacao_a_resultados("XYZ")
    assert len(r) == 1 and r.loc[0, "reacao_em"] == idx[i]
    assert r.loc[0, "variacao"] == pytest.approx(0.10) and r.loc[0, "volume_x"] == pytest.approx(4.0)


def test_reacao_a_resultados_sem_dados_vazio(base):
    assert bl.reacao_a_resultados("ZZZZ").empty
