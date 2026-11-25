"""Visão geral do mercado dos EUA: as maiores empresas listadas, com o sinal de decisão de cada uma.

A seleção (`selecionar_top`), o resumo por ação (`resumo_ticker`) e a amplitude (`amplitude`) não dependem da rede,
para serem testáveis; `buscar_top` e `baixar_lote` são a camada fina com o `yfinance`.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .data import normalizar_colunas
from .indicators import calcular_indicadores
from .sinal import calcular_decisao, tendencia_longa

INDICES = {"^GSPC": "S&P 500", "^IXIC": "Nasdaq", "^DJI": "Dow Jones"}

# ações preferenciais (-PC), warrants (-WS), units (-U) e rights (-R) não são "a empresa"
RE_NAO_ORDINARIA = re.compile(r"-(P[A-Z]?|W[A-Z]?|U|R)$")
RE_SUFIXOS = re.compile(r"\b(inc|incorporated|corp|corporation|company|co|ltd|limited|plc|holdings?|ordinary shares?|"
                        r"common stock|adr|ads|sponsored|class [a-z]|series [a-z]|new)\b")

# Plano B se o screener do Yahoo falhar (ordem aproximada por valor de mercado)
FALLBACK = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "AVGO", "TSLA", "BRK-B", "LLY", "WMT", "JPM", "V",
            "ORCL", "MA", "XOM", "NFLX", "COST", "JNJ", "HD", "PLTR", "ABBV", "BAC", "PG", "CVX", "UNH", "KO", "GE",
            "CSCO", "TMUS", "WFC", "AMD", "PM", "MS", "IBM", "ABT", "CRM", "LIN", "AXP", "MCD", "DIS", "GS", "MRK",
            "T", "RTX", "UBER", "NOW", "PEP", "CAT", "INTU", "VZ", "BKNG", "TSM", "ASML"]

ACAO_ICONE = {"COMPRAR": "▲", "MANTER": "●", "AGUARDAR": "◆", "REDUZIR": "▼", "VENDER": "▼"}


# --------------------------------------------------------------------------- #
# Seleção das maiores empresas
# --------------------------------------------------------------------------- #
def chave_empresa(nome: str) -> str:
    """Nome sem sufixos legais e classes de ação: GOOGL e GOOG ('Alphabet Inc.') dão a mesma chave."""
    n = re.sub(r"[^\w\s]", " ", (nome or "").lower())
    return re.sub(r"\s+", " ", RE_SUFIXOS.sub(" ", n)).strip()


def selecionar_top(quotes: list[dict], n: int = 50) -> list[dict]:
    """As `n` maiores EMPRESAS por valor de mercado: tira preferenciais/warrants e deixa uma ação por empresa
    (a de maior volume entre as classes). Cada item: ticker, nome, cap, bolsa."""
    candidatas = []
    for q in quotes:
        sym, cap = str(q.get("symbol") or ""), q.get("marketCap")
        if q.get("quoteType", "EQUITY") != "EQUITY" or not sym or RE_NAO_ORDINARIA.search(sym):
            continue
        if not isinstance(cap, (int, float)) or not cap > 0 or np.isnan(cap):
            continue
        candidatas.append({"ticker": sym, "nome": q.get("shortName") or q.get("longName") or sym, "cap": float(cap),
                           "bolsa": q.get("exchange", ""), "volume": float(q.get("averageDailyVolume3Month") or 0)})
    melhor: dict[str, dict] = {}
    for c in candidatas:                                  # uma por empresa: a classe mais negociada
        k = chave_empresa(c["nome"]) or c["ticker"]
        if k not in melhor or c["volume"] > melhor[k]["volume"]:
            melhor[k] = c
    top = sorted(melhor.values(), key=lambda c: -c["cap"])[:n]
    return [{k: v for k, v in c.items() if k != "volume"} for c in top]


def buscar_top(n: int = 50) -> tuple[list[dict], bool]:
    """(empresas, ao_vivo). Usa o screener do Yahoo; se falhar, devolve a lista fixa (ao_vivo=False)."""
    try:
        import yfinance as yf
        from yfinance import EquityQuery

        q = EquityQuery("and", [EquityQuery("eq", ["region", "us"]), EquityQuery("is-in", ["exchange", "NMS", "NYQ"])])
        quotes = yf.screen(q, sortField="intradaymarketcap", sortAsc=False, size=min(250, n * 2 + 30)).get("quotes", [])
        top = selecionar_top(quotes, n)
        if len(top) >= min(n, 20):
            return top, True
    except Exception:
        pass
    return [{"ticker": t, "nome": t, "cap": None, "bolsa": ""} for t in FALLBACK[:n]], False


# --------------------------------------------------------------------------- #
# Análise de cada ação
# --------------------------------------------------------------------------- #
def resumo_ticker(df: pd.DataFrame) -> dict | None:
    """Preço, variações, sinal de decisão e tendência de longo prazo de uma ação (df com Open/High/Low/Close/Volume)."""
    df = normalizar_colunas(df)
    if len(df) < 30:
        return None
    ind = calcular_indicadores(df)
    d = calcular_decisao(ind)
    c = ind["Close"]
    ma200 = c.rolling(200).mean().iloc[-1]
    longa = tendencia_longa(c)
    return {
        "preco": float(c.iloc[-1]),
        "var_dia": float(c.iloc[-1] / c.iloc[-2] - 1) if len(c) > 1 else np.nan,
        "var_mes": float(c.iloc[-1] / c.iloc[-22] - 1) if len(c) > 22 else np.nan,
        "var_ano": float(c.iloc[-1] / c.iloc[-253] - 1) if len(c) > 253 else np.nan,
        "acao": d.acao, "score": float(d.score), "confianca": d.confianca,
        "tendencia": longa.rotulo, "tendencia_dir": longa.direcao,
        "dist_ma200": float(c.iloc[-1] / ma200 - 1) if not np.isnan(ma200) else np.nan,
    }


def baixar_lote(tickers: list[str], periodo: str = "2y") -> dict[str, pd.DataFrame]:
    """Baixa o histórico de várias ações de uma vez; devolve {ticker: DataFrame} só dos que vieram."""
    import yfinance as yf

    raw = yf.download(tickers, period=periodo, interval="1d", group_by="ticker", auto_adjust=False,
                      progress=False, threads=True)
    out = {}
    for t in tickers:
        try:
            df = normalizar_colunas(raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw)
        except KeyError:
            continue
        if not df.empty:
            out[t] = df
    return out


def montar_tabela(top: list[dict], dados: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Tabela final na ordem do valor de mercado (posição 1 = maior). Ações sem dados ficam de fora."""
    linhas = []
    for pos, emp in enumerate(top, start=1):
        r = resumo_ticker(dados[emp["ticker"]]) if emp["ticker"] in dados else None
        if r is None:
            continue
        linhas.append({"posicao": pos, "ticker": emp["ticker"], "empresa": emp["nome"], "cap": emp["cap"], **r})
    return pd.DataFrame(linhas)


def resumo_indices(periodo: str = "2y") -> pd.DataFrame:
    """Preço, variação do dia e tendência de longo prazo dos principais índices."""
    dados = baixar_lote(list(INDICES), periodo)
    linhas = []
    for sim, nome in INDICES.items():
        r = resumo_ticker(dados[sim]) if sim in dados else None
        if r:
            linhas.append({"indice": nome, "ticker": sim, **r})
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------- #
# Amplitude
# --------------------------------------------------------------------------- #
def amplitude(tabela: pd.DataFrame, n_destaques: int = 5) -> dict:
    """Panorama das empresas da tabela: quantas em cada sinal, % acima da MA200, % em alta no dia e destaques."""
    if tabela.empty:
        return {"n": 0, "por_sinal": {}, "pct_acima_ma200": None, "pct_alta_dia": None, "maiores_altas": [],
                "maiores_quedas": []}
    por_sinal = {a: int((tabela["acao"] == a).sum()) for a in ("COMPRAR", "MANTER", "AGUARDAR", "REDUZIR", "VENDER")}
    dist = tabela["dist_ma200"].dropna()
    dia = tabela["var_dia"].dropna()
    ordenado = tabela.dropna(subset=["var_dia"]).sort_values("var_dia")
    return {
        "n": len(tabela), "por_sinal": por_sinal,
        "pct_acima_ma200": float((dist > 0).mean()) if len(dist) else None,
        "pct_alta_dia": float((dia > 0).mean()) if len(dia) else None,
        "maiores_altas": ordenado.tail(n_destaques).iloc[::-1][["ticker", "var_dia"]].to_dict("records"),
        "maiores_quedas": ordenado.head(n_destaques)[["ticker", "var_dia"]].to_dict("records"),
    }


def leitura_mercado(a: dict) -> str:
    """Frase curta sobre o fôlego do mercado (amplitude), sem prometer nada."""
    p = a.get("pct_acima_ma200")
    if p is None:
        return ""
    if p >= 0.70:
        base = f"Mercado com bom fôlego: {p:.0%} das maiores empresas estão acima da média de 200 dias."
    elif p >= 0.40:
        base = f"Mercado dividido: {p:.0%} das maiores empresas estão acima da média de 200 dias."
    else:
        base = f"Mercado fraco: só {p:.0%} das maiores empresas estão acima da média de 200 dias."
    return base
