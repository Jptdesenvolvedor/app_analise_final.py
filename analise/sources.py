"""Fontes oficiais e links de validação por ativo (dados em fontes.json)."""

from __future__ import annotations

import json
from pathlib import Path

_DADOS = json.loads((Path(__file__).with_name("fontes.json")).read_text(encoding="utf-8"))

FONTES_OFICIAIS: dict[str, dict[str, str]] = _DADOS["FONTES_OFICIAIS"]
CRIPTO_SLUGS: dict[str, str] = _DADOS["CRIPTO_SLUGS"]
FONTES_GERAIS: dict[str, list[list[str]]] = _DADOS["FONTES_GERAIS"]


def eh_ticker_br(ticker: str) -> bool:
    return ticker.upper().endswith(".SA")


def eh_ticker_cripto(ticker: str) -> bool:
    return ticker.upper().endswith("-USD")


def montar_links_cruzados(ticker: str) -> dict[str, list[tuple[str, str]]]:
    """Links de cotação/dados para o ticker em provedores conhecidos.

    Como o app aceita qualquer ticker, os links seguem o padrão de URL de cada
    provedor — confira sempre se a página aberta corresponde ao ativo certo.
    """
    ticker = ticker.strip().upper()
    grupos: dict[str, list[tuple[str, str]]] = {}

    if eh_ticker_br(ticker):
        tv = f"BMFBOVESPA-{ticker.replace('.SA', '')}"
    elif eh_ticker_cripto(ticker):
        tv = ticker.replace("-", "")
    else:
        tv = ticker

    grupos["Cotação (confira o gráfico e o preço)"] = [
        ("Yahoo Finance", f"https://finance.yahoo.com/quote/{ticker}"),
        ("TradingView", f"https://www.tradingview.com/symbols/{tv}/"),
        ("Investing.com (busca)", f"https://www.investing.com/search/?q={ticker}"),
    ]

    if eh_ticker_cripto(ticker):
        base = ticker.split("-")[0]
        slug = CRIPTO_SLUGS.get(ticker)
        if slug:
            grupos["Dados de criptoativos"] = [
                ("CoinGecko", f"https://www.coingecko.com/en/coins/{slug}"),
                ("CoinMarketCap", f"https://coinmarketcap.com/currencies/{slug}/"),
            ]
        else:
            grupos["Dados de criptoativos"] = [
                ("CoinGecko (busca)", f"https://www.coingecko.com/en/search?query={base}"),
                ("CoinMarketCap (busca)", f"https://coinmarketcap.com/search/?q={base}"),
            ]
    elif eh_ticker_br(ticker):
        codigo = ticker.replace(".SA", "")
        grupos["Fundamentos e regulação (Brasil)"] = [
            ("StatusInvest", f"https://statusinvest.com.br/acoes/{codigo.lower()}"),
            ("Fundamentus", f"https://www.fundamentus.com.br/detalhes.php?papel={codigo}"),
            ("StockAnalysis.com (B3)", f"https://stockanalysis.com/quote/bvmf/{codigo}/"),
            ("Google Finance (B3)", f"https://www.google.com/finance/quote/{codigo}:BVMF"),
            ("CVM — Consulta de documentos (RAD)",
             "https://www.rad.cvm.gov.br/ENET/frmConsultaExternaCVM.aspx"),
        ]
    else:
        grupos["Fundamentos e regulação (EUA/outros)"] = [
            ("StockAnalysis.com", f"https://stockanalysis.com/stocks/{ticker.lower()}/"),
            ("SEC EDGAR — Full Text Search", f"https://www.sec.gov/edgar/search/#/q={ticker}"),
        ]
    return grupos


def todas_as_urls_estaticas() -> list[str]:
    """URLs fixas do JSON (usadas no teste de links)."""
    urls = [v["url"] for v in FONTES_OFICIAIS.values()]
    for itens in FONTES_GERAIS.values():
        urls += [u for _, u in itens]
    return urls
