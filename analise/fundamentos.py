"""Fundamentos básicos via Yahoo Finance (`Ticker.info`)."""

from __future__ import annotations

import yfinance as yf

CAMPOS = {
    "longName": "Nome",
    "sector": "Setor",
    "industry": "Indústria",
    "currency": "Moeda",
    "marketCap": "Valor de mercado",
    "trailingPE": "P/L (12m)",
    "forwardPE": "P/L (projetado)",
    "priceToBook": "P/VP",
    "trailingEps": "LPA (12m)",
    "dividendYield": "Dividend yield (como informado pelo Yahoo)",
    "beta": "Beta",
    "fiftyTwoWeekHigh": "Máxima 52 semanas",
    "fiftyTwoWeekLow": "Mínima 52 semanas",
}


def buscar_fundamentos(ticker: str) -> dict[str, object]:
    """Retorna {rótulo: valor} dos campos disponíveis; vazio se o Yahoo não responder."""
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return {}
    return {rotulo: info[chave] for chave, rotulo in CAMPOS.items() if info.get(chave) is not None}


def formatar_valor(rotulo: str, valor: object) -> str:
    if isinstance(valor, (int, float)):
        if "mercado" in rotulo.lower():
            for lim, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
                if abs(valor) >= lim:
                    return f"{valor / lim:,.2f}{suf}"
        return f"{valor:,.2f}"
    return str(valor)
