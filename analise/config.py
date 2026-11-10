"""Constantes e regras de período × intervalo."""

from __future__ import annotations

from typing import NamedTuple, Optional

ATIVOS_FIXOS = {
    "Criptomoedas": {
        "Bitcoin (BTC)": "BTC-USD",
        "Ethereum (ETH)": "ETH-USD",
        "Solana (SOL)": "SOL-USD",
    },
    "Ações Americanas": {
        "Apple (AAPL)": "AAPL",
        "Nvidia (NVDA)": "NVDA",
        "Microsoft (MSFT)": "MSFT",
    },
    "Ações Brasil": {
        "Petrobras (PETR4)": "PETR4.SA",
        "Vale (VALE3)": "VALE3.SA",
        "Itaú (ITUB4)": "ITUB4.SA",
    },
}

PERIODOS_UI = {
    "7 dias": "7d",
    "1 mês": "1mo",
    "3 meses": "3mo",
    "6 meses": "6mo",
    "1 ano": "1y",
    "2 anos": "2y",
    "5 anos": "5y",
    "10 anos": "10y",
}

INTERVALOS_UI = {
    "15 minutos": "15m",
    "30 minutos": "30m",
    "1 hora": "1h",
    "4 horas (resample)": "4h",  # obtido via resample de dados de 1h
    "1 dia": "1d",
    "1 semana": "1wk",
    "1 mês": "1mo",
}

# Limites de histórico do Yahoo Finance por intervalo intraday (em dias).
# 15m/30m: ~60 dias; 1h: ~730 dias (usamos margem de segurança).
LIMITE_DIAS_POR_INTERVALO = {"15m": 60, "30m": 60, "1h": 700}
INTERVALOS_INTRADAY = set(LIMITE_DIAS_POR_INTERVALO)

# Mínimo de candles para cada indicador ficar confiável.
MIN_CANDLES = {"MA21": 21, "MA200": 200, "EMA72": 72, "EMA305": 305}


class ParametrosConsulta(NamedTuple):
    periodo: str
    intervalo_real: str
    resample_4h: bool
    aviso: Optional[str]


def periodo_em_dias(periodo: str) -> int:
    """Converte um período do estilo yfinance ('7d', '1mo', '1y') em dias aproximados."""
    if periodo.endswith("mo"):
        return int(periodo[:-2]) * 30
    if periodo.endswith("wk"):
        return int(periodo[:-2]) * 7
    if periodo.endswith("d"):
        return int(periodo[:-1])
    if periodo.endswith("y"):
        return int(periodo[:-1]) * 365
    return 30


def compatibilizar_periodo_intervalo(
    periodo: str, intervalo_ui: str, intervalo_label: str
) -> ParametrosConsulta:
    """Garante que período × intervalo é aceito pelo Yahoo Finance.

    Se o período excede o limite do intervalo intraday, escolhe o maior período
    disponível dentro do limite e devolve uma mensagem de aviso.
    """
    resample_4h = intervalo_ui == "4h"
    intervalo_real = "1h" if resample_4h else intervalo_ui

    limite = LIMITE_DIAS_POR_INTERVALO.get(intervalo_real)
    if limite is not None and periodo_em_dias(periodo) > limite:
        candidatos = [p for p in PERIODOS_UI.values() if periodo_em_dias(p) <= limite]
        ajustado = max(candidatos, key=periodo_em_dias) if candidatos else "7d"
        msg = (
            f"⚠️ O intervalo **{intervalo_label}** só é suportado pelo Yahoo Finance para "
            f"até **{limite} dias** de histórico. Ajustei o período para **{ajustado}**."
        )
        return ParametrosConsulta(ajustado, intervalo_real, resample_4h, msg)

    return ParametrosConsulta(periodo, intervalo_real, resample_4h, None)
