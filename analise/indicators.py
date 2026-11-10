"""Indicadores técnicos, Fibonacci e suporte/resistência."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FIB_RATIOS = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)


def rsi_wilder(close: pd.Series, periodo: int = 14) -> pd.Series:
    """RSI com suavização de Wilder (padrão de TradingView e similares).

    Antes de haver `periodo` variações o valor é NaN (nunca 100). Sem perdas no
    período → 100; sem ganhos nem perdas (preço parado) → 50.
    """
    delta = close.diff()
    ganho = delta.clip(lower=0)
    perda = -delta.clip(upper=0)
    alpha = 1.0 / periodo
    media_ganho = ganho.ewm(alpha=alpha, adjust=False, min_periods=periodo).mean()
    media_perda = perda.ewm(alpha=alpha, adjust=False, min_periods=periodo).mean()

    rs = media_ganho / media_perda.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.where(~((media_perda == 0) & (media_ganho > 0)), 100.0)
    rsi = rsi.where(~((media_perda == 0) & (media_ganho == 0)), 50.0)
    return rsi


def macd(close: pd.Series, rapida: int = 12, lenta: int = 26, sinal: int = 9):
    ema_r = close.ewm(span=rapida, adjust=False, min_periods=rapida).mean()
    ema_l = close.ewm(span=lenta, adjust=False, min_periods=lenta).mean()
    linha = ema_r - ema_l
    linha_sinal = linha.ewm(span=sinal, adjust=False, min_periods=sinal).mean()
    return linha, linha_sinal, linha - linha_sinal


def atr(df: pd.DataFrame, periodo: int = 14) -> pd.Series:
    pc = df["Close"].shift()
    tr = pd.concat(
        [df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / periodo, adjust=False, min_periods=periodo).mean()


def estocastico(df: pd.DataFrame, k: int = 14, d: int = 3):
    menor = df["Low"].rolling(k).min()
    maior = df["High"].rolling(k).max()
    faixa = (maior - menor).replace(0, np.nan)
    pct_k = 100 * (df["Close"] - menor) / faixa
    return pct_k, pct_k.rolling(d).mean()


def vwap_sessao(df: pd.DataFrame) -> pd.Series:
    """VWAP reiniciado a cada dia (só faz sentido em dados intraday)."""
    if "Volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"].astype(float)
    dia = pd.Series(df.index.date, index=df.index)
    pv = (tp * vol).groupby(dia).cumsum()
    v = vol.groupby(dia).cumsum().replace(0, np.nan)
    return pv / v


def calcular_indicadores(df: pd.DataFrame, intraday: bool = False) -> pd.DataFrame:
    """Adiciona todos os indicadores. Janelas curtas demais resultam em NaN."""
    df = df.copy()
    close = df["Close"]

    df["MA21"] = close.rolling(21).mean()
    df["MA200"] = close.rolling(200).mean()
    for span in (17, 72, 305):  # PhiCube
        df[f"EMA{span}"] = close.ewm(span=span, adjust=False, min_periods=span).mean()

    df["RSI"] = rsi_wilder(close)
    df["MACD"], df["Signal"], df["MACD_Hist"] = macd(close)

    ma20 = close.rolling(20).mean()
    desvio = close.rolling(20).std(ddof=0)
    df["BB_Meio"], df["BB_Sup"], df["BB_Inf"] = ma20, ma20 + 2 * desvio, ma20 - 2 * desvio

    if {"High", "Low"}.issubset(df.columns):
        df["ATR"] = atr(df)
        df["Stoch_K"], df["Stoch_D"] = estocastico(df)
    else:
        df["ATR"] = df["Stoch_K"] = df["Stoch_D"] = np.nan

    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    df["Volume_Financeiro"] = df["Volume"] * close
    df["Volume_MM20"] = df["Volume"].rolling(20).mean()
    df["VWAP"] = vwap_sessao(df) if intraday and {"High", "Low"}.issubset(df.columns) else np.nan
    return df


# --------------------------------------------------------------------------- #
# Fibonacci
# --------------------------------------------------------------------------- #
@dataclass
class Fibonacci:
    tendencia: str  # "alta" ou "baixa" (direção do movimento medido)
    niveis: dict[str, float]


def calcular_fibonacci(df: pd.DataFrame) -> Fibonacci | None:
    """Retração de Fibonacci sobre máxima/mínima do período.

    Alta (mínima antes da máxima): 0% na máxima, 100% na mínima.
    Baixa (máxima antes da mínima): 0% na mínima, 100% na máxima.
    """
    if df.empty:
        return None
    topo_s = df["High"] if "High" in df.columns else df["Close"]
    fundo_s = df["Low"] if "Low" in df.columns else df["Close"]
    topo, fundo = float(topo_s.max()), float(fundo_s.min())
    dif = topo - fundo
    alta = fundo_s.idxmin() <= topo_s.idxmax()
    niveis = {}
    for r in FIB_RATIOS:
        valor = topo - r * dif if alta else fundo + r * dif
        niveis[f"{r * 100:.1f}%"] = valor
    return Fibonacci("alta" if alta else "baixa", niveis)


# --------------------------------------------------------------------------- #
# Pivôs e suporte/resistência
# --------------------------------------------------------------------------- #
def encontrar_pivots(serie: pd.Series, janela: int = 5, tipo: str = "topo") -> pd.Series:
    """Pivôs confirmados: extremos locais numa janela de `janela` candles de cada lado."""
    w = 2 * janela + 1
    ext = serie.rolling(w, center=True).max() if tipo == "topo" else serie.rolling(w, center=True).min()
    return serie[serie.eq(ext) & ext.notna()]


def _agrupar_niveis(valores: list[float], tolerancia: float) -> list[tuple[float, int]]:
    grupos: list[list[float]] = []
    for v in sorted(valores):
        if grupos and v <= np.mean(grupos[-1]) * (1 + tolerancia):
            grupos[-1].append(v)
        else:
            grupos.append([v])
    return [(float(np.mean(g)), len(g)) for g in grupos]


def suporte_resistencia(
    df: pd.DataFrame, janela: int = 5, tolerancia: float = 0.01, max_niveis: int = 3
) -> dict[str, list[tuple[float, int]]]:
    """Níveis (preço, nº de toques) agrupados a partir dos pivôs de topo/fundo."""
    if df.empty or "High" not in df.columns:
        return {"suportes": [], "resistencias": []}
    topos = encontrar_pivots(df["High"], janela, "topo")
    fundos = encontrar_pivots(df["Low"], janela, "fundo")
    niveis = _agrupar_niveis(list(topos.values) + list(fundos.values), tolerancia)
    preco = float(df["Close"].iloc[-1])

    def escolher(lista):
        melhores = sorted(lista, key=lambda x: (-x[1], abs(x[0] - preco)))[:max_niveis]
        return sorted(melhores)

    return {
        "suportes": escolher([n for n in niveis if n[0] < preco]),
        "resistencias": escolher([n for n in niveis if n[0] >= preco]),
    }
