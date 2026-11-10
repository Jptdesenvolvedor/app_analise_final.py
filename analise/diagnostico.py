"""Diagnóstico textual e detecção de eventos (alertas) a partir dos indicadores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .indicators import encontrar_pivots


@dataclass
class ItemDiagnostico:
    titulo: str
    texto: str
    tom: str  # "alta", "baixa" ou "neutro"


def _ultimo(df: pd.DataFrame, col: str, i: int = -1) -> float:
    if col not in df.columns or len(df) < abs(i):
        return float("nan")
    return float(df[col].iloc[i])


def _divergencia_rsi(df: pd.DataFrame, janela: int = 3, olhar: int = 80) -> str | None:
    """Divergência entre preço e RSI nos dois últimos pivôs (dentro de `olhar` candles)."""
    if "RSI" not in df.columns or len(df) < 2 * janela + 5:
        return None
    rec = df.tail(olhar)
    for tipo, serie in (("topo", rec["Close"]), ("fundo", rec["Close"])):
        piv = encontrar_pivots(serie, janela, tipo)
        if len(piv) < 2:
            continue
        (i1, p1), (i2, p2) = list(piv.items())[-2:]
        r1, r2 = rec.loc[i1, "RSI"], rec.loc[i2, "RSI"]
        if pd.isna(r1) or pd.isna(r2):
            continue
        if tipo == "topo" and p2 > p1 and r2 < r1:
            return "baixa"
        if tipo == "fundo" and p2 < p1 and r2 > r1:
            return "alta"
    return None


def diagnosticar(df: pd.DataFrame) -> list[ItemDiagnostico]:
    itens: list[ItemDiagnostico] = []
    preco = _ultimo(df, "Close")

    # RSI
    rsi = _ultimo(df, "RSI")
    if np.isnan(rsi):
        itens.append(ItemDiagnostico("RSI (14)", "Indisponível — dados insuficientes.", "neutro"))
    elif rsi < 30:
        itens.append(ItemDiagnostico("RSI (14)", f"{rsi:.1f} — sobrevenda (possível repique).", "alta"))
    elif rsi > 70:
        itens.append(ItemDiagnostico("RSI (14)", f"{rsi:.1f} — sobrecompra (possível correção).", "baixa"))
    else:
        itens.append(ItemDiagnostico("RSI (14)", f"{rsi:.1f} — zona neutra.", "neutro"))

    # MACD
    m, s, mp, sp = (_ultimo(df, "MACD"), _ultimo(df, "Signal"),
                    _ultimo(df, "MACD", -2), _ultimo(df, "Signal", -2))
    if np.isnan([m, s, mp, sp]).any():
        itens.append(ItemDiagnostico("MACD", "Indisponível — dados insuficientes.", "neutro"))
    elif mp <= sp and m > s:
        itens.append(ItemDiagnostico("MACD", "Cruzou a linha de sinal para cima.", "alta"))
    elif mp >= sp and m < s:
        itens.append(ItemDiagnostico("MACD", "Cruzou a linha de sinal para baixo.", "baixa"))
    elif m > s:
        itens.append(ItemDiagnostico("MACD", "Acima da linha de sinal.", "alta"))
    else:
        itens.append(ItemDiagnostico("MACD", "Abaixo da linha de sinal.", "baixa"))

    # Tendência de longo prazo
    ma200 = _ultimo(df, "MA200")
    if np.isnan(ma200):
        itens.append(ItemDiagnostico("Tendência (MA200)", "Histórico curto demais para a MA200.", "neutro"))
    elif preco >= ma200:
        itens.append(ItemDiagnostico("Tendência (MA200)", "Preço acima da MA200 — viés de longo prazo de alta.", "alta"))
    else:
        itens.append(ItemDiagnostico("Tendência (MA200)", "Preço abaixo da MA200 — viés de longo prazo de baixa.", "baixa"))

    # Alinhamento PhiCube
    e17, e72, e305 = _ultimo(df, "EMA17"), _ultimo(df, "EMA72"), _ultimo(df, "EMA305")
    if np.isnan([e17, e72, e305]).any():
        itens.append(ItemDiagnostico("EMAs PhiCube", "EMA305 indisponível — histórico curto.", "neutro"))
    elif e17 > e72 > e305:
        itens.append(ItemDiagnostico("EMAs PhiCube", "Alinhadas para cima (17 > 72 > 305).", "alta"))
    elif e17 < e72 < e305:
        itens.append(ItemDiagnostico("EMAs PhiCube", "Alinhadas para baixo (17 < 72 < 305).", "baixa"))
    else:
        itens.append(ItemDiagnostico("EMAs PhiCube", "Desalinhadas — sem tendência clara.", "neutro"))

    # Bollinger
    sup, inf = _ultimo(df, "BB_Sup"), _ultimo(df, "BB_Inf")
    if np.isnan([sup, inf]).any():
        itens.append(ItemDiagnostico("Bollinger (20, 2)", "Indisponível — dados insuficientes.", "neutro"))
    elif preco > sup:
        itens.append(ItemDiagnostico("Bollinger (20, 2)", "Preço acima da banda superior (esticado).", "baixa"))
    elif preco < inf:
        itens.append(ItemDiagnostico("Bollinger (20, 2)", "Preço abaixo da banda inferior (esticado).", "alta"))
    else:
        itens.append(ItemDiagnostico("Bollinger (20, 2)", "Preço dentro das bandas.", "neutro"))

    # Volume
    vol, vmm = _ultimo(df, "Volume"), _ultimo(df, "Volume_MM20")
    if np.isnan(vmm) or vmm == 0:
        itens.append(ItemDiagnostico("Volume", "Sem média de volume disponível.", "neutro"))
    else:
        razao = vol / vmm
        itens.append(ItemDiagnostico(
            "Volume", f"{razao:.1f}× a média de 20 candles"
            + (" — volume elevado." if razao >= 1.5 else "."), "neutro"))

    # Volatilidade
    atr = _ultimo(df, "ATR")
    if not np.isnan(atr) and preco:
        itens.append(ItemDiagnostico("Volatilidade (ATR 14)", f"{atr:,.2f} ({atr / preco:.1%} do preço).", "neutro"))

    # Divergência
    div = _divergencia_rsi(df)
    if div == "alta":
        itens.append(ItemDiagnostico("Divergência RSI", "Preço fez fundo mais baixo, RSI mais alto (divergência de alta).", "alta"))
    elif div == "baixa":
        itens.append(ItemDiagnostico("Divergência RSI", "Preço fez topo mais alto, RSI mais baixo (divergência de baixa).", "baixa"))
    else:
        itens.append(ItemDiagnostico("Divergência RSI", "Nenhuma divergência recente.", "neutro"))

    return itens


def _cruzou_acima(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.shift() <= b.shift()) & (a > b)


def _cruzou_abaixo(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.shift() >= b.shift()) & (a < b)


def detectar_eventos(df: pd.DataFrame) -> pd.DataFrame:
    """Eventos históricos (cruzamentos) com colunas: data, evento, tom."""
    if df.empty:
        return pd.DataFrame(columns=["data", "evento", "tom"])
    c = df["Close"]
    regras = [
        (_cruzou_abaixo(df["RSI"], pd.Series(30.0, index=df.index)), "RSI entrou em sobrevenda (<30)", "alta"),
        (_cruzou_acima(df["RSI"], pd.Series(70.0, index=df.index)), "RSI entrou em sobrecompra (>70)", "baixa"),
        (_cruzou_acima(df["MACD"], df["Signal"]), "MACD cruzou sinal para cima", "alta"),
        (_cruzou_abaixo(df["MACD"], df["Signal"]), "MACD cruzou sinal para baixo", "baixa"),
        (_cruzou_acima(df["EMA17"], df["EMA72"]), "EMA17 cruzou EMA72 para cima", "alta"),
        (_cruzou_abaixo(df["EMA17"], df["EMA72"]), "EMA17 cruzou EMA72 para baixo", "baixa"),
        (_cruzou_acima(c, df["MA200"]), "Preço cruzou a MA200 para cima", "alta"),
        (_cruzou_abaixo(c, df["MA200"]), "Preço cruzou a MA200 para baixo", "baixa"),
        (_cruzou_acima(c, df["BB_Sup"]), "Preço rompeu a banda superior", "baixa"),
        (_cruzou_abaixo(c, df["BB_Inf"]), "Preço rompeu a banda inferior", "alta"),
    ]
    linhas = []
    for mask, texto, tom in regras:
        for data in df.index[mask.fillna(False).to_numpy()]:
            linhas.append((data, texto, tom))
    out = pd.DataFrame(linhas, columns=["data", "evento", "tom"])
    return out.sort_values("data", ascending=False, kind="stable").reset_index(drop=True)
