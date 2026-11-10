"""Backtest simples (long-only) de regras técnicas, para dar contexto histórico ao diagnóstico."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _posicao_de_condicao(cond: pd.Series) -> pd.Series:
    return cond.fillna(False).astype(int)


def estrategia_macd(df: pd.DataFrame) -> pd.Series:
    return _posicao_de_condicao(df["MACD"] > df["Signal"])


def estrategia_ema(df: pd.DataFrame) -> pd.Series:
    return _posicao_de_condicao(df["EMA17"] > df["EMA72"])


def estrategia_ma200(df: pd.DataFrame) -> pd.Series:
    return _posicao_de_condicao(df["Close"] > df["MA200"])


def estrategia_rsi(df: pd.DataFrame) -> pd.Series:
    """Compra quando RSI < 30, zera quando RSI > 70, mantém entre os dois."""
    estado = pd.Series(np.nan, index=df.index)
    estado[df["RSI"] < 30] = 1.0
    estado[df["RSI"] > 70] = 0.0
    return estado.ffill().fillna(0).astype(int)


ESTRATEGIAS = {
    "Cruzamento MACD × Sinal": estrategia_macd,
    "Cruzamento EMA17 × EMA72": estrategia_ema,
    "Preço acima da MA200": estrategia_ma200,
    "RSI: compra <30, sai >70": estrategia_rsi,
}


@dataclass
class ResultadoBacktest:
    curva_estrategia: pd.Series
    curva_buy_hold: pd.Series
    retorno_estrategia: float
    retorno_buy_hold: float
    n_trades: int
    taxa_acerto: float | None
    max_drawdown: float
    tempo_posicionado: float
    trades: pd.DataFrame


def _max_drawdown(curva: pd.Series) -> float:
    if curva.empty:
        return 0.0
    return float((curva / curva.cummax() - 1).min())


def executar_backtest(df: pd.DataFrame, posicao: pd.Series, custo: float = 0.0) -> ResultadoBacktest:
    """Executa a posição no candle seguinte ao sinal (sem look-ahead).

    `custo` é a fração cobrada a cada entrada/saída (ex.: 0.001 = 0,1%).
    """
    pos = posicao.reindex(df.index).shift(1).fillna(0)
    ret = df["Close"].pct_change().fillna(0)
    trocas = pos.diff().abs().fillna(pos.abs())
    ret_est = pos * ret - trocas * custo

    curva = (1 + ret_est).cumprod()
    bh = (1 + ret).cumprod()

    grupo = (pos.diff().fillna(pos) != 0).cumsum()
    linhas = []
    for _, idx in ret_est[pos == 1].groupby(grupo[pos == 1]).groups.items():
        r = float((1 + ret_est.loc[idx]).prod() - 1)
        linhas.append({"entrada": idx[0], "saida": idx[-1], "candles": len(idx), "retorno": r})
    trades = pd.DataFrame(linhas, columns=["entrada", "saida", "candles", "retorno"])

    acerto = float((trades["retorno"] > 0).mean()) if len(trades) else None
    return ResultadoBacktest(
        curva_estrategia=curva,
        curva_buy_hold=bh,
        retorno_estrategia=float(curva.iloc[-1] - 1) if len(curva) else 0.0,
        retorno_buy_hold=float(bh.iloc[-1] - 1) if len(bh) else 0.0,
        n_trades=len(trades),
        taxa_acerto=acerto,
        max_drawdown=_max_drawdown(curva),
        tempo_posicionado=float((pos == 1).mean()) if len(pos) else 0.0,
        trades=trades,
    )
