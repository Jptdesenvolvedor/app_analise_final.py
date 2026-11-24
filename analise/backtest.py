"""Backtest simples (long-only) de regras técnicas, para dar contexto histórico ao diagnóstico."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .regime import estado_defensivo, serie_tendencia


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


def estrategia_defensiva(df: pd.DataFrame) -> pd.Series:
    """Núcleo defensivo: MA200 com banda de 2% (histerese). Poucas operações, protege em quedas longas."""
    return estado_defensivo(df["Close"]).astype(int)


def estrategia_tendencia(df: pd.DataFrame) -> pd.Series:
    """Modelo de tendência (Minervini/O'Neil): 6 critérios de médias e máximas/mínimas de 52 semanas."""
    return serie_tendencia(df["Close"])


ESTRATEGIAS = {
    "Cruzamento MACD × Sinal": estrategia_macd,
    "Cruzamento EMA17 × EMA72": estrategia_ema,
    "Preço acima da MA200": estrategia_ma200,
    "RSI: compra <30, sai >70": estrategia_rsi,
    "Núcleo defensivo: MA200 com banda 2%": estrategia_defensiva,
    "Modelo de tendência (Minervini)": estrategia_tendencia,
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
    cagr: float | None = None
    sharpe: float | None = None
    cagr_buy_hold: float | None = None
    sharpe_buy_hold: float | None = None
    max_drawdown_buy_hold: float | None = None


def _max_drawdown(curva: pd.Series) -> float:
    if curva.empty:
        return 0.0
    return float((curva / curva.cummax() - 1).min())


def aplicar_stop_atr(df: pd.DataFrame, posicao: pd.Series, mult: float = 3.0) -> pd.Series:
    """Stop móvel (chandelier): sai se o fechamento cair `mult`×ATR abaixo da máxima desde a entrada.

    Depois de stopado, só volta a entrar quando o sinal original passar por 0 e voltar a 1
    (evita comprar e ser stopado em sequência). Causal: usa só dados até o candle atual.
    """
    from .indicators import atr as _atr

    close, a = df["Close"].to_numpy(float), _atr(df).to_numpy(float)
    sinal = posicao.reindex(df.index).fillna(0).to_numpy()
    saida, comprado, bloqueado, topo = np.zeros(len(df), dtype=int), False, False, 0.0
    for i in range(len(df)):
        if sinal[i] == 0:
            comprado, bloqueado = False, False
        elif comprado:
            topo = max(topo, close[i])
            if not np.isnan(a[i]) and close[i] < topo - mult * a[i]:
                comprado, bloqueado = False, True
        elif not bloqueado:
            comprado, topo = True, close[i]
        saida[i] = int(comprado)
    return pd.Series(saida, index=df.index)


def _metricas_retorno(ret: pd.Series, passos_ano: float) -> tuple[float | None, float | None]:
    """(CAGR, Sharpe) a partir dos retornos por candle. None quando não há dados suficientes."""
    if len(ret) < 3:
        return None, None
    total = float((1 + ret).prod())
    anos = len(ret) / passos_ano
    cagr = total ** (1 / anos) - 1 if anos > 0 and total > 0 else None
    sd = float(ret.std())
    sharpe = float(ret.mean() / sd * np.sqrt(passos_ano)) if sd > 0 else None
    return cagr, sharpe


def _passos_por_ano(index: pd.Index) -> float:
    if len(index) < 3:
        return 252.0
    dias = (index[-1] - index[0]).total_seconds() / 86400
    return max(len(index) / dias * 365.25, 1.0) if dias > 0 else 252.0


def executar_backtest(df: pd.DataFrame, posicao: pd.Series, custo: float = 0.0,
                      stop_atr: float | None = None) -> ResultadoBacktest:
    """Executa a posição no candle seguinte ao sinal (sem look-ahead).

    `custo` é a fração cobrada a cada entrada/saída (ex.: 0.001 = 0,1%).
    `stop_atr`: se informado, aplica stop móvel de N×ATR (ver `aplicar_stop_atr`).
    """
    if stop_atr:
        posicao = aplicar_stop_atr(df, posicao, stop_atr)
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
    passos = _passos_por_ano(df.index)
    cagr, sharpe = _metricas_retorno(ret_est, passos)
    cagr_bh, sharpe_bh = _metricas_retorno(ret, passos)
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
        cagr=cagr,
        sharpe=sharpe,
        cagr_buy_hold=cagr_bh,
        sharpe_buy_hold=sharpe_bh,
        max_drawdown_buy_hold=_max_drawdown(bh),
    )


def executar_treino_teste(df: pd.DataFrame, posicao: pd.Series, custo: float = 0.0,
                          stop_atr: float | None = None, fracao_treino: float = 0.6) -> dict[str, ResultadoBacktest]:
    """Roda o mesmo sinal em duas janelas consecutivas (treino e teste).

    O sinal é calculado sobre a série inteira (os indicadores são causais), mas cada janela é
    avaliada separadamente: uma regra que só funciona num dos períodos não é confiável.
    """
    corte = int(len(df) * fracao_treino)
    out = {}
    for nome, sl in (("treino", slice(0, corte)), ("teste", slice(corte, None))):
        sub = df.iloc[sl]
        if len(sub) < 30:
            continue
        sinal = posicao.reindex(df.index).iloc[sl]
        out[nome] = executar_backtest(sub, sinal, custo, stop_atr)
    return out
