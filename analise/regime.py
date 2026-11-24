"""Regime de mercado, modelo de tendência, momentum e dimensionamento de posição.

Regras validadas em backtests longos (SPY/QQQ e ~4.000 ações, treino 2017-22 × teste 2023-26, sem viés
de sobrevivência). Todas são causais: o valor de um candle só usa dados até ele.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BANDA_PADRAO = 0.02  # histerese de 2% em torno da MA200


def estado_defensivo(close: pd.Series, janela: int = 200, banda: float = BANDA_PADRAO) -> pd.Series:
    """True = investido, False = em caixa. Regra do "núcleo defensivo".

    Entra quando o preço fecha acima da média × (1 + banda) e só sai abaixo de média × (1 − banda).
    Dentro da banda mantém o estado anterior — evita trocas por ruído (poucas operações por ano).
    """
    ma = close.rolling(janela).mean()
    sobe, desce = (close > ma * (1 + banda)).to_numpy(), (close < ma * (1 - banda)).to_numpy()
    estado, saida = False, np.zeros(len(close), dtype=bool)
    for i in range(len(close)):
        if sobe[i]:
            estado = True
        elif desce[i]:
            estado = False
        saida[i] = estado
    out = pd.Series(saida, index=close.index)
    return out & ma.notna()


@dataclass
class CriterioTendencia:
    nome: str
    ok: bool
    detalhe: str


def _series_tendencia(close: pd.Series) -> dict[str, pd.Series]:
    ma50, ma150, ma200 = (close.rolling(n).mean() for n in (50, 150, 200))
    max52, min52 = close.rolling(252).max(), close.rolling(252).min()
    return {
        "Preço acima da MA50": close > ma50,
        "MA50 acima da MA150": ma50 > ma150,
        "MA150 acima da MA200": ma150 > ma200,
        "MA200 subindo (21 candles)": ma200 > ma200.shift(21),
        "Até 25% abaixo da máxima de 52 sem.": close > max52 * 0.75,
        "Pelo menos 30% acima da mínima de 52 sem.": close > min52 * 1.30,
    }


def serie_tendencia(close: pd.Series) -> pd.Series:
    """1 quando os 6 critérios do modelo de tendência (Minervini/O'Neil) estão satisfeitos."""
    crit = _series_tendencia(close)
    ok = pd.concat(crit.values(), axis=1).all(axis=1)
    return (ok & close.rolling(252).max().notna()).astype(int)


def checklist_tendencia(close: pd.Series) -> list[CriterioTendencia] | None:
    """Situação atual de cada critério; None se não houver 252 candles."""
    if len(close) < 252:
        return None
    return [CriterioTendencia(nome, bool(s.iloc[-1]), "atendido" if bool(s.iloc[-1]) else "não atendido")
            for nome, s in _series_tendencia(close).items()]


def momentum_12_1(close: pd.Series) -> float:
    """Retorno de 12 meses pulando o último mês (252 e 21 candles diários). NaN se faltar histórico."""
    if len(close) < 253:
        return float("nan")
    return float(close.iloc[-22] / close.iloc[-253] - 1)


@dataclass
class PlanoPosicao:
    quantidade: float
    entrada: float
    stop: float
    alvo: float
    risco_valor: float
    valor_posicao: float
    pct_capital: float


def dimensionar_posicao(capital: float, preco: float, atr: float, risco_pct: float = 0.01,
                        mult_atr: float = 2.0, max_pct: float = 0.20,
                        reward_risk: float = 3.0, fracionario: bool = False) -> PlanoPosicao | None:
    """Tamanho de posição por risco fixo: arrisca `risco_pct` do capital até o stop (preço − mult×ATR),
    limitado a `max_pct` do capital em um único ativo.

    `fracionario=True` (cripto) permite frações de unidade (6 casas). Caso contrário usa unidades
    inteiras e devolve None se não couber nem 1.
    """
    dist = mult_atr * atr
    if not (capital > 0 and preco > 0 and dist > 0) or np.isnan([preco, atr]).any():
        return None
    qtd = min(capital * risco_pct / dist, capital * max_pct / preco)
    qtd = float(np.floor(qtd * 1e6) / 1e6) if fracionario else int(qtd)
    if qtd <= 0 or (not fracionario and qtd < 1):
        return None
    return PlanoPosicao(qtd, preco, round(preco - dist, 2), round(preco + reward_risk * dist, 2),
                        qtd * dist, qtd * preco, qtd * preco / capital)
