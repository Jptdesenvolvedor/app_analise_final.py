"""Sinal de decisão (comprar / manter / aguardar / reduzir / vender) e tendência por prazo.

O sinal é a soma ponderada de regras que sobreviveram a backtests longos fora da amostra:

    regime de longo prazo (MA200 com banda 2%) ..... ±40
    modelo de tendência (6 critérios Minervini) .... ±20
    momentum 12-1 ................................. ±10
    tendência de médio prazo (MA50) ............... ±15
    timing de curto prazo (EMA17/72, MACD, RSI) ... ±15

Regras que faltam histórico saem do cálculo e o resultado é reescalado para −100…+100; a confiança cai.
É um sinal automático baseado em regras históricas — não é recomendação de investimento.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .regime import BANDA_PADRAO, checklist_tendencia, estado_defensivo, momentum_12_1

ACOES = {
    "COMPRAR": "alta",
    "MANTER": "alta",
    "AGUARDAR": "neutro",
    "REDUZIR": "baixa",
    "VENDER": "baixa",
}


@dataclass
class Tendencia:
    prazo: str          # "Curto", "Médio", "Longo", "Mercado"
    direcao: str        # "alta" | "baixa" | "lateral" | "indefinida"
    forca: str          # "forte" | "moderada" | "fraca" | ""
    detalhe: str

    @property
    def rotulo(self) -> str:
        if self.direcao == "indefinida":
            return "Indefinida"
        nome = {"alta": "Alta", "baixa": "Baixa", "lateral": "Lateral"}[self.direcao]
        return f"{nome} {self.forca}".strip() if self.forca else nome


@dataclass
class Motivo:
    tom: str            # "alta" | "baixa" | "neutro"
    titulo: str
    texto: str


@dataclass
class Decisao:
    acao: str
    tom: str
    score: float                 # −100 … +100
    confianca: str               # "Alta" | "Média" | "Baixa"
    resumo: str
    tendencias: list[Tendencia] = field(default_factory=list)
    motivos: list[Motivo] = field(default_factory=list)
    stop: float | None = None
    alvo: float | None = None


def _v(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or df.empty:
        return float("nan")
    return float(df[col].iloc[-1])


# --------------------------------------------------------------------------- #
# Tendência por prazo
# --------------------------------------------------------------------------- #
def tendencia_longa(close: pd.Series) -> Tendencia:
    """Preço × MA200 (com banda de 2%) e inclinação da média; 'forte' se o modelo de tendência passa 6/6."""
    if len(close) < 200:
        return Tendencia("Longo", "indefinida", "", "Precisa de 200 candles.")
    ma = close.rolling(200).mean()
    dist = float(close.iloc[-1] / ma.iloc[-1] - 1)
    sobe = len(ma.dropna()) > 21 and ma.iloc[-1] > ma.iloc[-22]
    crit = checklist_tendencia(close)
    forte = crit is not None and all(c.ok for c in crit)
    if dist > BANDA_PADRAO and sobe:
        return Tendencia("Longo", "alta", "forte" if forte else "moderada",
                         f"Preço {dist:+.1%} da MA200, que está subindo.")
    if dist < -BANDA_PADRAO and not sobe:
        return Tendencia("Longo", "baixa", "forte" if dist < -0.10 else "moderada",
                         f"Preço {dist:+.1%} da MA200, que está caindo.")
    return Tendencia("Longo", "lateral", "", f"Preço {dist:+.1%} da MA200 — sem direção definida.")


def tendencia_media(close: pd.Series) -> Tendencia:
    if len(close) < 60:
        return Tendencia("Médio", "indefinida", "", "Precisa de 60 candles.")
    ma = close.rolling(50).mean()
    dist = float(close.iloc[-1] / ma.iloc[-1] - 1)
    sobe = ma.iloc[-1] > ma.iloc[-11]
    if close.iloc[-1] > ma.iloc[-1] and sobe:
        return Tendencia("Médio", "alta", "forte" if dist > 0.05 else "moderada",
                         f"Preço {dist:+.1%} da MA50, que está subindo.")
    if close.iloc[-1] < ma.iloc[-1] and not sobe:
        return Tendencia("Médio", "baixa", "forte" if dist < -0.05 else "moderada",
                         f"Preço {dist:+.1%} da MA50, que está caindo.")
    return Tendencia("Médio", "lateral", "", f"Preço {dist:+.1%} da MA50 — médias sem confirmação.")


def tendencia_curta(df: pd.DataFrame) -> Tendencia:
    e17, e72, macd, sig, close = (_v(df, c) for c in ("EMA17", "EMA72", "MACD", "Signal", "Close"))
    if np.isnan([e17, e72, macd, sig]).any():
        return Tendencia("Curto", "indefinida", "", "Precisa de EMA72 e MACD (≥ 72 candles).")
    ups = int(e17 > e72) + int(macd > sig) + int(close > e17)
    detalhe = f"{ups} de 3 a favor (EMA17>EMA72, MACD>sinal, preço>EMA17)."
    if ups == 3:
        return Tendencia("Curto", "alta", "forte", detalhe)
    if ups == 2:
        return Tendencia("Curto", "alta", "moderada", detalhe)
    if ups == 1:
        return Tendencia("Curto", "baixa", "moderada", detalhe)
    return Tendencia("Curto", "baixa", "forte", detalhe)


def tendencia_mercado(close_bench: pd.Series, nome: str) -> Tendencia:
    """Tendência do mercado geral (benchmark) pela mesma regra de longo prazo."""
    t = tendencia_longa(close_bench)
    return Tendencia(f"Mercado ({nome})", t.direcao, t.forca, t.detalhe)


# --------------------------------------------------------------------------- #
# Decisão
# --------------------------------------------------------------------------- #
def _acao_do_score(score: float) -> str:
    if score >= 50:
        return "COMPRAR"
    if score >= 15:
        return "MANTER"
    if score > -15:
        return "AGUARDAR"
    if score > -50:
        return "REDUZIR"
    return "VENDER"


def calcular_decisao(df: pd.DataFrame) -> Decisao:
    """Sinal, tendências e motivos a partir do DataFrame com indicadores (`calcular_indicadores`)."""
    close = df["Close"]
    preco = float(close.iloc[-1]) if len(close) else float("nan")
    pontos, disponivel, motivos = [], 0.0, []

    def somar(peso: float, valor: float) -> None:
        nonlocal disponivel
        pontos.append(peso * valor)
        disponivel += peso

    # 1) regime de longo prazo
    if len(close) >= 200:
        ma = float(close.rolling(200).mean().iloc[-1])
        investido = bool(estado_defensivo(close).iloc[-1])
        dist = preco / ma - 1
        dentro = abs(dist) <= BANDA_PADRAO
        # dentro da banda o estado é só herdado (sem confirmação): meio peso se investido, zero se em caixa
        valor = (0.5 if investido else 0.0) if dentro else (1.0 if investido else -1.0)
        somar(40, valor)
        estado = "Investido" if investido else "Em caixa"
        nota = " Dentro da banda: estado herdado, sem confirmação." if dentro else ""
        motivos.append(Motivo(
            "alta" if valor > 0 else ("baixa" if valor < 0 else "neutro"), "Regime de longo prazo",
            f"{estado}: preço {dist:+.1%} da MA200 "
            f"(entra acima de +{BANDA_PADRAO:.0%}, sai abaixo de −{BANDA_PADRAO:.0%}).{nota}"))
    else:
        motivos.append(Motivo("neutro", "Regime de longo prazo", "Histórico curto (< 200 candles) — regra principal indisponível."))

    # 2) modelo de tendência
    crit = checklist_tendencia(close)
    if crit is not None:
        ok = sum(c.ok for c in crit)
        somar(20, (ok - 3) / 3)
        motivos.append(Motivo("alta" if ok >= 5 else ("baixa" if ok <= 2 else "neutro"), "Modelo de tendência",
                              f"{ok} de {len(crit)} critérios atendidos (médias alinhadas, perto da máxima de 52 semanas)."))

    # 3) momentum 12-1
    mom = momentum_12_1(close)
    if not np.isnan(mom):
        somar(10, 1.0 if mom > 0 else -1.0)
        motivos.append(Motivo("alta" if mom > 0 else "baixa", "Momentum 12-1",
                              f"{mom:+.1%} em 12 meses (sem o último mês)."))

    # 4) médio prazo e 5) curto prazo
    t_longa, t_media, t_curta = tendencia_longa(close), tendencia_media(close), tendencia_curta(df)
    if t_media.direcao != "indefinida":
        somar(15, {"alta": 1.0, "baixa": -1.0, "lateral": 0.0}[t_media.direcao])
        motivos.append(Motivo({"alta": "alta", "baixa": "baixa", "lateral": "neutro"}[t_media.direcao],
                              "Médio prazo", t_media.detalhe))
    if t_curta.direcao != "indefinida":
        base = {("alta", "forte"): 1.0, ("alta", "moderada"): 0.5,
                ("baixa", "moderada"): -0.5, ("baixa", "forte"): -1.0}[(t_curta.direcao, t_curta.forca)]
        somar(15, base)
        motivos.append(Motivo("alta" if base > 0 else "baixa", "Curto prazo", t_curta.detalhe))

    # RSI: só avisa se esticado (não pesa no score)
    rsi = _v(df, "RSI")
    esticado = not np.isnan(rsi) and rsi > 75
    if esticado:
        motivos.append(Motivo("baixa", "RSI esticado", f"RSI {rsi:.0f} — melhor esperar um recuo para entrar."))
    elif not np.isnan(rsi) and rsi < 30:
        motivos.append(Motivo("alta", "RSI em sobrevenda", f"RSI {rsi:.0f} — possível repique, mas contra a tendência."))

    score = float(np.clip(sum(pontos) / disponivel * 100, -100, 100)) if disponivel else 0.0
    acao = _acao_do_score(score) if disponivel >= 30 else "AGUARDAR"
    if acao == "COMPRAR" and esticado:
        acao = "MANTER"

    # confiança: cobertura dos dados e alinhamento entre as regras
    alinhadas = np.mean([np.sign(p) == np.sign(score) for p in pontos if p != 0]) if any(p != 0 for p in pontos) else 0
    if disponivel >= 85 and alinhadas >= 0.75:
        confianca = "Alta"
    elif disponivel >= 60 and alinhadas >= 0.6:
        confianca = "Média"
    else:
        confianca = "Baixa"

    atr = _v(df, "ATR")
    stop = alvo = None
    if acao in ("COMPRAR", "MANTER") and not np.isnan(atr) and atr > 0:
        stop, alvo = round(preco - 2 * atr, 2), round(preco + 6 * atr, 2)

    return Decisao(acao, ACOES[acao], score, confianca, _resumo(acao, t_longa, t_media, t_curta, disponivel, esticado),
                   [t_longa, t_media, t_curta], motivos, stop, alvo)


def _resumo(acao: str, longa: Tendencia, media: Tendencia, curta: Tendencia, disponivel: float, esticado: bool) -> str:
    if disponivel < 30:
        return "Histórico insuficiente para um sinal confiável — carregue um período maior (diário, 2 anos ou mais)."
    trio = ", ".join(f"{t.prazo.lower()}: {t.rotulo.lower()}" for t in (curta, media, longa))
    frase = {
        "COMPRAR": "Tendência e regime favoráveis; timing de entrada razoável.",
        "MANTER": "Quadro positivo; quem já está posicionado mantém." + (" Esticado: espere um recuo para entrar." if esticado else
                  " Para entrar, o ideal é esperar um recuo ou a confirmação do curto prazo."),
        "AGUARDAR": "Sinais mistos, sem vantagem clara — melhor aguardar.",
        "REDUZIR": "Quadro enfraquecendo; considere reduzir a exposição.",
        "VENDER": "Regime e tendência desfavoráveis; evitar posição comprada.",
    }[acao]
    return f"{frase} ({trio})."
