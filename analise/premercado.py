"""Pré-mercado e after-hours de ações dos EUA (Yahoo Finance).

Sessões de Nova York: pré-mercado 04:00–09:30, regular 09:30–16:00, after-hours 16:00–20:00 (horário de Nova York).
A montagem (`montar_sessao`, `classificar_sessoes`, `historico_premercado`) não depende da rede, para ser testável;
`buscar_sessao` e `buscar_intraday` são a camada fina com o `yfinance`.
Só faz sentido para ações/ETFs dos EUA: B3 tem outro calendário e cripto negocia 24h.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NY = "America/New_York"
BRT = "America/Sao_Paulo"
ROTULOS = {"PRE": "Pré-mercado", "REGULAR": "Mercado aberto", "POST": "After-hours", "FECHADO": "Mercado fechado"}


def suporta(ticker: str) -> bool:
    """Pré-mercado só existe para ações/ETFs dos EUA (sem sufixo de bolsa e sem '-USD' de cripto)."""
    t = ticker.upper()
    return not (t.endswith(".SA") or t.endswith("-USD") or "." in t or t.startswith("^") or "=" in t)


@dataclass
class Sessao:
    estado: str                         # PRE | REGULAR | POST | FECHADO
    preco_regular: float | None         # último preço da sessão regular (fechamento, se já encerrou)
    fechamento_anterior: float | None
    preco_estendido: float | None       # preço do pré-mercado (estado PRE) ou do after-hours (estado POST)
    var_estendida: float | None         # variação (fração) do preço estendido
    referencia: str                     # "fechamento de ontem" | "fechamento de hoje"
    horario: pd.Timestamp | None        # horário do dado estendido, em Brasília

    @property
    def rotulo(self) -> str:
        return ROTULOS[self.estado]

    @property
    def tem_estendido(self) -> bool:
        return self.preco_estendido is not None and self.var_estendida is not None


def _num(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(x) else x


def _estado(info: dict) -> str:
    s = str(info.get("marketState") or "").upper()
    if s in ("PRE", "PREPRE"):
        return "PRE"
    if s == "REGULAR":
        return "REGULAR"
    if s in ("POST", "POSTPOST"):
        return "POST" if s == "POST" else "FECHADO"
    return "FECHADO"


def montar_sessao(info: dict) -> Sessao:
    """Interpreta o `info` do Yahoo. Os campos *ChangePercent do Yahoo vêm em pontos percentuais (−0,05 = −0,05%)."""
    estado = _estado(info)
    regular, anterior = _num(info.get("regularMarketPrice")), _num(info.get("regularMarketPreviousClose"))
    pre, pos = _num(info.get("preMarketPrice")), _num(info.get("postMarketPrice"))
    estendido = var = hora = None
    ref = "fechamento de ontem"
    if estado == "PRE" and pre is not None:
        estendido, ref = pre, "fechamento de ontem"
        v = _num(info.get("preMarketChangePercent"))
        var = v / 100 if v is not None else (pre / regular - 1 if regular else None)
        hora = _num(info.get("preMarketTime"))
    elif estado in ("POST", "FECHADO") and pos is not None:
        estendido, ref = pos, "fechamento de hoje"
        v = _num(info.get("postMarketChangePercent"))
        var = v / 100 if v is not None else (pos / regular - 1 if regular else None)
        hora = _num(info.get("postMarketTime"))
    ts = pd.Timestamp(hora, unit="s", tz="UTC").tz_convert(BRT) if hora else None
    return Sessao(estado, regular, anterior, estendido, var, ref, ts)


def classificar_sessoes(hist: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta a coluna `sessao` (PRE | REGULAR | POST) e converte o índice para Nova York."""
    h = hist.copy()
    idx = pd.DatetimeIndex(h.index)
    h.index = idx.tz_localize("UTC").tz_convert(NY) if idx.tz is None else idx.tz_convert(NY)
    minutos = h.index.hour * 60 + h.index.minute
    h["sessao"] = np.select([minutos < 9 * 60 + 30, minutos < 16 * 60], ["PRE", "REGULAR"], default="POST")
    return h


def historico_premercado(hist: pd.DataFrame) -> pd.DataFrame:
    """Por dia: variação do pré-mercado, abertura, dia regular e se o pré-mercado apontou a direção do dia.

    gap_pre   = último preço do pré-mercado / fechamento regular do dia anterior − 1
    gap_aber  = abertura da sessão regular / fechamento anterior − 1
    dia       = fechamento regular / fechamento anterior − 1
    acertou   = sinal de gap_pre == sinal de dia
    """
    h = classificar_sessoes(hist)
    h["dia"] = h.index.date
    linhas, fech_ant = [], None
    for dia, g in h.groupby("dia"):
        reg, pre = g[g["sessao"] == "REGULAR"], g[g["sessao"] == "PRE"]
        if reg.empty:
            continue
        fech, abert = float(reg["Close"].iloc[-1]), float(reg["Open"].iloc[0])
        if fech_ant is not None and not pre.empty:
            gp = float(pre["Close"].iloc[-1]) / fech_ant - 1
            dia_ret = fech / fech_ant - 1
            linhas.append({"data": pd.Timestamp(dia), "gap_pre": gp, "gap_aber": abert / fech_ant - 1,
                           "dia": dia_ret, "acertou": bool(np.sign(gp) == np.sign(dia_ret)) if gp != 0 and dia_ret != 0 else np.nan})
        fech_ant = fech
    return pd.DataFrame(linhas, columns=["data", "gap_pre", "gap_aber", "dia", "acertou"]).sort_values(
        "data", ascending=False).reset_index(drop=True)


def taxa_de_acerto(h: pd.DataFrame, gap_minimo: float = 0.0) -> tuple[float | None, int]:
    """(% dos dias em que o sinal do pré-mercado coincidiu com o do dia, nº de dias), para |gap| ≥ `gap_minimo`."""
    s = h[h["gap_pre"].abs() >= gap_minimo]["acertou"].dropna()
    return (float(s.mean()), len(s)) if len(s) else (None, 0)


def ultimas_sessoes(hist: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    """Candles dos últimos `n` dias de negociação, já classificados por sessão (para o gráfico)."""
    h = classificar_sessoes(hist)
    dias = sorted(set(h.index.date))[-n:]
    return h[[d in dias for d in h.index.date]]


def fechamento_anterior_do_ultimo_dia(hist: pd.DataFrame) -> float | None:
    """Último preço da sessão regular do dia anterior ao último dia presente nos dados (linha de referência)."""
    h = classificar_sessoes(hist)
    dias = sorted(set(h.index.date))
    if len(dias) < 2:
        return None
    ant = h[(h.index.date == dias[-2]) & (h["sessao"] == "REGULAR")]
    return float(ant["Close"].iloc[-1]) if not ant.empty else None


# ---------------------------------------------------------------- rede (yfinance)
def buscar_sessao(ticker: str) -> Sessao | None:
    import yfinance as yf

    try:
        return montar_sessao(yf.Ticker(ticker).info or {})
    except Exception:
        return None


DIAS_INTRADAY = 57   # o Yahoo só guarda ~60 dias de candles de 5 min; margem para não estourar o limite


def inicio_intraday(agora=None, dias: int = DIAS_INTRADAY) -> str:
    """Data inicial (AAAA-MM-DD) da busca de candles de 5 min.

    Usa `start=` e não `period=`: com `period="60d"` o Yahoo devolve os dados só até o último pregão
    encerrado e deixa de fora o pré-mercado do dia em andamento, justamente o que a aba precisa mostrar."""
    t = pd.Timestamp.now() if agora is None else pd.Timestamp(agora)
    return (t - pd.Timedelta(days=dias)).strftime("%Y-%m-%d")


def buscar_intraday(ticker: str) -> pd.DataFrame:
    """Candles de 5 minutos com horário estendido; vazio se o Yahoo não responder."""
    import yfinance as yf

    try:
        h = yf.Ticker(ticker).history(start=inicio_intraday(), interval="5m", prepost=True)
    except Exception:
        return pd.DataFrame()
    return h if h is not None and not h.empty else pd.DataFrame()
