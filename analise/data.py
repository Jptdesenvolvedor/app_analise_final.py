"""Download e preparação dos dados (sem dependência do Streamlit, para ser testável)."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import yfinance as yf

from .config import periodo_em_dias


class DadosIndisponiveis(Exception):
    """Levantada quando o Yahoo não retorna dados (assim o resultado não é cacheado)."""


def normalizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    """Achata colunas MultiIndex (campo, ticker) e limpa linhas inválidas."""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    if "Close" not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=["Close"])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _download(ticker: str, **kwargs) -> pd.DataFrame:
    df = yf.download(
        ticker, auto_adjust=False, progress=False, threads=False, **kwargs
    )
    return normalizar_colunas(df)


def baixar_dados(
    ticker: str, periodo: str, intervalo_real: str, tentativas: int = 3
) -> pd.DataFrame:
    """Baixa OHLCV com retentativas e fallback por start/end.

    Levanta `DadosIndisponiveis` se nada for retornado.
    """
    ultimo_erro: Optional[Exception] = None

    for tentativa in range(1, tentativas + 1):
        try:
            df = _download(ticker, period=periodo, interval=intervalo_real)
            if not df.empty:
                return df
            break  # vazio sem exceção: tentar de novo não adianta
        except Exception as e:  # falhas de rede/rate-limit são transitórias
            ultimo_erro = e
            if tentativa < tentativas:
                time.sleep(1.5 * tentativa)

    try:
        fim = datetime.now(timezone.utc)
        inicio = fim - timedelta(days=periodo_em_dias(periodo))
        df = _download(ticker, start=inicio, end=fim, interval=intervalo_real)
        if not df.empty:
            return df
    except Exception as e:
        ultimo_erro = e

    if ultimo_erro is not None:
        raise DadosIndisponiveis(
            f"Falha ao consultar o Yahoo Finance para '{ticker}': {ultimo_erro}"
        )
    raise DadosIndisponiveis(f"Nenhum dado encontrado para '{ticker}'.")


def resample_ohlcv(df: pd.DataFrame, rule: str, cripto: bool = False) -> pd.DataFrame:
    """Reamostra OHLCV.

    Ações: o agrupamento começa no primeiro candle (`origin='start'`), assim as
    velas de 4h acompanham a sessão (ex.: 10h–14h/14h–18h na B3). Cripto negocia
    24h, então usa a grade UTC (`origin='epoch'`).
    """
    if df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last",
           "Adj Close": "last", "Volume": "sum"}
    agg = {k: v for k, v in agg.items() if k in df.columns}
    origin = "epoch" if cripto else "start"
    out = df.resample(rule, origin=origin).agg(agg)
    return out.dropna(subset=["Close"])


def normalizar_base100(series: pd.Series) -> pd.Series:
    """Série normalizada (primeiro valor = 100), usada na comparação de ativos."""
    s = series.dropna()
    return s / s.iloc[0] * 100 if len(s) else s
