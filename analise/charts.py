"""Gráficos interativos (Plotly)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .backtest import ResultadoBacktest
from .indicators import Fibonacci

ALTA, BAIXA = "#26a69a", "#ef5350"


@dataclass
class OpcoesGrafico:
    candles: bool = True
    medias: bool = True       # MA21 / MA200
    emas_phi: bool = True     # EMA17 / 72 / 305
    bollinger: bool = False
    vwap: bool = False
    fibonacci: bool = True
    suporte_resistencia: bool = True
    estocastico: bool = False
    volume_financeiro: bool = False


def _sem_tz(df: pd.DataFrame) -> pd.DataFrame:
    """Plotly ignora o fuso; usar o horário local da bolsa (sem tz) evita deslocamentos."""
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    return df


def _rangebreaks(df: pd.DataFrame, cripto: bool, intraday_horario: bool) -> list[dict]:
    """Esconde fins de semana (e horas fora do pregão em ≤1h) para não deixar buracos."""
    if cripto or df.empty:
        return []
    quebras: list[dict] = [dict(bounds=["sat", "mon"])]
    if intraday_horario:
        h = df.index.hour
        ini, fim = int(h.min()), int(h.max()) + 1
        if ini > 0 and fim < 24:
            quebras.append(dict(pattern="hour", bounds=[fim, ini]))
    return quebras


def grafico_principal(
    df: pd.DataFrame,
    titulo: str,
    opcoes: OpcoesGrafico,
    fib: Fibonacci | None = None,
    sr: dict | None = None,
    cripto: bool = False,
    intraday_horario: bool = False,
) -> go.Figure:
    df = _sem_tz(df)
    x = df.index

    linhas = ["preco", "volume", "rsi", "macd"] + (["stoch"] if opcoes.estocastico else [])
    pesos = {"preco": 0.46, "volume": 0.12, "rsi": 0.14, "macd": 0.14, "stoch": 0.14}
    fig = make_subplots(
        rows=len(linhas), cols=1, shared_xaxes=True, vertical_spacing=0.02,
        row_heights=[pesos[k] for k in linhas],
    )
    r = {k: i + 1 for i, k in enumerate(linhas)}

    # --- Preço ---
    if opcoes.candles and {"Open", "High", "Low"}.issubset(df.columns):
        fig.add_trace(go.Candlestick(
            x=x, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name="Preço", increasing_line_color=ALTA, decreasing_line_color=BAIXA,
        ), row=r["preco"], col=1)
    else:
        fig.add_trace(go.Scatter(x=x, y=df["Close"], name="Preço", line=dict(color="#222", width=1.6)),
                      row=r["preco"], col=1)

    def linha(col, nome, **estilo):
        if col in df.columns and df[col].notna().any():
            fig.add_trace(go.Scatter(x=x, y=df[col], name=nome, line=dict(width=1.2, **estilo)),
                          row=r["preco"], col=1)

    if opcoes.medias:
        linha("MA21", "MA21", color="#f9a825")
        linha("MA200", "MA200", color="#6d4c41")
    if opcoes.emas_phi:
        linha("EMA17", "EMA17 (Phi)", color="#1e88e5", dash="dash")
        linha("EMA72", "EMA72 (Phi)", color="#8e24aa", dash="dash")
        linha("EMA305", "EMA305 (Phi)", color="#e53935", dash="dash")
    if opcoes.bollinger:
        linha("BB_Sup", "Bollinger sup.", color="#90a4ae", dash="dot")
        linha("BB_Inf", "Bollinger inf.", color="#90a4ae", dash="dot")
    if opcoes.vwap:
        linha("VWAP", "VWAP (sessão)", color="#00897b", dash="dashdot")

    if opcoes.fibonacci and fib:
        for lvl, val in fib.niveis.items():
            fig.add_hline(y=val, line=dict(dash="dot", width=1, color="rgba(120,120,120,.6)"),
                          annotation_text=f"Fib {lvl}", annotation_position="top left",
                          annotation_font_size=9, row=r["preco"], col=1)
    if opcoes.suporte_resistencia and sr:
        for nivel, toques in sr.get("suportes", []):
            fig.add_hline(y=nivel, line=dict(width=1.2, color=ALTA),
                          annotation_text=f"S {nivel:,.2f} ({toques}x)", annotation_position="bottom right",
                          annotation_font_size=9, row=r["preco"], col=1)
        for nivel, toques in sr.get("resistencias", []):
            fig.add_hline(y=nivel, line=dict(width=1.2, color=BAIXA),
                          annotation_text=f"R {nivel:,.2f} ({toques}x)", annotation_position="top right",
                          annotation_font_size=9, row=r["preco"], col=1)

    # --- Volume ---
    col_vol = "Volume_Financeiro" if opcoes.volume_financeiro else "Volume"
    cores = [ALTA if c >= o else BAIXA for o, c in zip(df.get("Open", df["Close"]), df["Close"])]
    fig.add_trace(go.Bar(x=x, y=df[col_vol], marker_color=cores, name=col_vol.replace("_", " "),
                         showlegend=False), row=r["volume"], col=1)
    fig.update_yaxes(title_text="Vol. financeiro" if opcoes.volume_financeiro else "Volume",
                     row=r["volume"], col=1)

    # --- RSI ---
    fig.add_trace(go.Scatter(x=x, y=df["RSI"], name="RSI (14)", line=dict(color="#7e57c2", width=1.2)),
                  row=r["rsi"], col=1)
    for nivel, cor in ((70, BAIXA), (30, ALTA)):
        fig.add_hline(y=nivel, line=dict(dash="dash", width=1, color=cor), row=r["rsi"], col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=r["rsi"], col=1)

    # --- MACD ---
    hist_cor = [ALTA if (v or 0) >= 0 else BAIXA for v in df["MACD_Hist"]]
    fig.add_trace(go.Bar(x=x, y=df["MACD_Hist"], marker_color=hist_cor, name="Histograma", opacity=0.5),
                  row=r["macd"], col=1)
    fig.add_trace(go.Scatter(x=x, y=df["MACD"], name="MACD", line=dict(color="#1e88e5", width=1.2)),
                  row=r["macd"], col=1)
    fig.add_trace(go.Scatter(x=x, y=df["Signal"], name="Sinal", line=dict(color="#fb8c00", width=1.2)),
                  row=r["macd"], col=1)
    fig.update_yaxes(title_text="MACD", row=r["macd"], col=1)

    # --- Estocástico ---
    if opcoes.estocastico:
        fig.add_trace(go.Scatter(x=x, y=df["Stoch_K"], name="%K", line=dict(color="#1e88e5", width=1.2)),
                      row=r["stoch"], col=1)
        fig.add_trace(go.Scatter(x=x, y=df["Stoch_D"], name="%D", line=dict(color="#fb8c00", width=1.2)),
                      row=r["stoch"], col=1)
        for nivel, cor in ((80, BAIXA), (20, ALTA)):
            fig.add_hline(y=nivel, line=dict(dash="dash", width=1, color=cor), row=r["stoch"], col=1)
        fig.update_yaxes(title_text="Estoc.", range=[0, 100], row=r["stoch"], col=1)

    fig.update_xaxes(rangebreaks=_rangebreaks(df, cripto, intraday_horario), rangeslider_visible=False)
    fig.update_layout(
        title=titulo, height=420 + 170 * (len(linhas) - 1), hovermode="x unified",
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font=dict(size=10)),
    )
    return fig


def grafico_comparacao(series: dict[str, pd.Series]) -> go.Figure:
    fig = go.Figure()
    for nome, s in series.items():
        if getattr(s.index, "tz", None) is not None:
            s = s.copy()
            s.index = s.index.tz_localize(None)
        fig.add_trace(go.Scatter(x=s.index, y=s, name=nome, mode="lines"))
    fig.add_hline(y=100, line=dict(dash="dot", width=1, color="gray"))
    fig.update_layout(title="Desempenho relativo (base 100)", hovermode="x unified", height=450,
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


def grafico_backtest(res: ResultadoBacktest, nome: str) -> go.Figure:
    fig = go.Figure()
    for serie, rotulo, cor in ((res.curva_estrategia, nome, "#1e88e5"),
                               (res.curva_buy_hold, "Buy & hold", "#9e9e9e")):
        s = serie
        if getattr(s.index, "tz", None) is not None:
            s = s.copy()
            s.index = s.index.tz_localize(None)
        fig.add_trace(go.Scatter(x=s.index, y=s, name=rotulo, line=dict(color=cor)))
    fig.update_layout(title="Curva de capital (início = 1,0)", hovermode="x unified", height=400,
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig
