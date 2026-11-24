"""Base local de dados (Alpaca SIP + notícias Alpaca + RI/balanços da SEC), usada como fonte opcional.

Estrutura esperada em `ANALISE_BASE_LOCAL` (padrão: E:/Robo Trade 1), gerada pelo Robo Trade:

    data_sip/<TICKER>.csv          candles diários (timestamp,open,high,low,close,volume), ~10 anos
    news/<TICKER>.csv              notícias (created_at,headline,summary,source)
    edgar/<TICKER>_filings.csv     comunicados 8-K/10-Q/10-K (data e hora de aceite na SEC)
    edgar/<TICKER>_facts.csv       balanços XBRL (metric,end,start,val,filed,form...)

Sem a pasta, `disponivel()` é False e o app segue só com o Yahoo (é o caso do Streamlit Cloud).
Sem dependência do Streamlit, para ser testável.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import periodo_em_dias
from .data import DadosIndisponiveis

RAIZ_PADRAO = "E:/Robo Trade 1"
INTERVALOS_SUPORTADOS = {"1d": None, "1wk": "W-FRI", "1mo": "ME"}

ITENS_8K = {
    "1.01": "Contrato relevante", "1.02": "Fim de contrato", "2.01": "Aquisição ou venda de ativos",
    "2.02": "Resultados trimestrais", "2.03": "Nova dívida", "2.05": "Reestruturação",
    "2.06": "Redução de valor de ativos", "3.01": "Aviso de listagem", "3.02": "Venda de ações não registrada",
    "4.01": "Troca de auditor", "4.02": "Balanços anteriores não confiáveis", "5.01": "Mudança de controle",
    "5.02": "Mudança de diretoria", "5.03": "Alteração de estatuto", "5.07": "Votação de acionistas",
    "7.01": "Divulgação (Reg FD)", "8.01": "Outros eventos", "9.01": "Anexos e demonstrações",
}
TIPOS = {"8-K": "Fato relevante (8-K)", "10-Q": "Balanço trimestral (10-Q)", "10-K": "Balanço anual (10-K)",
         "10-Q/A": "Balanço trimestral (retificação)", "10-K/A": "Balanço anual (retificação)"}


# --------------------------------------------------------------------------- #
# Localização
# --------------------------------------------------------------------------- #
def raiz() -> Path | None:
    """Pasta da base local, ou None se não existir/estiver vazia."""
    p = Path(os.getenv("ANALISE_BASE_LOCAL", RAIZ_PADRAO))
    return p if (p / "data_sip").is_dir() else None


def disponivel() -> bool:
    return raiz() is not None


def _arquivo(sub: str, nome: str) -> Path | None:
    r = raiz()
    p = r / sub / nome if r else None
    return p if p is not None and p.is_file() else None


def ticker_base(ticker: str) -> str:
    """Ticker no formato da base (Alpaca): sem '.SA', sem '-USD'; só ações dos EUA existem lá."""
    return ticker.upper().strip()


def tem_candles(ticker: str) -> bool:
    return _arquivo("data_sip", f"{ticker_base(ticker)}.csv") is not None


def tem_ri(ticker: str) -> bool:
    return _arquivo("edgar", f"{ticker_base(ticker)}_filings.csv") is not None


def tem_noticias(ticker: str) -> bool:
    return _arquivo("news", f"{ticker_base(ticker)}.csv") is not None


def _ler(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:   # empresa estrangeira sem us-gaap: arquivo vazio
        return pd.DataFrame()


# --------------------------------------------------------------------------- #
# Candles
# --------------------------------------------------------------------------- #
def candles_diarios(ticker: str) -> pd.DataFrame:
    """Candles diários completos no formato do app (Open/High/Low/Close/Volume)."""
    p = _arquivo("data_sip", f"{ticker_base(ticker)}.csv")
    if p is None:
        raise DadosIndisponiveis(f"'{ticker}' não está na base local (só ações/ETFs dos EUA).")
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    df.columns = [c.capitalize() for c in df.columns]
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index().dropna(subset=["Close"])
    if df.empty:
        raise DadosIndisponiveis(f"Base local sem candles para '{ticker}'.")
    return df


def baixar_local(ticker: str, periodo: str, intervalo_real: str) -> pd.DataFrame:
    """Mesma assinatura de `data.baixar_dados`, lendo da base local (diário, semanal ou mensal)."""
    if intervalo_real not in INTERVALOS_SUPORTADOS:
        raise DadosIndisponiveis("A base local só tem candles diários (1d, 1wk e 1mo).")
    df = candles_diarios(ticker)
    corte = df.index[-1] - pd.Timedelta(days=periodo_em_dias(periodo))
    df = df[df.index >= corte]
    regra = INTERVALOS_SUPORTADOS[intervalo_real]
    if not regra:
        return df
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    return df.resample(regra).agg(agg).dropna(subset=["Close"])   # semanas/meses de calendário


# --------------------------------------------------------------------------- #
# Notícias
# --------------------------------------------------------------------------- #
def noticias(ticker: str, limite: int = 100) -> pd.DataFrame:
    """Notícias mais recentes (colunas: data, titulo, fonte, resumo). Vazio se não houver."""
    df = _ler(_arquivo("news", f"{ticker_base(ticker)}.csv"))
    if df.empty:
        return pd.DataFrame(columns=["data", "titulo", "fonte", "resumo"])
    df["data"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["data"]).drop_duplicates("id").sort_values("data", ascending=False)
    out = df.rename(columns={"headline": "titulo", "source": "fonte", "summary": "resumo"})
    return out[["data", "titulo", "fonte", "resumo"]].head(limite).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# RI: comunicados, resultados e reação do preço
# --------------------------------------------------------------------------- #
def _descricao_itens(itens) -> str:
    if pd.isna(itens) or not str(itens).strip():
        return ""
    nomes = [ITENS_8K.get(i.strip(), f"Item {i.strip()}") for i in str(itens).split(",")]
    # 9.01 (anexos) só aparece junto de outro assunto: esconde se houver algo mais informativo
    uteis = [n for n in nomes if n != ITENS_8K["9.01"]] or nomes
    return "; ".join(uteis)


def comunicados(ticker: str, limite: int = 40) -> pd.DataFrame:
    """Comunicados à SEC (colunas: quando, tipo, assunto, resultado), do mais recente ao mais antigo.

    `quando` é o momento de aceite na SEC em horário de Brasília; `resultado` marca o 8-K de resultados (item 2.02).
    """
    df = _ler(_arquivo("edgar", f"{ticker_base(ticker)}_filings.csv"))
    if df.empty:
        return pd.DataFrame(columns=["quando", "tipo", "assunto", "resultado"])
    df["quando"] = pd.to_datetime(df["acceptanceDateTime"], utc=True, errors="coerce").dt.tz_convert("America/Sao_Paulo")
    df = df.dropna(subset=["quando"]).sort_values("quando", ascending=False)
    df["tipo"] = df["form"].map(TIPOS).fillna(df["form"])
    df["assunto"] = df["items"].map(_descricao_itens)
    df.loc[df["form"].isin(["10-Q", "10-K"]), "assunto"] = "Demonstrações financeiras do período"
    df["resultado"] = df["items"].astype(str).str.contains("2.02", regex=False) & (df["form"] == "8-K")
    return df[["quando", "tipo", "assunto", "resultado"]].head(limite).reset_index(drop=True)


def _trimestrais(facts: pd.DataFrame, metrica: str) -> pd.Series:
    """Valor de cada trimestre (~3 meses) pela PRIMEIRA publicação, indexado pela data de fim do período."""
    if facts.empty or "metric" not in facts:
        return pd.Series(dtype=float)
    x = facts[facts["metric"] == metrica].dropna(subset=["start", "end", "filed"]).copy()
    for c in ("start", "end", "filed"):
        x[c] = pd.to_datetime(x[c])
    x = x[(x["end"] - x["start"]).dt.days.between(80, 100)]
    x = x.sort_values("filed").drop_duplicates("end", keep="first").sort_values("end")
    s = pd.Series(x["val"].to_numpy(float), index=pd.DatetimeIndex(x["end"]))
    s.attrs["publicado"] = pd.Series(pd.DatetimeIndex(x["filed"]), index=s.index)
    return s


def _cresc_aa(s: pd.Series, eps: bool) -> pd.Series:
    """Crescimento contra o mesmo trimestre do ano anterior (LPA usa denominador mínimo para não explodir)."""
    if s.empty:
        return s
    ant = s.reindex(s.index - pd.DateOffset(years=1), method="nearest", tolerance=pd.Timedelta(days=12))
    ant.index = s.index
    den = ant.abs().clip(lower=0.10) if eps else ant.abs().replace(0, np.nan)
    return (s - ant) / den


def resultados_trimestrais(ticker: str, n: int = 8) -> pd.DataFrame:
    """Últimos `n` trimestres: receita, lucro líquido, LPA diluído e crescimento contra o ano anterior."""
    facts = _ler(_arquivo("edgar", f"{ticker_base(ticker)}_facts.csv"))
    rec, lucro, lpa = (_trimestrais(facts, m) for m in ("revenue", "net_income", "eps_diluted"))
    if lpa.empty and rec.empty:
        return pd.DataFrame()
    base = (lpa if not lpa.empty else rec).index
    pub = (lpa if not lpa.empty else rec).attrs["publicado"]
    out = pd.DataFrame({
        "trimestre": base, "publicado": pub.to_numpy(),
        "receita": rec.reindex(base).to_numpy(), "lucro_liquido": lucro.reindex(base).to_numpy(),
        "lpa": lpa.reindex(base).to_numpy(),
        "cresc_receita": _cresc_aa(rec, eps=False).reindex(base).to_numpy(),
        "cresc_lpa": _cresc_aa(lpa, eps=True).reindex(base).to_numpy(),
    })
    return out.sort_values("trimestre", ascending=False).head(n).reset_index(drop=True)


def reacao_a_resultados(ticker: str, n: int = 8) -> pd.DataFrame:
    """Como o preço reagiu aos últimos resultados (8-K item 2.02).

    Resultado divulgado após as 16h de Nova York → a reação é no pregão seguinte. Retorna: data da divulgação,
    dia da reação, variação do fechamento anterior ao fechamento da reação e volume em múltiplos da média de 50 dias.
    """
    cm = comunicados(ticker, limite=400)
    cm = cm[cm["resultado"]]
    if cm.empty or not tem_candles(ticker):
        return pd.DataFrame(columns=["divulgado", "reacao_em", "variacao", "volume_x"])
    df = candles_diarios(ticker)
    idx, close, vol = df.index, df["Close"].to_numpy(), df["Volume"].to_numpy(float)
    linhas = []
    for quando in cm["quando"]:
        ny = quando.tz_convert("America/New_York")
        i = int(idx.searchsorted(pd.Timestamp(ny.date()), side="left"))
        if ny.hour >= 16:
            i += 1
        if i < 51 or i >= len(idx):
            continue
        media = vol[i - 50:i].mean()
        linhas.append({"divulgado": quando, "reacao_em": idx[i], "variacao": close[i] / close[i - 1] - 1,
                       "volume_x": vol[i] / media if media > 0 else np.nan})
    return pd.DataFrame(linhas, columns=["divulgado", "reacao_em", "variacao", "volume_x"]).head(n)
