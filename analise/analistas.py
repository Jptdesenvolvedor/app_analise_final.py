"""Preço-alvo e recomendações dos analistas (consenso do Yahoo Finance).

A montagem dos números (`montar_analistas`) é independente da rede, para ser testável; `buscar_analistas` é a
camada fina que consulta o `yfinance`. Cobre ações dos EUA e várias do Brasil; cripto e muitos ETFs não têm cobertura.

ATENÇÃO: preços-alvo de analistas são, em geral, otimistas e raramente acertam o preço em 12 meses. O app os mostra como
termômetro de consenso e NÃO os usa no sinal de decisão (não há histórico para validar isso).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ROTULO_RECOMENDACAO = {
    "strong_buy": "Compra forte", "buy": "Compra", "hold": "Neutro",
    "underperform": "Desempenho abaixo", "sell": "Venda", "strong_sell": "Venda forte",
    "none": "Sem recomendação",
}
NOMES_GRUPOS = (("strongBuy", "Compra forte"), ("buy", "Compra"), ("hold", "Neutro"),
                ("sell", "Venda"), ("strongSell", "Venda forte"))
ACAO_ROTULO = {"up": "Elevou a recomendação", "down": "Rebaixou a recomendação", "init": "Iniciou cobertura",
               "main": "Manteve", "reit": "Reiterou"}
ALVO_ROTULO = {"Raises": "Subiu o alvo", "Lowers": "Baixou o alvo", "Maintains": "Manteve o alvo",
               "Announces": "Definiu o alvo", "Initiates": "Iniciou"}


@dataclass
class Analistas:
    preco_atual: float | None
    media: float | None
    mediana: float | None
    alto: float | None
    baixo: float | None
    n: int | None
    recomendacao: str | None              # rótulo em português
    nota: float | None                    # 1 = compra forte … 5 = venda forte
    distribuicao: dict[str, int] = field(default_factory=dict)       # mês atual
    distribuicao_anterior: dict[str, int] = field(default_factory=dict)   # ~3 meses antes
    revisoes: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(
        columns=["data", "casa", "de", "para", "acao", "alvo_atual", "alvo_anterior", "alvo_acao"]))

    # ---- derivados -------------------------------------------------------- #
    def _up(self, alvo: float | None) -> float | None:
        if alvo is None or not self.preco_atual:
            return None
        return alvo / self.preco_atual - 1

    @property
    def upside_media(self) -> float | None:
        return self._up(self.media)

    @property
    def upside_mediana(self) -> float | None:
        return self._up(self.mediana)

    @property
    def upside_alto(self) -> float | None:
        return self._up(self.alto)

    @property
    def upside_baixo(self) -> float | None:
        return self._up(self.baixo)

    @property
    def dispersao(self) -> float | None:
        """(alto − baixo) / média: quanto os analistas divergem entre si."""
        if None in (self.alto, self.baixo, self.media) or not self.media:
            return None
        return (self.alto - self.baixo) / self.media

    @property
    def pct_compra(self) -> float | None:
        total = sum(self.distribuicao.values())
        return (self.distribuicao.get("Compra forte", 0) + self.distribuicao.get("Compra", 0)) / total if total else None

    def tendencia_alvos(self, dias: int = 90) -> dict[str, int]:
        """Revisões de preço-alvo nos últimos `dias`: quantos subiram, baixaram e a variação mediana."""
        r = self.revisoes
        if r.empty:
            return {"subiu": 0, "baixou": 0}
        recente = r[r["data"] >= r["data"].max() - pd.Timedelta(days=dias)]
        return {"subiu": int((recente["alvo_acao"] == "Raises").sum()),
                "baixou": int((recente["alvo_acao"] == "Lowers").sum())}

    @property
    def leitura(self) -> str:
        """Frase curta e honesta sobre o consenso."""
        partes = []
        u = self.upside_media
        if u is not None:
            if u >= 0.10:
                partes.append(f"alvo médio {u:+.0%} acima do preço")
            elif u <= -0.05:
                partes.append(f"preço já acima do alvo médio ({u:+.0%})")
            else:
                partes.append("preço perto do alvo médio")
        d = self.dispersao
        if d is not None and d > 0.6:
            partes.append("opiniões muito divergentes")
        t = self.tendencia_alvos()
        if t["subiu"] + t["baixou"] >= 3:
            partes.append("alvos sendo revisados para cima" if t["subiu"] > t["baixou"] else
                          "alvos sendo revisados para baixo" if t["baixou"] > t["subiu"] else "revisões divididas")
        return "; ".join(partes).capitalize() + "." if partes else ""


def _num(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(x) or x <= 0 else x


def _linha_distribuicao(resumo: pd.DataFrame, periodo: str) -> dict[str, int]:
    if resumo is None or resumo.empty or "period" not in resumo:
        return {}
    linha = resumo[resumo["period"] == periodo]
    if linha.empty:
        return {}
    r = linha.iloc[0]
    return {nome: int(r.get(chave, 0) or 0) for chave, nome in NOMES_GRUPOS}


def montar_analistas(info: dict, alvos: dict | None, resumo: pd.DataFrame | None,
                     revisoes: pd.DataFrame | None, max_revisoes: int = 60) -> Analistas | None:
    """Monta o objeto a partir dos dicionários/tabelas do Yahoo. None se não houver nenhum preço-alvo."""
    alvos = alvos or {}
    media = _num(alvos.get("mean", info.get("targetMeanPrice")))
    mediana = _num(alvos.get("median", info.get("targetMedianPrice")))
    alto = _num(alvos.get("high", info.get("targetHighPrice")))
    baixo = _num(alvos.get("low", info.get("targetLowPrice")))
    if media is None and mediana is None and alto is None and baixo is None:
        return None
    n = info.get("numberOfAnalystOpinions")
    chave = info.get("recommendationKey")
    nota = _num(info.get("recommendationMean"))

    tabela = pd.DataFrame(columns=["data", "casa", "de", "para", "acao", "alvo_atual", "alvo_anterior", "alvo_acao"])
    if revisoes is not None and len(revisoes):
        r = revisoes.reset_index().rename(columns={
            revisoes.index.name or "index": "data", "Firm": "casa", "FromGrade": "de", "ToGrade": "para",
            "Action": "acao", "currentPriceTarget": "alvo_atual", "priorPriceTarget": "alvo_anterior",
            "priceTargetAction": "alvo_acao"})
        r["data"] = pd.to_datetime(r["data"], errors="coerce")
        r = r.dropna(subset=["data"]).sort_values("data", ascending=False).head(max_revisoes)
        for c in ("alvo_atual", "alvo_anterior"):
            r[c] = r[c].map(_num)                      # 0,0 = "sem alvo anterior" → None
        tabela = r.reindex(columns=tabela.columns).reset_index(drop=True)

    return Analistas(
        preco_atual=_num(alvos.get("current", info.get("currentPrice"))), media=media, mediana=mediana, alto=alto,
        baixo=baixo, n=int(n) if n else None,
        recomendacao=ROTULO_RECOMENDACAO.get(str(chave).lower(), None) if chave else None,
        nota=nota, distribuicao=_linha_distribuicao(resumo, "0m"),
        distribuicao_anterior=_linha_distribuicao(resumo, "-3m"), revisoes=tabela)


def buscar_analistas(ticker: str) -> Analistas | None:
    """Consulta o Yahoo; None se o ativo não tem cobertura de analistas ou a consulta falha."""
    import yfinance as yf

    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
    except Exception:
        return None
    partes = {}
    for nome, getter in (("alvos", lambda: tk.analyst_price_targets), ("resumo", lambda: tk.recommendations_summary),
                         ("revisoes", lambda: tk.upgrades_downgrades)):
        try:
            partes[nome] = getter()
        except Exception:
            partes[nome] = None
    try:
        return montar_analistas(info, dict(partes["alvos"] or {}), partes["resumo"], partes["revisoes"])
    except Exception:
        return None
