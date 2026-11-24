"""Componentes visuais (HTML/CSS) do painel de decisão. Sem dependência do Streamlit, para ser testável.

Cores são translúcidas e o texto herda a cor do tema, então funcionam em tema claro e escuro.
"""

from __future__ import annotations

from html import escape

from .analistas import Analistas
from .premercado import Sessao
from .sinal import Decisao, Tendencia

COR = {"alta": "#16a34a", "baixa": "#dc2626", "neutro": "#6b7280", "aviso": "#d97706"}
ACAO_COR = {"COMPRAR": COR["alta"], "MANTER": "#65a30d", "AGUARDAR": COR["neutro"],
            "REDUZIR": COR["aviso"], "VENDER": COR["baixa"]}
ACAO_ICONE = {"COMPRAR": "▲", "MANTER": "●", "AGUARDAR": "◆", "REDUZIR": "▼", "VENDER": "▼"}
SETA = {"alta": "↗", "baixa": "↘", "lateral": "→", "indefinida": "·"}
TOM_ICONE = {"alta": "🟢", "baixa": "🔴", "neutro": "⚪"}

CSS = """
<style>
.rt-card{border:1px solid rgba(128,128,128,.28);border-radius:16px;padding:18px 20px;margin-bottom:10px;
  background:linear-gradient(135deg,rgba(128,128,128,.07),rgba(128,128,128,.02));}
.rt-top{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:baseline;gap:6px 16px;}
.rt-nome{font-size:1.05rem;font-weight:600;opacity:.9}
.rt-preco{font-size:1.5rem;font-weight:700}
.rt-var{font-size:.95rem;font-weight:600;margin-left:6px}
.rt-sub{font-size:.8rem;opacity:.65;margin-top:2px}
.rt-badge{display:inline-flex;align-items:center;gap:10px;border-radius:14px;padding:6px 20px;margin:14px 0 6px;
  font-size:2.1rem;font-weight:800;letter-spacing:.5px;line-height:1.2}
.rt-meta{font-size:.85rem;opacity:.75;margin-left:4px}
.rt-resumo{font-size:.98rem;line-height:1.45;margin:8px 0 4px}
.rt-bar{position:relative;display:flex;height:10px;border-radius:6px;margin:14px 0 4px;gap:2px}
.rt-seg{height:100%}
.rt-seg:first-child{border-radius:6px 0 0 6px}.rt-seg:nth-child(5){border-radius:0 6px 6px 0}
.rt-pin{position:absolute;top:-5px;width:4px;height:20px;border-radius:2px;background:#111;border:1px solid #fff}
.rt-escala{display:flex;gap:2px;font-size:.7rem;opacity:.65}
.rt-escala span{text-align:center}
.rt-chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:6px}
.rt-chip{flex:1 1 140px;border-radius:12px;padding:10px 12px;border:1px solid rgba(128,128,128,.25)}
.rt-chip-t{font-size:.72rem;text-transform:uppercase;letter-spacing:.6px;opacity:.7}
.rt-chip-v{font-size:1.15rem;font-weight:700;margin-top:2px}
.rt-chip-d{font-size:.74rem;opacity:.7;margin-top:2px;line-height:1.3}
.rt-plano{display:flex;flex-wrap:wrap;gap:18px;margin-top:10px;font-size:.9rem}
.rt-plano b{font-weight:700}
.rt-analistas{margin-top:10px;padding:9px 12px;border-radius:10px;font-size:.88rem;
  background:rgba(142,36,170,.08);border:1px solid rgba(142,36,170,.30)}
.rt-sessao{display:inline-flex;align-items:center;gap:8px;margin-top:6px;padding:4px 12px;border-radius:999px;
  font-size:.82rem;border:1px solid rgba(128,128,128,.35)}
.rt-aviso{font-size:.72rem;opacity:.6;margin-top:10px}
</style>
"""


def _rgba(hex_cor: str, alpha: float) -> str:
    h = hex_cor.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _cor_tendencia(t: Tendencia) -> str:
    return {"alta": COR["alta"], "baixa": COR["baixa"], "lateral": COR["aviso"]}.get(t.direcao, COR["neutro"])


def chip_tendencia(t: Tendencia) -> str:
    cor = _cor_tendencia(t)
    return (
        f'<div class="rt-chip" style="background:{_rgba(cor, .10)};border-color:{_rgba(cor, .45)}">'
        f'<div class="rt-chip-t">{escape(t.prazo)}</div>'
        f'<div class="rt-chip-v" style="color:{cor}">{SETA[t.direcao]} {escape(t.rotulo)}</div>'
        f'<div class="rt-chip-d">{escape(t.detalhe)}</div></div>'
    )


# Zonas da pontuação (−100…+100), com a mesma largura dos limiares de `sinal._acao_do_score`.
ZONAS = (("Vender", 50, "#dc2626"), ("Reduzir", 35, "#d97706"), ("Aguardar", 30, "#9ca3af"),
         ("Manter", 35, "#65a30d"), ("Comprar", 50, "#16a34a"))


def barra_score(score: float) -> str:
    """Barra de cinco zonas proporcionais aos limiares do sinal, com um pino na pontuação atual."""
    pos = (max(-100.0, min(100.0, score)) + 100) / 2  # 0…100 %
    segs = "".join(f'<div class="rt-seg" style="flex:{w};background:{c}"></div>' for _, w, c in ZONAS)
    rotulos = "".join(f'<span style="flex:{w}">{n}</span>' for n, w, _ in ZONAS)
    return (f'<div class="rt-bar">{segs}<div class="rt-pin" style="left:calc({pos:.1f}% - 2px)"></div></div>'
            f'<div class="rt-escala">{rotulos}</div>')


def faixa_sessao(sess: Sessao | None) -> str:
    """Pílula com o estado do mercado e, se houver, o preço do pré-mercado/after-hours."""
    if sess is None:
        return ""
    ponto = {"PRE": "#fb8c00", "REGULAR": COR["alta"], "POST": "#8e24aa", "FECHADO": COR["neutro"]}[sess.estado]
    txt = f'<span style="color:{ponto}">●</span> <b>{escape(sess.rotulo)}</b>'
    if sess.tem_estendido:
        cor = COR["alta"] if sess.var_estendida >= 0 else COR["baixa"]
        hora = f" às {sess.horario:%H:%M} (Brasília)" if sess.horario is not None else ""
        txt += (f' · {sess.preco_estendido:,.2f} <span style="color:{cor};font-weight:700">'
                f'({sess.var_estendida:+.2%})</span> vs. {escape(sess.referencia)}{hora}')
    return f'<div class="rt-sessao">{txt}</div>'


def linha_analistas(a: Analistas | None) -> str:
    """Faixa 'Alvo dos analistas' do cartão; vazia se não há cobertura."""
    if a is None or a.media is None:
        return ""
    up = a.upside_media
    cor = COR["alta"] if (up or 0) >= 0 else COR["baixa"]
    partes = [f'🎯 Alvo dos analistas <b>{a.media:,.2f}</b>'
              + (f' <span style="color:{cor};font-weight:700">({up:+.1%})</span>' if up is not None else "")]
    if a.baixo is not None and a.alto is not None:
        partes.append(f"faixa {a.baixo:,.2f}–{a.alto:,.2f}")
    if a.n:
        partes.append(f"{a.n} analistas")
    if a.recomendacao:
        partes.append(f"recomendação: {escape(a.recomendacao)}")
    return f'<div class="rt-analistas">{" · ".join(partes)}</div>'


def cartao_decisao(d: Decisao, nome: str, ticker: str, preco: float, variacao: float | None,
                   quando: str, extras: list[Tendencia] | None = None, analistas: Analistas | None = None,
                   sessao: Sessao | None = None) -> str:
    """HTML completo do painel: ativo, selo do sinal, barra, resumo, tendências e plano de risco."""
    cor = ACAO_COR[d.acao]
    var_txt = "" if variacao is None or variacao != variacao else (
        f'<span class="rt-var" style="color:{COR["alta"] if variacao >= 0 else COR["baixa"]}">{variacao:+.2%}</span>')
    chips = "".join(chip_tendencia(t) for t in [*d.tendencias, *(extras or [])])
    plano = ""
    if d.stop is not None and d.alvo is not None:
        plano = (f'<div class="rt-plano"><span>Stop sugerido <b>{d.stop:,.2f}</b> '
                 f'({(d.stop / preco - 1):+.1%})</span><span>Alvo (3× o risco) <b>{d.alvo:,.2f}</b> '
                 f'({(d.alvo / preco - 1):+.1%})</span></div>')
    return (
        CSS + '<div class="rt-card">'
        '<div class="rt-top">'
        f'<div><span class="rt-nome">{escape(nome)} · {escape(ticker)}</span></div>'
        f'<div><span class="rt-preco">{preco:,.2f}</span>{var_txt}</div></div>'
        f'<div class="rt-sub">Último candle: {escape(quando)}</div>{faixa_sessao(sessao)}'
        f'<div class="rt-badge" style="color:{cor};background:{_rgba(cor, .14)};border:2px solid {_rgba(cor, .55)}">'
        f'{ACAO_ICONE[d.acao]} {d.acao}</div>'
        f'<span class="rt-meta">Pontuação {d.score:+.0f}/100 · Confiança {escape(d.confianca)}</span>'
        f'{barra_score(d.score)}'
        f'<div class="rt-resumo">{escape(d.resumo)}</div>'
        f'<div class="rt-chips">{chips}</div>{plano}{linha_analistas(analistas)}'
        '<div class="rt-aviso">Sinal automático baseado em regras históricas (regime MA200, tendência, momentum e '
        'timing). Não é recomendação de investimento.</div></div>'
    )


def lista_motivos(d: Decisao) -> str:
    """Motivos em Markdown (uma linha por regra)."""
    return "\n".join(f"- {TOM_ICONE[m.tom]} **{m.titulo}** — {m.texto}" for m in d.motivos)
