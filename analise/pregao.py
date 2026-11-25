"""Quando o pregão está aberto (para decidir se o app deve se atualizar sozinho).

 - Ações/ETFs dos EUA (e índices): 09:30–16:00, horário de Nova York, de segunda a sexta.
 - B3 (ações com sufixo .SA): 10:00–18:00, horário de Brasília (inclui o call de fechamento), de segunda a sexta.
 - Cripto (-USD): sempre aberto.

O horário não conhece feriados. Para os EUA, se o Yahoo informar o estado do mercado, ele manda: um feriado em dia
útil aparece como "CLOSED" mesmo dentro do horário. Sem dependência do Streamlit, para ser testável.
"""

from __future__ import annotations

import pandas as pd

NY, BRT = "America/New_York", "America/Sao_Paulo"
US, BR, CRIPTO = "US", "BR", "CRIPTO"


def tipo_de_mercado(ticker: str) -> str:
    t = ticker.upper().strip()
    if t.endswith(".SA"):
        return BR
    if t.endswith("-USD"):
        return CRIPTO
    return US


def _agora(agora) -> pd.Timestamp:
    return pd.Timestamp.now("UTC") if agora is None else (
        pd.Timestamp(agora).tz_localize("UTC") if pd.Timestamp(agora).tzinfo is None else pd.Timestamp(agora))


def pregao_aberto(ticker: str, agora=None, estado_yahoo: str | None = None) -> bool:
    """True se o pregão regular do ticker está aberto agora. `agora` sem fuso é tratado como UTC."""
    tipo = tipo_de_mercado(ticker)
    if tipo == CRIPTO:
        return True
    t = _agora(agora)
    if tipo == BR:
        local = t.tz_convert(BRT)
        return local.weekday() < 5 and 10 * 60 <= local.hour * 60 + local.minute < 18 * 60
    local = t.tz_convert(NY)
    no_horario = local.weekday() < 5 and 9 * 60 + 30 <= local.hour * 60 + local.minute < 16 * 60
    if estado_yahoo:                                      # o Yahoo sabe dos feriados
        return no_horario and estado_yahoo.upper() == "REGULAR"
    return no_horario


INTERVALOS_AUTO = {"30 segundos": 30, "1 minuto": 60, "5 minutos": 300}
MINIMO_MERCADO = 300   # a página do mercado faz ~55 consultas ao Yahoo: no mínimo a cada 5 minutos


def intervalo_efetivo(segundos: int, pagina_mercado: bool) -> int:
    """Intervalo real de atualização: a página do mercado nunca atualiza mais rápido que a cada 5 minutos."""
    return max(segundos, MINIMO_MERCADO) if pagina_mercado else segundos


def balde(segundos: int, agora_ts: float | None = None) -> int:
    """Número do 'balde' de tempo atual: muda a cada `segundos`. Usado como parte da chave do cache, assim cada
    intervalo de atualização força uma consulta nova em vez de reaproveitar dados mais antigos."""
    import time

    return int((time.time() if agora_ts is None else agora_ts) // max(1, int(segundos)))


def horas_dos_eua_em_brasilia(agora=None) -> dict[str, str]:
    """Horários do pregão dos EUA convertidos para Brasília, conforme o horário de verão dos EUA na data.

    Inverno dos EUA (nov–mar): pré 06h, abre 11h30, fecha 18h, after-hours até 22h.
    Verão dos EUA (mar–nov):   pré 05h, abre 10h30, fecha 17h, after-hours até 21h.
    """
    dia = _agora(agora).tz_convert(NY).normalize()

    def hm(hora: int, minuto: int = 0) -> str:
        t = (dia + pd.Timedelta(hours=hora, minutes=minuto)).tz_convert(BRT)
        return f"{t.hour}h" if t.minute == 0 else f"{t.hour}h{t.minute:02d}"

    return {"pre": hm(4), "abre": hm(9, 30), "fecha": hm(16), "pos": hm(20)}
