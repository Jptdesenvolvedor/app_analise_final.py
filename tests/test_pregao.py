import pandas as pd
import pytest

from analise import pregao as pg


def ny(texto):
    """Horário de Nova York → UTC (o app chama com 'agora' em UTC)."""
    return pd.Timestamp(texto, tz="America/New_York").tz_convert("UTC")


def brt(texto):
    return pd.Timestamp(texto, tz="America/Sao_Paulo").tz_convert("UTC")


def test_tipo_de_mercado():
    assert pg.tipo_de_mercado("AAPL") == pg.US and pg.tipo_de_mercado("^GSPC") == pg.US
    assert pg.tipo_de_mercado("petr4.sa") == pg.BR and pg.tipo_de_mercado("VALE3.SA") == pg.BR
    assert pg.tipo_de_mercado("BTC-USD") == pg.CRIPTO and pg.tipo_de_mercado("eth-usd") == pg.CRIPTO


# ----------------------------- EUA ------------------------------------------
@pytest.mark.parametrize("quando,esperado", [
    ("2026-11-25 09:29:59", False),     # quarta, 1 segundo antes de abrir
    ("2026-11-25 09:30:00", True),      # abre
    ("2026-11-25 12:00:00", True),
    ("2026-11-25 15:59:59", True),
    ("2026-11-25 16:00:00", False),     # fecha (after-hours não conta como pregão)
    ("2026-11-25 04:30:00", False),     # pré-mercado
    ("2026-11-25 18:00:00", False),     # after-hours
    ("2026-11-28 11:00:00", False),     # sábado
    ("2026-11-29 11:00:00", False),     # domingo
    ("2026-11-23 10:00:00", True),      # segunda
    ("2026-11-27 10:00:00", True),      # sexta
])
def test_pregao_eua_no_horario_de_inverno(quando, esperado):
    assert pg.pregao_aberto("AAPL", ny(quando)) is esperado


def test_pregao_eua_respeita_o_horario_de_verao():
    # em julho NY está UTC-4 (13:30 UTC = 09:30); em novembro UTC-5 (14:30 UTC = 09:30)
    assert pg.pregao_aberto("AAPL", "2026-07-15 13:30:00") is True
    assert pg.pregao_aberto("AAPL", "2026-07-15 13:29:00") is False
    assert pg.pregao_aberto("AAPL", "2026-11-25 14:30:00") is True
    assert pg.pregao_aberto("AAPL", "2026-11-25 14:29:00") is False
    assert pg.pregao_aberto("AAPL", "2026-07-15 20:00:00") is False      # 16:00 em NY (verão)


def test_yahoo_manda_nos_feriados_dos_eua():
    quinta_acao_de_gracas = ny("2026-11-26 11:00:00")                    # dia útil e dentro do horário, mas feriado
    assert pg.pregao_aberto("AAPL", quinta_acao_de_gracas) is True        # só pelo relógio
    assert pg.pregao_aberto("AAPL", quinta_acao_de_gracas, estado_yahoo="CLOSED") is False
    assert pg.pregao_aberto("AAPL", ny("2026-11-25 11:00:00"), estado_yahoo="regular") is True
    assert pg.pregao_aberto("AAPL", ny("2026-11-25 18:00:00"), estado_yahoo="REGULAR") is False   # fora do horário: fechado


# ----------------------------- B3 -------------------------------------------
@pytest.mark.parametrize("quando,esperado", [
    ("2026-11-25 09:59:59", False), ("2026-11-25 10:00:00", True), ("2026-11-25 14:00:00", True),
    ("2026-11-25 17:59:59", True), ("2026-11-25 18:00:00", False), ("2026-11-28 11:00:00", False),
])
def test_pregao_b3(quando, esperado):
    assert pg.pregao_aberto("PETR4.SA", brt(quando)) is esperado


def test_b3_ignora_o_estado_do_yahoo_dos_eua():
    assert pg.pregao_aberto("VALE3.SA", brt("2026-11-25 11:00:00"), estado_yahoo="CLOSED") is True


# ----------------------------- cripto e entradas ----------------------------
def test_cripto_sempre_aberto():
    for q in ("2026-11-29 03:00:00", "2026-11-25 12:00:00", "2026-12-25 00:00:00"):
        assert pg.pregao_aberto("BTC-USD", q) is True


def test_agora_sem_fuso_e_tratado_como_utc_e_default_e_o_relogio():
    assert pg.pregao_aberto("AAPL", pd.Timestamp("2026-11-25 15:00:00")) is True        # 10:00 em NY
    assert pg.pregao_aberto("AAPL", pd.Timestamp("2026-11-25 12:00:00")) is False       # 07:00 em NY
    assert isinstance(pg.pregao_aberto("AAPL"), bool)


# ----------------------------- intervalo ------------------------------------
def test_intervalo_efetivo_protege_a_pagina_do_mercado():
    assert pg.intervalo_efetivo(30, pagina_mercado=False) == 30
    assert pg.intervalo_efetivo(30, pagina_mercado=True) == 300
    assert pg.intervalo_efetivo(60, pagina_mercado=True) == 300
    assert pg.intervalo_efetivo(300, pagina_mercado=True) == 300
    assert set(pg.INTERVALOS_AUTO.values()) == {30, 60, 300}


def test_balde_muda_a_cada_intervalo():
    assert pg.balde(60, 120.0) == pg.balde(60, 179.9) == 2
    assert pg.balde(60, 180.0) == 3
    assert pg.balde(300, 299.0) == 0 and pg.balde(300, 300.0) == 1
    assert pg.balde(0, 10.0) == 10                      # intervalo inválido não divide por zero
    assert isinstance(pg.balde(60), int)


def test_horas_dos_eua_em_brasilia_inverno_e_verao():
    inverno = pg.horas_dos_eua_em_brasilia("2026-11-25 15:00:00")        # depois do 1º domingo de novembro: UTC-5 em NY
    assert inverno == {"pre": "6h", "abre": "11h30", "fecha": "18h", "pos": "22h"}
    verao = pg.horas_dos_eua_em_brasilia("2026-07-15 15:00:00")          # horário de verão dos EUA: UTC-4
    assert verao == {"pre": "5h", "abre": "10h30", "fecha": "17h", "pos": "21h"}
    antes_da_virada = pg.horas_dos_eua_em_brasilia("2026-10-30 15:00:00")  # a virada de 2026 é em 1º/nov
    assert antes_da_virada["abre"] == "10h30"
    depois = pg.horas_dos_eua_em_brasilia("2026-11-02 15:00:00")
    assert depois["abre"] == "11h30"
    assert set(pg.horas_dos_eua_em_brasilia()) == {"pre", "abre", "fecha", "pos"}


def test_abertura_calculada_bate_com_pregao_aberto():
    """O horário em Brasília informado ao usuário é exatamente o que a regra considera aberto."""
    for dia, abre in (("2026-11-25", "11:30"), ("2026-07-15", "10:30")):
        h = pg.horas_dos_eua_em_brasilia(f"{dia} 15:00:00")
        assert h["abre"].replace("h", ":") == abre
        assert pg.pregao_aberto("AAPL", brt(f"{dia} {abre}:00")) is True
        assert pg.pregao_aberto("AAPL", brt(f"{dia} {abre}:00") - pd.Timedelta(minutes=1)) is False
