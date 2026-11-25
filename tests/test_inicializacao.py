"""Testa a inicialização do app (limpeza de módulos antigos e exibição do erro real de importação).

As funções são extraídas do arquivo do app (que não pode ser importado sem rodar a interface) e executadas com
`sys` e `st` falsos, para não mexer nos módulos reais carregados pelo pytest.
"""

import ast
import pathlib
import types

import pytest

APP = pathlib.Path(__file__).resolve().parents[1] / "app_analise_final.py"


def carregar_varias(nomes, **ambiente):
    """Executa as funções (do topo do app) `nomes` num mesmo namespace e o devolve."""
    arvore = ast.parse(APP.read_text(encoding="utf-8"))
    funcs = [n for n in arvore.body if isinstance(n, ast.FunctionDef) and n.name in nomes]
    assert {f.name for f in funcs} == set(nomes), f"funções ausentes no app: {set(nomes) - {f.name for f in funcs}}"
    ns = dict(ambiente)
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(APP), "exec"), ns)
    return ns


def carregar(nome_funcao: str, **ambiente):
    return carregar_varias([nome_funcao], **ambiente)[nome_funcao]


class SysFalso:
    def __init__(self, nomes):
        self.modules = {n: types.ModuleType(n) for n in nomes}


# ----------------------------- limpeza de módulos antigos -------------------
def test_purga_so_os_modulos_do_pacote_analise():
    falso = SysFalso(["analise", "analise.charts", "analise.sub.profundo", "analiseX", "analises", "pandas",
                      "streamlit", "app"])
    purgar = carregar("_purgar_modulos_antigos", sys=falso)
    estado = {}
    assert purgar(estado) is True
    assert set(falso.modules) == {"analiseX", "analises", "pandas", "streamlit", "app"}   # prefixos parecidos ficam
    assert estado["_modulos_ok"] is True


def test_purga_so_uma_vez_por_sessao():
    falso = SysFalso(["analise", "analise.charts"])
    purgar = carregar("_purgar_modulos_antigos", sys=falso)
    estado = {}
    assert purgar(estado) is True
    falso.modules["analise.charts"] = types.ModuleType("analise.charts")      # reimportado depois da limpeza
    assert purgar(estado) is False                                            # reexecução da mesma sessão: não limpa de novo
    assert "analise.charts" in falso.modules
    assert purgar({}) is True                                                 # sessão nova: limpa outra vez


def test_purga_sem_modulos_carregados_nao_quebra():
    falso = SysFalso(["pandas"])
    assert carregar("_purgar_modulos_antigos", sys=falso)({}) is True
    assert set(falso.modules) == {"pandas"}


def test_cenario_do_deploy_modulo_antigo_deixa_de_quebrar_a_importacao():
    """Reproduz o ImportError da nuvem: módulo antigo em memória sem os nomes novos."""
    falso = SysFalso(["analise", "analise.charts"])
    assert not hasattr(falso.modules["analise.charts"], "grafico_alvos")       # versão antiga (sem funções novas)
    carregar("_purgar_modulos_antigos", sys=falso)({})
    assert "analise.charts" not in falso.modules                               # o próximo import lê o arquivo atual


# ----------------------------- erro real de importação ----------------------
class StFalso:
    class _Parou(Exception):
        pass

    def __init__(self):
        self.chamadas = []

    def set_page_config(self, **kw):
        self.chamadas.append(("config", kw))

    def error(self, msg):
        self.chamadas.append(("error", msg))

    def code(self, texto, language=None):
        self.chamadas.append(("code", texto))

    def info(self, msg):
        self.chamadas.append(("info", msg))

    def stop(self):
        self.chamadas.append(("stop", None))
        raise self._Parou()


def test_mostra_o_erro_real_e_para_o_app():
    st = StFalso()
    mostrar = carregar("_mostrar_erro_de_importacao", st=st, traceback=__import__("traceback"))
    try:
        raise ImportError("cannot import name 'grafico_alvos' from 'analise.charts'")
    except ImportError as e:
        with pytest.raises(StFalso._Parou):
            mostrar(e)
    tipos = [c[0] for c in st.chamadas]
    assert tipos[0] == "config"                                                # set_page_config tem de ser o primeiro comando
    assert tipos[-1] == "stop" and {"error", "code", "info"} <= set(tipos)
    codigo = next(c[1] for c in st.chamadas if c[0] == "code")
    assert "ImportError" in codigo and "grafico_alvos" in codigo and "Traceback" in codigo
    assert "Reboot app" in next(c[1] for c in st.chamadas if c[0] == "info")


def test_o_app_protege_as_importacoes_do_pacote():
    """O bloco de importações do pacote `analise` precisa estar dentro do try/except ImportError."""
    arvore = ast.parse(APP.read_text(encoding="utf-8"))
    tentativas = [n for n in arvore.body if isinstance(n, ast.Try)]
    assert tentativas, "sem bloco try no topo do app"
    importados = {n.module for n in tentativas[0].body if isinstance(n, ast.ImportFrom)}
    assert {"analise", "analise.charts", "analise.analistas", "analise.apresentacao"} <= importados
    assert any(isinstance(h.type, ast.Name) and h.type.id == "ImportError" for h in tentativas[0].handlers)
    fora = [n for n in arvore.body if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("analise")]
    assert not fora, f"importações do pacote fora do bloco protegido: {[n.module for n in fora]}"


# ----------------------------- relógio de atualização automática -----------
class StRelogio:
    """`st` falso: guarda o fragmento agendado e conta as reexecuções pedidas."""

    def __init__(self):
        self.session_state = {}
        self.fragmentos = []
        self.reruns = 0
        self.run_every = []

    def fragment(self, run_every=None):
        def decorador(func):
            self.run_every.append(run_every)
            self.fragmentos.append(func)
            return func
        return decorador

    def rerun(self):
        self.reruns += 1


def _montar_relogio():
    st = StRelogio()
    ns = carregar_varias(["_tique_de_atualizacao", "_agendar_atualizacao"], st=st)
    return st, ns["_agendar_atualizacao"]


def test_tique_primeira_execucao_so_arma_e_a_seguinte_pede_rerun():
    tique = carregar("_tique_de_atualizacao")
    estado = {}
    assert tique(estado) is False          # logo após a execução completa: só arma
    assert tique(estado) is True           # passou o intervalo: pede a reexecução
    assert tique(estado) is True
    estado["_tique_armado"] = False        # nova execução completa reinicia
    assert tique(estado) is False


def test_relogio_nao_entra_em_laco_depois_de_reexecutar():
    st, agendar = _montar_relogio()
    agendar(60)                            # execução completa nº 1: define o fragmento e o executa uma vez
    assert st.run_every == ["60s"] and st.reruns == 0
    fragmento = st.fragmentos[0]
    fragmento()                            # passou 1 intervalo → pede reexecução do app
    assert st.reruns == 1
    agendar(60)                            # o app reexecutou: o relógio é reiniciado e NÃO pede outra reexecução na hora
    assert st.reruns == 1
    st.fragmentos[-1]()                    # só no próximo intervalo
    assert st.reruns == 2


def test_relogio_usa_o_intervalo_pedido():
    for seg in (30, 60, 300):
        st, agendar = _montar_relogio()
        agendar(seg)
        assert st.run_every == [f"{seg}s"]


def test_interacao_do_usuario_reinicia_a_contagem():
    st, agendar = _montar_relogio()
    agendar(60)
    st.fragmentos[0]()                     # um tique já armado → pediria reexecução
    assert st.reruns == 1
    agendar(60)                            # o usuário clicou em algo (execução completa): contagem recomeça
    assert st.session_state["_tique_armado"] is True and st.reruns == 1
