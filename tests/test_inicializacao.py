"""Testa a inicialização do app (limpeza de módulos antigos e exibição do erro real de importação).

As funções são extraídas do arquivo do app (que não pode ser importado sem rodar a interface) e executadas com
`sys` e `st` falsos, para não mexer nos módulos reais carregados pelo pytest.
"""

import ast
import pathlib
import types

import pytest

APP = pathlib.Path(__file__).resolve().parents[1] / "app_analise_final.py"


def carregar(nome_funcao: str, **ambiente):
    arvore = ast.parse(APP.read_text(encoding="utf-8"))
    func = next(n for n in arvore.body if isinstance(n, ast.FunctionDef) and n.name == nome_funcao)
    modulo = ast.Module(body=[func], type_ignores=[])
    ns = dict(ambiente)
    exec(compile(modulo, str(APP), "exec"), ns)
    return ns[nome_funcao]


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
