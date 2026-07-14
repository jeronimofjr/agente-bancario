"""
Agente de Entrevista de Crédito.

Módulo responsável pela construção do Agente de Entrevista de Crédito.
O Agente realiza uma entrevista para coletar dados financeiros e recalcular o score de crédito do cliente.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Annotated

from langchain.agents import create_agent
from langchain.tools import InjectedState, InjectedToolCallId, tool
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from model import get_model
from schemas.errors import CreditError
from state import BankState

logger = logging.getLogger(__name__)

customers_path = Path(__file__).parent.parent / "data" / "clientes.csv"

peso_renda = 30
peso_emprego = {"formal": 300, "autônomo": 200, "desempregado": 0}
peso_dependentes = {0: 100, 1: 80, 2: 60, "3+": 30}
peso_dividas = {"sim": -100, "não": 100}
score_min = 0
score_max = 1000

interview_checkpointer = InMemorySaver()

def calcular_score(
    renda_mensal: float,
    tipo_emprego: str,
    despesas: float,
    num_dependentes: int,
    tem_dividas: str,
) -> int:
    """Calcula o novo score de crédito do cliente (0 a 1000)."""
    
    dependentes = num_dependentes if num_dependentes < 3 else "3+"

    score = (
        (renda_mensal / (despesas + 1)) * peso_renda
        + peso_emprego[tipo_emprego]
        + peso_dependentes[dependentes]
        + peso_dividas[tem_dividas]
    )
    
    print(f"score: {score}")

    if score < score_min:
        score = score_min
    elif score > score_max:
        score = score_max
    
    logger.debug(
        "Score calculado: renda_mensal=%s, tipo_emprego=%s, despesas=%s, "
        "num_dependentes=%s, tem_dividas=%s -> score=%s",
        renda_mensal, tipo_emprego, despesas, num_dependentes, tem_dividas, score,
    )

    return int(score)


def atualizar_score_cliente(cpf: str, novo_score: int) -> None:
    """Atualiza o score de crédito do cliente em `clientes.csv`."""

    if not customers_path.exists():
        raise FileNotFoundError(
            f"Dados dos clientes não encontrados em {customers_path}"
        )

    clientes = pd.read_csv(customers_path, dtype={"cpf": str})

    if not (clientes["cpf"] == cpf).any():
        raise ValueError(f"CPF {cpf} não encontrado em clientes.csv")

    clientes.loc[clientes["cpf"] == cpf, "score"] = novo_score
    clientes.to_csv(customers_path, index=False)
    
    logger.info("Score atualizado em clientes.csv: cpf=%s, novo_score=%s", cpf, novo_score)


@tool
def registrar_entrevista_credito(
    renda_mensal: float,
    tipo_emprego: str,
    despesas: float,
    num_dependentes: int,
    tem_dividas: str,
    state: Annotated[BankState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Registra as respostas da entrevista de crédito, calcula o novo score e atualiza a base em clientes.csv"""

    try:
        cpf_cliente = state["cpf_cliente"]
        tipo_emprego = tipo_emprego.strip().lower()
        tem_dividas = tem_dividas.strip().lower()

        logger.debug(
            "Entrevista de crédito recebida: cpf=%s, renda_mensal=%s, tipo_emprego=%s, "
            "despesas=%s, num_dependentes=%s, tem_dividas=%s",
            cpf_cliente, renda_mensal, tipo_emprego, despesas, num_dependentes, tem_dividas,
        )

        erros = []

        if tipo_emprego not in peso_emprego:
            erros.append(
                f"tipo_emprego inválido: '{tipo_emprego}'. Use 'formal', "
                f"'autônomo' ou 'desempregado'."
            )
        if tem_dividas not in peso_dividas:
            erros.append(f"tem_dividas inválido: '{tem_dividas}'. Use 'sim' ou 'não'.")
        if renda_mensal < 0:
            erros.append("renda_mensal não pode ser negativa.")
        if despesas < 0:
            erros.append("despesas não pode ser negativa.")
        if num_dependentes < 0:
            erros.append("num_dependentes não pode ser negativo.")

        if erros:
            logger.warning(
                "Entrevista rejeitada por validação: cpf=%s, erros=%s",
                cpf_cliente, erros,
            )

            texto = "Não foi possível registrar a entrevista: " + " ".join(erros)
            return Command(
                update={"messages": [ToolMessage(content=texto, tool_call_id=tool_call_id)]}
            )

        novo_score = calcular_score(
            renda_mensal=renda_mensal,
            tipo_emprego=tipo_emprego,
            despesas=despesas,
            num_dependentes=num_dependentes,
            tem_dividas=tem_dividas,
        )

        try:
            atualizar_score_cliente(cpf=cpf_cliente, novo_score=novo_score)
        except (FileNotFoundError) as e:
            logger.error("Falha ao atualizar score do cliente %s: %s", cpf_cliente, e)
            texto = (
                "As respostas foram recebidas, mas houve um erro ao atualizar "
                "seu cadastro. Por favor, tente novamente mais tarde."
            )
            return Command(
                update={"messages": [ToolMessage(content=texto, tool_call_id=tool_call_id)]}
            )

        texto = (
            f"Entrevista concluída! Seu novo score de crédito é {novo_score:.0f} "
            f"(de um máximo de {score_max}). Vou te encaminhar de volta para o "
            f"Agente de Crédito para uma nova análise do seu pedido."
        )

        return Command(
            update={
                "score_cliente": novo_score,
                "messages": [ToolMessage(content=texto, tool_call_id=tool_call_id)],
            }
        )
    except (KeyError) as e:
            logger.error("Erro nos dados ao registrar entrevista de crédito: %s", e)
            texto = """Não foi possível processar sua solicitação no momento.
                Por favor, tente novamente mais tarde."""
            return Command(
                update={"messages": [ToolMessage(content=texto, tool_call_id=tool_call_id)]}
            )
    
    except Exception as e:
        logger.error("Erro inesperado ao registrar entrevista de crédito: %s", e)
        texto = """Não foi possível processar sua solicitação no momento.
                    Por favor, tente novamente mais tarde."""
        return Command(
            update={"messages": [ToolMessage(content=texto, tool_call_id=tool_call_id)]}
        )

credit_interview_agent_system_prompt = """
    Você é o Credit Interview Agent (Agente de Entrevista de Crédito) de um sistema bancário multi-agente.

    ## Contexto
    O cliente já foi autenticado antes de chegar até você, não peça CPF, data de nascimento ou qualquer outra credencial novamente; essa informação 
    já está disponível internamente. Você chega a esta conversa porque o Credit Agent identificou que o score atual do cliente não é suficiente para 
    o limite de crédito solicitado, e o cliente concordou em passar por uma entrevista financeira para tentar melhorar o score.

    ## Função
    Sua responsabilidade é conduzir uma entrevista conversacional, coletando exatamente 5 informações do cliente, uma pergunta por vez.

    ## Responsabilidades
    - Perguntar, em ordem e uma de cada vez (nunca listar todas de uma vez):
      1. Renda mensal (em reais).
      2. Tipo de emprego, normalizar para "formal", "autônomo" ou "desempregado" (ex.: "CLT" -> "formal"; "MEI" ou "freelancer" -> "autônomo").
      3. Despesas fixas mensais (em reais).
      4. Número de dependentes.
      5. Se possui dívidas ativas atualmente ("sim" ou "não").
    - Após coletar as 5 respostas, usar a ferramenta `registrar_entrevista_credito` para calcular o novo score e atualizar o cadastro do cliente.
    - Informar o cliente sobre o novo score e explicar que ele será encaminhado de volta ao Agente de Crédito para uma nova análise do pedido de aumento de limite.

    ## Regras
    - Faça as perguntas em ordem, uma de cada vez, de forma natural e cordial.
    - Se a resposta do cliente for ambígua ou não puder ser normalizada com segurança para os valores esperados, peça esclarecimento antes de seguir em frente.
    - Só utilize a ferramenta `registrar_entrevista_credito` depois de ter as 5 respostas.
    - Nunca calcule o score você mesmo, nem invente ou estime o resultado, a ferramenta é quem calcula e persiste o valor.
    - Se a ferramenta retornar um erro, não exiba mensagens de erro internas. Em vez disso, explique o problema ao cliente de forma clara e peça a informação novamente, se for o caso.
    - Se ocorrer algum erro interno, sempre informe ao cliente que o serviço está temporariamente indisponível ou que não foi possível processar a solicitação, e sugira tentar novamente mais tarde. Não mostre mensagens técnicas ou detalhes internos.
    - Sempre responda no mesmo idioma usado pelo cliente.

    ## Estilo de Resposta
    - Seja conciso, profissional e cordial.
"""

def build_credit_interview_agent():
    """Compila o credit_interview_agent."""

    logger.debug("Construindo credit_interview_agent")
    
    return create_agent(
            model=get_model(),
            tools=[registrar_entrevista_credito],
            name="credit_interview_agent",
            system_prompt=credit_interview_agent_system_prompt,
            state_schema=BankState,
            checkpointer=interview_checkpointer
        )
    