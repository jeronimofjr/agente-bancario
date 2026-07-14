"""
Agente Principal: coordena os agentes especialistas via ferramentas (padrão subagents).
"""

import logging
from typing import Annotated

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import InjectedState, InjectedToolCallId, tool
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agents.credit_agent import build_credit_agent
from agents.credit_interview_agent import build_credit_interview_agent
from agents.currency_agent import build_currency_agent
from model import get_model
from state import BankState

logger = logging.getLogger(__name__)

load_dotenv()

credit_agent = build_credit_agent()
currency_agent = build_currency_agent()
credit_interview_agent = build_credit_interview_agent()


@tool
def call_currency_agent(
    query: Annotated[str, "O que o cliente quer saber sobre câmbio, em linguagem natural."],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Encaminha a solicitação para o especialista em câmbio (cotações, conversão de moeda)."""
    logger.info("Encaminhando solicitação ao Agente de Câmbio")
    
    result = currency_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return Command(
        update={"messages": [ToolMessage(content=result["messages"][-1].content, tool_call_id=tool_call_id)]}
    )


@tool
def call_credit_agent(
    query: Annotated[str, "O que o cliente quer sobre limite de crédito, em linguagem natural."],
    state: Annotated[BankState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Encaminha a solicitação para o especialista em crédito (consulta/aumento de limite)."""
    logger.info("Encaminhando solicitação ao Agente de Crédito")
    
    credit_thread_id = f"credit-agent-{state['cpf_cliente']}"
    
    result = credit_agent.invoke({
        "messages": [{"role": "user", "content": query}],
        "cpf_cliente": state["cpf_cliente"],
        "nome_cliente": state["nome_cliente"],
        "limite_atual": state["limite_atual"],
        "score_cliente": state["score_cliente"],
    },
    config={"configurable": {"thread_id": credit_thread_id}},
    )
    
    return Command(
        update={
            "limite_atual": result.get("limite_atual", state["limite_atual"]),
            "messages": [ToolMessage(content=result["messages"][-1].content, tool_call_id=tool_call_id)],
        }
    )


@tool
def call_credit_interview_agent(
    query: Annotated[str, "Contexto de por que o cliente precisa da entrevista de crédito."],
    state: Annotated[BankState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Encaminha a solicitação para o especialista em entrevista de crédito (recalcula o score)."""
    logger.info("Encaminhando solicitação ao Agente de entrevista de Crédito")
    
    interview_thread_id = f"credit-interview-{state['cpf_cliente']}"
    
    result = credit_interview_agent.invoke({
        "messages": [{"role": "user", "content": query}],
        "cpf_cliente": state["cpf_cliente"],
        "nome_cliente": state["nome_cliente"],
        "limite_atual": state["limite_atual"],
        "score_cliente": state["score_cliente"],
    },
        config={"configurable": {"thread_id": interview_thread_id}},
    )
    return Command(
        update={
            "score_cliente": result.get("score_cliente", state["score_cliente"]),
            "messages": [ToolMessage(content=result["messages"][-1].content, tool_call_id=tool_call_id)],
        }
    )


main_agent_system_prompt = """
    Você é o agente principal responsável por coordenar uma equipe de agentes bancários especializados.

    ## Função
    Determinar qual especialista deve tratar cada solicitação, chamando a ferramenta correspondente,
    e apresentar o resultado ao cliente.

    ## Ferramentas Disponíveis
    - `call_currency_agent`: conversões de moeda e cotações.
    - `call_credit_agent`: consultas de limite de crédito e solicitações de aumento de limite.
    - `call_credit_interview_agent`: entrevista financeira para recalcular o score de crédito.

    ## Responsabilidades
    - Encaminhar solicitações de câmbio para `call_currency_agent`.
    - Encaminhar solicitações de crédito para `call_credit_agent`.
    - Encaminhar para `call_credit_interview_agent` somente quando o `call_credit_agent` indicar que a
      entrevista é necessária e o cliente concordar.
    - Depois de uma entrevista concluída, se o cliente quiser prosseguir com o aumento, chame
      `call_credit_agent` novamente para reavaliar com o novo score.
    - Se nada disso se aplicar, responda diretamente, sem inventar informações de domínio.

    ## Regras
    - Nunca responda perguntas de domínio você mesmo, sempre use a ferramenta apropriada.
    - Sempre preserve o contexto da conversa.
    - Não mencione ao cliente que a resposta veio de uma "ferramenta" ou "agente".

    ## Estilo de Resposta
    - Seja conciso e combine os resultados em uma resposta única e coerente.
"""


def build_main_agent():
    """Compila o agente principal com um checkpointer na memória."""
    return create_agent(
        model=get_model(),
        tools=[call_currency_agent, call_credit_agent, call_credit_interview_agent],
        system_prompt=main_agent_system_prompt,
        state_schema=BankState,
        checkpointer=InMemorySaver(),
    )