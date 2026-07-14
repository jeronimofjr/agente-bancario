"""
Agente de Triagem.

Módulo responsável pela construção do Agente de Triagem.
O Agente utentica o cliente e prepara o encaminhamento ao Agente Principal..
"""

import logging
from pathlib import Path
from typing import Annotated, Optional

from langchain.agents import create_agent
from langchain.tools import InjectedState, InjectedToolCallId, tool
from langchain_core.tools import InjectedToolCallId, tool
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from model import get_model
from state import TriageState
from utils.utils import carregar_clientes, normalize_cpf
from schemas.customer import Customer

logger = logging.getLogger(__name__)

customers_path = Path(__file__).parent.parent / "data" / "clientes.csv"
MAX_ATTEMPTS = 3


triage_checkpointer = InMemorySaver()

def authenticate(cpf: str, data_nascimento: str) -> Optional[Customer]:
    """Retorna o cliente se CPF e data de nascimento forem válidos."""
    
    customers = carregar_clientes()
    cpf_norm = normalize_cpf(cpf)
    nasc_norm = data_nascimento.strip()

    for customer in customers:
        if (
            normalize_cpf(customer["cpf"]) == cpf_norm
            and customer["data_nascimento"].strip() == nasc_norm
        ):
            return customer
    return None


@tool
def autenticar_cliente(
    cpf: str,
    data_nascimento: str,
    state: Annotated[TriageState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Autentica um cliente usando CPF e data de nascimento.
    Chame somente quando ambos os dados já tiverem sido informados."""

    logger.info(
        "autenticar_cliente chamado: cpf=%r, data_nascimento=%r, tentativas_atuais=%s",
        cpf, data_nascimento, state.get("tentativas", 0),
    )
    
    tentativa_atual = state.get("tentativas", 0) + 1
    customer = authenticate(cpf, data_nascimento)

    if customer:
        texto = f"Cliente autenticado com sucesso: {customer['nome']}."
        return Command(
            update={
                "tentativas": tentativa_atual,
                "autenticado": True,
                "cpf_cliente": customer["cpf"],
                "nome_cliente": customer["nome"],
                "limite_atual": float(customer["limite_atual"]),
                "score_cliente": int(customer["score"]),
                "messages": [ToolMessage(content=texto, tool_call_id=tool_call_id)],
            }
        )

    if tentativa_atual >= MAX_ATTEMPTS:
        texto = "Autenticação falhou após o número máximo de tentativas permitidas."
        return Command(
            update={
                "tentativas": tentativa_atual,
                "encerrado": True,
                "messages": [ToolMessage(content=texto, tool_call_id=tool_call_id)],
            }
        )

    restantes = MAX_ATTEMPTS - tentativa_atual
    texto = f"CPF ou data de nascimento não conferem. Restam {restantes} tentativa(s)."
    return Command(
        update={
            "tentativas": tentativa_atual,
            "messages": [ToolMessage(content=texto, tool_call_id=tool_call_id)],
        }
    )


triage_agent_system_prompt = """
    Você é o Agente de Triagem de um sistema bancário multi-agente, a porta de entrada
    do atendimento.

    ## Fluxo
    1. Cumprimente o cliente de forma breve e cordial.
    2. Peça o CPF do cliente (somente números). Faça essa pergunta sozinha, sem
       pedir mais nada junto.
    3. Depois que o cliente responder o CPF, peça a data de nascimento (AAAA-MM-DD)
       em uma pergunta separada.
       - Exceção: se o cliente já informar CPF e data de nascimento juntos, numa
         única mensagem, use os dois diretamente sem pedir de novo.
    4. Assim que tiver os dois valores (CPF e data de nascimento), chame a função
       autenticar_cliente imediatamente. Nunca decida sozinho se os dados estão
       corretos, a função é quem decide isso.
    5. Se a função indicar falha e ainda houver tentativas, peça os dados novamente,
       de forma cordial, informando quantas tentativas restam. Volte ao passo 2.
    6. Se a função indicar falha definitiva (sem mais tentativas), informe o cliente
       de forma educada que não foi possível autenticar e encerre a conversa.
    7. Se a função confirmar autenticação, cumprimente o cliente pelo nome e
       pergunte como pode ajudar hoje.

    ## Regras
    - Nunca peça CPF e data de nascimento na mesma pergunta, exceto se o cliente
      já os tiver informado juntos por conta própria.
    - Nunca valide o formato ou a veracidade do CPF/data de nascimento você mesmo —
      apenas colete os valores exatamente como o cliente informou e deixe a
      validação a cargo da função autenticar_cliente.
    - Nunca invente se o cliente está autenticado ou não — sempre confie no
      resultado da função autenticar_cliente.
    - Nunca mencione nomes de funções/ferramentas ao cliente.
    - Seja conciso, profissional e cordial.
"""

def build_triage_agent():
    """Compila o triage agent."""
    
    logger.debug("Construindo triage_agent")
    
    return create_agent(
        model=get_model(),
        tools=[autenticar_cliente],
        name="triage_agent",
        system_prompt=triage_agent_system_prompt,
        state_schema=TriageState,
        checkpointer=triage_checkpointer,
    )