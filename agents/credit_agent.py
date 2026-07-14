"""
Agente de Crédito.

Módulo responsável pela construção do Agente de Crédito.
O Agente permite que o cliente solicite um aumento de crédito.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, List, Optional, Union

import pandas as pd
from langchain_core.messages import ToolMessage
from langchain.agents import create_agent
from langchain.tools import InjectedState, InjectedToolCallId, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from model import get_model
from schemas.customer import Customer
from schemas.errors import CreditError
from state import BankState
from utils.utils import carregar_clientes

logger = logging.getLogger(__name__)

score_limite_path = Path(__file__).parent.parent / "data" /  "score_limite.csv"
solicitacoes_path = Path(__file__).parent.parent / "data" /  "solicitacoes_aumento_limite.csv"
customers_path = Path(__file__).parent.parent / "data" / "clientes.csv"

score_data = pd.read_csv(score_limite_path)

credit_checkpointer = InMemorySaver()

def consultar_score_e_limite(cpf: str) -> Optional[dict]:
    """Consulta limite_atual + score para um CPF. Retorna None se não encontrado."""

    try:
        customers: List[Customer] = carregar_clientes()

        for customer in customers:
            if customer["cpf"] == cpf:
                credit_data = {
                    "limite_atual": float(customer["limite_atual"]),
                    "score": int(customer["score"]),
                }
                logger.debug("Score e limite encontrados para cpf=%s: %s", cpf, credit_data)
                return credit_data
        
        logger.warning("Nenhum registro de score/limite encontrado para cpf=%s", cpf)
        return None
    
    except (FileNotFoundError) as e:
        logger.error("Falha ao consultar dados de crédito do cliente %s: %s", cpf, e)
        return None

    except Exception as e:
        logger.error("Erro inesperado ao consultar score e limite do cliente %s: %s", cpf,e,)
        return None


def limite_maximo_permitido(score: int) -> Optional[float]:
    """Encontra o limite máximo aprovável para um determinado score, com base no arquivo score_limite.csv."""

    limite = score_data.loc[
        score_data["score_minimo"].le(score) & score_data["score_maximo"].ge(score),
        "limite_maximo_permitido",
    ]
    return float(limite.squeeze()) if not limite.empty else None


def registrar_solicitacao(
    cpf_cliente: str,
    limite_atual: float,
    novo_limite_solicitado: float,
    status_pedido: str,
) -> None:
    """Anexa uma solicitação de crédito a `solicitacoes_aumento_limite.csv`."""

    df = pd.DataFrame(
        [
            {
                "cpf_cliente": cpf_cliente,
                "data_hora_solicitacao": datetime.now(timezone.utc).isoformat(),
                "limite_atual": limite_atual,
                "novo_limite_solicitado": novo_limite_solicitado,
                "status_pedido": status_pedido,
            }
        ]
    )

    df.to_csv(
        solicitacoes_path,
        mode="a",
        header=not solicitacoes_path.exists(),
        index=False,
    )
    
    logger.info(
    "Solicitação registrada: cpf=%s, limite_atual=R$ %.2f, "
    "novo_limite_solicitado=R$ %.2f, status=%s",
    cpf_cliente, limite_atual, novo_limite_solicitado, status_pedido)



@tool
def consultar_limite_disponivel(state: Annotated[BankState, InjectedState]) -> Union[str, dict]:
    """Consulta o limite de crédito atual disponível para o cliente autenticado."""

    try:
        logger.debug("Consulta de limite disponível: cpf=%s", state.get("cpf_cliente"))
        
        limite_atual = float(state["limite_atual"])
        
        return f"O limite de crédito atual é de R$ {limite_atual:.2f}."
    
    except (KeyError) as e:
        logger.error("Erro ao consultar limite disponível no estado do cliente: %s", e)
        return CreditError(error="""Não foi possível consultar o limite no momento. "
                Por favor, tente novamente mais tarde.""").model_dump()

    except Exception as e:
        logger.error("Erro inesperado ao consultar limite disponível: %s", e)
        return CreditError(error="""Não foi possível consultar o limite no momento. "
                Por favor, tente novamente mais tarde.""").model_dump()



@tool
def solicitar_aumento_limite(
    novo_limite_solicitado: float,
    state: Annotated[BankState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Registra formalmente um pedido de aumento de limite de crédito e o
    avalia automaticamente com base no score do cliente autenticado.
    """

    try:
        cpf_cliente = state["cpf_cliente"]
        limite_atual = float(state["limite_atual"])
        score = int(state["score_cliente"])
        
        logger.debug(
            "Pedido de aumento de limite: cpf=%s, limite_atual=R$ %.2f, "
            "novo_limite_solicitado=R$ %.2f, score=%s",
            cpf_cliente, limite_atual, novo_limite_solicitado, score,
        )

        if novo_limite_solicitado <= limite_atual:
            logger.info(
                "Pedido rejeitado sem registro (valor solicitado <= limite atual): "
                "cpf=%s, novo_limite_solicitado=R$ %.2f, limite_atual=R$ %.2f",
                cpf_cliente, novo_limite_solicitado, limite_atual,
            )

            texto = (
                f"O novo limite solicitado (R$ {novo_limite_solicitado:.2f}) precisa ser "
                f"maior que o limite atual (R$ {limite_atual:.2f})."
            )
            return Command(
                update={"messages": [ToolMessage(content=texto, tool_call_id=tool_call_id)]}
            )

        limite_maximo = limite_maximo_permitido(score)
        status = (
            "aprovado"
            if (limite_maximo is not None and novo_limite_solicitado <= limite_maximo)
            else "rejeitado"
        )

        registrar_solicitacao(cpf_cliente, limite_atual, novo_limite_solicitado, status)

        update = {}
        if status == "aprovado":
            texto = (
                f"Pedido registrado e aprovado! O novo limite de crédito é "
                f"R$ {novo_limite_solicitado:.2f}."
            )
    
            update["limite_atual"] = novo_limite_solicitado
        elif limite_maximo is not None:
            logger.info(
                "Pedido rejeitado (acima do limite máximo permitido para o score): "
                "cpf=%s, novo_limite_solicitado=R$ %.2f, limite_maximo=R$ %.2f, score=%s",
                cpf_cliente, novo_limite_solicitado, limite_maximo, score,
            )

            texto = (
                f"Pedido registrado, porém rejeitado com base no score atual. "
                f"Para o score atual, o limite máximo permitido é R$ {limite_maximo:.2f}. "
                f"Deseja ser encaminhado ao Agente de Entrevista de Crédito para tentar "
                f"melhorar seu score?"
            )
        else:
            logger.warning(
                "Pedido rejeitado por falta de faixa de score correspondente: "
                "cpf=%s, score=%s",
                cpf_cliente, score,
            )

            texto = (
                "Pedido registrado, porém rejeitado: não foi possível determinar o limite "
                "máximo permitido para o score atual. Deseja ser encaminhado ao Agente de "
                "Entrevista de Crédito?"
            )

        update["messages"] = [ToolMessage(content=texto, tool_call_id=tool_call_id)]
        return Command(update=update)

    except (FileNotFoundError) as e:
        logger.error("Erro de integração ao processar aumento de limite: %s", e)
        texto = """Não foi possível processar sua solicitação no momento. 
                        Por favor, tente novamente mais tarde."""
        return Command(
            update={"messages": [ToolMessage(content=texto, tool_call_id=tool_call_id)]}
        )

    except Exception as e:
        logger.error("Erro inesperado ao processar aumento de limite: %s", e)
        texto = """Não foi possível processar sua solicitação no momento. 
                Por favor, tente novamente mais tarde."""
        return Command(
            update={"messages": [ToolMessage(content=texto, tool_call_id=tool_call_id)]}
        )


credit_agent_system_prompt = """
    Você é o Credit Agent (Agente de Crédito) de um sistema bancário multi-agente.

    ## Contexto
    O cliente já foi autenticado antes de chegar até você, não peça CPF, data de nascimento ou qualquer outra credencial novamente; essa informação 
    já está disponível internamente.

    ## Função
    Sua responsabilidade é gerenciar consultas de limite de crédito e solicitações de aumento de limite.

    ## Responsabilidades
    - Se o cliente perguntar sobre seu limite de crédito, chame a função consultar_limite_disponivel imediatamente. Nunca responda em texto que você "vai consultar" ou peça para o cliente aguardar, apenas chame a função e use o resultado retornado como base da sua resposta.
    - Se o cliente informar o novo limite desejado, chame a função solicitar_aumento_limite (basta passar o valor desejado, o restante é resolvido internamente). Ela já registra o pedido em CSV e retorna o resultado (aprovado ou rejeitado).
      - Se aprovado: informar o cliente do novo limite e finalizar a interação.
      - Se rejeitado: informar o motivo e oferecer a opção de encaminhamento ao Credit Interview Agent para tentar ajustar o score.
        - Se o cliente aceitar, informar que sua parte está encerrada e que o próximo passo será a entrevista de crédito.
        - Se o cliente recusar, encerrar a conversa educadamente ou perguntar se há mais alguma coisa em que possa ajudar.

    ## Regras
    - Nunca invente valores de limite ou de score, use sempre as funções disponíveis.
    - Nunca mencione o nome de nenhuma função/ferramenta na sua resposta ao cliente. O cliente não deve saber que ferramentas existem, apenas veja o resultado.
    - Nunca diga que "vai verificar" ou "vai consultar" sem imediatamente chamar a função correspondente na mesma resposta.
    - Se a ferramenta retornar um erro, não exiba mensagens de erro internas. Em vez disso, informe ao cliente que o serviço está temporariamente indisponível ou que não foi possível processar a solicitação, e sugira tentar novamente mais tarde.
    - Se ocorrer algum erro interno, sempre informe ao cliente que o serviço está temporariamente indisponível ou que não foi possível processar a solicitação, e sugira tentar novamente mais tarde. Não mostre mensagens técnicas ou detalhes internos.
    
    ## Estilo de Resposta
    - Seja conciso, profissional e cordial.
"""


def build_credit_agent():
    """Compila o credit agent."""

    logger.debug("Construindo credit_agent")
     
    return create_agent(
            model=get_model(),
            tools=[consultar_limite_disponivel, solicitar_aumento_limite],
            name="credit_agent",
            system_prompt=credit_agent_system_prompt,
            state_schema=BankState,
            checkpointer=credit_checkpointer
        )
    
