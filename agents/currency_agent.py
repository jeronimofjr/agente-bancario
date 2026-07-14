"""
Agente de Câmbio.

Módulo responsável pela construção do Agente de Câmbio.
O Agente permite que o cliente consulte a cotação de moedas
em tempo real.
"""

import logging
import os

import httpx
from langchain.agents import create_agent
from langchain.tools import tool

from model import get_model
from schemas.errors import ExchangeRateError

logger = logging.getLogger(__name__)

frankfurter_api_base = os.environ.get(
    "FRANKFURTER_API_BASE", "https://api.frankfurter.dev/v1"
)


@tool
def get_exchange_rate(
    currency_from: str = "USD",
    currency_to: str = "BRL",
    currency_date: str = "latest",
) -> dict:
    """Obtenha a taxa de câmbio entre duas moedas.

    Args:
        currency_from: Código de três letras da moeda de origem. ex.: "USD".
        currency_to: Código de três letras da moeda de destino.  ex.: "BRL".
        currency_date: "latest" ou uma data no formato ISO (AAAA-MM-DD) para uma
        taxa histórica.
    """

    currency_from = currency_from.strip().upper()
    currency_to = currency_to.strip().upper()

    logger.debug(
        "get_exchange_rate chamado: from=%s, to=%s, date=%s",
        currency_from,
        currency_to,
        currency_date,
    )

    try:
        url = f"{frankfurter_api_base}/{currency_date}"
        params = {"from": currency_from, "to": currency_to}

        response = httpx.get(
            url,
            params=params,
            timeout=10.0,
        )
        response.raise_for_status()
        data = response.json()

        if "rates" not in data or not data["rates"]:
            logger.error("Formato de resposta da API inesperado: %s", data)

            return ExchangeRateError(
                error="Formato de resposta da API inválido"
            ).model_dump()

        target, rate = list(data["rates"].items())[0]

        logger.info(
            "Taxa obtida com sucesso: %s -> %s = %s (data=%s)",
            data["base"],
            target,
            rate,
            data["date"],
        )

        return {
            "base": data["base"],
            "target": target,
            "rate": rate,
            "date": data["date"],
        }
    except httpx.HTTPError as e:
        logger.error("A solicitação à API de taxas de câmbio falhou.: %s", e)
        return ExchangeRateError(error=f"A solicitação à API falhou").model_dump()
    except ValueError as e:
        logger.error("A API de taxas de câmbio retornou um JSON inválido.: %s", e)
        return ExchangeRateError(
            error=f"A API retornou uma resposta inválida."
        ).model_dump()


currency_agent_system_prompt = """
    Você é um agente de câmbio de um sistema bancário.

    ## Função
    Sua única responsabilidade é fornecer informações sobre taxas de câmbio.

    ## Responsabilidades
    - Obter taxas de câmbio atuais ou históricas usando a ferramenta `get_exchange_rate`.
    - Apresentar a taxa de câmbio solicitada de forma clara e precisa.
    - Encerrar a interação com uma mensagem de despedida breve e cordial.

    ## Regras
    - Você deve utilizar a ferramenta `get_exchange_rate` sempre que informações sobre taxas de câmbio forem solicitadas.
    - Nunca invente, estime ou fabrique taxas de câmbio.
    - Você só pode responder a perguntas sobre taxas de câmbio atuais ou históricas se a informação estiver disponível por meio da ferramenta.
    - Não faça previsões, projeções nem especulações sobre taxas de câmbio futuras.
    - Não forneça aconselhamento financeiro ou de investimento.
    - Interprete nomes comuns de moedas, símbolos e códigos ISO de moedas sempre que possível (ex.: dólar → USD, euro → EUR, libra esterlina → GBP, iene → JPY, real → BRL).
    - Se a solicitação do usuário não contiver informações suficientes para identificar as moedas de origem e de destino, faça uma pergunta de esclarecimento antes de chamar a ferramenta.
    - Se a solicitação do usuário contiver apenas informações sobre a moeda de origem, presuponha que a moeda de destino é o real -> BRL.
    - Se a moeda solicitada não for suportada, explique que ela está indisponível.
    - Se a ferramenta retornar um erro, não exiba mensagens de erro internas. Em vez disso, informe ao usuário que a taxa de câmbio está temporariamente indisponível e sugira tentar novamente mais tarde.
    - Se o usuário perguntar sobre assuntos não relacionados a câmbio, explique educadamente que você só pode ajudar com informações sobre taxas de câmbio.

    ## Estilo de Resposta
    - Seja conciso, profissional e cordial.
    - Responda sempre no mesmo idioma utilizado pelo usuário.
"""


def build_currency_agent(model=None):
    """Compila o currency agent."""

    logger.debug(
        "Construindo currency_agent (model=%s)",
        "custom" if model else "default (get_model())",
    )

    return create_agent(
        model=model or get_model(),
        tools=[get_exchange_rate],
        name="currency_agent",
        system_prompt=(currency_agent_system_prompt),
    )
