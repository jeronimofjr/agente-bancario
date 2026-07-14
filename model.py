"""
Modulo responsável pela configuração dos modelos de LLM.
"""

from os import getenv

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

def get_model(model_env_var: str = "OPENAI_MODEL") -> ChatOpenAI:
    """Cria um cliente ChatOpenAI configurado para apontar para a API da OpenAI."""

    api_key = getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY não está configurada. Copie .env.example para .env e "
            "preencha com sua API key gerada em platform.openai.com/api-keys."
        )

    model_name = getenv(model_env_var, "gpt-4o-mini")

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        temperature=0.3,
        max_retries=2,
        timeout=30,
    )