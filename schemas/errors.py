"""
Módulo responsável pela definição de schemas relativos a possivéis erros durante a execução do sistema.
"""

from pydantic import BaseModel

class ExchangeRateError(BaseModel):
    """Contrato tipado para o payload de erro da função `get_exchange_rate`."""

    error: str


class CreditError(BaseModel):
    """Contrato tipado para o payload de erro das ferramentas do Agente de Crédito."""

    error: str
