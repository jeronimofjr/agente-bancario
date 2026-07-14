"""
Módulo responsável pela definição de schemas relativos aos dados dos clientes.
"""

from typing import TypedDict

class Customer(TypedDict):
    cpf: str
    data_nascimento: str
    nome: str
    limite: float
    score: float