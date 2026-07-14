"""
Módulo responsável por funcões auxiliares.
"""

from pathlib import Path
from typing import List

import pandas as pd

from schemas.customer import Customer

customers_path = Path(__file__).parent.parent / "data" / "clientes.csv"

def carregar_clientes() -> List[dict]:
    """Carrega a base de clientes disponível em `data/clientes.csv`."""
    
    data = pd.read_csv(customers_path, dtype={"cpf": str})
    customers: List[Customer] = data.to_dict('records')
    
    return customers

def formatar_moeda(valor: float) -> str:
    return (
        f"R$ {valor:,.2f}"
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )

def normalize_cpf(cpf: str) -> str:
    return "".join(ch for ch in cpf if ch.isdigit())
