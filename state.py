from langchain.agents import AgentState

class BankState(AgentState):
    cpf_cliente: str
    nome_cliente: str
    limite_atual: float
    score_cliente: int

class TriageState(AgentState):
    tentativas: int
    autenticado: bool
    encerrado: bool
    cpf_cliente: str
    nome_cliente: str
    limite_atual: float
    score_cliente: int