"""
Interface de chat do Streamlit para o sistema bancário multiagente.
"""

import logging

from config.config_logging import setup_logging

setup_logging()

logger = logging.getLogger(__name__)

import streamlit as st

from agents.triage_agent import build_triage_agent
from main_agent import build_main_agent

st.set_page_config(page_title="Banco Ágil", page_icon="🏦")
st.title("🏦 Banco Ágil")


@st.cache_resource
def get_triage_agent():
    logger.info("Construindo triage_agent")
    return build_triage_agent()


@st.cache_resource
def get_main_agent():
    logger.info("Construindo main_agent")
    return build_main_agent()

def init_session():
    st.session_state.phase = "triage"
    st.session_state.messages = [
        ("assistant", "Olá! Bem-vindo(a). Para começar, informe seu CPF e sua data de nascimento (AAAA-MM-DD).")
    ]
    st.session_state.main_started = False
    st.session_state.triage_started = False
    st.session_state.customer_state = None
    st.session_state.thread_config = {"configurable": {"thread_id": "streamlit-session-1"}}


if "phase" not in st.session_state:
    init_session()


with st.sidebar:
    if st.button("🔄 Reiniciar atendimento"):
        init_session()
        st.rerun()


for role, msg in st.session_state.messages:
    st.chat_message(role).write(msg)

if "triage_started" not in st.session_state:
        st.session_state.triage_started = False

if st.session_state.phase != "ended":
    user_input = st.chat_input("Digite aqui...")
else:
    user_input = None
    st.info("Atendimento encerrado. Use o botão 'Reiniciar atendimento' na barra lateral para começar de novo.")

if user_input:
    st.session_state.messages.append(("user", user_input))
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                if st.session_state.phase == "triage":
                    if not st.session_state.triage_started:
                        payload = {
                            "messages": [{"role": "user", "content": user_input}],
                            "tentativas": 0,
                            "autenticado": False,
                            "encerrado": False,
                            "cpf_cliente": "",
                            "nome_cliente": "",
                            "limite_atual": 0.0,
                            "score_cliente": 0,
                        }
                        st.session_state.triage_started = True
                    else:
                        payload = {"messages": [{"role": "user", "content": user_input}]}
                        
                    result = get_triage_agent().invoke(payload, config=st.session_state.thread_config)
                    reply = result["messages"][-1].content
                             
                    if result.get("encerrado"):
                        st.session_state.phase = "ended"
                    elif result.get("autenticado"):
                        st.session_state.customer_state = {
                            "cpf_cliente": result["cpf_cliente"],
                            "nome_cliente": result["nome_cliente"],
                            "limite_atual": result["limite_atual"],
                            "score_cliente": result["score_cliente"],
                        }
                        st.session_state.phase = "chat"

                elif st.session_state.phase == "chat":
                    if user_input.lower() in {"exit", "quit", "sair"}:
                        reply = "Atendimento encerrado. Obrigado!"
                        st.session_state.phase = "ended"
                    else:
                        if not st.session_state.main_started:
                            payload = {
                                "messages": [{"role": "user", "content": user_input}],
                                **st.session_state.customer_state,
                            }
                            st.session_state.main_started = True
                        else:
                            payload = {"messages": [{"role": "user", "content": user_input}]}

                        result = get_main_agent().invoke(payload, config=st.session_state.thread_config)
                        reply = result["messages"][-1].content

            except Exception:
                logger.exception("Erro ao processar a mensagem do usuário")
                reply = "Ocorreu um erro inesperado ao processar sua solicitação. Veja o terminal para detalhes."

        st.write(reply)

    st.session_state.messages.append(("assistant", reply))
    st.rerun()