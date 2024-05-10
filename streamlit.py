import argparse
import streamlit as st
import os
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.chains.conversation.memory import ConversationBufferMemory
from rags import LangchainModel
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.prompts import MessagesPlaceholder
from langsmith import Client


client = Client()

st.set_page_config(
    page_title="ChatLangChain",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="collapsed",
)
"# Chat🦜🔗"


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Langchain Model with different model types.')
    parser.add_argument('--directory', default='./visa_data', help='Ingesting files Directory')
    parser.add_argument('--model_type',
                        choices=['react_agent', 'gpt-4', 'gpt-4-vision', 'mistral', "llama3-70b", "llama:7b", "gemma",
                                 "mixtral", "bakllava", "llama_agent", "command-r", "agentic_rag", "adaptive_rag",
                                 "code_assistant"],
                        default="gpt-4", help='Model type for processing')
    parser.add_argument('--file_formats', nargs='+', default=['pdf', 'txt'],
                        help='List of file formats for loading documents')
    args = parser.parse_args()
    return args.directory, args.model_type, args.file_formats


# @st.cache_resource(ttl="1h")
# LLM Model init
directory, model_type, file_formats = parse_arguments()
llm = LangchainModel(llm_model=model_type)
llm.model_chain_init(directory, data_types=file_formats)
outputs = {}

# Token Memory for Streamlit
if "agent" in model_type:
    memory = AgentTokenBufferMemory(llm=llm.llm)
else:
    memory = ConversationBufferMemory(llm=llm.llm)

# Streamlit Interface
starter_message = "Ask me anything!"
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [AIMessage(content=starter_message)]


def send_feedback(run_id, score):
    client.create_feedback(run_id, "user_score", score=score)


for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    memory.chat_memory.add_message(msg)

if prompt := st.chat_input(placeholder=starter_message):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response, results = llm.query_inferences(prompt)
        outputs["output"] = response
        st.session_state.messages.append(AIMessage(content=response))
        st.write(response)
        print(results)
        memory.save_context({"input": prompt}, results)
        st.session_state["messages"] = memory.buffer

        # Feedback setup
        run_id = results["__run"].run_id

        col_blank, col_text, col1, col2 = st.columns([10, 2, 1, 1])
        with col_text:
            st.text("Feedback:")

        with col1:
            st.button("👍", on_click=send_feedback, args=(run_id, 1))

        with col2:
            st.button("👎", on_click=send_feedback, args=(run_id, 0))
