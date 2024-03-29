"""Python file to serve as the frontend"""
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
import argparse
from langchain_llm import LangchainModel


class StreamlitRAG:
    def __init__(self):
        self.llm = None
        directory, model_type, file_formats = parse_arguments()
        self.llm = LangchainModel(llm_model=model_type)
        self.llm.embedding_chunks(directory, data_types=file_formats)

    def page_config(self):
        st.set_page_config(page_title="Langchain RAG", page_icon="🤖")
        st.title("Chat with Documents")

        # sidebar
        with st.sidebar:
            st.header("Settings")

        # session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(content="Hello, I am a bot. How can I help you?"),
            ]

        # user input
        user_query = st.chat_input("Ask your question...")
        if user_query is not None and user_query != "":
            with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
                response = self.llm.query_inferences(user_query)
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))

        # conversation
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Langchain Model with different model types.')
    parser.add_argument('--directory', default='./data', help='Ingesting files Directory')
    parser.add_argument('--model_type', choices=['gpt-4', 'gpt-3.5', 'mistral', "llama-7b", "gemma", "mixtral"],
                        default='mistral', help='Model type for processing')
    parser.add_argument('--file_formats', nargs='+', default=['txt', 'pdf'],
                        help='List of file formats for loading documents')
    args = parser.parse_args()
    return args.directory, args.model_type, args.file_formats


if __name__ == "__main__":
    st_main_component = StreamlitRAG()
    st_main_component.page_config()
