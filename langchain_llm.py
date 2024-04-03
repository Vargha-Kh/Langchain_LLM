import argparse
import os
import sys
import warnings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFDirectoryLoader, PythonLoader, \
    UnstructuredURLLoader, CSVLoader, UnstructuredCSVLoader
from langchain_community.embeddings import GPT4AllEmbeddings, OllamaEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import (
    CombinedMemory,
    ConversationBufferMemory,
    ConversationKGMemory,
    ConversationSummaryMemory,
)
from langchain_experimental.tools import PythonREPLTool
from langchain_core.prompts import PromptTemplate
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama, OpenAI
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI, OpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent, initialize_agent, Tool, ZeroShotAgent, \
    ConversationalChatAgent
from langchain.tools.retriever import create_retriever_tool
from langchain.chains import LLMChain
from langchain import hub
from langchain_community.utilities import GoogleSearchAPIWrapper
import os.path

warnings.filterwarnings("ignore")

# Set OpenAI API Key
openai_key = "OPENAI_API_KEY"
os.environ["OPENAI_API_KEY"] = openai_key
os.environ["GOOGLE_CSE_ID"] = "d3bbbf1c807c64465"
os.environ["GOOGLE_API_KEY"] = "AIzaSyAJqnjqhi1N0Fx5VpK04NyQCi890z-xUBc"


class LangchainModel:
    """
    Langchain Model class to handle different types of language models.
    """

    def __init__(self, llm_model):
        self.loader = None
        self.llm = OpenAI(tempurature=0)
        self.results = None
        self.model_type = llm_model
        self.text_splitter = None
        self.model = None
        self.temperature = 0.1
        self.chain = None
        self.chat_history = []
        self.openai_tools_prompt = hub.pull("hwchase17/openai-tools-agent")
        self.retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        self.retrieval_prompt = """
              You are an assistant for question-answering tasks.Use the following pieces of context to answer the question at the end. 
              If you don't know the answer, just say that you don't know, don't try to make up an answer. 
              {chat_history}
              
              {context}
              Question: {question}
              Helpful Answer:
              """
        self.conversation_prompt = """You are an assistant for question-answering tasks, Use the following pieces of context to answer the question. The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
            {context}
            Summary of conversation:
            {history}
            Current conversation:
            {chat_history_lines}
            Human: {question}
            AI:"""

    def documents_loader(self, data_path, data_types):
        """
        Load documents from the given directory based on the file mode.
        """
        all_texts = []
        for data_type in data_types:
            if data_type == 'py':
                self.loader = DirectoryLoader(data_path, glob="**/*.py", loader_cls=PythonLoader,
                                              use_multithreading=True)
            elif data_type == "pdf":
                self.loader = PyPDFDirectoryLoader(data_path)
            elif data_type == "url":
                urls = []
                with open(os.path.join(data_path, 'urls.txt'), 'r') as file:
                    for line in file:
                        urls.append(line.strip())
                self.loader = UnstructuredURLLoader(urls=urls)
            elif data_type == "csv":
                self.loader = DirectoryLoader(data_path, glob="**/*.csv", loader_cls=UnstructuredCSVLoader,
                                              use_multithreading=True)
            elif data_type == "txt":
                text_loader_kwargs = {'autodetect_encoding': True}
                self.loader = DirectoryLoader(data_path, glob="**/*.txt", loader_cls=TextLoader,
                                              loader_kwargs=text_loader_kwargs, use_multithreading=True)
            if self.loader is not None:
                texts = self.loader.load()
                all_texts.extend(texts)
            else:
                raise ValueError("Data file format is Not correct")
        return all_texts

    def chroma_initialization(self, data_path, text_chunks, embedding_function):
        if os.path.isfile(os.path.join(data_path, 'chroma.sqlite3')):
            vector_store = Chroma(persist_directory=data_path, embedding_function=embedding_function)
            vector_store.persist()
        else:
            vector_store = Chroma.from_documents(text_chunks, embedding=embedding_function,
                                                 persist_directory=data_path)
        return vector_store

    def ollama_create_chain(self, data_path, text_chunks):
        vector_store = self.chroma_initialization(data_path, text_chunks, OllamaEmbeddings(model=self.model_type))

        # LLM init
        llm = Ollama(model=self.model_type, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

        # Conversational Memory Buffer
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=False
        )
        # Knowledge Graph Memory
        # kg_memory = ConversationKGMemory(llm=llm)

        prompt_template = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template=self.retrieval_prompt,
        )

        self.chain = ConversationalRetrievalChain.from_llm(
            llm,
            memory=memory,
            retriever=vector_store.as_retriever(),
            combine_docs_chain_kwargs={
                "prompt": prompt_template,
            },
            get_chat_history=lambda h: h,
            verbose=True
        )

    def embedding_chunks(self, data_path, data_types):
        """
        Embed documents into chunks based on the model type.
        """
        embedding_data = self.documents_loader(data_path, data_types)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=100)
        text_chunks = text_splitter.split_documents(embedding_data)

        if self.model_type == "agent_gpt":
            vectordb = Chroma.from_text(text_chunks, embedding=OpenAIEmbeddings(), persist_directory=data_path)
            vectordb.persist()

            # Conversational Memory Buffer
            conv_memory = ConversationBufferMemory(
                memory_key="chat_history_lines", input_key="question"
            )
            # Summarization Memory
            summary_memory = ConversationSummaryMemory(llm=self.llm)
            # Memory Modules Combined
            memory = CombinedMemory(memories=[conv_memory, summary_memory])

            # Prompt template Definition
            prompt_template = PromptTemplate(
                input_variables=["history", "input", "chat_history_lines"],
                template=self.conversation_prompt,
            )

            # Google Search API tool initialization
            search = GoogleSearchAPIWrapper()
            search_tool = Tool(
                name="Search",
                func=search.run,
                description="useful for when you need to answer questions about current events",
            )

            chain_qa = RetrievalQA.from_chain_type(
                llm=self.llm, chain_type="stuff", retriever=vectordb.as_retriever()
            )

            qa_retrieval_tool = Tool(
                name=f"{os.path.basename(data_path)}",
                func=chain_qa.run,
                description=f"useful for when you need to answer questions about {os.path.basename(data_path)}. Input should be a fully formed question.",
            )

            # Agent's tools initialization for retrievers
            retriever_tool = create_retriever_tool(
                vectordb.as_retriever(),
                f"{os.path.basename(data_path)}",
                f"Searches and returns answers from {os.path.basename(data_path)} document.",
            )
            tools = [retriever_tool, search_tool, qa_retrieval_tool, PythonREPLTool()]

            # openai agent creation
            agent = create_openai_tools_agent(ChatOpenAI(temperature=0), tools, self.openai_tools_prompt)
            # llm_chain = LLMChain(llm=ChatOpenAI(temperature=0), prompt=self.openai_tools_prompt)
            # agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
            self.chain = AgentExecutor(
                agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True
            )

        elif self.model_type == "gpt-3.5":
            vectordb = Chroma.from_documents(text_chunks, embedding=OpenAIEmbeddings(), persist_directory=data_path)
            vectordb.persist()

            # Conversational Memory Buffer
            memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=False
            )
            # Knowledge Graph Memory
            # kg_memory = ConversationKGMemory(llm=llm)

            prompt_template = PromptTemplate(
                input_variables=["context", "question", "chat_history"],
                template=self.retrieval_prompt,
            )

            self.chain = ConversationalRetrievalChain.from_llm(
                ChatOpenAI(temperature=self.temperature, model_name="gpt-3.5"),
                retriever=vectordb.as_retriever(),
                return_source_documents=False,
                memory=memory,
                combine_docs_chain_kwargs={"prompt": prompt_template},
                get_chat_history=lambda h: h,
                verbose=True
            )

        elif self.model_type == "mistral" or "llama-7b" or "gemma" or "mixtral":
            self.ollama_create_chain(data_path, text_chunks)

    def query_inferences(self, query_input):
        """
        Perform inference based on the query input and the model type.
        """
        if self.model_type == "agent_gpt":
            self.result = self.chain.invoke({"input": query_input})
            self.results = self.result["output"]
            self.chat_history.append((query_input, self.results))
        elif self.model_type == "gpt-3.5":
            result = self.chain({"question": query_input, "chat_history": self.chat_history})
            self.results = result["answer"]
            self.chat_history.append((query_input, self.results))
        elif self.model_type == "mistral" or "llama-7b" or "gemma" or "mixtral":
            self.results = self.chain.run({"question": query_input})
            self.chat_history.append((query_input, self.results))
        print(self.results)
        return self.results


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Langchain Model with different model types.')
    parser.add_argument('--directory', default='./data', help='Ingesting files Directory')
    parser.add_argument('--model_type', choices=['agent_gpt', 'gpt-3.5', 'mistral', "llama-7b", "gemma", "mixtral"],
                        default='mistral', help='Model type for processing')
    parser.add_argument('--file_formats', nargs='+', default=['txt', 'pdf'],
                        help='List of file formats for loading documents')
    args = parser.parse_args()
    return args.directory, args.model_type, args.file_formats


def main():
    """
    Main function to run Langchain Model.
    """
    directory, model_type, file_formats = parse_arguments()
    llm = LangchainModel(llm_model=model_type)
    llm.embedding_chunks(directory, data_types=file_formats)
    while True:
        query = input("Please ask your question! ")
        llm.query_inferences(query)


if __name__ == "__main__":
    main()
