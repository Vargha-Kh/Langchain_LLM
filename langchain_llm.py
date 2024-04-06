import argparse
import os
import warnings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFDirectoryLoader, PythonLoader, \
    UnstructuredURLLoader, CSVLoader, UnstructuredCSVLoader
from langchain_community.embeddings import GPT4AllEmbeddings, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import (
    CombinedMemory,
    ConversationBufferMemory,
    ConversationKGMemory,
    ConversationSummaryMemory,
)
from langchain_community.tools.tavily_search import TavilySearchResults, TavilyAnswer
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_experimental.tools import PythonREPLTool
from langchain_core.prompts import PromptTemplate
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama, OpenAI
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_openai_tools_agent, create_openai_functions_agent, initialize_agent, \
    Tool, ZeroShotAgent, \
    ConversationalChatAgent, create_react_agent, create_self_ask_with_search_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_experimental.autonomous_agents import AutoGPT
from langchain import hub
from utils import get_resized_images, resize_base64_image, img_prompt_func
from langchain_community.utilities import GoogleSearchAPIWrapper
import os.path

warnings.filterwarnings("ignore")

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
os.environ["GOOGLE_CSE_ID"] = "YOUR_GOOGLE_CSE_ID"
os.environ["GOOGLE_API_KEY"] = "YOUR-GOOGLE-API-KEY"
os.environ["TAVILY_API_KEY"] = "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"


class LangchainModel:
    """
    Langchain Model class to handle different types of language models.
    """

    def __init__(self, llm_model):
        """
            Class to handle interactions with different types of language models provided by Langchain.
        """
        self.loader = None
        self.llm = OpenAI()
        self.results = None
        self.model_type = llm_model
        self.text_splitter = None
        self.model = None
        self.temperature = 0.1
        self.chain = None
        self.chat_history = []
        self.openai_tools_prompt = hub.pull("hwchase17/openai-tools-agent")
        self.openai_functions_prompt = hub.pull("hwchase17/openai-functions-agent")
        # self.retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        # self.rag_prompt = hub.pull("rlm/rag-prompt")
        self.ask_search_prompt = hub.pull("hwchase17/self-ask-with-search")
        self.react_prompt = hub.pull("hwchase17/react-chat")
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
        Load documents from a given directory and return a list of texts.
        The method supports multiple data types including python files, PDFs, URLs, CSVs, and text files.
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
                if data_type == "txt":
                    all_texts.extend(texts[0])
                else:
                    all_texts.extend(texts)
            else:
                raise ValueError("Data file format is Not correct")
        return all_texts

    def ollama_chain_init(self, data_path, data_types):
        vector_store = self.chroma_embeddings(data_path, data_types, OllamaEmbeddings(model=self.model_type))

        # LLM init
        llm = Ollama(model=self.model_type, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

        # Conversational Memory Buffer
        memory = ConversationBufferMemory(
            memory_key="chat_history", input_key="question", return_messages=True
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

    def chroma_embeddings(self, data_path, data_types, embedding_function):
        try:
            if os.path.isfile(os.path.join(data_path, 'chroma.sqlite3')):
                vector_store = Chroma(persist_directory=data_path, embedding_function=embedding_function)
                vector_store.persist()
            else:
                embedding_data = self.documents_loader(data_path, data_types)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=100)
                text_chunks = text_splitter.split_documents(embedding_data)
                vector_store = Chroma.from_documents(text_chunks, embedding=embedding_function,
                                                     persist_directory=data_path)
        except InvalidDimensionException:
            Chroma().delete_collection()
            embedding_data = self.documents_loader(data_path, data_types)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=100)
            text_chunks = text_splitter.split_documents(embedding_data)
            vector_store = Chroma.from_documents(text_chunks, embedding=embedding_function,
                                                 persist_directory=data_path)

        return vector_store

    def model_chain_init(self, data_path, data_types):
        """
        Embed documents into chunks based on the model type.
        """

        if self.model_type == "gpt-4-vision":
            # Load chroma
            multi_modal_vectorstore = Chroma(
                collection_name="multi-modal-rag",
                persist_directory=data_path,
                embedding_function=OpenCLIPEmbeddings(
                    model_name="ViT-H-14", checkpoint="laion2b_s32b_b79k"
                ),
            )
            # Initialize the multi-modal Large Language Model with specific parameters
            model = ChatOpenAI(temperature=0, model="gpt-4-vision-preview", max_tokens=1024)

            # Define the RAG pipeline
            self.chain = (
                    {
                        "context": multi_modal_vectorstore.as_retriever | RunnableLambda(get_resized_images),
                        "question": RunnablePassthrough(),
                    }
                    | RunnableLambda(img_prompt_func)
                    | model
                    | StrOutputParser()
            )
        if self.model_type == "agent_gpt":
            vector_db = self.chroma_embeddings(data_path, data_types, OpenAIEmbeddings())

            # Conversational Memory Buffer
            conv_memory = ConversationBufferMemory(
                memory_key="chat_history", input_key="input"
            )

            # Summarization Memory
            # summary_memory = ConversationSummaryMemory(llm=self.llm)
            # Memory Modules Combined
            # memory = CombinedMemory(memories=[conv_memory, summary_memory])

            # Google Search API tool initialization
            # google_search = GoogleSearchAPIWrapper()
            # google_search_tool = Tool(
            #     name="Search",
            #     func=google_search.run,
            #     description="useful for when you need to answer questions about current events",
            # )

            search_tool = TavilySearchResults(max_results=1)

            chain_qa = RetrievalQA.from_chain_type(
                llm=self.llm, chain_type="stuff", retriever=vector_db.as_retriever()
            )

            qa_retrieval_tool = Tool(
                name=f"{os.path.basename(data_path)}",
                func=chain_qa.run,
                description=f"useful for when you need to answer questions about {os.path.basename(data_path)}. Input should be a fully formed question.",
            )

            # Agent's tools initialization for retrievers
            retriever_tool = create_retriever_tool(
                vector_db.as_retriever(),
                f"{os.path.basename(data_path)}",
                f"Searches and returns answers from {os.path.basename(data_path)} document.",
            )
            tools = [retriever_tool, search_tool, qa_retrieval_tool, PythonREPLTool()]

            # Openai function Agent
            openai_agent = create_openai_functions_agent(ChatOpenAI(model="gpt-3.5-turbo-1106"), tools,
                                                         self.openai_functions_prompt)

            # Create ReAct Agent
            react_agent = create_react_agent(ChatOpenAI(temperature=0), tools, self.react_prompt)

            # Create self-ask search Agent
            self_ask_agent = create_self_ask_with_search_agent(ChatOpenAI(temperature=0), [
                TavilyAnswer(max_results=1, name="Intermediate Answer")], self.ask_search_prompt)

            self.chain = AgentExecutor(
                agent=react_agent, tools=tools, memory=conv_memory, verbose=True, handle_parsing_errors=True,
                return_intermediate_steps=True
            )

        elif self.model_type == "gpt-3.5":
            vector_db = self.chroma_embeddings(data_path, data_types, OpenAIEmbeddings())

            # Conversational Memory Buffer
            memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=False
            )
            # Knowledge Graph Memory
            # kg_memory = ConversationKGMemory(llm=self.llm)

            prompt_template = PromptTemplate(
                input_variables=["context", "question", "chat_history"],
                template=self.retrieval_prompt,
            )

            self.chain = ConversationalRetrievalChain.from_llm(
                ChatOpenAI(temperature=self.temperature, model_name="gpt-3.5-turbo"),
                retriever=vector_db.as_retriever(),
                return_source_documents=False,
                memory=memory,
                combine_docs_chain_kwargs={"prompt": prompt_template},
                get_chat_history=lambda h: h,
                verbose=True
            )
        elif self.model_type == "mixtral_agent":
            vector_db = self.chroma_embeddings(data_path, data_types, OllamaEmbeddings(model="mixtral"))

            # Ollama init
            llm = ChatOllama(model='mixtral', callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

            # Conversational Memory Buffer
            conv_memory = ConversationBufferMemory(
                memory_key="chat_history", input_key="input"
            )

            # Summarization Memory
            # summary_memory = ConversationSummaryMemory(llm=self.llm)
            # Memory Modules Combined
            # memory = CombinedMemory(memories=[conv_memory, summary_memory])

            search_tool = TavilySearchResults(max_results=1)

            chain_qa = RetrievalQA.from_chain_type(
                llm=llm, chain_type="stuff", retriever=vector_db.as_retriever()
            )

            qa_retrieval_tool = Tool(
                name=f"{os.path.basename(data_path)}",
                func=chain_qa.run,
                description=f"useful for when you need to answer questions about {os.path.basename(data_path)}. Input should be a fully formed question.",
            )

            # Agent's tools initialization for retrievers
            retriever_tool = create_retriever_tool(
                vector_db.as_retriever(),
                f"{os.path.basename(data_path)}",
                f"Searches and returns answers from {os.path.basename(data_path)} document.",
            )
            tools = [retriever_tool, search_tool, qa_retrieval_tool, PythonREPLTool()]

            # Create ReAct Agent
            react_agent = create_react_agent(llm, tools, self.react_prompt)

            self.chain = AgentExecutor(
                agent=react_agent, tools=tools, memory=conv_memory, verbose=True, handle_parsing_errors=True,
                return_intermediate_steps=True
            )


        elif self.model_type == "mistral" or "llama-7b" or "gemma" or "mixtral":
            self.ollama_chain_init(data_path, data_types)
        elif self.model_type == "bakllava":
            # Load chroma
            multi_modal_vectorstore = Chroma(
                collection_name="multi-modal-rag",
                persist_directory=data_path,
                embedding_function=OpenCLIPEmbeddings(
                    model_name="ViT-H-14", checkpoint="laion2b_s32b_b79k"
                ),
            )

            # LLM init
            model = Ollama(model=self.model_type, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

            # Define the RAG pipeline
            self.chain = (
                    {
                        "context": multi_modal_vectorstore.as_retriever | RunnableLambda(get_resized_images),
                        "question": RunnablePassthrough(),
                    }
                    | RunnableLambda(img_prompt_func)
                    | model
                    | StrOutputParser()
            )

    def query_inferences(self, query_input):
        """
        Perform inference based on the query input and the model type.
        """
        if self.model_type == "agent_gpt" or "mixtral_agent":
            self.result = self.chain.invoke({"input": query_input, "chat_history": self.chat_history})
            self.results = self.result["output"]
            self.chat_history.append((query_input, self.results))
        elif self.model_type == "gpt-3.5":
            result = self.chain({"question": query_input, "chat_history": self.chat_history})
            self.results = result["answer"]
            self.chat_history.append((query_input, self.results))
        elif self.model_type == "mistral" or "llama-7b" or "gemma" or "mixtral":
            self.results = self.chain.run({"question": query_input})
            self.chat_history.append((query_input, self.results))
        if self.model_type == "gpt-4-vision":
            self.result = self.chain.invoke({"input": query_input})
            self.results = self.result["output"]
        print(self.results)
        return self.results


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Langchain Model with different model types.')
    parser.add_argument('--directory', default='./data', help='Ingesting files Directory')
    parser.add_argument('--model_type',
                        choices=['agent_gpt', 'gpt-3.5', 'gpt-4-vision', 'mistral', "llama-7b", "gemma", "mixtral",
                                 "bakllava", "mixtral_agent"],
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
    llm.model_chain_init(directory, data_types=file_formats)
    while True:
        query = input("Please ask your question! ")
        llm.query_inferences(query)


if __name__ == "__main__":
    main()
