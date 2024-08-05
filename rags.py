import argparse
import os.path
import warnings

from elasticsearch import Elasticsearch
from langchain.agents import create_react_agent
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain, LLMChain, RetrievalQAWithSourcesChain
from langchain.chains.combine_documents.refine import RefineDocumentsChain
from langchain.memory import (
    ConversationBufferMemory,
    ReadOnlySharedMemory
)
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama, OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
from utils import get_resized_images, img_prompt_func, AgenticRAG, AdaptiveRAG, CRAG, CodeAssistant, \
    get_prompt, SelfRAG, get_vectorstores
from utils.tools import *

warnings.filterwarnings("ignore")

# Set API Keys
# os.environ["OPENAI_API_KEY"] = getpass.getpass("Your OpenAI API key: ")


class LangchainModel:
    """
    Langchain Model class to handle different types of language models.
    """

    def __init__(self, llm_model, vectorstore_name):
        """
        Initialize the LangchainModel class with the specified LLM model type.

        Args:
            llm_model (str): The type of LLM model to use.
        """
        self.loader = None
        self.llm = OpenAI()
        self.results = None
        self.model_type = llm_model
        self.text_splitter = None
        self.model = None
        self.temperature = 0.1
        self.chain = None
        self.result = None
        self.results = None
        self.chat_history = []
        self.vectorstore_name = vectorstore_name
        self.create_db = False

    def model_chain_init(self, data_path, data_types):
        """
        Initialize the model chain based on the specified model type.

        Args:
            data_path (str): The path to the data directory.
            data_types (list): The list of data types to process.
        """
        if self.model_type == "agentic_rag":
            self._init_agentic_rag_chain(data_path, data_types)
        elif self.model_type == "adaptive_rag":
            self._init_adaptive_rag_chain(data_path, data_types)
        elif self.model_type == "crag":
            self._init_crag_chain(data_path, data_types)
        elif self.model_type == "self_rag":
            self._init_self_rag_chain(data_path, data_types)
        elif self.model_type == "code_assistant":
            self._init_code_assistant_chain(data_path, data_types)
        elif self.model_type == "react_agent":
            self._init_react_agent_chain(data_path, data_types)
        elif self.model_type == "gpt-4":
            self._init_gpt4_chain(data_path, data_types)
        elif self.model_type == "gpt-4o":
            self._init_gpt4o_chain(data_path, data_types)
        elif self.model_type == "claude":
            self._init_claude_chain(data_path, data_types)
        elif self.model_type == "mixtral_agent":
            self._init_mixtral_agent_chain(data_path, data_types)
        elif self.model_type in ["mistral", "llama:7b", "llama3:70b", "gemma", "mixtral", "command-r", "llama3:8b"]:
            self.ollama_chain_init(data_path, data_types)
        elif self.model_type == "bakllava":
            self._init_bakllava_chain(data_path)
        elif self.model_type == "gpt-4-vision":
            self._init_gpt4_vision_chain(data_path)

    def ollama_chain_init(self, data_path, data_types):
        """
        Initialize the Ollama chain with Qdrant embeddings.

        Args:
            data_path (str): The path to the data directory.
            data_types (list): The list of data types to process.
        """
        vector_store = get_vectorstores(self.vectorstore_name, data_path, data_types,
                                        OllamaEmbeddings(model=self.model_type), self.create_db)

        # Initialize the LLM
        self.llm = ChatOllama(model=self.model_type, streaming=True,
                              callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

        # Initialize conversational memory
        memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", return_messages=False)

        # Define the prompt template
        prompt_template = PromptTemplate(input_variables=["context", "question", "chat_history"],
                                         template=get_prompt("retrieval"))

        # Create the conversational retrieval chain
        self.chain = ConversationalRetrievalChain.from_llm(
            self.llm,
            memory=memory,
            retriever=vector_store.as_retriever(),
            combine_docs_chain_kwargs={"prompt": prompt_template},
            get_chat_history=lambda h: h,
            verbose=True
        )

    def _init_agentic_rag_chain(self, data_path, data_types):
        """
        Initialize the AgenticRAG chain.

        Args:
            data_path (str): The path to the data directory.
            data_types (list): The list of data types to process.
        """
        # Initialize vector database with embeddings
        vector_store = get_vectorstores(self.vectorstore_name, data_path, data_types, OpenAIEmbeddings(),
                                        self.create_db)

        # Create a retriever tool for the agent
        retriever_tool = create_retriever_tool(vector_store.as_retriever(), f"{os.path.basename(data_path)}",
                                               f"Searches and returns answers from {os.path.basename(data_path)} document.")

        # Initialize AgenticRAG chain with the retriever tool
        self.chain = AgenticRAG(retriever_tool)
        self.chain.create_graph()

    def _init_adaptive_rag_chain(self, data_path, data_types):
        """
        Initialize the AdaptiveRAG chain.

        Args:
            data_path (str): The path to the data directory.
            data_types (list): The list of data types to process.
        """
        # Initialize vector database with embeddings
        vector_store = get_vectorstores(self.vectorstore_name, data_path, data_types, OpenAIEmbeddings(),
                                        self.create_db)

        # Initialize AdaptiveRAG chain with the retriever tool
        self.chain = AdaptiveRAG(vector_store.as_retriever())
        self.chain.create_graph()

    def _init_crag_chain(self, data_path, data_types):
        """
        Initialize the CRAG chain.

        Args:
            data_path (str): The path to the data directory.
            data_types (list): The list of data types to process.
        """
        # Initialize vector database with embeddings
        vector_store = get_vectorstores(self.vectorstore_name, data_path, data_types, OpenAIEmbeddings(),
                                        self.create_db)

        # Initialize CRAG chain with the retriever tool
        self.chain = CRAG(vector_store.as_retriever())
        self.chain.create_graph()

    def _init_self_rag_chain(self, data_path, data_types):
        """
        Initialize the SelfRAG chain.

        Args:
            data_path (str): The path to the data directory.
            data_types (list): The list of data types to process.
        """
        # Initialize vector database with embeddings
        vector_store = get_vectorstores(self.vectorstore_name, data_path, data_types, OpenAIEmbeddings(),
                                        self.create_db)

        # Initialize SelfRAG chain with the retriever tool
        self.chain = SelfRAG(vector_store.as_retriever())
        self.chain.create_graph()

    def _init_code_assistant_chain(self, data_path, data_types):
        """
        Initialize the CodeAssistant chain.

        Args:
            data_path (str): The path to the data directory.
            data_types (list): The list of data types to process.
        """
        # Initialize vector database with embeddings
        # vector_db = chroma_embeddings(data_path, data_types, OpenAIEmbeddings(), self.create_db)

        # Initialize CodeAssistant chain with the retriever tool
        self.chain = CodeAssistant()
        self.chain.create_graph()

    def _init_react_agent_chain(self, data_path, data_types):
        """
        Initialize the ReAct agent chain.

        Args:
            data_path (str): The path to the data directory.
            data_types (list): The list of data types to process.
        """
        # Initialize vector database with embeddings
        vector_store = get_vectorstores(self.vectorstore_name, data_path, data_types, OpenAIEmbeddings(),
                                        self.create_db)

        # Initialize conversation memory buffer
        conv_memory = ConversationBufferMemory(memory_key="chat_history", input_key="input")

        # Define the prompt template for summary
        prompt_template = PromptTemplate(input_variables=["input", "chat_history"], template=get_prompt("summary"))

        # Initialize read-only memory
        read_only_memory = ReadOnlySharedMemory(memory=conv_memory)

        # Create the summary chain
        summary_chain = LLMChain(llm=self.llm, prompt=prompt_template, verbose=True, memory=read_only_memory)

        # Define summary memory tool
        summary_memory_tool = Tool(name="Summary", func=summary_chain.run,
                                   description="Useful for summarizing a conversation. The input should be a string representing who will read this summary.")

        # Create search and retrieval tools
        search_tool = create_search_tool("tavily")
        qa_retrieval_tool = retrieval_qa_tool(os.path.basename(data_path), vector_store, self.llm)
        retriever_tool = vectorstore_retriever_tool(os.path.basename(data_path), vector_store)

        # Combine all tools for the agent
        tools = [retriever_tool, search_tool, qa_retrieval_tool, summary_memory_tool]

        # Create ReAct agent with the tools
        react_agent = create_react_agent(ChatOpenAI(temperature=0, streaming=True, model="gpt-4"), tools,
                                         get_prompt("react"))

        # Initialize AgentExecutor with the ReAct agent
        self.chain = AgentExecutor(agent=react_agent, tools=tools, memory=conv_memory, verbose=True,
                                   handle_parsing_errors=True, return_intermediate_steps=True, include_run_info=True)

    def _init_gpt4_chain(self, data_path, data_types):
        """
        Initialize the GPT-4 chain.

        Args:
            data_path (str): The path to the data directory.
            data_types (list): The list of data types to process.
        """
        # Initialize vector database with embeddings
        vector_store = get_vectorstores(self.vectorstore_name, data_path, data_types, OpenAIEmbeddings(),
                                        self.create_db)

        # Initialize conversation memory buffer
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False)

        # Define the prompt template
        prompt_template = PromptTemplate(input_variables=["context", "question", "chat_history"],
                                         template=get_prompt("visa"))

        # Create the conversational retrieval chain with GPT-4
        self.chain = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(temperature=self.temperature, streaming=True, model_name="gpt-4-turbo"),
            retriever=vector_store.as_retriever(),
            return_source_documents=False,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt_template},
            get_chat_history=lambda h: h,
            verbose=True
        )

    def _init_gpt4o_chain(self, data_path, data_types):
        """
        Initialize the GPT-4o chain.

        Args:
            data_path (str): The path to the data directory.
            data_types (list): The list of data types to process.
        """
        # Initialize vector database with embeddings
        vector_store = get_vectorstores(self.vectorstore_name, data_path, data_types, OpenAIEmbeddings(),
                                        self.create_db)

        # Define the prompt template
        prompt = PromptTemplate(template=get_prompt("visa"), input_variables=["context", "question"])

        # Create an LLMChain with the prompt
        llm_chain = LLMChain(llm=ChatOpenAI(temperature=self.temperature, streaming=True, model_name=self.model_type),
                             prompt=prompt)

        # Create a RefineDocumentsChain as the combine_docs_chain
        combine_docs_chain = RefineDocumentsChain(
            initial_llm_chain=llm_chain,
            refine_llm_chain=llm_chain,
            document_variable_name="context",
            initial_response_name="initial_answer"
        )

        # Initialize conversation memory buffer
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False, output_key="answer")

        # Create the RetrievalQAWithSourcesChain with the refined document chain
        self.chain = RetrievalQAWithSourcesChain(
            combine_documents_chain=combine_docs_chain,
            memory=memory,
            retriever=vector_store.as_retriever(),
            return_source_documents=True,
            verbose=True
        )

    def _init_claude_chain(self, data_path, data_types):
        """
        Initialize the Claude chain.

        Args:
            data_path (str): The path to the data directory.
            data_types (list): The list of data types to process.
        """
        # Initialize vector database with embeddings
        vector_store = get_vectorstores(self.vectorstore_name, data_path, data_types, OpenAIEmbeddings(),
                                        self.create_db)
        # Define the LLM for Claude
        llm = ChatAnthropic(model="claude-3-opus-20240229", streaming=True)

        # Initialize conversation memory buffer
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False)

        # Define the prompt template
        prompt_template = PromptTemplate(input_variables=["context", "question", "chat_history"],
                                         template=get_prompt("visa"))

        # Create the conversational retrieval chain with Claude
        self.chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=vector_store.as_retriever(),
            return_source_documents=False,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt_template},
            get_chat_history=lambda h: h,
            verbose=True
        )

    def _init_mixtral_agent_chain(self, data_path, data_types):
        """
        Initialize the Mixtral agent chain.

        Args:
            data_path (str): The path to the data directory.
            data_types (list): The list of data types to process.
        """
        # Initialize vector database with embeddings
        vector_store = get_vectorstores(self.vectorstore_name, data_path, data_types, OllamaEmbeddings(model="mixtral"),
                                        self.create_db)

        # Initialize Ollama LLM for Mixtral
        self.llm = ChatOllama(model='mixtral', streaming=True,
                              callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

        # Initialize conversation memory buffer
        conv_memory = ConversationBufferMemory(memory_key="chat_history", input_key="input")

        # Create search tool and retriever tool
        search_tool = create_search_tool(engine="arxiv")
        retriever_tool = vectorstore_retriever_tool(os.path.basename(data_path), vector_store)

        # Combine tools for the agent
        tools = [retriever_tool, search_tool]

        # Create ReAct agent with the tools
        react_agent = create_react_agent(self.llm, tools, get_prompt("react"))

        # Initialize AgentExecutor with the ReAct agent
        self.chain = AgentExecutor(agent=react_agent, tools=tools, memory=conv_memory, verbose=True,
                                   handle_parsing_errors=True, return_intermediate_steps=True, include_run_info=True)

    def _init_bakllava_chain(self, data_path):
        """
        Initialize the Bakllava chain.

        Args:
            data_path (str): The path to the data directory.
        """
        # Initialize multi-modal vector store with OpenCLIP embeddings
        multi_modal_vectorstore = get_vectorstores(self.vectorstore_name, data_path, "image", OpenAIEmbeddings(),
                                                   self.create_db)

        # Initialize the LLM with streaming callback
        self.llm = Ollama(model=self.model_type, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

        # Define the Bakllava chain with image resizing and prompt functions
        self.chain = (
                {
                    "context": multi_modal_vectorstore.as_retriever | RunnableLambda(get_resized_images),
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(img_prompt_func)
                | self.llm
                | StrOutputParser()
        )

    def _init_gpt4_vision_chain(self, data_path):
        """
        Initialize the GPT-4 Vision chain.

        Args:
            data_path (str): The path to the data directory.
        """
        # Initialize multi-modal vector store with OpenCLIP embeddings
        multi_modal_vectorstore = get_vectorstores(self.vectorstore_name, data_path, "image", OpenAIEmbeddings(),
                                                   self.create_db)

        # Initialize GPT-4 Vision model
        model = ChatOpenAI(temperature=0, model="gpt-4-vision-preview", max_tokens=1024)

        # Define the GPT-4 Vision chain with image resizing and prompt functions
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

        Args:
            query_input (str): The query input for inference.
        """
        if self.model_type in ["react_agent", "mixtral_agent"]:
            # Invoke the chain with the query input for ReAct and Mixtral agents
            self.result = self.chain.invoke({"input": query_input, "chat_history": self.chat_history},
                                            include_run_info=True)
            self.results = self.result["output"]
            self.chat_history.append((query_input, self.results))

        elif self.model_type in ["gpt-4", "claude"]:
            # Perform inference with the query input for GPT-4 and Claude models
            self.result = self.chain({"question": query_input, "chat_history": self.chat_history},
                                     include_run_info=True)
            self.results = self.result["answer"]
            self.result["output"] = self.results
            self.chat_history.append((query_input, self.results))

        elif self.model_type == "gpt-4o":
            # Perform inference with the query input for GPT-4o model
            self.result = self.chain({"question": query_input, "chat_history": self.chat_history})
            self.results = self.result["answer"]
            self.chat_history.append((query_input, self.results))

        elif self.model_type in ["mistral", "llama:7b", "llama3:70b", "gemma", "mixtral", "command-r", "llama3:8b"]:
            # Run the chain with the query input for Mistral, Llama, Gemma, and Mixtral models
            self.results = self.chain.run({"question": query_input})
            self.chat_history.append((query_input, self.results))

        elif self.model_type in ["gpt-4-vision", "bakllava"]:
            # Invoke the chain with the query input for GPT-4 Vision and Bakllava models
            self.results = self.chain.invoke({"question": query_input})

        elif self.model_type in ["agentic_rag", "adaptive_rag", "code_assistant", "self_rag", "crag"]:
            # Invoke the chain with the query input for AgenticRAG, AdaptiveRAG, CodeAssistant, SelfRAG, and CRAG models
            self.results = self.chain.invoke(query_input)

        # Print and return the results
        print(self.results)
        return self.results, self.result


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        tuple: A tuple containing the directory, model type, and file formats.
    """
    parser = argparse.ArgumentParser(description='Langchain Models with different LLM.')
    parser.add_argument('--directory', default='./visa_data', help='Ingesting files Directory')
    parser.add_argument('--model_type',
                        choices=['react_agent', 'gpt-4', 'gpt-4o', 'gpt-4-vision', 'mistral', "llama3:70b",
                                 "llama:7b", "gemma", "crag", "mixtral", "self_rag", "bakllava", "mixtral_agent",
                                 "command-r", "agentic_rag", "llama3:8b",
                                 "adaptive_rag", "claude", "code_assistant"],
                        default="mistral", help='Model type for processing')
    parser.add_argument('--vectorstore', default="chroma", help='Embeddings Vectorstore', choices=["chroma",
                                                                                                   "milvus",
                                                                                                   "weaviate",
                                                                                                   "qdrant",
                                                                                                   "pinecone",
                                                                                                   "faiss",
                                                                                                   "elasticsearch",
                                                                                                   "opensearch",
                                                                                                   "openclip",
                                                                                                   "vectara",
                                                                                                   "neo4j"])
    parser.add_argument('--file_formats', nargs='+', default=['txt'],
                        help='List of file formats for loading documents')
    args = parser.parse_args()
    return args.directory, args.model_type, args.vectorstore, args.file_formats


def main():
    """
    Main function to run Langchain Model.
    """
    directory, model_type, vectorstore, file_formats = parse_arguments()
    # Langchain model init
    llm = LangchainModel(llm_model=model_type, vectorstore_name=vectorstore)
    llm.model_chain_init(directory, data_types=file_formats)
    while True:
        query = input("Please ask your question! ")
        llm.query_inferences(query)


if __name__ == "__main__":
    main()
