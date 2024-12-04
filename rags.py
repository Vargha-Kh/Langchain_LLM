import argparse
import os.path
import warnings
from elasticsearch import Elasticsearch
from langchain.agents import create_react_agent, AgentExecutor
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain, LLMChain, RetrievalQAWithSourcesChain
from langchain.chains.combine_documents.refine import RefineDocumentsChain
from langchain.memory import (
    ConversationBufferMemory,
    ReadOnlySharedMemory
)
from typing import List
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama, OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import HypotheticalDocumentEmbedder
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import ContextualCompressionRetriever, MultiQueryRetriever
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.retrievers.document_compressors import CohereRerank
from langchain_elasticsearch import ElasticsearchEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from langchain_pinecone import PineconeEmbeddings
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from utils import get_resized_images, img_prompt_func, AgenticRAG, AdaptiveRAG, CRAG, CodeAssistant, \
    get_prompt, SelfRAG, get_vectorstores
from utils.tools import *

warnings.filterwarnings("ignore")


# Set API Keys
# os.environ["OPENAI_API_KEY"] = getpass.getpass("Your OpenAI API key: ")

# Output parser will split the LLM result into a list of queries
class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # Remove empty lines


class LangchainModel:
    """
    Langchain Model class to handle different types of language models.
    """

    def __init__(self, llm_model, vectorstore_name, embeddings_model="openai", use_mongo_memory=False,
                 use_cohere_rank=False, use_multi_query_retriever=False, use_contextual_compression=False,
                 use_hyde=False):
        """
        Initialize the LangchainModel class with the specified LLM model type and options.

        Args:
            llm_model (str): The type of LLM model to use.
            vectorstore_name (str): The name of the vector store to use.
            embeddings_model (str): The embeddings model to use.
            use_mongo_memory (bool): Whether to use MongoDB for chat history.
            use_cohere_rank (bool): Whether to use Cohere Rank for retriever compression.
            use_multi_query_retriever (bool): Whether to use MultiQueryRetriever.
            use_contextual_compression (bool): Whether to use Contextual Compression Retriever.
            use_hyde (bool): Whether to Hypothetical Embedding for documents
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
        self.database_collection_name = "RAG"
        self.chunk_size = 5000
        self.use_mongo_memory = use_mongo_memory
        self.use_cohere_rank = use_cohere_rank
        self.use_multi_query_retriever = use_multi_query_retriever
        self.use_contextual_compression = use_contextual_compression
        self.use_hyde = use_hyde
        self.embeddings_model = embeddings_model

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
        elif self.model_type in ["mistral", "llama:7b", "llama3:70b", "gemma", "mixtral", "command-r", "llama3:8b",
                                 "gpt-4o", "gpt-4o-mini", "gpt-4"]:
            self._init_rag_chain(data_path, data_types)
        elif self.model_type == "mixtral_agent":
            self._init_mixtral_agent_chain(data_path, data_types)
        # elif self.model_type in ["mistral", "llama:7b", "llama3:70b", "gemma", "mixtral", "command-r", "llama3:8b"]:
        #     self.ollama_chain_init(data_path, data_types)
        elif self.model_type == "bakllava":
            self._init_bakllava_chain(data_path)
        elif self.model_type == "gpt-4-vision":
            self._init_gpt4_vision_chain(data_path)

    def _select_embeddings_model(self):
        """
        Select the embeddings model based on the embeddings_model attribute.

        Returns:
            BaseEmbeddings: The selected embeddings instance.
        """
        if self.embeddings_model == "elasticsearch":
            return ElasticsearchEmbeddings.from_credentials(
                model_id="your_model_id",
                es_cloud_id=os.getenv("ES_CLOUD_ID"),
                es_user=os.getenv("ES_USER"),
                es_password=os.getenv("ES_PASSWORD"),
            )
        elif self.embeddings_model == "mistralai":
            return MistralAIEmbeddings(model="mistral-embed")
        elif self.embeddings_model == "pinecone":
            return PineconeEmbeddings(model="multilingual-e5-large")
        elif self.embeddings_model == "voyage":
            return VoyageAIEmbeddings(
                voyage_api_key=os.getenv("VOYAGE_API_KEY"),
                model="voyage-law-2",
            )
        elif self.embeddings_model == "fastembed":
            return FastEmbedEmbeddings()
        elif self.embeddings_model == "huggingface":
            return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        elif self.embeddings_model == "ollama":
            return OllamaEmbeddings(model=self.model_type)
        else:
            # Default to OpenAI Embeddings
            return OpenAIEmbeddings(model="text-embedding-3-small")

    def _init_agentic_rag_chain(self, data_path, data_types):
        """
        Initialize the AgenticRAG chain.

        Args:
            data_path (str): The path to the data directory.
            data_types (list): The list of data types to process.
        """
        # Initialize vector database with embeddings
        vector_store = get_vectorstores(self.vectorstore_name, data_path, data_types, OpenAIEmbeddings(),
                                        self.database_collection_name, self.chunk_size, self.create_db)

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
                                        self.database_collection_name, self.chunk_size, self.create_db)

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
                                        self.database_collection_name, self.chunk_size, self.create_db)

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
                                        self.database_collection_name, self.chunk_size, self.create_db)

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
                                        self.database_collection_name, self.chunk_size, self.create_db)

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

    def _init_rag_chain(self, data_path, data_types):
        """
        Initialize the GPT-4o chain.

        Args:
            data_path (str): The path to the data directory.
            data_types (list): The list of data types to process.
        """
        # Initialize vector database with embeddings
        # Determine the embeddings based on model type and use_hyde flag
        if self.use_hyde:
            # Use HYDE embeddings
            hyde_llm = OpenAI(n=5, best_of=5)
            if self.model_type in [
                "llama3:8b",
                "gemma",
                "mistral",
                "mixtral",
                "llama3:70b",
                "llama3.1",
            ]:
                # Use OllamaEmbeddings with HYDE
                base_embeddings = OllamaEmbeddings(model=self.model_type)
            else:
                # Use OpenAIEmbeddings with HYDE
                base_embeddings = OpenAIEmbeddings(model="text-embedding-ada-003")
            embeddings = HypotheticalDocumentEmbedder.from_llm(
                llm=hyde_llm,
                embeddings=base_embeddings,
                prompt_template="web_search",
            )
        else:
            # Use standard embeddings without HYDE
            embeddings = self._select_embeddings_model()

        # Now call get_vectorstores once with the determined embeddings
        vector_store = get_vectorstores(
            self.vectorstore_name,
            data_path,
            data_types,
            embeddings,
            self.database_collection_name,
            self.chunk_size,
            self.create_db
        )

        # Set up the chat model based on the model_choice
        if self.model_type in ['gpt-4', 'gpt-4o', 'gpt-4o-mini', 'o1-preview']:
            self.chat_model = ChatOpenAI(temperature=self.temperature, streaming=True, model_name=self.model_type,
                                         callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
            input_variable = "input"
        elif self.model_type == 'claude':
            self.chat_model = ChatAnthropic(model="claude-3-opus-20240229", streaming=True)
        elif self.model_type in ['llama3:8b', 'gemma', 'mistral', 'mixtral', 'llama3:70b', 'llama3.1']:
            self.chat_model = ChatOllama(model=self.model_type, streaming=True,
                                         callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
            input_variable = "question"
        else:
            raise ValueError(f"Unsupported model choice: {self.model_type}")

        # Define a helper function to retrieve chat history
        def get_chat_history(session_id=None):
            if self.use_mongo_memory:
                # Initialize MongoDB chat history based on session ID
                self.chat_history = MongoDBChatMessageHistory(
                    session_id=session_id,
                    connection_string="mongodb://localhost:27017/",  # MongoDB connection
                    database_name="chat_history",  # Database name for chat history
                    collection_name="operation_chat_histories",  # Collection name for storing histories
                )
            else:
                # Use in-memory chat history
                self.chat_history = []
            return self.chat_history

        # Get the chat history (session_id can be None or a generated ID)
        self.chat_history = get_chat_history()

        # Optionally set up MultiQueryRetriever
        if self.use_multi_query_retriever:
            # Create the output parser for the MultiQueryRetriever
            output_parser = LineListOutputParser()

            # Create the MultiQueryRetriever
            multi_query_prompt = ChatPromptTemplate.from_messages([
                ("system", get_prompt("multi_query")),
                ("human", "{input}")
            ])

            # Create the LLM chain for the MultiQueryRetriever
            llm_chain = LLMChain(llm=self.chat_model, prompt=multi_query_prompt)

            # Create the MultiQueryRetriever
            multi_query_retriever = MultiQueryRetriever(
                retriever=vector_store.as_retriever(),
                llm_chain=llm_chain,
                parser=output_parser,
                combine_results=True,
                k=5,
            )
            retriever = multi_query_retriever
        else:
            # Use the basic retriever
            retriever = vector_store.as_retriever()

        # Optionally set up ContextualCompressionRetriever
        if self.use_contextual_compression:
            # Optionally use Cohere Ranker
            if self.use_cohere_rank:
                base_compressor = CohereRerank()
            else:
                base_compressor = LLMChainFilter.from_llm(
                    self.chat_model)  # You can set a default compressor or leave it as None

            if base_compressor:
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=base_compressor,
                    base_retriever=retriever
                )
                retriever = compression_retriever
        else:
            # If no contextual compression is used, keep the retriever as is
            pass

        # Define condense question prompt
        condense_question_prompt = ChatPromptTemplate.from_messages([
            ("system", get_prompt("condense_question")),
            ("placeholder", "{chat_history}"),
            ("human", "{input}")
        ])

        # Create a history-aware retriever to manage context from previous chats
        history_aware_retriever = create_history_aware_retriever(
            self.chat_model,  # Language model
            retriever,  # Vector store retriever
            condense_question_prompt,  # Prompt template to condense questions
        )

        # Define the prompt template for question answering
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", get_prompt("retrieval")),  # System message for setting the context
            ("placeholder", "{chat_history}"),  # Placeholder for chat history
            ("human", "{input}")  # Placeholder for user's input
        ])

        # Create a chain to process and return answers based on documents retrieved
        qa_chain = create_stuff_documents_chain(self.chat_model, qa_prompt)

        # Set up the overall chain with message history management
        if self.use_mongo_memory:
            self.chain = RunnableWithMessageHistory(
                create_retrieval_chain(history_aware_retriever, qa_chain),  # Combine retriever and QA chain
                lambda session_id: get_chat_history(session_id),  # Fetch chat history per session
                input_messages_key="input",  # Key for input messages (user's questions)
                history_messages_key="chat_history",  # Key for accessing chat history
            )
        else:
            # Use in-memory chat history
            self.chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    def _init_mixtral_agent_chain(self, data_path, data_types):
        """
        Initialize the Mixtral agent chain.

        Args:
            data_path (str): The path to the data directory.
            data_types (list): The list of data types to process.
        """
        # Initialize vector database with embeddings
        vector_store = get_vectorstores(self.vectorstore_name, data_path, data_types, OllamaEmbeddings(model="mixtral"),
                                        self.database_collection_name, self.chunk_size, self.create_db)

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
                                                   self.database_collection_name, self.chunk_size, self.create_db)

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
                                                   self.database_collection_name, self.chunk_size, self.create_db)

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

        elif self.model_type in ["gpt-4o", "gpt-4o-mini", "mistral", "llama:7b", "llama3:70b", "gemma", "mixtral",
                                 "command-r", "llama3:8b"]:
            # Perform inference with the query input for GPT-4o model
            if self.use_mongo_memory:
                self.result = self.chain.invoke(
                    {"input": query_input, "chat_history": []}, config={"configurable": {"session_id": self.user_id}})
                self.results = self.result["answer"]
                self.chat_history.append((query_input, self.results))
            else:
                self.results = self.chain.invoke({"input": query_input, "chat_history": self.chat_history})
                self.results = self.results["answer"]
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
                        choices=['react_agent', 'gpt-4', 'gpt-4o', 'gpt-4o-mini', 'o1-preview', 'gpt-4-vision',
                                 'mistral', "llama3:70b",
                                 "llama3.1", "gemma", "crag", "mixtral", "self_rag", "bakllava", "mixtral_agent",
                                 "command-r", "agentic_rag", "llama3:8b",
                                 "adaptive_rag", "claude", "code_assistant"],
                        default="mistral", help='Model type for processing')
    parser.add_argument('--vectorstore', default="chroma", help='Embeddings Vectorstore', choices=["chroma",
                                                                                                   "milvus",
                                                                                                   "weaviate",
                                                                                                   "weaviate_hybrid"
                                                                                                   "qdrant",
                                                                                                   "pinecone",
                                                                                                   "faiss",
                                                                                                   "elasticsearch",
                                                                                                   "opensearch",
                                                                                                   "openclip",
                                                                                                   "vectara",
                                                                                                   "neo4j"])
    parser.add_argument(
        "--embeddings_model",
        default="ollama",
        choices=[
            "elasticsearch",
            "mistralai",
            "pinecone",
            "voyage",
            "fastembed",
            "huggingface",
            "ollama",
            "openai",  # Added for default
        ],
        help="Choose the embeddings model to use")
    parser.add_argument('--file_formats', nargs='+', default=['txt'],
                        help='List of file formats for loading documents')
    parser.add_argument('--use_mongo_memory', action='store_true',
                        help='Use MongoDB for chat history')
    parser.add_argument('--use_cohere_rank', action='store_true',
                        help='Use Cohere Rank for retriever compression')
    parser.add_argument('--use_multi_query_retriever', action='store_true',
                        help='Use MultiQueryRetriever')
    parser.add_argument('--use_contextual_compression', action='store_true',
                        help='Use Contextual Compression Retriever')
    parser.add_argument('--use_hyde', action='store_true',
                        help='Use Hypothetical Documents Embedder')

    args = parser.parse_args()
    return args.directory, args.model_type, args.vectorstore, args.file_formats, args.use_mongo_memory, args.use_cohere_rank, args.use_multi_query_retriever, args.use_contextual_compression, args.use_hyde, args.embeddings_model


def main():
    """
    Main function to run Langchain Model.
    """
    directory, model_type, vectorstore, file_formats, use_mongo_memory, use_cohere_rank, use_multi_query_retriever, use_contextual_compression, use_hyde, embeddings_model = parse_arguments()
    # Langchain model init
    llm = LangchainModel(llm_model=model_type, vectorstore_name=vectorstore, embeddings_model=embeddings_model,
                         use_mongo_memory=use_mongo_memory, use_cohere_rank=use_cohere_rank,
                         use_multi_query_retriever=use_multi_query_retriever,
                         use_contextual_compression=use_contextual_compression, use_hyde=use_hyde)
    llm.model_chain_init(directory, data_types=file_formats)
    while True:
        query = input("Please ask your question! ")
        llm.query_inferences(query)


if __name__ == "__main__":
    main()
