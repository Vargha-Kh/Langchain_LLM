import argparse
import os.path
import warnings
from langchain.agents import create_react_agent, create_self_ask_with_search_agent
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain, LLMChain, RetrievalQA, RetrievalQAWithSourcesChain
from langchain.chains.combine_documents.map_rerank import MapRerankDocumentsChain
from langchain.chains.combine_documents.refine import RefineDocumentsChain
from langchain.memory import (
    ConversationBufferMemory,
    ReadOnlySharedMemory
)
from langchain.output_parsers import RegexParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama, OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_elasticsearch import ElasticsearchEmbeddings
from utils import get_resized_images, img_prompt_func, AgenticRAG, AdaptiveRAG, CodeAssistant, \
    get_prompt, SelfRAG, chroma_embeddings, milvus_embeddings, pinecone_embeddings, openclip_embeddings, \
    faiss_embeddings, qdrant_embeddings, create_retriever_tool
from utils.tools import *

warnings.filterwarnings("ignore")

# Set API Keys
# os.environ["OPENAI_API_KEY"] = getpass.getpass("Your OpenAI API key: ")



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
        self.result = None
        self.results = None
        self.chat_history = []
        self.create_db = True

    def ollama_chain_init(self, data_path, data_types):
        vector_store = qdrant_embeddings(data_path, data_types, OllamaEmbeddings(model=self.model_type),
                                         self.create_db)

        # LLM init
        self.llm = ChatOllama(model=self.model_type, streaming=True,
                              callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

        # Conversational Memory Buffer
        memory = ConversationBufferMemory(
            memory_key="chat_history", input_key="question", return_messages=False
        )
        # Knowledge Graph Memory
        # kg_memory = ConversationKGMemory(llm=llm)

        prompt_template = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template=get_prompt("retrieval"),
        )

        self.chain = ConversationalRetrievalChain.from_llm(
            self.llm,
            memory=memory,
            retriever=vector_store.as_retriever(),
            combine_docs_chain_kwargs={
                "prompt": prompt_template,
            },
            get_chat_history=lambda h: h,
            verbose=True
        )

    def model_chain_init(self, data_path, data_types):
        """
        Embed documents into chunks based on the model type.
        """
        if self.model_type == "gpt-4-vision":
            # Load chroma
            multi_modal_vectorstore = openclip_embeddings(data_path)
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

        elif self.model_type == "agentic_rag":
            vector_db = chroma_embeddings(data_path, data_types, OpenAIEmbeddings(), self.create_db)

            # Agent's tools initialization for retrievers
            retriever_tool = create_retriever_tool(
                vector_db.as_retriever(),
                f"{os.path.basename(data_path)}",
                f"Searches and returns answers from {os.path.basename(data_path)} document.",
            )

            # initialize AgenticRAG graphs
            self.chain = AgenticRAG(retriever_tool)
            self.chain.create_graph()

        if self.model_type == "adaptive_rag":
            # Chroma Vectorstore
            vector_db = chroma_embeddings(data_path, data_types, OpenAIEmbeddings(), self.create_db)
            # initialize Adaptive RAG
            self.chain = AdaptiveRAG(vector_db.as_retriever())
            self.chain.create_graph()

        if self.model_type == "self_rag":
            # Chroma Vectorstore
            vector_db = chroma_embeddings(data_path, data_types, OpenAIEmbeddings(), self.create_db)
            # initialize Adaptive RAG
            self.chain = SelfRAG(vector_db.as_retriever())
            self.chain.create_graph()

        if self.model_type == "code_assistant":
            # Chroma Vectorstore
            vector_db = chroma_embeddings(data_path, data_types, OpenAIEmbeddings(), self.create_db)
            # initialize Adaptive RAG
            self.chain = CodeAssistant(vector_db.as_retriever())
            self.chain.create_graph()

        if self.model_type == "react_agent":
            vector_db = chroma_embeddings(data_path, data_types, OpenAIEmbeddings(), self.create_db)

            # Conversational Memory Buffer
            conv_memory = ConversationBufferMemory(
                memory_key="chat_history", input_key="input"
            )

            # Summary Memory Module
            prompt_template = PromptTemplate(input_variables=["input", "chat_history"], template=get_prompt("summary"))
            read_only_memory = ReadOnlySharedMemory(memory=conv_memory)

            summary_chain = LLMChain(
                llm=self.llm,
                prompt=prompt_template,
                verbose=True,
                memory=read_only_memory,  # use the read-only memory to prevent the tool from modifying the memory
            )

            # Define summary memory tool
            summary_memory_tool = Tool(
                name="Summary",
                func=summary_chain.run,
                description="useful for when you summarize a conversation. The input to this tool should be a string, representing who will read this summary.",
            )

            search_tool = create_search_tool("tavily")

            # Agent's tools initialization for retrievers
            qa_retrieval_tool = retrieval_qa_tool(os.path.basename(data_path), vector_db, self.llm)
            retriever_tool = vectorstore_retriever_tool(os.path.basename(data_path), vector_db)

            tools = [retriever_tool, search_tool, qa_retrieval_tool, summary_memory_tool]

            # Create Openai function Agent
            # openai_agent = create_openai_functions_agent(ChatOpenAI(model="gpt-3.5-turbo-1106"), tools,
            #                                              self.openai_functions_prompt)

            # Create ReAct Agent
            react_agent = create_react_agent(ChatOpenAI(temperature=0, streaming=True, model="gpt-4"), tools,
                                             get_prompt("react"))

            # Create self-ask search Agent
            self_ask_agent = create_self_ask_with_search_agent(ChatOpenAI(temperature=0, streaming=True, model="gpt-4"),
                                                               [
                                                                   TavilyAnswer(max_results=1,
                                                                                name="Intermediate Answer")],
                                                               get_prompt("ask_search"))

            self.chain = AgentExecutor(
                agent=react_agent, tools=tools, memory=conv_memory, verbose=True, handle_parsing_errors=True,
                return_intermediate_steps=True, include_run_info=True
            )

        elif self.model_type == "gpt-4":
            vector_db = chroma_embeddings(data_path, data_types, OpenAIEmbeddings(), self.create_db)

            # Conversational Memory Buffer
            memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=False
            )
            # Knowledge Graph Memory
            # kg_memory = ConversationKGMemory(llm=self.llm)

            prompt_template = PromptTemplate(
                input_variables=["context", "question", "chat_history"],
                template=get_prompt("visa"),
            )

            self.chain = ConversationalRetrievalChain.from_llm(
                ChatOpenAI(temperature=self.temperature, streaming=True, model_name="gpt-4-turbo"),
                retriever=vector_db.as_retriever(),
                return_source_documents=False,
                memory=memory,
                combine_docs_chain_kwargs={"prompt": prompt_template},
                get_chat_history=lambda h: h,
                verbose=True
            )

        elif self.model_type == "gpt-4o":
            vector_db = chroma_embeddings(data_path, data_types, OpenAIEmbeddings(), self.create_db)

            # Define your prompt template
            prompt = PromptTemplate(template=get_prompt("visa"), input_variables=["context", "question"])

            # Create an LLMChain with the prompt
            llm_chain = LLMChain(llm=ChatOpenAI(temperature=self.temperature, streaming=True, model_name="gpt-4o"),
                                 prompt=prompt)

            # Create a RefineDocumentsChain as the combine_docs_chain
            combine_docs_chain = RefineDocumentsChain(
                initial_llm_chain=llm_chain,
                refine_llm_chain=llm_chain,
                document_variable_name="context",
                initial_response_name="initial_answer",
            )

            # Conversational Memory Buffer
            memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=False, output_key="answer"
            )

            prompt_template = PromptTemplate(
                input_variables=["context", "question", "chat_history"],
                template=get_prompt("visa"),
            )

            llm = ChatOpenAI(temperature=self.temperature, streaming=True, model_name="gpt-4o")

            self.chain = RetrievalQAWithSourcesChain(
                combine_documents_chain=combine_docs_chain,
                memory=memory,
                retriever=vector_db.as_retriever(),
                return_source_documents=True,
                verbose=True,
            )

        elif self.model_type == "claude":
            vector_db = milvus_embeddings(data_path, data_types, OpenAIEmbeddings(), self.create_db)

            # LLM definition
            llm = ChatAnthropic(model="claude-3-opus-20240229", streaming=True)
            # Conversational Memory Buffer
            memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=False
            )
            # Knowledge Graph Memory
            # kg_memory = ConversationKGMemory(llm=self.llm)

            prompt_template = PromptTemplate(
                input_variables=["context", "question", "chat_history"],
                template=get_prompt("visa"),
            )

            self.chain = ConversationalRetrievalChain.from_llm(
                llm,
                retriever=vector_db.as_retriever(),
                return_source_documents=False,
                memory=memory,
                combine_docs_chain_kwargs={"prompt": prompt_template},
                get_chat_history=lambda h: h,
                verbose=True
            )

        elif self.model_type == "mixtral_agent":
            vector_db = chroma_embeddings(data_path, data_types, OllamaEmbeddings(model="mixtral"), self.create_db)

            # Ollama init
            self.llm = ChatOllama(model='mixtral', streaming=True,
                                  callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

            # Conversational Memory Buffer
            conv_memory = ConversationBufferMemory(
                memory_key="chat_history", input_key="input"
            )

            # Summarization Memory
            # summary_memory = ConversationSummaryMemory(llm=self.llm)
            # Memory Modules Combined
            # memory = CombinedMemory(memories=[conv_memory, summary_memory])

            search_tool = create_search_tool(engine="arxiv")
            retriever_tool = vectorstore_retriever_tool(os.path.basename(data_path), vector_db)

            tools = [retriever_tool, search_tool]

            # Create ReAct Agent
            react_agent = create_react_agent(self.llm, tools, get_prompt("react"))

            self.chain = AgentExecutor(
                agent=react_agent, tools=tools, memory=conv_memory, verbose=True, handle_parsing_errors=True,
                return_intermediate_steps=True, include_run_info=True
            )
        elif self.model_type == self.model_type == "mistral" or self.model_type == "llama-7b" or self.model_type == "gemma" or self.model_type == "mixtral" or self.model_type == "command-r" or self.model_type == "llama3:70b":
            self.ollama_chain_init(data_path, data_types)

        elif self.model_type == "bakllava":
            # Load chroma
            multi_modal_vectorstore = openclip_embeddings(data_path)

            # LLM init
            self.llm = Ollama(model=self.model_type,
                              callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

            # Define the RAG pipeline
            self.chain = (
                    {
                        "context": multi_modal_vectorstore.as_retriever | RunnableLambda(get_resized_images),
                        "question": RunnablePassthrough(),
                    }
                    | RunnableLambda(img_prompt_func)
                    | self.llm
                    | StrOutputParser()
            )

    def query_inferences(self, query_input):
        """
        Perform inference based on the query input and the model type.
        """
        if self.model_type == "react_agent" or self.model_type == "mixtral_agent":
            self.result = self.chain.invoke({"input": query_input, "chat_history": self.chat_history},
                                            include_run_info=True)
            self.results = self.result["output"]
            self.chat_history.append((query_input, self.results))
        elif self.model_type == "gpt-4" or "claude":
            self.result = self.chain({"question": query_input, "chat_history": self.chat_history},
                                     include_run_info=True)
            self.results = self.result["answer"]
            self.result["output"] = self.results
            self.chat_history.append((query_input, self.results))
        elif self.model_type == "gpt-4o":
            self.result = self.chain({"input": query_input, "chat_history": self.chat_history})
            self.results = self.result["result"]
            self.chat_history.append((query_input, self.results))
        elif self.model_type == "mistral" or self.model_type == "llama-7b" or self.model_type == "llama3:70b" or self.model_type == "gemma" or self.model_type == "mixtral" or self.model_type == "command-r":
            self.results = self.chain.run({"question": query_input})
            self.chat_history.append((query_input, self.results))
        elif self.model_type == "gpt-4-vision" or self.model_type == "bakllava":
            self.results = self.chain.invoke({"question": query_input})
        elif self.model_type == "agentic_rag" or self.model_type == "adaptive_rag" or self.model_type == "code_assistant" or self.model_type == "self_rag":
            self.results = self.chain.invoke(query_input)
        print(self.results)
        return self.results, self.result


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Langchain Model with different model types.')
    parser.add_argument('--directory', default='./visa_data', help='Ingesting files Directory')
    parser.add_argument('--model_type',
                        choices=['react_agent', 'gpt-4', 'gpt-4o', 'gpt-4-vision', 'mistral', "llama3-70b", "llama:7b",
                                 "gemma",
                                 "mixtral", "self_rag", "bakllava", "mixtral_agent", "command-r", "agentic_rag",
                                 "adaptive_rag", "claude", "code_assistant"],
                        default="gpt-4o", help='Model type for processing')
    parser.add_argument('--file_formats', nargs='+', default=['pdf'],
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
