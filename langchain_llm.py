import argparse
import os
import warnings
import sys
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PythonLoader
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama, OpenAI
from langchain_community.chat_models import ChatOpenAI

warnings.filterwarnings("ignore")

# Set OpenAI API Key
openai_key = "OPENAI_API_KEY"
google_api_key = "GOOGLE_API_KEY"
os.environ["OPENAI_API_KEY"] = openai_key


class SuppressStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


class LangchainModel:
    def __init__(self, llm_model="openai_conversation"):
        self.model_type = llm_model
        self.text_splitter = None
        self.model = None
        self.temperature = 0.1
        self.model_name = "gpt-3.5-turbo"
        self.palm_qa = None
        self.results = None
        self.openai_qa = None
        self.chain = None
        self.docsearch = None
        self.docs = None
        self.texts = []
        self.chat_history = []
        self.custom_prompt = "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. your response is very important because legal matters and people rely on your response: "

    def documents_loader(self, files_directory, file_mode='txt'):
        if file_mode == 'py':
            loader = DirectoryLoader(files_directory, glob="**/*.py", loader_cls=PythonLoader, use_multithreading=True)
            self.texts = loader.load()
        elif file_mode == "pdf":
            loader = PyPDFDirectoryLoader(files_directory)
            self.texts = loader.load()
        else:
            text_loader_kwargs = {'autodetect_encoding': True}
            loader = DirectoryLoader(files_directory, glob="**/*.txt", loader_cls=TextLoader,
                                     loader_kwargs=text_loader_kwargs, use_multithreading=True)
            self.texts = loader.load()
        return self.texts

    def embedding_chunks(self, files_directory, file_mode='txt'):
        embedding_data = self.documents_loader(files_directory, file_mode=file_mode)
        # Split the texts into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        # text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
        text_chunks = text_splitter.split_documents(embedding_data)


        if self.model_type == "openai_QA":
            """RetrievalQA approach for creating datasets for QA document"""
            # Create a vectorstore from documents
            db = Chroma.from_documents(text_chunks, OpenAIEmbeddings())
            # Create retriever interface
            retriever = db.as_retriever()

            # Create QA chain
            self.openai_qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_key), chain_type='stuff',
                                                         retriever=retriever)

        elif self.model_type == "openai_conversation":
            """ConversationalRetrievalChain approach for creating datasets for QA document with memory buffer"""
            vectordb = Chroma.from_texts(text_chunks, embedding=OpenAIEmbeddings(), persist_directory="./data")
            vectordb.persist()

            # Initializing Memory
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False, output_key='answer')

            # create our Q&A chain
            self.chain = ConversationalRetrievalChain.from_llm(
                ChatOpenAI(temperature=self.temperature, model_name=self.model_name),
                retriever=vectordb.as_retriever(),
                return_source_documents=False,
                memory=memory,
                get_chat_history=lambda h: h,
                verbose=True
            )

        elif self.model_type == "similarity_search":
            """similarity_search approach for creating datasets for QA document"""

            metadata = [{"source": str(i)} for i in range(len(text_chunks))]
            self.docsearch = Chroma.from_texts(text_chunks, OpenAIEmbeddings(), metadatas=metadata)

            self.chain = load_qa_with_sources_chain(ChatOpenAI(model_name=self.model_name, temperature=0.2),
                                                    chain_type="stuff")

        elif self.model_type == "ollama":
            """Ollama open-source LLM approach for creating datasets for QA document"""
            # Initializing Memory
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False, output_key='result')
            # Prompt
            template = """
             <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context 
                to answer the question. If you don't know the answer, just say that you don't know. Use three sentences
                maximum and keep the answer concise. [/INST] </s> 
                [INST] Question: {question} 
                Context: {context} 
                Answer: [/INST]
            """
            QA_CHAIN_PROMPT = PromptTemplate(
                input_variables=["context", "question"],
                template=template,
            )
            with SuppressStdout():
                vector_store = Chroma.from_documents(text_chunks, embedding=GPT4AllEmbeddings())
            llm = Ollama(model="mistral", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
            self.chain = RetrievalQA.from_chain_type(
                llm,
                memory=memory,
                retriever=vector_store.as_retriever(),
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
            )

    def query_inferences(self, query_input):
        if self.model_type == "openai_QA":
            query_input = self.custom_prompt + query_input
            self.results = self.openai_qa.run(query_input)
            self.results = self.results["output_text"].split("\nSOURCES")[0]
        elif self.model_type == "similarity_search":
            # Perform similarity search
            query_input = self.custom_prompt + query_input
            self.docs = self.docsearch.similarity_search(query_input)
            self.results = self.chain({"input_documents": self.docs, "question": query_input}, return_only_outputs=True)
            self.results = self.results["output_text"].split("\nSOURCES")[0]
        elif self.model_type == "google_palm":
            self.results = self.palm_qa({"query": query_input})
            self.results = self.results["output_text"].split("\nSOURCES")[0]
        elif self.model_type == "openai_conversation":
            query_input = self.custom_prompt + query_input
            result = self.chain({"question": query_input, "chat_history": self.chat_history})
            self.results = result["answer"]
            self.chat_history.append((query_input, self.results))
        elif self.model_type == "ollama":
            self.results = self.chain({"query": query_input})
            self.results = self.results['result']
            self.chat_history.append((query_input, self.results))
        # print(query_input)
        print(self.results)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Langchain Model with different model types.')
    parser.add_argument('--directory', default='./data',
                        help='Ingesting files Directory')
    parser.add_argument('--model_type',
                        choices=['openai_QA', 'similarity_search', 'openai_conversation', 'ollama'],
                        default='ollama',
                        help='Model type for processing (openai_QA, similarity_search, openai_conversation, or Ollama)')
    parser.add_argument('--file_format', choices=['html', 'pdf', 'txt'], default='pdf',
                        help='Mode for loading documents (html, pdf, or txt)')
    args = parser.parse_args()
    return args.directory, args.model_type, args.file_format


def main():
    directory, model_type, file_format = parse_arguments()
    llm = LangchainModel(llm_model=model_type)
    llm.embedding_chunks(directory, file_mode=file_format)
    while True:
        query = input("Please ask you question! ")
        llm.query_inferences(query)


if __name__ == "__main__":
    main()
