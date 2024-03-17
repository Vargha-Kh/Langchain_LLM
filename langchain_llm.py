import argparse
import os
import sys
import warnings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFDirectoryLoader, PythonLoader, \
    UnstructuredURLLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama, OpenAI
from langchain_community.chat_models import ChatOpenAI

warnings.filterwarnings("ignore")

# Set OpenAI API Key
openai_key = "OPENAI_API_KEY"
os.environ["OPENAI_API_KEY"] = openai_key


class LangchainModel:
    """
    Langchain Model class to handle different types of language models.
    """

    def __init__(self, llm_model):
        self.loader = None
        self.results = None
        self.model_type = llm_model
        self.text_splitter = None
        self.model = None
        self.temperature = 0.1
        self.chain = None
        self.chat_history = []
        self.custom_prompt = """
              You are an assistant for question-answering tasks.Use the following pieces of context to answer the question at the end. 
              If you don't know the answer, just say that you don't know, don't try to make up an answer. 
              {context}
              Question: {question}
              Helpful Answer:
              """
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
            elif data_type == "txt":
                text_loader_kwargs = {'autodetect_encoding': True}
                self.loader = DirectoryLoader(data_path, glob="**/*.txt", loader_cls=TextLoader,
                                              loader_kwargs=text_loader_kwargs, use_multithreading=True)
            texts = self.loader.load()
            all_texts.extend(texts)
        return all_texts

    def embedding_chunks(self, data_path, data_types):
        """
        Embed documents into chunks based on the model type.
        """
        embedding_data = self.documents_loader(data_path, data_types)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        text_chunks = text_splitter.split_documents(embedding_data)

        if self.model_type == "gpt-4":
            vectordb = Chroma.from_texts(text_chunks, embedding=OpenAIEmbeddings(), persist_directory="./data")
            vectordb.persist()

            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False, output_key='answer')

            self.chain = ConversationalRetrievalChain.from_llm(
                ChatOpenAI(temperature=self.temperature, model_name=self.model_type),
                retriever=vectordb.as_retriever(),
                return_source_documents=False,
                memory=memory,
                get_chat_history=lambda h: h,
                verbose=True
            )

        elif self.model_type == "gpt-3.5":
            metadata = [{"source": str(i)} for i in range(len(text_chunks))]
            self.docsearch = Chroma.from_texts(text_chunks, OpenAIEmbeddings(), metadatas=metadata)
            self.chain = load_qa_with_sources_chain(ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2),
                                                    chain_type="stuff")

        elif self.model_type == "mistral":
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False, output_key='result')

            QA_CHAIN_PROMPT = PromptTemplate(
                input_variables=["context", "question"],
                template=self.custom_prompt,
            )
            vector_store = Chroma.from_documents(text_chunks, embedding=GPT4AllEmbeddings())
            llm = Ollama(model=self.model_type, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
            self.chain = RetrievalQA.from_chain_type(
                llm,
                memory=memory,
                retriever=vector_store.as_retriever(),
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
            )
        elif self.model_type == "gemma":
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False, output_key='result')
            QA_CHAIN_PROMPT = PromptTemplate(
                input_variables=["context", "question"],
                template=self.custom_prompt,
            )
            vector_store = Chroma.from_documents(text_chunks, embedding=GPT4AllEmbeddings())
            llm = Ollama(model=self.model_type, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
            self.chain = RetrievalQA.from_chain_type(
                llm,
                memory=memory,
                retriever=vector_store.as_retriever(),
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
            )
        elif self.model_type == "llama-7b":
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False, output_key='result')
            QA_CHAIN_PROMPT = PromptTemplate(
                input_variables=["context", "question"],
                template=self.custom_prompt,
            )
            vector_store = Chroma.from_documents(text_chunks, embedding=GPT4AllEmbeddings())
            llm = Ollama(model=self.model_type, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
            self.chain = RetrievalQA.from_chain_type(
                llm,
                memory=memory,
                retriever=vector_store.as_retriever(),
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
            )

    def query_inferences(self, query_input):
        """
        Perform inference based on the query input and the model type.
        """
        if self.model_type == "gpt-4":
            query_input = self.custom_prompt + query_input
            result = self.chain({"question": query_input, "chat_history": self.chat_history})
            self.results = result["answer"]
            self.chat_history.append((query_input, self.results))
        elif self.model_type == "gpt-3.5":
            query_input = self.custom_prompt + query_input
            self.docs = self.docsearch.similarity_search(query_input)
            self.results = self.chain({"input_documents": self.docs, "question": query_input}, return_only_outputs=True)
            self.results = self.results["output_text"].split("\nSOURCES")[0]
        elif self.model_type == "mistral" or "llama-7b" or "gemma":
            self.results = self.chain({"query": query_input})
            self.results = self.results['result']
            self.chat_history.append((query_input, self.results))
        print(self.results)


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Langchain Model with different model types.')
    parser.add_argument('--directory', default='./data', help='Ingesting files Directory')
    parser.add_argument('--model_type', choices=['gpt-4', 'gpt-3.5', 'mistral', "llama-7b", "gemma"],
                        default='mistral', help='Model type for processing')
    parser.add_argument('--file_formats', nargs='+', default=['txt', 'pdf', 'web'],
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
