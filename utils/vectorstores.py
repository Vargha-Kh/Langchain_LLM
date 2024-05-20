import glob
import os
from bs4 import BeautifulSoup as Soup
from chromadb.errors import InvalidDimensionException
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFDirectoryLoader, PythonLoader, \
    UnstructuredURLLoader, CSVLoader, UnstructuredCSVLoader, GitLoader, RecursiveUrlLoader, PDFPlumberLoader
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.vectorstores import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus
import weaviate
from langchain_community.vectorstores import Vectara
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_elasticsearch import ElasticsearchStore
from langchain_community.vectorstores import FAISS
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import Qdrant
from langchain_weaviate.vectorstores import WeaviateVectorStore


def documents_loader(data_path, data_types):
    """
    Load documents from a given directory and return a list of texts.
    The method supports multiple data types including python files, PDFs, URLs, CSVs, and text files.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=100)
    all_texts = []
    loader = None
    for data_type in data_types:
        if data_type == 'py':
            loader = DirectoryLoader(data_path, glob="**/*.py", loader_cls=PythonLoader,
                                     use_multithreading=True)
        elif data_type == "pdf":
            loader = PyPDFDirectoryLoader(data_path)
        elif data_type == "pdf_plumber":
            for file_path in glob.glob(os.path.join(data_path, "*.pdf"), recursive=True):
                loader = PDFPlumberLoader(file_path, extract_images=True)
                all_texts.extend(loader.load_and_split())
        elif data_type == "url":
            urls = []
            with open(os.path.join(data_path, 'urls.txt'), 'r') as file:
                for line in file:
                    urls.append(line.strip())
            loader = UnstructuredURLLoader(urls=urls)
        elif data_type == 'site':
            url = "https://immi.homeaffairs.gov.au/visas/working-in-australia/skill-occupation-list"
            loader = RecursiveUrlLoader(
                url=url, max_depth=3, extractor=lambda x: Soup(x, "html.parser").text, use_async=True
            )
        elif data_type == "csv":
            loader = DirectoryLoader(data_path, glob="**/*.csv", loader_cls=UnstructuredCSVLoader,
                                     use_multithreading=True)
        elif data_type == "txt":
            text_loader_kwargs = {'autodetect_encoding': True}
            loader = DirectoryLoader(data_path, glob="**/*.txt", loader_cls=TextLoader,
                                     loader_kwargs=text_loader_kwargs, use_multithreading=True)
        elif data_type == 'repo':
            # Clone
            repo_path = "./test_repo"
            # repo = Repo.clone_from("https://github.com/Vargha-Kh/INDE_577_Machine_Learning_Cookbooks/", to_path=repo_path)

            # Load
            loader = GenericLoader.from_filesystem(
                repo_path,
                glob="**/*",
                suffixes=[".py"],
                exclude=["**/non-utf8-encoding.py"],
                parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
            )

        if loader is not None:
            if data_type == "pdf_plumber":
                all_texts = all_texts
            else:
                splitted_texts = loader.load_and_split(text_splitter)
                all_texts.extend(splitted_texts)
        else:
            raise ValueError("Data file format is Not correct")
    return all_texts


def chroma_embeddings(data_path, data_types, embedding_function, create_db):
    try:
        if os.path.isfile(os.path.join(data_path, 'chroma.sqlite3')) and create_db is not True:
            vector_store = Chroma(persist_directory=data_path, embedding_function=embedding_function)
            vector_store.persist()
        else:
            text_chunks = documents_loader(data_path, data_types)
            vector_store = Chroma.from_documents(text_chunks, embedding=embedding_function,
                                                 persist_directory=data_path)
    except InvalidDimensionException:
        Chroma().delete_collection()
        os.remove(os.path.join(data_path, 'chroma.sqlite3'))
        text_chunks = documents_loader(data_path, data_types)
        vector_store = Chroma.from_documents(text_chunks, embedding=embedding_function,
                                             persist_directory=data_path)
    return vector_store


def milvus_embeddings(data_path, data_types, embedding_function, create_db):
    docstore = documents_loader(data_path, data_types)
    return Milvus.from_documents(
        docstore,
        embedding_function,
        collection_name=f"milvus",
        connection_args={"host": "127.0.0.1", "port": "19530"},
    )


def weaviate_embeddings(data_path, data_types, embedding_function, create_db):
    weaviate_client = weaviate.connect_to_local()
    docstore = documents_loader(data_path, data_types)
    return WeaviateVectorStore.from_documents(docstore, embedding_function, client=weaviate_client)


def qdrant_embeddings(data_path, data_types, embedding_function, create_db):
    docstore = documents_loader(data_path, data_types)
    return Qdrant.from_documents(
        docstore,
        embedding_function,
        path=data_path,
        collection_name=f"{data_path}",
    )


def pinecone_embeddings(data_path, data_types, embedding_function, create_db):
    index_name = "visa-rag"
    docstore = documents_loader(data_path, data_types)
    return PineconeVectorStore.from_documents(docstore, embedding_function, index_name=index_name)


def faiss_embeddings(data_path, data_types, embedding_function, create_db):
    docstore = documents_loader(data_path, data_types)
    return FAISS.from_documents(docstore, embedding_function)


def elasticsearch_embeddings(data_path, data_types, embedding_function, create_db):
    docstore = documents_loader(data_path, data_types)
    return ElasticsearchStore.from_documents(
        docstore,
        embedding_function,
        es_url="http://localhost:9200",
        index_name="rag",
    )


def opensearch_embeddings(data_path, data_types, embedding_function, create_db):
    docstore = documents_loader(data_path, data_types)
    return OpenSearchVectorSearch.from_documents(
        docstore, embedding_function, opensearch_url="http://localhost:9200"
    )


def openclip_embeddings(data_path):
    return Chroma(
        collection_name="multi-modal-rag",
        persist_directory=data_path,
        embedding_function=OpenCLIPEmbeddings(
            model_name="ViT-H-14", checkpoint="laion2b_s32b_b79k"
        )
    )


def vectara_embeddings(data_path, data_types, embedding_function, create_db):
    docstore = documents_loader(data_path, data_types)
    return Vectara.from_documents(
        documents=docstore,
        embedding=embedding_function,
    )
