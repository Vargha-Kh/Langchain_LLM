import glob
import os
from bs4 import BeautifulSoup as Soup
from chromadb.errors import InvalidDimensionException
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFDirectoryLoader, PythonLoader, \
    UnstructuredURLLoader, CSVLoader, UnstructuredCSVLoader, GitLoader, RecursiveUrlLoader, PDFPlumberLoader, \
    UnstructuredWordDocumentLoader
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.vectorstores import Chroma, Neo4jVector, Vectara, Qdrant, ElasticsearchStore, FAISS, Milvus, \
    OpenSearchVectorSearch
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
import weaviate
from langchain_elasticsearch import ElasticsearchStore
from langchain_pinecone import PineconeVectorStore
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_community.retrievers import (
    WeaviateHybridSearchRetriever,
)


def documents_loader(data_path, data_types, chunk_size):
    """
    Load documents from a given directory and return a list of texts.
    The method supports multiple data types including python files, PDFs, URLs, CSVs, and text files.
    """
    recursive_text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50)
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
                loader = PDFPlumberLoader(file_path)
                all_texts.extend(loader.load_and_split())
        elif data_type == "md":
            text_loader_kwargs = {'autodetect_encoding': True}
            loader = DirectoryLoader(data_path, glob="**/*.md", loader_cls=UnstructuredWordDocumentLoader,
                                     loader_kwargs=text_loader_kwargs,
                                     use_multithreading=True)
        elif data_type == "docx":
            text_loader_kwargs = {'autodetect_encoding': True}
            loader = DirectoryLoader(data_path, glob="**/*.docx", loader_cls=UnstructuredWordDocumentLoader,
                                     loader_kwargs=text_loader_kwargs,
                                     use_multithreading=True)
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
            text_loader_kwargs = {'autodetect_encoding': True}
            loader = DirectoryLoader(data_path, glob="**/*.csv", loader_cls=UnstructuredCSVLoader,
                                     loader_kwargs=text_loader_kwargs,
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
                splitted_texts = loader.load_and_split(recursive_text_splitter)
                all_texts.extend(splitted_texts)
        else:
            raise ValueError("Data file format is Not correct")
    return all_texts


def chroma_embeddings(data_path, data_types, embedding_function, index_name, chunk_size, create_db):
    try:
        if os.path.isfile(os.path.join(data_path, 'chroma.sqlite3')) and create_db is not True:
            vector_store = Chroma(persist_directory=data_path, embedding_function=embedding_function)
            vector_store.persist()
        else:
            docstore = documents_loader(data_path, data_types, chunk_size)
            vector_store = Chroma.from_documents(docstore, embedding=embedding_function,
                                                 persist_directory=data_path)
    except InvalidDimensionException:
        Chroma().delete_collection()
        os.remove(os.path.join(data_path, 'chroma.sqlite3'))
        docstore = documents_loader(data_path, data_types, chunk_size)
        vector_store = Chroma.from_documents(docstore, embedding=embedding_function,
                                             persist_directory=data_path)
    return vector_store


def milvus_embeddings(data_path, data_types, embedding_function, index_name, chunk_size, create_db):
    docstore = documents_loader(data_path, data_types, chunk_size)

    # Check if the Milvus collection exists
    try:
        vector_store = Milvus(
            embedding_function=embedding_function,
            connection_args={"host": "127.0.0.1", "port": "19530"},
            collection_name=index_name,
        )
        print("Using existing Milvus collection.")
    except:
        # Create a new Milvus collection
        vector_store = Milvus.from_documents(
            docstore,
            embedding_function,
            collection_name=index_name,
            connection_args={"host": "127.0.0.1", "port": "19530"},
        )
        print("Created new Milvus collection.")

    return vector_store


def weaviate_embeddings(data_path, data_types, embedding_function, index_name, chunk_size, create_db):
    weaviate_client = weaviate.connect_to_local()
    docstore = documents_loader(data_path, data_types, chunk_size)
    if create_db:
        return WeaviateVectorStore.from_documents(docstore, embedding_function, client=weaviate_client)
    else:
        return WeaviateVectorStore(client=weaviate_client, embedding=embedding_function, index_name=index_name,
                                   text_key="text")


def weaviate_hybrid_embeddings(data_path, data_types, embedding_function, index_name, chunk_size, create_db):
    docstore = documents_loader(data_path, data_types, chunk_size)
    client = weaviate.Client("http://localhost:8080", additional_headers={
        "X-Openai-Api-Key": os.environ.get("OPENAI_API_KEY")})
    vectorstore = WeaviateHybridSearchRetriever(
        client=client,
        index_name=index_name,
        text_key="text",
        embedding_function=embedding_function,
        attributes=[],
        create_schema_if_missing=True,
    )
    vectorstore.add_documents(docstore)
    return vectorstore


def qdrant_embeddings(data_path, data_types, embedding_function, index_name, chunk_size, create_db):
    docstore = documents_loader(data_path, data_types, chunk_size)
    return Qdrant.from_documents(
        docstore,
        embedding_function,
        path=data_path,
        force_recreate=True,
        collection_name=f"{data_path}",
    )


def pinecone_embeddings(data_path, data_types, embedding_function, index_name, chunk_size, create_db):
    index_name = "visa-rag"
    docstore = documents_loader(data_path, data_types, chunk_size)
    return PineconeVectorStore.from_documents(docstore, embedding_function, index_name=index_name)


def faiss_embeddings(data_path, data_types, embedding_function, index_name, chunk_size, create_db):
    docstore = documents_loader(data_path, data_types, chunk_size)
    return FAISS.from_documents(docstore, embedding_function)


def elasticsearch_embeddings(data_path, data_types, embedding_function, index_name, chunk_size, create_db):
    docstore = documents_loader(data_path, data_types, chunk_size)
    return ElasticsearchStore.from_documents(
        docstore,
        embedding_function,
        es_url="http://localhost:9200",
        index_name="rag",
    )


def opensearch_embeddings(data_path, data_types, embedding_function, index_name, chunk_size, create_db):
    docstore = documents_loader(data_path, data_types, chunk_size)
    return OpenSearchVectorSearch.from_documents(
        docstore, embedding_function, opensearch_url="http://localhost:9200"
    )


def openclip_embeddings(data_path, data_types, embedding_function, index_name, chunk_size, create_db):
    return Chroma(
        collection_name="multi-modal-rag",
        persist_directory=data_path,
        embedding_function=OpenCLIPEmbeddings(
            model_name="ViT-H-14", checkpoint="laion2b_s32b_b79k"
        )
    )


def vectara_embeddings(data_path, data_types, embedding_function, index_name, chunk_size, create_db):
    docstore = documents_loader(data_path, data_types, chunk_size)
    return Vectara.from_documents(
        documents=docstore,
        embedding=embedding_function,
    )


def neo4j_embeddings(data_path, data_types, embedding_function, index_name, chunk_size, create_db):
    docstore = documents_loader(data_path, data_types, chunk_size)
    if create_db:
        return Neo4jVector.from_documents(
            docstore, embedding_function
        )
    else:
        index_name = "vector"  # default index name
        return Neo4jVector.from_existing_index(
            embedding_function,
            index_name=index_name,
        )


# Dictionary to store all embedding functions
embeddings_dictionary = {
    "chroma": chroma_embeddings,
    "milvus": milvus_embeddings,
    "weaviate": weaviate_embeddings,
    "hybrid_weaviate": weaviate_hybrid_embeddings,
    "qdrant": qdrant_embeddings,
    "pinecone": pinecone_embeddings,
    "faiss": faiss_embeddings,
    "elasticsearch": elasticsearch_embeddings,
    "opensearch": opensearch_embeddings,
    "openclip": openclip_embeddings,
    "vectara": vectara_embeddings,
    "neo4j": neo4j_embeddings
}


# Function to call an embedding function by name
def get_vectorstores(vectorstore_name, data_path, data_types, embedding_function, collection_name, chunk_size,
                     create_db):
    """Retrieve and execute an embedding function by name."""
    if vectorstore_name in embeddings_dictionary:
        return embeddings_dictionary[vectorstore_name](data_path, data_types, embedding_function,
                                                       collection_name, chunk_size, create_db)
    else:
        return "Embedding function not found."
