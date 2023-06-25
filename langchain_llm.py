import os
from bs4 import BeautifulSoup
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chat_models import ChatOpenAI
import time
import glob
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
import warnings

warnings.filterwarnings("ignore")

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = "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"


class LangchainModel:
    def __init__(self):
        self.chain = None
        self.texts = []
        self.docs = None

    def documents_loader_txt(self, files, mode='files'):
        for file_path in files:
            with open(file_path) as f:
                if mode == 'html':
                    html_content = f.read()
                    soup = BeautifulSoup(html_content, 'html.parser')
                    text = soup.get_text(separator='\n')
                    self.texts.append(text)
                else:
                    text = f.read()
                    self.texts.append(text)

    def embedding_chunks(self):
        # Split the texts into chunks
        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
        text_chunks = text_splitter.split_text("\n".join(self.texts))

        # Create embeddings for the chunks using OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()
        docsearch = Chroma.from_texts(text_chunks, embeddings,
                                      metadatas=[{"source": str(i)} for i in range(len(text_chunks))])
        self.chain = load_qa_with_sources_chain(ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
                                                chain_type="stuff")
        return docsearch

    def query_inferences(self, query_input):
        docsearch = self.embedding_chunks()
        # Perform similarity search
        start = time.time()
        self.docs = docsearch.similarity_search(query_input)
        results = self.chain({"input_documents": self.docs, "question": query_input}, return_only_outputs=True)
        print(query_input)
        print(results["output_text"].split("\nSOURCES")[0])
        print(f"Elapsed time: {time.time() - start}")


if __name__ == "__main__":
    llm = LangchainModel()
    directory = "liquidmarket_data"  # Replace with the path to your directory
    file_names = glob.glob(directory + "/*")
    llm.documents_loader_txt(file_names)
    while True:
        query = input("Please enter your prompt here! ")
        llm.query_inferences(query)
