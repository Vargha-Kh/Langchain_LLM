import os
import glob
import warnings
import argparse
from pdf2text import PDFtoTXTConverter
from bs4 import BeautifulSoup
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI, GooglePalm
from langchain.prompts import PromptTemplate

warnings.filterwarnings("ignore")

# Set OpenAI API Key
openai_key = "OPENAI_API_KEY"
google_api_key = "GOOGLE_API_KEY"
os.environ["OPENAI_API_KEY"] = openai_key


# Define function to extract text from PDF
def extract_text_from_pdf(file_path):
    converter = PDFtoTXTConverter(file_path, model="easyocr")
    images = converter.convert_to_images()
    text = converter.perform_easy_ocr(images)
    return text


class LangchainModel:
    def __init__(self, llm_model="similarity_search"):
        self.palm_qa = None
        self.results = None
        self.openai_qa = None
        self.chain = None
        self.docsearch = None
        self.docs = None
        self.texts = []
        self.model_type = llm_model

    def documents_loader(self, files, file_mode='txt'):
        for file_path in files:
            with open(file_path) as f:
                if file_mode == 'html':
                    html_content = f.read()
                    soup = BeautifulSoup(html_content, 'html.parser')
                    text = soup.get_text(separator='\n')
                    self.texts.append(text)
                elif file_mode == "pdf":
                    self.texts.append(extract_text_from_pdf(file_path))
                else:
                    text = f.read()
                    self.texts.append(text)

    def embedding_chunks(self):
        # Split the texts into chunks
        text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
        text_chunks = text_splitter.split_text("\n".join(self.texts))
        # Select embeddings
        embeddings = OpenAIEmbeddings()

        if self.model_type == "openai_QA":
            """RetrievalQA approach for creating datasets for QA document"""
            texts = text_splitter.create_documents(text_chunks)

            # Create a vectorstore from documents
            db = Chroma.from_documents(texts, embeddings)
            # Create retriever interface
            retriever = db.as_retriever()

            # Create QA chain
            self.openai_qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_key), chain_type='stuff',
                                                         retriever=retriever)
        elif self.model_type == "similarity_search":
            """similarity_search approach for creating datasets for QA document"""

            metadata = [{"source": str(i)} for i in range(len(text_chunks))]
            self.docsearch = Chroma.from_texts(text_chunks, embeddings, metadatas=metadata)

            self.chain = load_qa_with_sources_chain(ChatOpenAI(model_name="gpt-4", temperature=0.2),
                                                    chain_type="stuff")
        elif self.model_type == "google_palm":
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=10,
                                                           separators=["\n\n", "\n", ",", " ", "."])
            texts = text_splitter.create_documents(text_chunks)
            # Create a vectorstore from documents
            db = Chroma.from_documents(texts, embeddings)
            # Create retriever interface
            retriever = db.as_retriever()

            # Build prompt
            template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
            {context}
            Question: {question}
            Helpful Answer:"""
            QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

            self.palm_qa = RetrievalQA.from_chain_type(
                llm=GooglePalm(google_api_key=google_api_key, temperature=0.1, max_output_tokens=512),
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

    def query_inferences(self, query_input):
        if self.model_type == "openai_QA":
            self.results = self.openai_qa.run(query_input)
        elif self.model_type == "similarity_search":
            # Perform similarity search
            self.docs = self.docsearch.similarity_search(query_input)
            self.results = self.chain({"input_documents": self.docs, "question": query_input}, return_only_outputs=True)
        elif self.model_type == "google_palm":
            self.results = self.palm_qa({"query": query_input})

        print(query_input)
        print(self.results["output_text"].split("\nSOURCES")[0])


def parse_arguments():
    parser = argparse.ArgumentParser(description='Langchain Model with different model types.')
    parser.add_argument('--directory', choices=['html', 'pdf', 'txt'], default='txt',
                        help='Mode for loading documents (html, pdf, or files)')
    parser.add_argument('--model_type', choices=['openai_QA', 'similarity_search', 'google_palm'],
                        default='similarity_search',
                        help='Model type for processing (openai_QA, similarity_search, or google_palm)')
    parser.add_argument('--file_format', choices=['html', 'pdf', 'txt'], default='files',
                        help='Mode for loading documents (html, pdf, or txt)')
    args = parser.parse_args()
    return args.directory, args.model_type, args.mode


if __name__ == "__main__":
    directory, model_type, file_format = parse_arguments()
    llm = LangchainModel(llm_model=model_type)
    file_names = glob.glob(directory + f"/*{file_format}")
    llm.documents_loader(file_names, file_mode=file_format)
    llm.embedding_chunks()
    while True:
        query = input("Please ask you question! ")
        llm.query_inferences(query)
