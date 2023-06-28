import os
import glob
import warnings
import pdfminer.high_level
import re
from bs4 import BeautifulSoup
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

warnings.filterwarnings("ignore")

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = "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"

# Set up Telegram bot
bot = telegram.Bot(token='YOUR_API_TOKEN')


class LangchainModel:
    def __init__(self):
        self.chain = None
        self.docsearch = None
        self.docs = None
        self.texts = []

    # Define function to extract text from PDF using pdfminer
    def extract_text_from_pdf(self, file_path):
        text = pdfminer.high_level.extract_text(file_path)
        # Remove newline and multiple spaces
        text = re.sub(r'\n|\s{2,}', ' ', text).replace("..", '')
        return text

    def documents_loader_txt(self, files, mode='files'):
        for file_path in files:
            with open(file_path) as f:
                if mode == 'html':
                    html_content = f.read()
                    soup = BeautifulSoup(html_content, 'html.parser')
                    text = soup.get_text(separator='\n')
                    self.texts.append(text)
                elif mode == "pdf":
                    self.texts.append(self.extract_text_from_pdf(file_path))
                else:
                    text = f.read()
                    self.texts.append(text)

    def embedding_chunks(self):
        # Split the texts into chunks
        text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
        text_chunks = text_splitter.split_text("\n".join(self.texts))

        # Create embeddings for the chunks using OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()
        metadata = [{"source": str(i)} for i in range(len(text_chunks))]
        self.docsearch = Chroma.from_texts(text_chunks, embeddings, metadatas=metadata)

        self.chain = load_qa_with_sources_chain(ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2),
                                                chain_type="stuff")

    def query_inferences(self, query_input):
        # Perform similarity search
        self.docs = self.docsearch.similarity_search(query_input)
        results = self.chain({"input_documents": self.docs, "question": query_input}, return_only_outputs=True)
        response = results["output_text"].split("\nSOURCES")[0]
        return response


# Define a function to handle incoming messages
def handle_message(update, context):
    query = update.message.text
    llm = LangchainModel()
    directory = "liquidmarket_data"  # Replace with the path to your directory
    file_names = glob.glob(directory + "/*")
    llm.documents_loader_txt(file_names)
    llm.embedding_chunks()
    response = llm.query_inferences(query)
    update.message.reply_text(response)


# Set up the Telegram bot handler
updater = Updater(token='YOUR_API_TOKEN', use_context=True)
dispatcher = updater.dispatcher
dispatcher.add_handler(MessageHandler(Filters.text, handle_message))

# Start the bot
updater.start_polling()
