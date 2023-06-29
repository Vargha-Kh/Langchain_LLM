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
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, CallbackContext

warnings.filterwarnings("ignore")

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = "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"

# Set up Telegram bot
token = "6035216410:AAEHm0keUv-ULYKUSE9nYiuaPdxZZvVEe-Q"
bot = telegram.Bot(token=token)


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

    async def query_inferences(self, update: Update, context: CallbackContext):
        # Perform similarity search
        query_input = update.message.text
        self.docs = self.docsearch.similarity_search(query_input)
        results = self.chain({"input_documents": self.docs, "question": query_input}, return_only_outputs=True)
        response = results["output_text"].split("\nSOURCES")[0]
        await context.bot.send_message(chat_id=update.effective_chat.id,
                                           text=response)
    
    async def start(self, update: Update, context: CallbackContext):
        await context.bot.send_message(chat_id=update.effective_chat.id,
                                               text="Ask your question!")

def run_bot():
    llm = LangchainModel()
    directory = "liquidmarket_data"  # Replace with the path to your directory
    file_names = glob.glob(directory + "/*")
    llm.documents_loader_txt(file_names)
    llm.embedding_chunks()
    while True:
        application = ApplicationBuilder().token(token).build()
        que = application.job_queue
        start_handler = CommandHandler('start', llm.start)
        input_handler = MessageHandler(filters.Text(), llm.query_inferences)

        application.add_handler(start_handler)
        application.add_handler(input_handler)

        application.run_polling()


if __name__ == "__main__":
    run_bot()
