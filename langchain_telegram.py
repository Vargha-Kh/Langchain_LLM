import os
import glob
import warnings
from langchain.llms import OpenAI
import re
import asyncio
from bs4 import BeautifulSoup
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from pdf2text import PDFtoTXTConverter
import telegram
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, CallbackContext, Updater

warnings.filterwarnings("ignore")

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = ""

# Set up Telegram bot
token = "Telegram_token"
bot = telegram.Bot(token=token)


class LangchainModel:
    def __init__(self):
        self.chain = None
        self.docsearch = None
        self.docs = None
        self.texts = []
        self.started_chats = {}

    # Define function to extract text from PDF
    def extract_text_from_pdf(self, file_path):
        converter = PDFtoTXTConverter(file_path, model="easyocr")
        images = converter.convert_to_images()
        text = converter.perform_easy_ocr(images)
        return text

    def documents_loader_txt(self, files, mode='files'):
        for file_path in files:
            with open(file_path, encoding='utf-8-sig') as f:
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

        self.chain = load_qa_with_sources_chain(ChatOpenAI(model_name="gpt-4", temperature=0.2),
                                                chain_type="stuff")

    async def query_inferences(self, update: Update, context: CallbackContext):
        if update.effective_chat.id not in self.started_chats:
            return
        pattern = re.compile(r'^(who|what|where|when|why|how|which|whose|whom|is|are|was|were|am|do|does|did)\b',
                             re.IGNORECASE)
        query_input = update.message.text
        if pattern.match(query_input) or "?" in query_input:
            # Perform similarity search
            self.docs = self.docsearch.similarity_search(query_input)
            results = self.chain({"input_documents": self.docs, "question": query_input}, return_only_outputs=True)
            response = results["output_text"].split("\nSOURCES")[0]
            await context.bot.send_message(chat_id=update.effective_chat.id,
                                           text=f"Q:{query_input}\nA:{response}")

    async def start(self, update: Update, context: CallbackContext):
        self.started_chats[update.effective_chat.id] = True
        await context.bot.send_message(chat_id=update.effective_chat.id,
                                       text="Ask your question!")

    def run_bot(self):
        dispatcher = ApplicationBuilder().token(token).build()
        dispatcher.add_handler(CommandHandler("start", self.start))
        dispatcher.add_handler(MessageHandler(filters.Text(), self.query_inferences))

        # Start the event loop
        loop = asyncio.get_event_loop()
        loop.create_task(dispatcher.run_polling())
        loop.run_forever()


def main():
    llm = LangchainModel()
    directory = "liquidmarket_data"  # Replace with the path to your directory
    file_names = glob.glob(directory + "/*")
    llm.documents_loader_txt(file_names)
    llm.embedding_chunks()
    llm.run_bot()


if __name__ == "__main__":
    main()
