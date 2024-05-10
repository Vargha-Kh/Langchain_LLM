import os
import glob
import warnings
import argparse
import re
import asyncio
import telegram
from rags import LangchainModel
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, CallbackContext, Updater

warnings.filterwarnings("ignore")

# Set OpenAI API Key
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# Set up Telegram bot
token = ""
bot = telegram.Bot(token=token)


class TelegramBot:
    def __init__(self):
        directory, model_type, file_formats = parse_arguments()
        self.llm = LangchainModel(llm_model=model_type)
        self.llm.model_chain_init(directory, data_types=file_formats)

    async def query_inferences(self, update: Update, context: CallbackContext):
        # if update.effective_chat.id not in self.started_chats:
        #     return
        pattern = re.compile(r'^(who|what|where|when|why|how|which|whose|whom|is|are|was|were|am|do|does|did)\b',
                             re.IGNORECASE)
        query_input = update.message.text
        if pattern.match(query_input) or "?" in query_input:
            # Query inference
            response, results = self.llm.query_inferences(query_input)
            await context.bot.send_message(chat_id=update.effective_chat.id,
                                           text=f"Q:{query_input}\nA:{response}")

    async def start(self, update: Update, context: CallbackContext):
        # self.started_chats[update.effective_chat.id] = True
        await context.bot.send_message(chat_id=update.effective_chat.id,
                                       text="I am a RAG ready to answer your questions you regarding to Australia Visa immigration. Ask your question please!")

    def run_bot(self):
        dispatcher = ApplicationBuilder().token(token).build()
        dispatcher.add_handler(CommandHandler("start", self.start))
        dispatcher.add_handler(MessageHandler(filters.Text(), self.query_inferences))

        # Start the event loop
        loop = asyncio.get_event_loop()
        loop.create_task(dispatcher.run_polling())
        loop.run_forever()


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Langchain Model with different model types.')
    parser.add_argument('--directory', default='./visa_data', help='Ingesting files Directory')
    parser.add_argument('--model_type',
                        choices=['react_agent', 'gpt-4', 'gpt-4-vision', 'mistral', "llama3:70b", "llama:7b", "gemma",
                                 "mixtral", "bakllava", "llama_agent", "command-r", "agentic_rag", "adaptive_rag",
                                 "code_assistant"],
                        default="claude", help='Model type for processing')
    parser.add_argument('--file_formats', nargs='+', default=['pdf', 'txt', 'site'],
                        help='List of file formats for loading documents')
    args = parser.parse_args()
    return args.directory, args.model_type, args.file_formats


def main():
    llm = TelegramBot()
    llm.run_bot()


if __name__ == "__main__":
    main()
