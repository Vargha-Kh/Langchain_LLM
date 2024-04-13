from langchain import hub

# Define Prompts
openai_tools_prompt = hub.pull("hwchase17/openai-tools-agent")
openai_functions_prompt = hub.pull("hwchase17/openai-functions-agent")
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
rag_prompt = hub.pull("rlm/rag-prompt")
ask_search_prompt = hub.pull("hwchase17/self-ask-with-search")
react_prompt = hub.pull("hwchase17/react-chat")

retrieval_prompt = """
             You are an assistant for question-answering tasks.Use the following pieces of context to answer the question at the end. 
             If you don't know the answer, just say that you don't know, don't try to make up an answer. 
             {chat_history}

             {context}
             Question: {question}
             Helpful Answer:
             """

conversation_prompt = """You are an assistant for question-answering tasks, Use the following pieces of context to answer the question. The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
           {context}
           Summary of conversation:
           {history}
           Current conversation:
           {chat_history_lines}
           Human: {question}
           AI:"""

summary_prompt = """This is a conversation between a human and a bot:
           {chat_history}
           Write a summary of the conversation for {input}:
           """

prompts_dictionary = {
    "openai_tools": openai_tools_prompt,
    "openai_functions": openai_functions_prompt,
    "retrieval_qa": retrieval_qa_chat_prompt,
    "rag": rag_prompt,
    "ask_search": ask_search_prompt,
    "react": react_prompt,
    "conversation": conversation_prompt,
    "summary": summary_prompt
}


# Function to call a prompt by name
def get_prompt(prompt_name):
    """Retrieve and execute a prompt function by name."""
    if prompt_name in prompts_dictionary:
        return prompts_dictionary[prompt_name]
    else:
        return "Prompt not found."
