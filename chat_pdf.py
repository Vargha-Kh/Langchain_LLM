import time
import openai
import pdfminer.high_level
import re

# Set up OpenAI API credentials
openai.api_key = "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"


# Define function to extract text from PDF using pdfminer
def extract_text_from_pdf(file_path):
    text = pdfminer.high_level.extract_text(file_path)
    # Remove newline and multiple spaces
    text = re.sub(r'\n|\s{2,}', ' ', text)
    return text


# Define function to perform question answering with ChatGPT
def chat_gpt_question_answer(pdf_file_path, question):
    # Extract text from PDF file
    text = extract_text_from_pdf(pdf_file_path)
    # Perform question answering with ChatGPT
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Q: {question}\nA:",
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
        frequency_penalty=0,
        presence_penalty=0
    )
    # Get answer from response
    return response.choices[0].text.strip()


# Example usage
pdf_file_path = "SkillsAssessmentGuidelinesforApplicants_OCR.pdf"
question = "How to do the Re-application process?"
start = time.time()
answer = chat_gpt_question_answer(pdf_file_path, question)
print(question)
print(f"Elapsed time: {time.time()-start} seconds")
print(answer)
