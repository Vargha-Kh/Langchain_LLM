# Langchain Model for Question-Answering (QA) and Document Retrieval using Langchain

This is a Python script that demonstrates how to use different language models for question-answering (QA) and document retrieval tasks using Langchain. The script utilizes various language models, including OpenAI's GPT, Google's Palm, and similarity search with Llama_index's retrieval approach, to provide answers to user queries based on the provided documents.

![](img/2.png)
<br>

## Setup
Before running the script, you need to set up the required credentials and install the necessary libraries.


## Install Required Libraries
You can install the required libraries using pip. Run the following command in your terminal or command prompt:

```
pip install -r requirements.txt
```

## Set Up API Keys
The script uses the OpenAI API key and, optionally, the Google API key for different models. You need to set these API keys as environment variables in your system. Replace OPENAI_API_KEY and GOOGLE_API_KEY with your actual API keys.


## Usage
1. File Format and Model Type:

    The script supports three file formats for loading documents: 'html', 'pdf', and 'txt'. You can specify the desired file format and model type using command-line arguments.

- --directory: Specify the directory containing the input files (e.g., 'html', 'pdf', or 'txt' files).
- --model_type: Choose the model type to use for processing (options: 'openai_QA', 'similarity_search', or 'google_palm').

2. Input Documents:

    Place your documents in the specified directory. The script will read the documents based on the provided file format (e.g., 'html', 'pdf', or 'txt').

3. Run the Script:

    Run the script in your terminal or command prompt with the desired command-line arguments. For example:

```
python langchain_llm.py --directory /path/to/documents --model_type similarity_search --file_format txt
```

4. Enter Queries:

    Once the script is running, it will prompt you to enter your question/query. It will then use the selected model to find relevant answers or retrieve relevant documents.

5. Results:

    The script will display the answer or retrieved documents based on the provided question/query.

![](img/1.png)

<br>

# Examples
## Example 1: Question-Answering with OpenAI GPT
```
python langchain_llm.py --directory /path/to/documents --model_type openai_QA --file_format pdf
```

## Example 2: Document Retrieval with Similarity Search
```
python langchain_llm.py --directory /path/to/documents --model_type similarity_search --file_format txt
```

## Example 3: Question-Answering with Google's Palm
```
python langchain_llm.py --directory /path/to/documents --model_type google_palm --file_format html
```

<br>

# Important Note
Please ensure that you have valid API keys and access to the required models before running the script. Additionally, make sure the input documents are placed in the specified directory and are in the correct format (html, pdf, or txt).
