import os
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from pinecone import Pinecone, ServerlessSpec


def process_pdf(pdf_file, openai_api_key, pinecone_api_key,pinecone_api_env,index_name, dimension):
    """with pdfplumber.open(pdf_file) as pdf:
        text += [page.extract_text() for page in pdf.pages]
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    PINECONE_API_KEY = pinecone_api_key
    PINECONE_API_ENV = pinecone_api_env
    pc = Pinecone(api_key=PINECONE_API_KEY)
    # pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
    docsearch = Pinecone.from_texts(pages, embeddings, index_name=index_name)
    
    return docsearch
"""
def get_answer(docsearch, question, openai_api_key):
    """docs = docsearch.similarity_search(question, include_metadata=True)

    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    chain = load_qa_chain(llm, chain_type="stuff")
    result = chain.run(input_documents=docs, question=question)

    return result"""
