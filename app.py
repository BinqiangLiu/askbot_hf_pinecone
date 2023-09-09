import streamlit as st
#from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
import pinecone
import requests
import sys
#from langchain.prompts.chat import (ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate)
from langchain.chains.question_answering import load_qa_chain
#from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from langchain import HuggingFaceHub
from PyPDF2 import PdfReader
#from langchain.document_loaders import TextLoader
#from sentence_transformers.util import semantic_search
from pathlib import Path
from time import sleep
#import pandas as pd
#import torch
import os
import random
import string
from dotenv import load_dotenv
load_dotenv()

#loader = UnstructuredPDFLoader("valuation.pdf") 
#loader = PyPDFLoader("valuation.pdf")
#data = loader.load() 

#text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#db_texts = text_splitter.split_documents(data)

data = PdfReader("valuation.pdf")
raw_text = ''
db_texts=''
for i, page in enumerate(data.pages):
    text = page.extract_text()
    if text:
        raw_text += text
        text_splitter = RecursiveCharacterTextSplitter(        
#            separator = "\n",
            chunk_size = 1000,
            chunk_overlap  = 100, #striding over the text
            length_function = len,
        )
        db_texts = text_splitter.split_text(raw_text)

class HFEmbeddings:
    def __init__(self, api_url, headers):
        self.api_url = api_url
        self.headers = headers

    def get_embeddings(self, texts):
        response = requests.post(self.api_url, headers=self.headers, json={"inputs": texts, "options": {"wait_for_model": True}})
        embeddings = response.json()
        return embeddings

    def embed_documents(self, texts):
        embeddings = self.get_embeddings(texts)
        return embeddings

    def __call__(self, texts):
        return self.embed_documents(texts)

model_id = "sentence-transformers/all-MiniLM-L6-v2"
hf_token = "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"

PINECONE_API_KEY = "5f07b52e-2a16-42a3-89c4-8899c584109e"
PINECONE_ENVIRONMENT = "asia-southeast1-gcp-free"
PINECONE_INDEX_NAME = "myindex-allminilm-l6-v2-384"

api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}

hf_embeddings = HFEmbeddings(api_url, headers)

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

index_name = PINECONE_INDEX_NAME
namespace = "askbot-hf-pinecone"

vector_db = Pinecone.from_texts(db_texts, hf_embeddings, index_name=index_name, namespace=namespace)
#vector_db = Pinecone.from_texts([t.page_content for t in db_texts], hf_embeddings, index_name=index_name, namespace=namespace)
#docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name, namespace=namespace)
print("***********************************")
print("Pinecone Vector/Embedding DB Ready.")
print()

repo_id = "HuggingFaceH4/starchat-beta"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

llm = HuggingFaceHub(repo_id=repo_id,
                     model_kwargs={"min_length":100,
                                   "max_new_tokens":1024, "do_sample":True,
                                   "temperature":0.1,
                                   "top_k":50,
                                   "top_p":0.95, "eos_token_id":49155})

chain = load_qa_chain(llm=llm, chain_type="stuff")

user_query=st.text_input("Enter your query:\n")  
if user_query !="" and not user_query.strip().isspace() and not user_query.isspace():
    print("Your query:\n"+user_query)
    print()
    vector_db_from_index = Pinecone.from_existing_index(index_name, hf_embeddings, namespace=namespace)
    ss_results = vector_db_from_index.similarity_search(query=user_query, namespace=namespace, k=5)
    print(f'Similarity Searched Contexts:\n')
    print(ss_results)
    print("***********************************")
    print()
    initial_ai_response=chain.run(input_documents=ss_results, question=user_query)
    temp_ai_response=initial_ai_response.partition('<|end|>')[0]
    final_ai_response = temp_ai_response.replace('\n', '')
    print("AI Response:")
    print(final_ai_response)
