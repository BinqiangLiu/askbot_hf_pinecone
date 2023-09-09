#import streamlit as st
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
import pinecone
import requests
import sys
#from langchain.prompts.chat import (ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate)
from langchain.chains.question_answering import load_qa_chain
#from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from langchain import HuggingFaceHub
#from PyPDF2 import PdfReader
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

loader = UnstructuredPDFLoader("D:/ChatGPTApps/AskBookLangChainOpenAI/TheAPP/valuation.pdf") 
data = loader.load()

print()
print(data)
print("***********************************")
print()

print (f'You have {len(data)} document(s) in your data')
print()
print (f'There are {len(data[0].page_content)} characters in your document')
print()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
print(text_splitter)
print()

db_texts = text_splitter.split_documents(data)
print(db_texts)
print("***********************************")
print()
print (f'Now you have {len(db_texts)} documents')

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
namespace = "valuation"

#vector_db = Pinecone.from_texts(db_texts, hf_embeddings, index_name=index_name, namespace=namespace)

vector_db = Pinecone.from_texts([t.page_content for t in db_texts], hf_embeddings, index_name=index_name, namespace=namespace)

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

while True:
    user_query=input("Enter your query:\n")
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

#使用这个while True，在cmd命令行窗口中可以实现持续询问，而不会重新生成向量并增加到Pinecone

#如果想要删除某个namespace，需要使用如下代码：唤起pinecone数据库 - 指明要删除的namespace所在的index，然后执行删除命令
#pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
#index_namespace_to_delete = pinecone.Index(index_name=index_name)
#index_namespace_to_delete.delete(delete_all=True, namespace=namespace)
#index_name.delete(delete_all=True, namespace="askpdfbot") #神奇，有时候这个代码可以删除namespace，但大多数时候报错：AttributeError: 'str' object has no attribute 'delete'

#查看Pinecone账户下的Index（名称）
pinecone.list_indexes()
index_name=pinecone.list_indexes()

#查看Pinecone的Index状态
#index = pinecone.Index(index_name=index_name)
#index.describe_index_stats() 

#What valuation methods are discussed and most recommended?
