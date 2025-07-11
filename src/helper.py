from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()



def load_pdf_file(data):
    loader=DirectoryLoader(data,glob="*.pdf",loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents


def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1024,chunk_overlap=64)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks



def download_huggingface_embeddings():
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.getenv('GOOGLE_API_KEY'))
    return embeddings