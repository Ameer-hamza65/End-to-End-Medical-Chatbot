from src.helper import load_pdf_file, text_split,download_huggingface_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import os


PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
os.environ['PINECONE_API_KEY']=PINECONE_API_KEY


extracted_data=load_pdf_file(data='data/')
text_chunks=text_split(extracted_data)
embeddings=download_huggingface_embeddings()


pc=Pinecone(api_key=PINECONE_API_KEY)

index_name='test-bot'

pc.create_index(
    name=index_name,
    dimension=384, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)        


docsearch=PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings
)