from src.helper import load_pdf_file, text_split, download_huggingface_embeddings
from langchain.vectorstores import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
import os

# Load and chunk text
extracted_data = load_pdf_file(data='data/')
text_chunks = text_split(extracted_data)

# Load HuggingFace embedding model
embeddings = download_huggingface_embeddings()
# (Expect an embedding object compatible with langchain_huggingface.HuggingFaceEmbeddings)

# Create or load local FAISS index
index_dir = "faiss_index"
if os.path.exists(index_dir):
    # Load from disk
    vector_store = FAISS.load_local(index_dir, embeddings)
    print("✅ Loaded existing FAISS index")
else:
    # Build fresh index
    vector_store = FAISS.from_documents(
        documents=text_chunks,
        embedding=embeddings
    )
    vector_store.save_local(index_dir)
    print("✅ Built and saved new FAISS index")
