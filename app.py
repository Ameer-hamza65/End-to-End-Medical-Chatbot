from flask import Flask, render_template, request
from src.helper import download_huggingface_embeddings, load_pdf_file, text_split
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from src.prompt import system_prompt
import os

load_dotenv()
app = Flask(__name__, template_folder='templates')

#  Embeddings
embeddings = download_huggingface_embeddings()

#  Load & chunk your PDFs once (e.g., at startup or as a background job)
extracted = load_pdf_file(data='data/')
chunks = text_split(extracted)

#  Build or load FAISS index
index_dir = "faiss_index"
if os.path.exists(index_dir):
    vector_store = FAISS.load_local(index_dir, embeddings,allow_dangerous_deserialization=True)  # load saved index :contentReference[oaicite:3]{index=3}
    print("Loaded FAISS index from disk.")
else:
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(index_dir)
    print("Built and saved FAISS index.")

#  Retriever
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

#  RAG with Gemini
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', api_key=GEMINI_API_KEY)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])
stuff_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, stuff_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "")
    print("User:", msg)
    try:
        resp = rag_chain.invoke({"input": msg})
        answer = resp.get("answer", "No answer found.")
        return answer
    except Exception as e:
        print("Error:", e)
        return "An error occurred."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
