from flask import Flask, render_template, jsonify, request
from src.helper import download_huggingface_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__, template_folder='C:/Users/Hamza/Desktop/End-to-End-Medical-Chatbot/template')

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY='YOUR_API_KEY'

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

embeddings = download_huggingface_embeddings()


index_name = "test-bot"

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


llm = ChatGroq(model='llama-3.3-70b-versatile',temperature=0.2, max_tokens=500)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg", "")  # Ensure safe retrieval of the message
    print(f"User Input: {msg}")
    
    try:
        # Call the retrieval-augmented generation chain
        response = rag_chain.invoke({"input": msg})
        print("Full Response: ", response)
        
        # Safely retrieve the answer from the response
        answer = response.get("answer", "Sorry, I couldn't find an answer.")
        return str(answer)
    except Exception as e:
        print(f"Error during processing: {e}")
        return str("An error occurred while processing your request.")





if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
