from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_TOKEN=os.environ.get('OPENAI_API_TOKEN')
OPENAI_API_BASE =os.environ.get("OPENAI_API_BASE")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_TOKEN"] = OPENAI_API_TOKEN
os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE

embeddings = download_hugging_face_embeddings()

# os.environ["OPENAI_API_KEY"] = "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"
# os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"  # Important!


index_name = "medicalbot"

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":10})


llm = ChatOpenAI(
    model="llama3-8b-8192",
    temperature=0.4,
    api_key=OPENAI_API_TOKEN,         
    base_url=OPENAI_API_BASE        
)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

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
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])




if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)