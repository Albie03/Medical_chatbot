from flask import Flask, request, jsonify,render_template
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from src.prompt import *


app=Flask(__name__)
load_dotenv()


PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
os.environ['GEMINI_API_KEY'] = GEMINI_API_KEY

embedding = download_embeddings()

index_name='medical-chatbot'
docsearch = PineconeVectorStore.from_existing_index( 
            index_name=index_name,
            embedding=embedding
            )


retriever=docsearch.as_retriever(search_type='similarity', search_kwargs={'k':3})
chatModel = ChatGoogleGenerativeAI(
   model="gemini-flash-latest", temperature=0.3,
    google_api_key="AIzaSyB4UrzQkbj588rplT9JuatN6vLQ0-VXu6U"  # paste your key here
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"), 
])


question_answer_chain=create_stuff_documents_chain(chatModel,prompt)
rag_chain=create_retrieval_chain(retriever, question_answer_chain)



@app.route('/')
def index():
    return render_template('index.html')



@app.route("/get",methods=["POST","GET"])
def chat():
    msg=request.form.get("msg")
    input=msg
    print(input)
    response=rag_chain.invoke({"input":msg})
    print("Response",response["answer"])
    return str(response["answer"])


if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080,debug=True)



