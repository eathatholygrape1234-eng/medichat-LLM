from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.vectorstores import Pinecone as PineconeStore
from langchain_pinecone import PineconeVectorStore 
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from pinecone import Pinecone
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
import langchain.chains
from dotenv import load_dotenv
from src.prompt import *
import os
app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
embeddings = download_hugging_face_embeddings()
index_name = "medichat"

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name="medichat"
index = pc.Index("medichat")
index.describe_index_stats() 
docsearch=PineconeVectorStore.from_existing_index(index_name, embeddings)
vectordb = PineconeStore(index, embeddings,"text")
llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':150,
                          'temperature':0.7})
# Define the system prompt to guide the model's behavior
system_prompt = (
    "You are a medical professional and act like one. Use the following pieces of information to answer the user's question accurately"
    "If you don't know the answer, just say that you don't know. Do not add any details by yourself.answer in short 3-5 lines "
    "Context: {context} ? "
)

# Create the prompt template with the system and human message structure
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Configure the retriever to get relevant documents from the vector database
retriever = vectordb.as_retriever(search_kwargs={'k': 2})

# Create the chain to combine the documents and generate a response using the language model
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Create the retrieval chain that integrates the retriever and the question-answer chain
chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        msg = request.form["msg"]
        # Invoke the chain with the input message
        result = chain.invoke({"input": msg})
        print(result["answer"])
        # Return the result as a JSON response
        return jsonify({"response": result["answer"]})
    except Exception as e:
        # Return the error as a JSON response
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
