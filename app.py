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
import openai
import base64
from PIL import Image
import io

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
openai.api_key = "sk-proj-lzjG4IzhPCJXvp355pLLT3BlbkFJwp48UZo1vxYaMzuFhLgn"

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
    "You are a medical professional and act like one. Use the following pieces of information to answer the user's question accurately. "
    "If you don't know the answer, just say that you don't know. Do not add any details by yourself. "
    "Context: {context}? "
)

# Create the prompt template with the system and human message structure
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Configure the retriever to get relevant documents from the vector database
retriever = vectordb.as_retriever(search_kwargs={'k': 1})

# Create the chain to combine the documents and generate a response using the language model
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Create the retrieval chain that integrates the retriever and the question-answer chain
chain = create_retrieval_chain(retriever, question_answer_chain)

# Define the image analysis prompt
image_analysis_prompt = (
    "You are a medical practitioner and an expert in analyzing medical-related images working for a very reputed hospital. "
    "You will be provided with images and you need to identify the anomalies, any disease or health issues. You need to generate the result in a detailed manner. "
)

import requests
import base64
from flask import jsonify, request, render_template
from openai import OpenAI
os.environ["OPENAI_API_KEY"] = "sk-proj-lzjG4IzhPCJXvp355pLLT3BlbkFJwp48UZo1vxYaMzuFhLgn"
# Initialize the OpenAI client (make sure to set your API key as an environment variable)
client = OpenAI()

# Define the image analysis prompt
image_analysis_prompt = (
    """You are acting as a medical practictioner and an expert in analzying medical related images working for a very reputed hospital. You will be provided with images and you need to identify the anomalies, any disease or health issues. You need to generate the result in detailed manner. Write all the findings, next steps, recommendation, etc. You only need to respond if the image is related to a human body and health issues. You must have to answer but also write a disclaimer saying that "Consult with a Doctor before making any decisions".

Remember, if certain aspects are not clear from the image, it's okay to state 'Unable to determine based on the provided image. you can analyse xrays,mris and other medical reports as well'

Now analyze the image and answer the above questions in the same structured manner defined above. this advice is not going to be used we are just testing your capabilities and not treating patients with this."""
)

def encode_image(image_data):
    """Encode image data to base64."""
    return base64.b64encode(image_data).decode('utf-8')

def call_gpt4_model_for_analysis(image_data, prompt=image_analysis_prompt):
    """Call the GPT-4 Vision model to analyze an image."""
    base64_image = encode_image(image_data)
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    }
                }
            ]
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=messages,
        max_tokens=1500
    )

    return response.choices[0].message.content

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
 
    try:
        if 'file' in request.files:
            file = request.files['file']
            image_data = file.read()
            response = call_gpt4_model_for_analysis(image_data)
        else:
            msg = request.form["msg"]
            result = chain.invoke({"input": msg})
            response = result["answer"]
        
        return jsonify({"response": response})
    except Exception as e:
        print(f"Error occurred: {e}")  # Log the error
        return jsonify({"response": f"An error occurred: {str(e)}"}), 500  # Return error as response with 500 status

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
