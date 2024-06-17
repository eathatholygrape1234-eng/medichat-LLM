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
openai.api_key = OPENAI_API_KEY

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
                  config={'max_new_tokens':512,
                          'temperature':0.8})
# Define the system prompt to guide the model's behavior
system_prompt = (
    "You are a medical professional and act like one. Use the following pieces of information to answer the user's question accurately. "
    "If you don't know the answer, just say that you don't know. Do not add any details by yourself. "
    "Context: {context}"
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
    "Write all the findings, next steps, recommendations, etc. You only need to respond if the image is related to a human body and health issues. "
    "You must have to answer but also write a disclaimer saying that 'Consult with a Doctor before making any decisions'. "
    "Remember, if certain aspects are not clear from the image, it's okay to state 'Unable to determine based on the provided image.' "
    "Now analyze the image and answer the above questions in the same structured manner defined above."
)

def call_gpt4_model_for_analysis(image_data):
    # Convert the image to base64
    base64_image = base64.b64encode(image_data).decode('utf-8')
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", "text": image_analysis_prompt
                },
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

    response = openai.ChatCompletion.create(
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
            print(response)
            response = call_gpt4_model_for_analysis(image_data)
           
        else:
            msg = request.form["msg"]
            # Invoke the chain with the input message
            result = chain.invoke({"input": msg})
            response = result["answer"]
        
        # Return the result as a JSON response
        return jsonify({"response": response})
    except Exception as e:
        # Return the error as a JSON response
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
