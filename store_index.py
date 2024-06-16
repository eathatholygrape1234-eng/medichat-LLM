from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore 

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
print(PINECONE_API_KEY)
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name="medichat"
index = pc.Index("medichat")
index.describe_index_stats() 

docsearch=PineconeVectorStore.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)
