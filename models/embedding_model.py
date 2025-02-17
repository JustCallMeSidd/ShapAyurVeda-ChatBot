import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()  # Load environment variables

def generate_embedding(text):
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        api_key=os.getenv("GOOGLE_API_KEY")
    )
    return embeddings_model.embed_query(text)