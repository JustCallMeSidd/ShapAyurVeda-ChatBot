import os
import json
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()  # Load environment variables

def process_medicines():
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        api_key=os.getenv("GOOGLE_API_KEY")
    )

    with open('data/medicines.json', 'r') as file:
        medicines = json.load(file)

    embeddings_with_metadata = []
    for idx, medicine in enumerate(medicines):
        try:
            medicine_str = json.dumps(medicine)
            embedding = embeddings_model.embed_query(medicine_str)
            
            embeddings_with_metadata.append({
                "id": idx,
                "medicine": medicine,
                "embedding": np.array(embedding).tolist()  # Ensure serialization
            })
            
        except Exception as e:
            print(f"Error processing medicine {idx}: {str(e)}")
            continue

    with open('embeddings/saved_embeddings/medicine_embeddings.json', 'w') as file:
        json.dump(embeddings_with_metadata, file)

if __name__ == "__main__":
    process_medicines()