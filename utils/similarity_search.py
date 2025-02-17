import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean

def validate_and_convert_embeddings(embeddings):
    validated = []
    for idx, emb in enumerate(embeddings):
        try:
            vector = np.array(emb['embedding'], dtype=np.float32)
            if np.linalg.norm(vector) == 0:  # Avoid zero vectors
                print(f"Skipping zero vector at index {idx}")
                continue
            vector = vector / np.linalg.norm(vector)  # Normalize embeddings
            validated.append({
                "id": emb['id'],
                "medicine": emb['medicine'],
                "embedding": vector
            })
        except (KeyError, ValueError) as e:
            print(f"Invalid embedding at index {idx}: {str(e)}")
    return validated

def load_embeddings(filepath):
    with open(filepath, 'r') as file:
        embeddings = json.load(file)
    return validate_and_convert_embeddings(embeddings)

def hybrid_similarity(query_embedding, embeddings, top_n=3, alpha=0.7):
    """
    Compute hybrid similarity using weighted cosine similarity and Euclidean distance.
    :param query_embedding: Query vector
    :param embeddings: List of embeddings
    :param top_n: Number of similar medicines to return
    :param alpha: Weight for cosine similarity (higher = more importance on cosine similarity)
    :return: List of top similar medicines
    """
    query_vec = np.array(query_embedding, dtype=np.float32)
    query_vec = query_vec / np.linalg.norm(query_vec)  # Normalize query vector

    embeddings_matrix = np.array([e['embedding'] for e in embeddings])

    # Compute cosine similarity
    cos_similarities = cosine_similarity(query_vec.reshape(1, -1), embeddings_matrix)[0]

    # Compute Euclidean distances (convert to similarity by taking inverse)
    euclidean_similarities = np.array([1 / (1 + euclidean(query_vec, emb)) for emb in embeddings_matrix])

    # Combine both similarities using weighted sum
    hybrid_scores = alpha * cos_similarities + (1 - alpha) * euclidean_similarities

    most_similar_indices = np.argsort(hybrid_scores)[::-1][:top_n]

    return [{
        "similarity": float(hybrid_scores[idx]),
        "medicine": embeddings[idx]['medicine']
    } for idx in most_similar_indices]

def find_similar_medicines(query_embedding, top_n=3):
    embeddings = load_embeddings('embeddings/saved_embeddings/medicine_embeddings.json')
    if not embeddings:
        return []
    return hybrid_similarity(query_embedding, embeddings, top_n=top_n)
