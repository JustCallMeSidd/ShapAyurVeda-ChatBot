import streamlit as st
import os
from models.embedding_model import generate_embedding
from utils.similarity_search import find_similar_medicines
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()  # Load environment variables

# Initialize Gemini model
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-pro')

def generate_explanation(user_query, medicine_data):
    """Generate natural language explanation using Gemini"""
    prompt = f"""
    Explain/tell why this Ayurvedic medicine might be suitable for the user's query. 
    Consider the following details:

    User Query: {user_query}
    Medicine Name: {medicine_data['name']}
    Key Uses: {', '.join(medicine_data['uses'])}
    Typical Dosage: {medicine_data['dosage']}
    Common Side Effects: {', '.join(medicine_data['side_effects'])}

    Provide a concise, natural-sounding explanation (2-3 sentences) that:
    1. Connects the user's symptoms/needs to the medicine's uses
    2. Mentions important considerations like dosage and side effects
    3. Maintains a professional but approachable tone
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Could not generate explanation: {str(e)}"

def main():
    st.title("🌱ShapAyurVeda ChatBot🧠")
    
    user_query = st.text_input("Enter Ingredient or medicine needed :")
    
    if user_query:
        try:
            # Generate embedding and find matches
            query_embedding = generate_embedding(user_query)
            results = find_similar_medicines(query_embedding, top_n=3)
            
            if not results:
                st.warning("No matching medicines found. Try different keywords.")
            else:
                # Filter results based on similarity score
                filtered_results = [result for result in results if result['similarity'] > 0.70]

                if not filtered_results:
                    st.info("No results found Please inquiry on shaporganices2002@gmail.com")  # Changed to info for clarity
                else:
                    # Display top recommendation with explanation
                    top_result = filtered_results[0]
                    st.subheader("Most Relevant Item We Got")
                    
                    # Generate natural language explanation
                    explanation = generate_explanation(user_query, top_result['medicine'])
                    
                    # Display explanation in highlighted section with black background and white text
                    st.markdown(f"""
                    <div style='padding: 15px; 
                                    background-color: black; 
                                    color: white; 
                                    border-radius: 10px;
                                    margin-bottom: 20px;'>
                        <h4 style='color: white;'>Where u lokking for this?</h4>
                        {explanation}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show detailed medicine information
                    # with st.expander(f"🔍 {top_result['medicine']['name']} Details (Similarity: {top_result['similarity']:.2f})"):
                    #     st.markdown(f"""
                    #     **Description**: {top_result['medicine']['description']} 
                    #     **Primary Uses**: {", ".join(top_result['medicine']['uses'])} 
                    #     **Recommended Dosage**: {top_result['medicine']['dosage']} 
                    #     **Possible Side Effects**: {", ".join(top_result['medicine']['side_effects'])}
                    #     """)

        except Exception as e:
            st.error(f"Error processing request: {str(e)}")

if __name__ == "__main__":
    main()
