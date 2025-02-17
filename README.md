# Ayurvedic Medicine Search & Recommendation System

This project is an Ayurvedic Medicine Search & Recommendation System built using Streamlit, LangChain, and the Google Gemini API. It allows users to search for Ayurvedic medicines based on their queries and provides recommendations based on similarity.

## Project Structure

```
ayurvedic-medicine-app/
├── app.py                     # Main Streamlit application file
├── data/
│   └── medicines.json         # Raw data of Ayurvedic medicines
├── embeddings/
│   ├── medicine_embeddings.py  # Script to generate and save embeddings
│   └── saved_embeddings/
│       └── medicine_embeddings.json  # Stored embeddings for similarity search
├── models/
│   └── embedding_model.py      # Logic for generating embeddings using Gemini API
├── utils/
│   └── similarity_search.py     # Script for performing similarity searches
├── requirements.txt            # List of project dependencies
└── README.md                   # Project documentation
```

## Setup Instructions

1. **Install Dependencies**: Ensure you have Python 3.8 or higher. Install the necessary packages by running:
   ```
   pip install streamlit langchain google-generativeai
   ```

2. **Obtain API Key**: Acquire a Google API key for accessing the Gemini models from the Google AI platform.

3. **Set Up Environment Variables**: Create a `.env` file in your project directory and add your Google API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## Running the Application

1. **Generate Embeddings**: Run the `medicine_embeddings.py` script to process the raw data and generate embeddings:
   ```
   python embeddings/medicine_embeddings.py
   ```

2. **Start Streamlit App**: Launch the Streamlit application:
   ```
   streamlit run app.py
   ```

## Usage

- Navigate to the Streamlit app in your browser.
- Enter a query related to Ayurvedic medicines.
- The system will process your query, perform a similarity search, and display the most relevant medicine recommendations.

## Additional Resources

- For a comprehensive guide on integrating LangChain with Google's Gemini API, refer to the quickstart guide.
- To understand how to create a Google Gemini chatbot with minimal code, see the tutorial.
- For a visual walkthrough on integrating Gemini with LangChain, you might find the video tutorial helpful.