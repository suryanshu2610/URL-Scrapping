import streamlit as st
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
from transformers import pipeline

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.write(f"Device set to use {device}")

# Load models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Storage
stored_texts = []
stored_embeddings = None

# Function to scrape and clean content from a URL
def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove scripts and styles
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()

        paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]
        text = '\n'.join(paragraphs)
        return text
    except Exception as e:
        return f"Error scraping {url}: {e}"

# Streamlit UI
st.title("📚 Web-based Q&A Tool")
st.write("Enter one or more webpage URLs, then ask questions based only on the scraped content.")

# URL Input
urls = st.text_area("Enter URLs (one per line):")
if st.button("Ingest Content"):
    url_list = urls.strip().splitlines()
    scraped_texts = []

    for url in url_list:
        text = extract_text_from_url(url)
        if "Error scraping" not in text:
            scraped_texts.append(text)
        else:
            st.warning(text)

    if scraped_texts:
        stored_texts.clear()
        stored_texts.extend(scraped_texts)

        # Split into chunks
        all_chunks = []
        for doc in scraped_texts:
            chunks = [doc[i:i+500] for i in range(0, len(doc), 500)]
            all_chunks.extend(chunks)

        # Embed
        embeddings = embedding_model.encode(all_chunks, convert_to_numpy=True)

        # Save to FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        stored_embeddings = (index, all_chunks)

        st.success("✅ Content ingested and indexed successfully.")

# Question Input
question = st.text_input("Ask a question based on the above pages:")
if question and stored_embeddings:
    index, all_chunks = stored_embeddings

    # Embed the question
    q_embedding = embedding_model.encode([question])
    D, I = index.search(np.array(q_embedding), k=5)

    context = " ".join([all_chunks[i] for i in I[0]])
    result = qa_pipeline(question=question, context=context)
    st.markdown("### 🧠 Answer")
    st.success(result['answer'])

elif question:
    st.warning("Please ingest some content first.")
