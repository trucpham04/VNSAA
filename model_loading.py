import time
from transformers import pipeline
import streamlit as st

@st.cache_resource
def load_model_pipeline():
    print("Loading model pipeline...")
    start_time = time.time()

    sentiment_pipeline = pipeline(
        task="sentiment-analysis",
        model="wonrax/phobert-base-vietnamese-sentiment"
    )

    end_time = time.time()
    print(f"Model has been loaded! (Took {end_time - start_time:.2f} seconds)")
    return sentiment_pipeline