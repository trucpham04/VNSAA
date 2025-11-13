import time
import joblib
from transformers import AutoModel, AutoTokenizer
import streamlit as st
from torch.utils.data import Dataset

@st.cache_resource
def load_model_pipeline():
    print("Loading model pipeline...")
    start_time = time.time()

    # sentiment_pipeline = pipeline(
    #     task="sentiment-analysis",
    #     # model="wonrax/phobert-base-vietnamese-sentiment"
    #     model="vinai/phobert-base-v2"
    # )
    model_name = "vinai/phobert-base-v2"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    clf = joblib.load("svm_phobert_sentiment.pkl")
    device = "cpu"
    model.to(device)
    model.eval()

    end_time = time.time()
    print(f"Model has been loaded! (Took {end_time - start_time:.2f} seconds)")
    # return sentiment_pipeline
    return {"model": model, "tokenizer": tokenizer, "classifier": clf, "device": device}

class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=100):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {"input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0)}