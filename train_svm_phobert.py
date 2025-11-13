import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

from preprocessing import correct_slang_words, standardize_text, tokenize_text

# =========================== Dataset ===========================
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=100):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "label": label}

# =========================== Load PhoBERT ===========================
def load_phobert_model(model_name="vinai/phobert-base-v2"):
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    return model, tokenizer

# =========================== Feature Extraction ===========================
def extract_features(model, tokenizer, texts, max_len=100, batch_size=16, device='cpu'):
    dataset = SentimentDataset(texts=texts, labels=[0]*len(texts), tokenizer=tokenizer, max_len=max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model = model.to(device)
    model.eval()
    
    features = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs[0][:, 0, :].cpu().numpy()
            features.append(cls_embeddings)
    features = np.vstack(features)
    return features

# =========================== Main Training Script ===========================
def main():
    start_time = time.time()

    print("> Loading dataset...")
    # 1. Load dataset CSV (cột: text, label 0/1/2)
    df = pd.read_csv("data_sentiment_vn.csv")  # label: 0=negative,1=neutral,2=positive
    df['text'] = df['text'].apply(lambda x: tokenize_text(correct_slang_words(standardize_text(x))))
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    # 2. Load PhoBERT
    print("> Loading PhoBERT model...")
    model, tokenizer = load_phobert_model()
    
    # 3. Extract features
    print("> Extracting features from PhoBERT embeddings...")
    features = extract_features(model, tokenizer, texts, max_len=150, batch_size=32, device='cpu')
    
    # 4. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)
    
    # 5. Train SVM
    print("> Training SVM classifier...")
    clf = SVC(kernel='linear', probability=True, gamma=0.125)
    clf.fit(X_train, y_train)
    
    # 6. Evaluate
    y_pred = clf.predict(X_test)
    print("> Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    # 7. Save classifier
    joblib.dump(clf, "svm_phobert_sentiment.pkl")
    print("> Saved classifier to svm_phobert_sentiment.pkl")

    end_time = time.time()
    print(f"> Training SVM completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
