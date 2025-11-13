import time
import joblib
from transformers import AutoModel, AutoTokenizer
import streamlit as st


@st.cache_resource
def load_model_pipeline():
    # Ghi nhận thời gian bắt đầu
    start_time = time.time()
    print("> Loading model pipeline...")

    # Tải mô hình PhoBERT-base-v2
    model_name = "vinai/phobert-base-v2"
    model = AutoModel.from_pretrained(model_name)

    # Tải tokenizer tương ứng với PhoBERT
    # use_fast=False vì PhoBERT sử dụng kiểu tokenizer đặc thù của BPE (Byte-Pair Encoding)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # Tải mô hình phân loại SVM (đã được train sẵn) từ file pickle
    # Mô hình này sẽ nhận embedding từ PhoBERT để dự đoán nhãn cảm xúc
    classifier = joblib.load("svm_phobert_sentiment.pkl")

    # Chọn thiết bị để chạy model
    device = "cpu"
    model.to(device)

    # Chuyển model sang chế độ đánh giá
    model.eval()

    # Ghi nhận thời gian kết thúc
    end_time = time.time()
    print(f"> Model has been loaded! (Took {end_time - start_time:.2f} seconds)")

    # Trả về dictionary chứa các thành phần cần thiết cho pipeline
    return {"model": model, 
            "tokenizer": tokenizer, 
            "classifier": classifier, 
            "device": device
    }