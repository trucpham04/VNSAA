import torch
from torch.utils.data import DataLoader
import numpy as np
from model_loading import InferenceDataset


def classify_sentiment(text: str, pipeline) -> dict:
    if pipeline is None:
        raise Exception("Pipeline has not been initialized.")
    
    model = pipeline["model"]
    tokenizer = pipeline["tokenizer"]
    classifier = pipeline["classifier"]
    device = pipeline["device"]

    dataset = InferenceDataset([text], tokenizer)
    loader = DataLoader(dataset, batch_size=1)

    features = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embedding = outputs[0][:, 0, :].cpu().numpy()
            features.append(cls_embedding)
    features = np.vstack(features)

    proba = classifier.predict_proba(features)[0]
    label_idx = classifier.predict(features)[0]
    label_map = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}

    return {"label": label_map[label_idx], "score": float(max(proba))}
    
    # # Lấy kết quả từ mô hình
    # raw_result = sentiment_pipeline(text)

    # # Lấy sentiment có xác suất cao nhất
    # sentiment = max(raw_result, key=lambda x: x['score'])

    # # Xử lý trường hợp xác suất thấp
    # if sentiment['score'] < 0.5:
    #     sentiment['label'] = "NEU"

    # # Chuyển tên nhãn sang dạng đầy đủ
    # def get_sentiment_label(label: str) -> str:
    #     match label:
    #         # case "POS":
    #         #     return "POSITIVE"
    #         # case "NEG":
    #         #     return "NEGATIVE"
    #         case "LABEL_1":
    #             return "POSITIVE"
    #         case "LABEL_0":
    #             return "NEGATIVE"
    #         case "NEU":
    #             return "NEUTRAL"
            
    # sentiment['label'] = get_sentiment_label(sentiment['label'])

    # return sentiment
