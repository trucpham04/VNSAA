import torch
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Dataset

def classify_sentiment(text: str, pipeline) -> dict:
    """
    Hàm phân loại cảm xúc cho một đoạn văn bản đầu vào.
    
    Args:
        text (str): Chuỗi văn bản cần phân loại cảm xúc.
        pipeline (dict): Chứa các thành phần đã được khởi tạo gồm:
            - model: Mô hình transformer
            - tokenizer: Bộ tách từ tương ứng
            - classifier: Bộ phân loại
            - device: Thiết bị chạy

    Returns:
        dict: Kết quả phân loại gồm nhãn ('label') và độ tin cậy ('score')
    """

    # Kiểm tra xem pipeline đã được khởi tạo chưa
    if pipeline is None:
        raise Exception("Pipeline has not been initialized.")
    
    # Lấy các thành phần từ pipeline
    model = pipeline["model"]
    tokenizer = pipeline["tokenizer"]
    classifier = pipeline["classifier"]
    device = pipeline["device"]

    # Tạo dataset chỉ chứa một mẫu văn bản
    dataset = InferenceDataset([text], tokenizer)
    # DataLoader giúp chia dữ liệu thành batch
    loader = DataLoader(dataset, batch_size=1)

    features = []

    # Tắt gradient để chỉ inference (dự đoán)
    with torch.no_grad():
        for batch in loader:
            # Đưa dữ liệu lên CPU
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Chạy mô hình transformer để lấy embedding
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Lấy embedding của token [CLS] (đại diện toàn câu)
            cls_embedding = outputs[0][:, 0, :].cpu().numpy()

            # Lưu embedding vào danh sách
            features.append(cls_embedding)

    # Ghép các embedding lại thành ma trận numpy
    features = np.vstack(features)

    # Dự đoán xác suất các lớp bằng SVM classifier
    proba = classifier.predict_proba(features)[0]

    # Lấy chỉ số lớp có xác suất cao nhất
    label_idx = classifier.predict(features)[0]
    # Lấy điểm tin cậy cao nhất
    score = float(max(proba))

    # Nếu độ tin cậy thấp hơn 0.5, gán nhãn là NEUTRAL
    if score < 0.5:
        label_idx = 1

    # Ánh xạ chỉ số lớp sang nhãn
    label_map = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
    label = label_map[label_idx]

    # Trả về nhãn cảm xúc và điểm tin cậy
    return {"label": label, "score": score}

class InferenceDataset(Dataset):
    """
    Dataset dùng cho inference (dự đoán).
    Mỗi phần tử trong dataset là một văn bản được mã hóa bởi tokenizer.
    """

    def __init__(self, texts, tokenizer, max_len=100):
        self.texts = texts              # Danh sách các chuỗi văn bản
        self.tokenizer = tokenizer      # Tokenizer để chuyển text thành token ID
        self.max_len = max_len          # Độ dài tối đa mỗi câu

    def __len__(self):
        # Trả về số lượng mẫu trong dataset
        return len(self.texts)

    def __getitem__(self, idx):
        # Lấy một văn bản theo chỉ số
        text = self.texts[idx]

        # Tokenize và mã hóa câu thành input cho model
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,     # Thêm token [CLS], [SEP]
            max_length=self.max_len,     # Giới hạn độ dài tối đa
            padding="max_length",        # Padding đến độ dài cố định
            truncation=True,             # Cắt bớt nếu vượt quá max_len
            return_attention_mask=True,  # Trả về attention mask
            return_tensors="pt",         # Trả về tensor PyTorch
        )

        # Trả về dict chứa input_ids và attention_mask (loại bỏ chiều batch=1)
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0)
        }