# Vietnamese Sentiment Analysis Application

Ứng dụng Streamlit phân tích cảm xúc tiếng Việt dựa trên PhoBERT-base-v2 (Hugging Face) và bộ phân loại SVM huấn luyện riêng. Văn bản đầu vào được chuẩn hóa, sửa từ lóng, tokenize rồi suy luận cảm xúc (NEGATIVE/NEUTRAL/POSITIVE), kết quả được lưu vào SQLite.

## Mục lục

- [Vietnamese Sentiment Analysis Application](#vietnamese-sentiment-analysis-application)
  - [Mục lục](#mục-lục)
  - [Tổng quan](#tổng-quan)
  - [Cấu trúc thư mục](#cấu-trúc-thư-mục)
  - [Mô hình sử dụng](#mô-hình-sử-dụng)
    - [PhoBERT + SVM](#phobert--svm)
  - [Pipeline xử lý](#pipeline-xử-lý)
    - [1. Tiền xử lý](#1-tiền-xử-lý)
    - [2. Phân loại](#2-phân-loại)
    - [3. Lưu trữ \& UI](#3-lưu-trữ--ui)
  - [Tính năng](#tính-năng)
  - [Hướng dẫn cài đặt](#hướng-dẫn-cài-đặt)
    - [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
    - [Các bước thực hiện](#các-bước-thực-hiện)
      - [Bước 1: Clone Github Repo](#bước-1-clone-github-repo)
      - [Bước 2: Môi trường ảo](#bước-2-môi-trường-ảo)
      - [Bước 3: Cài đặt](#bước-3-cài-đặt)
      - [Bước 4: Chạy](#bước-4-chạy)
    - [Bước 5: Sử dụng](#bước-5-sử-dụng)
  - [Cấu trúc Database](#cấu-trúc-database)
  - [Dependencies chính](#dependencies-chính)
  - [Huấn luyện Classifier](#huấn-luyện-classifier)
  - [Ghi chú](#ghi-chú)

## Tổng quan

Ứng dụng cung cấp pipeline hoàn chỉnh cho văn bản tiếng Việt:

- Chuẩn hóa, sửa viết tắt/từ lóng và tokenize bằng `underthesea`
- Sinh embedding PhoBERT-base-v2, phân loại bằng SVM đã huấn luyện
- Lưu lịch sử phân tích vào SQLite

## Cấu trúc thư mục

```text
VNSAA/
├── app.py                      # UI Streamlit + điều phối pipeline, phân trang lịch sử
├── preprocessing.py            # standardize → slang correction → tokenize
├── model_loading.py            # tải PhoBERT-base-v2 + tokenizer + SVM pickle
├── sentiment_classification.py # tạo embedding CLS, dự đoán label + score
├── database.py                 # SQLite CRUD, cursor pagination, đếm trang
├── utils.py                    # UI helper hiển thị kết quả & pipeline
├── constant.py                 # DB_NAME, giới hạn độ dài, từ điển sửa từ lóng
├── requirements.txt            # danh sách package
├── sentiment_data.db           # database tạo tự động khi chạy app
├── train_svm_phobert.py        # training SVM classifier
├── data_sentiment_vn.csv       # dataset để training SVM classifier
└── svm_phobert_sentiment.pkl   # classifier đã huấn luyện
```

## Mô hình sử dụng

### PhoBERT + SVM

- **Backbone**: `vinai/phobert-base-v2` (AutoModel + AutoTokenizer, chạy CPU)
- **Classifier**: SVM tuyến tính (`svm_phobert_sentiment.pkl`) huấn luyện trên embedding CLS
- **Nhãn**: NEGATIVE / NEUTRAL / POSITIVE, tự động chuyển về NEUTRAL nếu score < 0.5

## Pipeline xử lý

### 1. Tiền xử lý

- `standardize_text`: strip + lowercase
- `correct_slang_words`: thay thế bằng `CORRECTION_DICT`
- `tokenize_text`: `underthesea.word_tokenize`, thay khoảng trắng bằng `_`

### 2. Phân loại

- Dataset suy luận 1 mẫu → DataLoader batch 1
- PhoBERT tạo embedding CLS → numpy
- SVM dự đoán xác suất, ép `NEUTRAL` nếu score < 0.5

### 3. Lưu trữ & UI

- Ghi bản ghi đã tokenize + nhãn vào SQLite
- Lược sử hiển thị dạng bảng có phân trang
- Cột phải hiển thị icon cảm xúc + chi tiết pipeline

## Tính năng

- Nhập nhanh văn bản (5–50 ký tự), pipeline chạy đầy đủ, kết quả hiển thị tức thì
- Phần lịch sử cho phép làm mới, phân trang trước/sau, xem tổng số trang
- Có dialog xác nhận trước khi xóa toàn bộ lịch sử trong DB
- Thông báo lỗi thân thiện khi pipeline hoặc DB gặp sự cố

## Hướng dẫn cài đặt

### Yêu cầu hệ thống

- Python 3.8 trở lên
- Kết nối internet (để tải mô hình từ Hugging Face lần đầu)

### Các bước thực hiện

#### Bước 1: Clone Github Repo

```bash
git clone https://github.com/trucpham04/VNSAA.git
cd VNSAA
```

#### Bước 2: Môi trường ảo

```bash
python -m venv .venv
# Linux & MacOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

#### Bước 3: Cài đặt

```bash
pip install -r requirements.txt
```

> Torch + transformers + underthesea khá lớn, cần kết nối mạng ổn định.

#### Bước 4: Chạy

```bash
streamlit run app.py
```

> App tự tạo database (nếu chưa có), tải PhoBERT lần đầu rồi mở `http://localhost:8501`.

### Bước 5: Sử dụng

Nhập câu 5–50 ký tự → bấm “Phân tích” → xem kết qủa phân tích kèm quá trình chi tiết.

## Cấu trúc Database

- **Bảng**: `sentiments`
- `id` INTEGER, PRIMARY KEY, AUTOINCREMENT
- `text` TEXT, NOT NULL (chuỗi đã tokenize)
- `sentiment` TEXT, NOT NULL (NEGATIVE/NEUTRAL/POSITIVE)
- `timestamp` DATETIME, DEFAULT CURRENT_TIMESTAMP

## Dependencies chính

- `streamlit`: UI, cache model, quản lý state phân trang
- `transformers` + `torch`: tải PhoBERT và chạy embedding
- `underthesea`: word_tokenize tiếng Việt
- `scikit-learn` + `joblib`: huấn luyện & load SVM
- `pandas` + `sqlite3`: thao tác dữ liệu và DB nhẹ

## Huấn luyện Classifier

- **Dataset**:
  - `data_sentiment_vn.csv` (cột `text`, `label` với giá trị 0/1/2). Văn bản được chuẩn hóa, sửa từ lóng và tokenize giống pipeline suy luận.
  - Tập dữ liệu gồm 150 các câu thường dùng trong đời sống hằng ngày.
- **Feature extractor**: PhoBERT-base-v2 sinh embedding CLS (max_len 150, batch 32, CPU), lưu thành ma trận numpy.
- **Training**: Chia tập 80/20 (stratify), huấn luyện `SVC(kernel='linear', probability=True, gamma=0.125)`, in accuracy + classification report.
- **Xuất model**: Sau khi train, classifier được lưu tại `svm_phobert_sentiment.pkl` (joblib). Đảm bảo file này nằm ở thư mục gốc để `model_loading.py` sử dụng.
- **Chạy lại training**

```bash
python train_svm_phobert.py
```

> Lần chạy đầu sẽ tải PhoBERT và có thể tốn vài phút tùy kích thước dataset.

## Ghi chú

- Lần chạy đầu cần thời gian tải PhoBERT + dependencies; các lần sau dùng cache.
- DB SQLite và file `svm_phobert_sentiment.pkl` được đọc/ghi tại thư mục gốc dự án.
- Giữ số ký tự đầu vào trong khoảng 5–50 để tránh lỗi kiểm tra độ dài.
- Sẵn sàng mở rộng: chỉ cần cập nhật SVM pickle mới và/hoặc từ điển sửa từ lóng.
