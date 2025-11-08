# Vietnamese Sentiment Analysis Application

Ứng dụng phân tích cảm xúc tiếng Việt sử dụng mô hình PhoBERT và giao diện web Streamlit. Ứng dụng hỗ trợ phân tích cảm xúc (tích cực, tiêu cực, trung tính) từ văn bản tiếng Việt với khả năng lưu trữ lịch sử và phân trang hiệu quả.

## Mục lục

- [Tổng quan](#tổng-quan)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Mô hình sử dụng](#mô-hình-sử-dụng)
- [Pipeline xử lý](#pipeline-xử-lý)
- [Tính năng](#tính-năng)
- [Hướng dẫn cài đặt](#hướng-dẫn-cài-đặt)
- [Cấu trúc Database](#cấu-trúc-database)
- [Ghi chú](#ghi-chú)

## Tổng quan

Ứng dụng này cung cấp một pipeline hoàn chỉnh để phân tích cảm xúc văn bản tiếng Việt, bao gồm:

- **Tiền xử lý văn bản**: Chuẩn hóa, sửa lỗi chính tả, từ lóng, và phân đoạn từ
- **Phân loại cảm xúc**: Sử dụng mô hình PhoBERT được fine-tuned cho tiếng Việt
- **Lưu trữ lịch sử**: SQLite database với phân trang cursor-based tối ưu
- **Giao diện web**: Streamlit với UI thân thiện và trực quan

## Cấu trúc thư mục

```
seminar/
├── app.py                      # File chính chứa UI Streamlit và logic điều phối
├── model_loading.py            # Module tải và cache mô hình sentiment analysis
├── preprocessing.py            # Module tiền xử lý văn bản (chuẩn hóa, sửa lỗi, tokenize)
├── sentiment_classification.py # Module phân loại cảm xúc sử dụng mô hình
├── database.py                 # Module quản lý database SQLite và phân trang
├── utils.py                    # Module chứa các hàm tiện ích hiển thị
├── constant.py                 # File chứa các hằng số và từ điển sửa lỗi
├── requirements.txt            # Danh sách các package Python cần thiết
├── sentiment_data.db           # Database SQLite lưu trữ lịch sử phân tích
└── README.md                   # File tài liệu này
```

### Mô tả các file chính

- **app.py**: File entry point của ứng dụng, chứa giao diện Streamlit, quản lý state, và điều phối các module khác
- **model_loading.py**: Tải mô hình `wonrax/phobert-base-vietnamese-sentiment` từ Hugging Face và cache để tối ưu hiệu suất
- **preprocessing.py**: Xử lý văn bản đầu vào với 3 bước: normalize, correct slang, và tokenize
- **sentiment_classification.py**: Sử dụng pipeline từ transformers để phân loại cảm xúc và xử lý kết quả
- **database.py**: Quản lý SQLite database với các hàm CRUD và phân trang cursor-based
- **utils.py**: Các hàm helper để hiển thị kết quả và chi tiết pipeline
- **constant.py**: Chứa tên database, từ điển sửa lỗi từ lóng, và các hằng số khác

## Mô hình sử dụng

### PhoBERT Base Vietnamese Sentiment

Ứng dụng sử dụng mô hình **PhoBERT** được fine-tuned cho phân tích cảm xúc tiếng Việt:

- **Model**: `wonrax/phobert-base-vietnamese-sentiment`
- **Nguồn**: Hugging Face Model Hub
- **Kiến trúc**: PhoBERT (Vietnamese BERT) - một biến thể của BERT được pre-train trên corpus tiếng Việt lớn
- **Task**: Sentiment Analysis (3 lớp: POSITIVE, NEGATIVE, NEUTRAL)

### Đặc điểm mô hình

- **Input**: Văn bản tiếng Việt đã được tokenize
- **Output**: Nhãn cảm xúc (POSITIVE/NEGATIVE/NEUTRAL) kèm confidence score
- **Xử lý đặc biệt**: Nếu confidence score < 0.5, mô hình sẽ trả về NEUTRAL để tránh dự đoán không chắc chắn

## Pipeline xử lý

Pipeline xử lý văn bản gồm các bước sau:

### 1. Tiền xử lý (Preprocessing)

#### Bước 1.1: Chuẩn hóa văn bản (`normalize_text`)

- Loại bỏ khoảng trắng thừa ở đầu và cuối
- Chuyển toàn bộ văn bản sang chữ thường

#### Bước 1.2: Sửa từ lóng và viết tắt (`correct_slang_words`)

- Sử dụng từ điển `CORRECTION_DICT` trong `constant.py` để thay thế:
  - Từ viết tắt: "dk" → "được", "k" → "không", "vs" → "với"
  - Từ không dấu: "toi" → "tôi", "ban" → "bạn"
  - Từ lóng: "mik" → "mình", "iu" → "yêu"

#### Bước 1.3: Phân đoạn từ (`tokenize_text`)

- Sử dụng thư viện `underthesea` để tokenize văn bản tiếng Việt
- Thay thế khoảng trắng trong token bằng dấu gạch dưới để phù hợp với format của mô hình

### 2. Phân loại cảm xúc (Sentiment Classification)

#### Bước 2.1: Dự đoán (`classify_sentiment`)

- Đưa văn bản đã tokenize vào pipeline sentiment analysis
- Lấy nhãn có confidence score cao nhất

#### Bước 2.2: Xử lý kết quả

- Nếu confidence score < 0.5: gán nhãn NEUTRAL
- Chuyển đổi nhãn từ dạng viết tắt (POS/NEG/NEU) sang dạng đầy đủ (POSITIVE/NEGATIVE/NEUTRAL)

### 3. Lưu trữ và hiển thị

- Lưu kết quả vào SQLite database
- Hiển thị kết quả với icon và màu sắc
- Hiển thị chi tiết các bước xử lý

## Tính năng

### Phân tích cảm xúc

- Nhập văn bản tiếng Việt (5-50 ký tự)
- Phân tích tự động với pipeline hoàn chỉnh
- Hiển thị kết quả trực quan với icon và màu sắc

### Lịch sử phân tích

- Lưu trữ tất cả các kết quả phân tích vào database
- Hiển thị lịch sử với pagination cursor-based tối ưu
- Hỗ trợ phân trang khi có hơn 50 bản ghi
- Hiển thị số trang hiện tại / tổng số trang
- Nút "Làm mới" để reset về trang đầu
- Nút "Xóa tất cả" để xóa toàn bộ lịch sử

## Hướng dẫn cài đặt

### Yêu cầu hệ thống

- Python 3.8 trở lên
- Kết nối internet (để tải mô hình từ Hugging Face lần đầu)

### Bước 1: Clone repository

```bash
git clone https://github.com/trucpham04/VNSAA.git
cd VNSAA
```

### Bước 2: Tạo môi trường ảo (khuyến nghị)

```bash
# Sử dụng venv
python -m venv venv

# Kích hoạt môi trường ảo
# Trên Windows:
venv\Scripts\activate
# Trên Linux/Mac:
source venv/bin/activate
```

### Bước 3: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

**Lưu ý**: Quá trình cài đặt có thể mất vài phút do cần tải các package lớn như:

- `torch` (PyTorch)
- `transformers` (Hugging Face Transformers)
- `underthesea` (Vietnamese NLP library)

### Bước 4: Chạy ứng dụng

```bash
streamlit run app.py
```

Ứng dụng sẽ tự động:

- Tạo database SQLite nếu chưa có
- Tải mô hình từ Hugging Face (lần đầu chạy sẽ mất thời gian)
- Mở trình duyệt tại `http://localhost:8501`

### Bước 5: Sử dụng ứng dụng

1. Nhập văn bản tiếng Việt vào ô nhập liệu (5-50 ký tự)
2. Nhấn nút "Phân tích"
3. Xem kết quả ở cột bên phải
4. Xem lịch sử phân tích ở cột bên trái

## Cấu trúc Database

### Bảng `sentiments`

Lưu trữ tất cả các kết quả phân tích cảm xúc:

| Cột         | Kiểu dữ liệu                       | Mô tả                                    |
| ----------- | ---------------------------------- | ---------------------------------------- |
| `id`        | INTEGER PRIMARY KEY AUTOINCREMENT  | ID tự động tăng                          |
| `text`      | TEXT NOT NULL                      | Văn bản đã được tokenize                 |
| `sentiment` | TEXT NOT NULL                      | Nhãn cảm xúc (POSITIVE/NEGATIVE/NEUTRAL) |
| `timestamp` | DATETIME DEFAULT CURRENT_TIMESTAMP | Thời gian phân tích                      |

## Dependencies chính

- **streamlit**: Framework web UI
- **transformers**: Thư viện Hugging Face để sử dụng mô hình
- **torch**: PyTorch cho deep learning
- **underthesea**: Thư viện NLP tiếng Việt (tokenization)
- **pandas**: Xử lý dữ liệu và DataFrame
- **sqlite3**: Database SQLite (built-in Python)

## Ghi chú

- Lần đầu chạy ứng dụng, mô hình sẽ được tải từ Hugging Face và cache lại, có thể mất vài phút
- Database được tạo tự động khi chạy ứng dụng lần đầu
- Mô hình được cache trong session để tối ưu hiệu suất
- Ứng dụng hỗ trợ xử lý từ lóng và viết tắt phổ biến trong tiếng Việt
