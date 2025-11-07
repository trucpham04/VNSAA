import streamlit as st
from underthesea import word_tokenize
from transformers import pipeline
import time
from database import delete_all_records, initialize_database, save_to_sqlite, load_data_from_sqlite
from constant import CORRECTION_DICT
from utils import show_pipeline_steps, show_sentiment_result

# =========================== Model Loading ===========================
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

# =========================== Preprocessing ===========================
def normalize_text(text: str) -> str:
    normalized_sentence = text.strip().lower()
    return normalized_sentence

def correct_slang_words(text: str) -> str:
    words = text.split()
    corrected_words = [CORRECTION_DICT.get(w, w) for w in words]
    corrected_sentence = " ".join(corrected_words)
    return corrected_sentence

def tokenize_text(text: str) -> str:
    tokenized_list = word_tokenize(text)
    processed_tokens = []
    for token in tokenized_list:
        processed_tokens.append(token.replace(" ", "_"))
    final_text = " ".join(processed_tokens)
    return final_text

# =========================== Sentiment Classification ===========================
def classify_sentiment(text: str, sentiment_pipeline):
    if sentiment_pipeline is None:
        raise Exception("Pipeline has not been initialized.")
    
    raw_result = sentiment_pipeline(text)

    sentiment = max(raw_result, key=lambda x: x['score'])

    if sentiment['score'] < 0.5:
        sentiment['label'] = "NEU"

    def get_sentiment_label(label: str) -> str:
        match label:
            case "POS":
                return "POSITIVE"
            case "NEG":
                return "NEGATIVE"
            case "NEU":
                return "NEUTRAL"

    sentiment['label'] = get_sentiment_label(sentiment['label'])

    return sentiment

# =========================== Full Pipeline ===========================
def full_pipeline(text: str, sentiment_pipeline):
    try:
        # === Bước 1: Tiền xử lý

        # Chuẩn hóa câu
        normalized_sentence = normalize_text(text)

        # Sửa những từ không dấu, viết tắt, từ lóng
        corrected_text = correct_slang_words(normalized_sentence)

        # Phân đoạn từ
        tokenized_text = tokenize_text(corrected_text)

        # === Bước 2: Phân loại cảm xúc
        sentiment = classify_sentiment(tokenized_text, sentiment_pipeline)

        # === Bước 3: Hợp nhất và xử lý lỗi
        result = {
            "text": tokenized_text,
            "sentiment": sentiment['label'],
        }

        # Lưu kết quả vào database
        save_to_sqlite(result)

        # Kiểm tra hợp lệ
        if len(result["text"]) < 5:
            return None, "Câu không hợp lệ, vui lòng thử lại"
        
        # Hiển thị kết quả
        show_sentiment_result(result['sentiment'])

        # Hiển thị chi tiết các bước trong pipeline
        show_pipeline_steps(text, corrected_text, tokenized_text, sentiment, result)
        
        return result, None

    except Exception as e:
        return None, f"Pipeline error: {e}. Please try again."

# =========================== UI ===========================
initialize_database()
global_pipeline = load_model_pipeline()

st.set_page_config(page_title="Vietnamese Sentiment Assistant", layout="wide")

st.markdown("# Nhận diện cảm xúc tiếng Việt")

col_1, col_2 = st.columns([1, 1], gap="large")

with col_1:
    st.markdown("##### Nhập câu cần phân tích:")
    
    user_input = st.text_input(
        "Nhập câu (5-50 ký tự):", 
        max_chars=50, 
        key="user_input_text",
        label_visibility="collapsed"
    )

    analyze_button = st.button("Phân tích", type="primary", width="stretch")
   
    history_header_col1, history_header_col2, history_header_col3 = st.columns([4, 1, 1])

    with history_header_col1:
        st.markdown("#### Lịch sử")

    @st.dialog("Xác nhận xóa tất cả lịch sử?")
    def confirm_delete_all():
        if st.button("Submit"):
            delete_all_records()
            st.rerun()

    with history_header_col2:
        st.button("Xóa tất cả", type="tertiary", icon="🗑️", width="stretch", on_click=confirm_delete_all)
        
    with history_header_col3:
        refresh_button = st.button("Làm mới", icon="🔄", width="stretch")

    df_history = load_data_from_sqlite()
       
    if df_history.empty:
        st.info("Chưa có lịch sử!")
    else:
        df_display = df_history.copy()
        st.dataframe(df_display, 
                    hide_index=True, 
                    width="stretch",
                    selection_mode='single-row',
                    on_select='ignore',
                    column_config={
                        "id": st.column_config.NumberColumn("ID", width=25),
                        "text": st.column_config.TextColumn("Văn bản", width="large"),
                        "sentiment": st.column_config.TextColumn("Nhãn cảm xúc", width=50),
                        "timestamp": st.column_config.TextColumn("Thời gian", width=100),
                    })

with col_2:
    if analyze_button:
            result, error = full_pipeline(user_input, global_pipeline)
                        
            if error:
                st.error(f"Lỗi: {error}")
    else:
        st.info("Vui lòng nhập một câu và nhấn 'Phân tích' để đánh giá cảm xúc.")