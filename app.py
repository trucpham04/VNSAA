import streamlit as st
from database import delete_all_records, initialize_database, save_to_sqlite, load_data_from_sqlite, has_more_records, get_total_pages
from model_loading import load_model_pipeline
from preprocessing import normalize_text, correct_slang_words, tokenize_text
from sentiment_classification import classify_sentiment
from utils import show_pipeline_steps, show_sentiment_result

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

        # Kiểm tra hợp lệ
        if len(result["text"]) < 5:
            return None, "Câu không hợp lệ, vui lòng thử lại"

        # Lưu kết quả vào database
        save_to_sqlite(result)
        
        # Hiển thị kết quả
        show_sentiment_result(result['sentiment'], sentiment['score'])

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

# Initialize pagination state
if 'pagination_last_id' not in st.session_state:
    st.session_state.pagination_last_id = None
if 'pagination_history' not in st.session_state:
    st.session_state.pagination_history = []
if 'pagination_has_more' not in st.session_state:
    st.session_state.pagination_has_more = False

def reset_pagination():
    st.session_state.pagination_last_id = None
    st.session_state.pagination_history = []
    st.session_state.pagination_has_more = False

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

    @st.dialog("Xóa tất cả lịch sử?")
    def confirm_delete_all():
        if st.button("Xác nhận"):
            delete_all_records()
            # Reset pagination state
            reset_pagination()
            st.rerun()

    with history_header_col2:
        if st.button("Xóa tất cả", type="tertiary", icon="🗑️", width="stretch", on_click=confirm_delete_all):
            pass
        
    with history_header_col3:
        if st.button("Làm mới", icon="🔄", width="stretch", on_click=reset_pagination):
            pass

    # Load data with pagination
    df_history = load_data_from_sqlite(last_id=st.session_state.pagination_last_id)
    
    # Check if there are more records and calculate current last_id
    current_last_id = None
    if not df_history.empty:
        current_last_id = int(df_history.iloc[-1]['id'])
        st.session_state.pagination_has_more = has_more_records(current_last_id)
    else:
        st.session_state.pagination_has_more = False

    def go_to_next_page():
        """Navigate to next page"""
        if current_last_id is not None:
            # Save current last_id to history for "Previous" button
            if st.session_state.pagination_last_id is not None:
                st.session_state.pagination_history.append(st.session_state.pagination_last_id)
            # Update to new last_id
            st.session_state.pagination_last_id = current_last_id

    def go_to_previous_page():
        """Navigate to previous page"""
        if st.session_state.pagination_history:
            # Pop the last last_id from history
            st.session_state.pagination_last_id = st.session_state.pagination_history.pop()
        else:
            # Go back to first page
            st.session_state.pagination_last_id = None
       
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
        
        # Pagination controls
        pagination_col1, pagination_col2, pagination_col3, pagination_col4, pagination_col5 = st.columns([2, 1, 0.5, 1, 2 ])
        
        # Calculate current page and total pages
        is_first_page = st.session_state.pagination_last_id is None
        if is_first_page:
            current_page = 1
        else:
            current_page = len(st.session_state.pagination_history) + 2
        total_pages = get_total_pages()
        
        with pagination_col2:
            # Previous button - show if not on first page
            if st.button("◀ Trước", disabled=is_first_page, use_container_width=True):
                go_to_previous_page()
                st.rerun()
        
        with pagination_col3:
            # Show current page / total pages in the middle
            st.write(f"{current_page}/{total_pages}")
        
        with pagination_col4:
            # Next button - show if there are more records
            if st.button("Tiếp theo ▶", disabled=not st.session_state.pagination_has_more, use_container_width=True):
                go_to_next_page()
                st.rerun()

with col_2:
    if analyze_button:
            # Reset pagination to show latest result after new analysis
            reset_pagination()
            result, error = full_pipeline(user_input, global_pipeline)
                        
            if error:
                st.error(f"Lỗi: {error}")
    else:
        st.info("Vui lòng nhập một câu và nhấn 'Phân tích' để đánh giá cảm xúc.")