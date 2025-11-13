import streamlit as st
from database import delete_all_records, initialize_database, save_to_sqlite, load_data_from_sqlite, has_more_records, get_total_pages
from model_loading import load_model_pipeline
from preprocessing import correct_slang_words, standardize_text, tokenize_text
from sentiment_classification import classify_sentiment
from utils import show_pipeline_steps, show_sentiment_result

# =========================== Full Pipeline ===========================
def full_pipeline(text: str, sentiment_pipeline):
    try:
        # === Bước 1: Tiền xử lý

        # Chuẩn hóa văn bản
        standardized_text = standardize_text(text)

        # Sửa những từ không dấu, viết tắt, từ lóng
        corrected_text = correct_slang_words(standardized_text)

        # Tách từ
        tokenized_text = tokenize_text(corrected_text)

        # === Bước 2: Phân loại cảm xúc
        sentiment = classify_sentiment(tokenized_text, sentiment_pipeline)

        # === Bước 3: Hợp nhất và xử lý lỗi
        result = {
            "text": tokenized_text,
            "sentiment": sentiment['label'],
        }

        # Kiểm tra hợp lệ
        if len(result["text"]) < 5 or len(result["text"]) > 50:
            return None, None, "Độ dài câu không hợp lệ, vui lòng thử lại (5-50 ký tự)"

        # Lưu kết quả vào database
        save_to_sqlite(result)

        # Thông tin hiển thị
        display_result = {
            "original_text": text,
            "corrected_text": corrected_text,
            "tokenized_text": tokenized_text,
            "sentiment_label": sentiment['label'],
            "sentiment_score": round(sentiment['score'] * 100, 2),
        }
               
        # Trả về kết quả
        return result, display_result, None

    except Exception as e:
        return None, f"Pipeline error: {e}. Please try again."

# =========================== UI ===========================
initialize_database()
global_pipeline = load_model_pipeline()

st.set_page_config(page_title="Vietnamese Sentiment Assistant", layout="wide")

st.markdown("# Nhận diện cảm xúc tiếng Việt")

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
            reset_pagination()
            st.rerun()

    with history_header_col2:
        if st.button("Xóa tất cả", type="tertiary", icon="🗑️", width="stretch", on_click=confirm_delete_all):
            pass
        
    with history_header_col3:
        if st.button("Làm mới", icon="🔄", width="stretch", on_click=reset_pagination):
            pass

    df_history = load_data_from_sqlite(last_id=st.session_state.pagination_last_id)
    
    current_last_id = None
    if not df_history.empty:
        current_last_id = int(df_history.iloc[-1]['id'])
        st.session_state.pagination_has_more = has_more_records(current_last_id)
    else:
        st.session_state.pagination_has_more = False

    def go_to_next_page():
        if current_last_id is not None:
            if st.session_state.pagination_last_id is not None:
                st.session_state.pagination_history.append(st.session_state.pagination_last_id)
            st.session_state.pagination_last_id = current_last_id

    def go_to_previous_page():
        if st.session_state.pagination_history:
            st.session_state.pagination_last_id = st.session_state.pagination_history.pop()
        else:
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
        
        pagination_col1, pagination_col2, pagination_col3, pagination_col4, pagination_col5 = st.columns([2, 1, 0.5, 1, 2 ])
        
        is_first_page = st.session_state.pagination_last_id is None
        if is_first_page:
            current_page = 1
        else:
            current_page = len(st.session_state.pagination_history) + 2
        total_pages = get_total_pages()
        
        with pagination_col2:
            if st.button("◀ Trước", disabled=is_first_page, use_container_width=True):
                go_to_previous_page()
                st.rerun()
        
        with pagination_col3:
            st.markdown(
                f"<div style='text-align:center; font-weight:600'>{current_page}/{total_pages}</div>",
                unsafe_allow_html=True,
            )
        
        with pagination_col4:
            if st.button("Tiếp theo ▶", disabled=not st.session_state.pagination_has_more, use_container_width=True):
                go_to_next_page()
                st.rerun()

with col_2:
    if analyze_button:
            reset_pagination()
            result, display_result, error = full_pipeline(user_input, global_pipeline)

            if result and display_result:
                # Hiển thị kết quả
                show_sentiment_result(result['sentiment'], display_result['sentiment_score'])

                # Hiển thị chi tiết các bước trong pipeline
                show_pipeline_steps(display_result['original_text'], display_result['corrected_text'], display_result['tokenized_text'], display_result['sentiment_label'], result)

            if error:
                st.error(f"Lỗi: {error}")
    else:
        st.info("Vui lòng nhập một câu và nhấn 'Phân tích' để đánh giá cảm xúc.")