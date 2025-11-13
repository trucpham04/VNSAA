import streamlit as st

def show_sentiment_result(label: str, score: float):
    st.markdown("### Kết quả phân tích")
    match label:
        case "POSITIVE":
            return st.success(f"Tích cực - {score}%", icon="😊")
        case "NEGATIVE":
            return st.error(f"Tiêu cực - {score}%", icon="😠")
        case "NEUTRAL":
            return st.warning(f"Trung tính - {score}%", icon="😐")

def show_pipeline_steps(text, corrected_text, tokenized_text, sentiment, result):
    with st.expander("Xem chi tiết luồng xử lý", expanded=True):
        st.markdown("##### 1. Câu ban đầu")
        st.code(text)

        st.markdown("##### 2. Tiền xử lý")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("###### Chuẩn hóa")
            st.code(corrected_text)
        with col2:
            st.markdown("###### Phân đoạn từ")
            st.code(tokenized_text)

        st.markdown("##### 3. Phân loại cảm xúc")
        st.code(sentiment['label'])
        
        st.markdown("##### 4. Hợp nhất kết quả")
        st.json(result)