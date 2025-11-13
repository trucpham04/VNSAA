def classify_sentiment(text: str, sentiment_pipeline):
    if sentiment_pipeline is None:
        raise Exception("Pipeline has not been initialized.")
    
    # Lấy kết quả từ mô hình
    raw_result = sentiment_pipeline(text)

    # Lấy sentiment có xác suất cao nhất
    sentiment = max(raw_result, key=lambda x: x['score'])

    # Xử lý trường hợp xác suất thấp
    if sentiment['score'] < 0.5:
        sentiment['label'] = "NEU"

    # Chuyển tên nhãn sang dạng đầy đủ
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
