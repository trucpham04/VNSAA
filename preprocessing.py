from underthesea import word_tokenize
from constant import CORRECTION_DICT

def standardize_text(text: str) -> str:
    standardized_text = text.strip().lower()
    return standardized_text

def correct_slang_words(text: str) -> str:
    words = text.split()
    corrected_words = [CORRECTION_DICT.get(w, w) for w in words]
    corrected_text = " ".join(corrected_words)
    return corrected_text

def tokenize_text(text: str) -> str:
    # Tách từ sử dụng underthesea
    tokenized_list = word_tokenize(text)

    # Thay khoảng trắng trong token bằng dấu gạch dưới
    processed_tokens = []
    for token in tokenized_list:
        processed_tokens.append(token.replace(" ", "_"))
    final_text = " ".join(processed_tokens)

    return final_text