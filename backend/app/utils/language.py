import re


def detect_language(text: str) -> str:
    """
    Detect language of input citizen report text.
    Supports: English, Hindi (Devanagari script), Hinglish (Romanized Hindi).
    """
    if not text or not text.strip():
        return "English"

    # Check for Devanagari Unicode character range (U+0900 to U+097F)
    devanagari_count = len(re.findall(r'[\u0900-\u097F]', text))
    total_chars = len(text.strip())
    
    if devanagari_count > 0 and (devanagari_count / max(total_chars, 1)) > 0.1:
        return "Hindi"

    # Hinglish common words heuristic
    hinglish_keywords = {
        "hai", "hain", "ho", "hu", "hoon", "ko", "se", "me", "mein", "par",
        "ka", "ki", "ke", "kya", "aur", "ya", "nahi", "nahin", "nhi", "bohot",
        "bahut", "bhi", "yeh", "voh", "woh", "kab", "kahan", "kaise", "log",
        "paani", "aag", "madad", "bachao", "ghar", "phasa", "phase", "makan"
    }
    
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    if words:
        matching = sum(1 for w in words if w in hinglish_keywords)
        if matching / len(words) > 0.15:
            return "Hinglish"

    return "English"
