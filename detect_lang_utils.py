import json
from pathlib import Path

# Optional: enable only what you need
USE_METHOD_1 = "langdetect"   # options: langdetect, lingua, gcld3, pycld2
USE_METHOD_2 = "lingua"       # can be same or different

# ========================
# IMPORT LANGUAGE DETECTORS (comment out what you don't want)
# ========================

try:
    from langdetect import detect as ld_detect
except ImportError:
    ld_detect = None

try:
    from lingua import LanguageDetectorBuilder, Language
    lingua_detector = LanguageDetectorBuilder.from_all_languages().build()
except ImportError:
    lingua_detector = None

try:
    import gcld3
    gcld3_detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=1000)
except ImportError:
    gcld3_detector = None

try:
    import pycld2 as cld2
except ImportError:
    cld2 = None

# ========================
# ADAPTER FUNCTIONS
# ========================

def detect_lang(text, method):
    if not text.strip():
        return "unknown"

    try:
        if method == "langdetect" and ld_detect:
            return ld_detect(text)
        elif method == "lingua" and lingua_detector:
            lang = lingua_detector.detect_language_of(text)
            return lang.iso_code_639_1.name.lower() if lang else "unknown"
        elif method == "gcld3" and gcld3_detector:
            result = gcld3_detector.FindLanguage(text)
            return result.language if result and result.language else "unknown"
        elif method == "pycld2" and cld2:
            isReliable, _, details = cld2.detect(text)
            return details[0][1].lower() if isReliable else "unknown"
        else:
            return "unknown"
    except Exception:
        return "unknown"

def detect_two_langs(text):
    lang1 = detect_lang(text, USE_METHOD_1)
    lang2 = detect_lang(text, USE_METHOD_2)
    return lang1, lang2
