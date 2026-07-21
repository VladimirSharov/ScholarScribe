# text_stats.py

def summarize_lengths(lengths):
    return {
        "count": len(lengths),
        "min": min(lengths) if lengths else 0,
        "max": max(lengths) if lengths else 0,
        "average": round(sum(lengths) / len(lengths), 2) if lengths else 0
    }

def compute_text_stats(data):
    title_word_lengths = []
    title_char_lengths = []
    abstract_word_lengths = []
    abstract_char_lengths = []
    lang_counter = {}

    for entry in data:
        title = entry.get("thesis_title", [""])[0]
        abstract = entry.get("abstract", [""])[0]
        lang = entry.get("abstract_language", "unknown")

        title_words = title.split()
        abstract_words = abstract.split()

        title_word_lengths.append(len(title_words))
        title_char_lengths.append(len(title))
        abstract_word_lengths.append(len(abstract_words))
        abstract_char_lengths.append(len(abstract))

        lang_counter[lang] = lang_counter.get(lang, 0) + 1

    return {
        "title_lengths": {
            "word_count": summarize_lengths(title_word_lengths),
            "char_count": summarize_lengths(title_char_lengths)
        },
        "abstract_lengths": {
            "word_count": summarize_lengths(abstract_word_lengths),
            "char_count": summarize_lengths(abstract_char_lengths)
        },
        "abstract_language_distribution": lang_counter
    }
