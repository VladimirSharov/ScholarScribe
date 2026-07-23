import json
import pandas as pd
from datetime import datetime
from collections import Counter
from langdetect import detect, LangDetectException

# Load the original results
with open('generated_tags_results_union_tags.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

results = data['results']
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"tag_language_analysis_{timestamp}.json"

def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

language_analysis = {
    "metadata": {
        "source_file": "generated_tags_results.json",
        "analysis_timestamp": datetime.now().isoformat(),
        "total_items_analyzed": len(results),
        "language_detection_method": "langdetect"
    },
    "results": []
}

# Language confusion matrix data
language_matrix = Counter()

for item in results:
    identifier = item["identifier"]
    
    # Analyze original subject tags
    original_subject_tags = item.get("original_subject_tags", [])
    original_subject_langs = {}
    for tag in original_subject_tags:
        lang = detect_language(tag)
        original_subject_langs[tag] = lang
    
    # Analyze original additional tags
    original_additional_tags = item.get("original_additional_tags", [])
    original_additional_langs = {}
    for tag in original_additional_tags:
        lang = detect_language(tag)
        original_additional_langs[tag] = lang
    
    # Analyze generated tags
    generated_tags = item.get("generated_tags", [])
    generated_langs = {}
    for tag in generated_tags:
        lang = detect_language(tag)
        generated_langs[tag] = lang
    
    # Count languages
    original_lang_counts = Counter([lang for lang in list(original_subject_langs.values()) + list(original_additional_langs.values())])
    generated_lang_counts = Counter(generated_langs.values())
    
    # Determine dominant languages
    dominant_original_lang = original_lang_counts.most_common(1)[0][0] if original_lang_counts else "unknown"
    dominant_generated_lang = generated_lang_counts.most_common(1)[0][0] if generated_lang_counts else "unknown"
    
    # Update confusion matrix
    language_matrix[(dominant_original_lang, dominant_generated_lang)] += 1
    
    # Create analysis result
    analysis_result = {
        "identifier": identifier,
        "original_identifier": item.get("original_identifier", ""),
        "original_subject_tags_with_lang": [{"tag": tag, "language": lang} for tag, lang in original_subject_langs.items()],
        "original_additional_tags_with_lang": [{"tag": tag, "language": lang} for tag, lang in original_additional_langs.items()],
        "generated_tags_with_lang": [{"tag": tag, "language": lang} for tag, lang in generated_langs.items()],
        "original_language_counts": dict(original_lang_counts),
        "generated_language_counts": dict(generated_lang_counts),
        "dominant_original_language": dominant_original_lang,
        "dominant_generated_language": dominant_generated_lang,
        "language_consistency": dominant_original_lang == dominant_generated_lang
    }
    
    language_analysis["results"].append(analysis_result)

# Convert confusion matrix to a format suitable for JSON and visualization
formatted_matrix = []
for (orig_lang, gen_lang), count in language_matrix.items():
    formatted_matrix.append({
        "original_language": orig_lang,
        "generated_language": gen_lang,
        "count": count
    })

language_analysis["language_matrix"] = formatted_matrix

# Calculate summary stats
language_analysis["summary"] = {
    "language_consistency_rate": sum(1 for r in language_analysis["results"] if r["language_consistency"]) / len(language_analysis["results"]),
    "original_languages": dict(Counter([r["dominant_original_language"] for r in language_analysis["results"]])),
    "generated_languages": dict(Counter([r["dominant_generated_language"] for r in language_analysis["results"]])),
}

# Save to file
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(language_analysis, f, ensure_ascii=False, indent=2)

print(f"Analysis saved to {output_filename}")