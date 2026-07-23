# tag_stats.py
from collections import Counter

def compute_tag_stats(data):
    subject_tags = []
    additional_tags = []
    subject_count_per_item = []
    additional_count_per_item = []

    for entry in data:
        subs = entry.get("subject_tags", [])
        adds = entry.get("additional_tags", [])

        subject_tags.extend(subs)
        additional_tags.extend(adds)

        subject_count_per_item.append(len(subs))
        additional_count_per_item.append(len(adds))

    def summarize(tags, counts):
        tag_freq = Counter(tags)
        return {
            "total_unique_tags": len(tag_freq),
            "total_tag_instances": len(tags),
            "average_tags_per_entity": round(sum(counts) / len(counts),2) if counts else 0,
            "min_tags_per_entity": min(counts) if counts else 0,
            "max_tags_per_entity": max(counts) if counts else 0,
            "top_10_tags": tag_freq.most_common(10),
            "rare_tags": [tag for tag, count in tag_freq.items() if count == 1]
        }

    return {
        "subject_tags": summarize(subject_tags, subject_count_per_item),
        "additional_tags": summarize(additional_tags, additional_count_per_item)
    }
