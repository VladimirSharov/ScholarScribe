import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
import os
from collections import Counter

# Create images directory if it doesn't exist
os.makedirs('images', exist_ok=True)

# Load the dataset
file_path = 'data_split_v4/full_dataset_v3_test.json'
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Function to extract and count tags
def count_tags(data, tag_field):
    tag_counter = Counter()
    for item in data:
        if tag_field in item and item[tag_field]:
            for tag in item[tag_field]:
                tag_counter[tag] += 1
    return tag_counter

# Process different tag types
tag_fields = ['additional_tags', 'subject_tags']
for tag_field in tag_fields:
    print(f"\nAnalyzing {tag_field}...")
    
    # Count tags
    tag_counts = count_tags(data, tag_field)
    
    if not tag_counts:
        print(f"No {tag_field} found in the dataset.")
        continue
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame({'tag': list(tag_counts.keys()), 
                       'count': list(tag_counts.values())})
    
    # Sort by count in descending order
    df = df.sort_values('count', ascending=False).reset_index(drop=True)
    
    # Calculate total tags and percentage covered by top 10
    total_tags = df['count'].sum()
    top_10 = df.head(10)
    top_10_sum = top_10['count'].sum()
    top_10_percentage = (top_10_sum / total_tags) * 100
    
    print(f"Total unique {tag_field}: {len(df)}")
    print(f"Total tag occurrences: {total_tags}")
    print(f"Top 10 {tag_field} cover {top_10_percentage:.2f}% of all occurrences")
    
    # 1. Plot long-tailed distribution
    plt.figure(figsize=(12, 6))
    plt.bar(range(min(100, len(df))), df['count'].head(100), color='skyblue')
    plt.title(f'Long-tailed Distribution of {tag_field}\nTop 10 tags cover {top_10_percentage:.2f}% of all occurrences', 
              fontsize=14)
    plt.xlabel('Tag Rank', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'images/{tag_field}_long_tail_distribution.png', dpi=300)
    
    # 2. Create tag cloud for least common tags (reverse top)
    plt.figure(figsize=(10, 10))
    
    # Option 1: Least common tags (reverse top)
    least_common = df.tail(100)
    wordcloud_data = {row['tag']: row['count'] for _, row in least_common.iterrows()}
    
    # Option 2: Random sampling of tags (uncomment to use)
    # random_sample = df.sample(min(100, len(df)))
    # wordcloud_data = {row['tag']: row['count'] for _, row in random_sample.iterrows()}
    
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          min_font_size=10,
                          max_font_size=150,
                          colormap='viridis').generate_from_frequencies(wordcloud_data)
    
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.title(f'Cloud of Least Common {tag_field}', fontsize=16)
    plt.savefig(f'images/{tag_field}_rare_tags_cloud.png', dpi=300)
    
    # 3. Create a horizontal bar chart for top 10 tags
    plt.figure(figsize=(10, 6))
    plt.barh(top_10['tag'], top_10['count'], color='lightcoral')
    plt.title(f'Top 10 Most Common {tag_field}', fontsize=14)
    plt.xlabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'images/{tag_field}_top10_horizontal.png', dpi=300)
    
    print(f"Visualizations for {tag_field} saved to images folder.")

# Additional visualization for publication years (date_issued)
date_counter = Counter()
for item in data:
    if 'date_issued' in item and item['date_issued']:
        for date in item['date_issued']:
            try:
                # Try to extract just the year
                year = date.strip()[:4]
                if year.isdigit() and 1900 <= int(year) <= 2030:
                    date_counter[year] += 1
            except:
                continue

if date_counter:
    # Convert to DataFrame
    df_dates = pd.DataFrame({'year': list(date_counter.keys()), 
                           'count': list(date_counter.values())})
    
    # Sort by year
    df_dates['year'] = pd.to_numeric(df_dates['year'])
    df_dates = df_dates.sort_values('year').reset_index(drop=True)
    
    # Plot publication years distribution
    plt.figure(figsize=(12, 6))
    plt.bar(df_dates['year'], df_dates['count'], color='green')
    plt.title('Distribution of Publication Years', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('images/publication_years_distribution.png', dpi=300)
    print("Publication years visualization saved to images folder.")

print("\nAll visualizations complete!")