import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

# Load your analysis results
with open("tag_language_analysis_20250421_152813.json", 'r', encoding='utf-8') as f:
    analysis = json.load(f)
timestamp = "20250421_152813"

# 1. Language Distribution Comparison
plt.figure(figsize=(15, 5))

# Function to plot language distribution
def plot_lang_dist(ax, data, title):
    labels = list(data.keys())
    sizes = list(data.values())
    ax.bar(labels, sizes)
    ax.set_title(title)
    ax.set_ylabel('Count')
    # Rotate x labels if needed
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

# Create three subplots
ax1 = plt.subplot(1, 3, 1)
plot_lang_dist(ax1, analysis['summary']['instructed_languages'], 'Instructed Languages')

ax2 = plt.subplot(1, 3, 2)
plot_lang_dist(ax2, analysis['summary']['original_languages'], 'Original Tags Languages')

ax3 = plt.subplot(1, 3, 3)
plot_lang_dist(ax3, analysis['summary']['generated_languages'], 'Generated Tags Languages')

plt.tight_layout()
plt.savefig(f"language_distributions_{timestamp}.png")
plt.close()

# 2. Match Rates Comparison
plt.figure(figsize=(10, 6))
match_rates = analysis['summary']['match_rates']
labels = ['Instructed vs\nOriginal', 'Instructed vs\nGenerated', 'Original vs\nGenerated', 'All Match']
values = [match_rates['instructed_vs_original'], 
          match_rates['instructed_vs_generated'], 
          match_rates['original_vs_generated'],
          match_rates['all_match']]

plt.bar(labels, values)
plt.ylim(0, 1)
plt.ylabel('Match Rate')
plt.title('Language Consistency Match Rates')
for i, v in enumerate(values):
    plt.text(i, v + 0.01, f'{v:.2%}', ha='center')
plt.savefig(f"language_match_rates_{timestamp}.png")
plt.close()

# 3. Confusion Matrices (3 different matrices)
def plot_confusion_matrix(matrix_data, title, filename):
    # Extract unique languages
    languages = sorted(list(set([item['source_language'] for item in matrix_data] + 
                               [item['target_language'] for item in matrix_data])))
    
    # Create a proper matrix
    conf_matrix = np.zeros((len(languages), len(languages)))
    lang_to_idx = {lang: idx for idx, lang in enumerate(languages)}
    
    for item in matrix_data:
        src_idx = lang_to_idx[item['source_language']]
        tgt_idx = lang_to_idx[item['target_language']]
        conf_matrix[src_idx, tgt_idx] = item['count']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='.0f', xticklabels=languages, 
                yticklabels=languages, cmap='YlGnBu')
    plt.xlabel('Target Language')
    plt.ylabel('Source Language')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Plot each confusion matrix
plot_confusion_matrix(
    analysis['confusion_matrices']['instructed_to_original'],
    'Instructed → Original Language Transitions',
    f"inst_to_orig_matrix_{timestamp}.png"
)

plot_confusion_matrix(
    analysis['confusion_matrices']['instructed_to_generated'],
    'Instructed → Generated Language Transitions',
    f"inst_to_gen_matrix_{timestamp}.png"
)

plot_confusion_matrix(
    analysis['confusion_matrices']['original_to_generated'],
    'Original → Generated Language Transitions',
    f"orig_to_gen_matrix_{timestamp}.png"
)

print("Visualizations saved.")