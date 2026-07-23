import matplotlib.pyplot as plt
import numpy as np

# Fixed Prompt Distribution (Before)
fixed_distribution = {
    "0.0": 1278,
    "0.1-0.2": 182,
    "0.2-0.3": 120,
    "0.3-0.4": 46,
    "0.4-0.5": 11,
    "0.5-0.6": 11,
    "0.6-0.7": 2,
    "0.7-0.8": 1,
    "0.8-0.9": 0,
    "0.9-1.0": 0,
    "1.0": 0
}

# Adaptive Prompt Distribution (After)
adaptive_distribution = {
    "0.0": 262,
    "0.1-0.2": 544,
    "0.2-0.3": 380,
    "0.3-0.4": 203,
    "0.4-0.5": 82,
    "0.5-0.6": 59,
    "0.6-0.7": 17,
    "0.7-0.8": 2,
    "0.8-0.9": 1,
    "0.9-1.0": 0,
    "1.0": 0
}

# Categories
bins = list(fixed_distribution.keys())
fixed_values = [fixed_distribution[b] for b in bins]
adaptive_values = [adaptive_distribution[b] for b in bins]

# Plot
x = np.arange(len(bins))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bar1 = ax.bar(x - width/2, fixed_values, width, label='Fixed Prompt')
bar2 = ax.bar(x + width/2, adaptive_values, width, label='Adaptive Prompt')

# Labels and formatting
ax.set_xlabel('Exact Match Score Range')
ax.set_ylabel('Number of Samples')
ax.set_title('Exact Match Score Distribution: Fixed vs Adaptive Prompting')
ax.set_xticks(x)
ax.set_xticklabels(bins, rotation=45)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Annotate counts on bars
for i, v in enumerate(fixed_values):
    if v > 0:
        ax.text(i - width/2, v + 10, str(v), ha='center', va='bottom', fontsize=8)
for i, v in enumerate(adaptive_values):
    if v > 0:
        ax.text(i + width/2, v + 10, str(v), ha='center', va='bottom', fontsize=8)

# Save figure
plt.tight_layout()
plt.savefig("exact_match_distribution_comparison.png", dpi=300)
plt.show()
