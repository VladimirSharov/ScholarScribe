import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Configure matplotlib for better compatibility and appearance
plt.rcParams.update({
    'font.family': 'serif',      # Use serif font (similar to Times)
    'font.size': 12,             # Base font size for Times New Roman 12 compatibility
    'axes.titlesize': 14,        # Title
    'axes.labelsize': 12,        # Axis labels
    'xtick.labelsize': 11,       # X-tick labels
    'ytick.labelsize': 11,       # Y-tick labels
    'legend.fontsize': 11,       # Legend
    'figure.dpi': 300,           # High resolution for better text rendering
    'savefig.dpi': 300,          # High resolution for saved figures
    'savefig.bbox': 'tight'      # Remove extra whitespace
})

def load_data(file_path):
    """Load JSON data from file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def analyze_length_distribution(data):
    """
    Analyze length distributions of titles, abstracts, and tags
    
    Args:
        data (list): List of dictionary entries
    
    Returns:
        dict: Containing length distributions
    """
    # Lengths of various fields
    title_lengths = []
    abstract_lengths = []
    yso_tag_counts = []
    other_tag_counts = []
    
    for entry in data:
        # Title lengths
        if 'thesis_title' in entry:
            title_lengths.extend([len(title) for title in entry['thesis_title']])
        
        # Abstract lengths
        if 'abstract' in entry:
            abstract_lengths.extend([len(abstract) for abstract in entry['abstract']])
        
        # Tag counts
        if 'subject_tags' in entry:
            yso_tag_counts.append(len(entry['subject_tags']))
        
        if 'additional_tags' in entry:
            other_tag_counts.append(len(entry['additional_tags']))
    
    return {
        'title_lengths': title_lengths,
        'abstract_lengths': abstract_lengths,
        'yso_tag_counts': yso_tag_counts,
        'other_tag_counts': other_tag_counts
    }

def visualize_title_lengths(title_lengths):
    """Create visualization for title lengths"""
    plt.figure(figsize=(10, 6))
    
    # Calculate statistics
    min_len = min(title_lengths)
    max_len = max(title_lengths)
    avg_len = np.mean(title_lengths)
    median_len = np.median(title_lengths)
    
    # Create histogram with better styling
    plt.hist(title_lengths, bins=30, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
    
    # Add vertical lines for statistics
    plt.axvline(avg_len, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_len:.1f}')
    plt.axvline(median_len, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_len:.1f}')
    
    plt.title(f'Distribution of Title Lengths\n(Min: {min_len}, Max: {max_len})')
    plt.xlabel('Title Length (characters)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('title_lengths_distribution.png')
    plt.close()

def visualize_abstract_lengths(abstract_lengths):
    """Create visualization for abstract lengths with better scaling"""
    plt.figure(figsize=(10, 6))
    
    # Calculate statistics
    min_len = min(abstract_lengths)
    max_len = max(abstract_lengths)
    avg_len = np.mean(abstract_lengths)
    median_len = np.median(abstract_lengths)
    
    # Use square root transformation instead of log for better visualization
    # This compresses large values while keeping smaller values more visible
    sqrt_lengths = np.sqrt(abstract_lengths)
    
    plt.hist(sqrt_lengths, bins=40, alpha=0.7, color='lightgreen', edgecolor='black', linewidth=0.5)
    
    # Add vertical lines for statistics (transformed)
    plt.axvline(np.sqrt(avg_len), color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_len:.0f}')
    plt.axvline(np.sqrt(median_len), color='orange', linestyle='--', linewidth=2, label=f'Median: {median_len:.0f}')
    
    plt.title(f'Distribution of Abstract Lengths (Square Root Scale)\n(Min: {min_len}, Max: {max_len:,})')
    plt.xlabel('√(Abstract Length) (√characters)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add second x-axis with original values
    ax1 = plt.gca()
    ax2 = ax1.twiny()
    
    # Create tick positions and labels for original scale
    sqrt_ticks = np.linspace(0, max(sqrt_lengths), 6)
    original_ticks = (sqrt_ticks ** 2).astype(int)
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(sqrt_ticks)
    ax2.set_xticklabels([f'{tick:,}' for tick in original_ticks])
    ax2.set_xlabel('Abstract Length (characters)')
    
    plt.savefig('abstract_lengths_distribution.png')
    plt.close()

def visualize_subject_tags(yso_tag_counts):
    """Create visualization for subject tag counts"""
    plt.figure(figsize=(10, 6))
    
    # Calculate statistics
    min_count = min(yso_tag_counts)
    max_count = max(yso_tag_counts)
    avg_count = np.mean(yso_tag_counts)
    median_count = np.median(yso_tag_counts)
    
    plt.hist(yso_tag_counts, bins=range(min_count, max_count + 2), 
             alpha=0.7, color='lightcoral', edgecolor='black', linewidth=0.5,
             align='left')
    
    # Add vertical lines for statistics
    plt.axvline(avg_count, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_count:.1f}')
    plt.axvline(median_count, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_count:.1f}')
    
    plt.title(f'Distribution of Subject Tag Counts\n(Min: {min_count}, Max: {max_count})')
    plt.xlabel('Number of Subject Tags')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Set integer ticks
    plt.xticks(range(min_count, max_count + 1))
    
    plt.savefig('subject_tags_distribution.png')
    plt.close()

def visualize_additional_tags(other_tag_counts):
    """Create visualization for additional tag counts"""
    plt.figure(figsize=(10, 6))
    
    # Calculate statistics
    min_count = min(other_tag_counts) if other_tag_counts else 0
    max_count = max(other_tag_counts) if other_tag_counts else 0
    avg_count = np.mean(other_tag_counts) if other_tag_counts else 0
    median_count = np.median(other_tag_counts) if other_tag_counts else 0
    
    plt.hist(other_tag_counts, bins=range(min_count, max_count + 2), 
             alpha=0.7, color='plum', edgecolor='black', linewidth=0.5,
             align='left')
    
    # Add vertical lines for statistics
    if other_tag_counts:
        plt.axvline(avg_count, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_count:.1f}')
        plt.axvline(median_count, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_count:.1f}')
    
    plt.title(f'Distribution of Additional Tag Counts\n(Min: {min_count}, Max: {max_count})')
    plt.xlabel('Number of Additional Tags')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Set integer ticks
    if max_count > 0:
        plt.xticks(range(min_count, max_count + 1))
    
    plt.savefig('additional_tags_distribution.png')
    plt.close()

def analyze_temporal_distribution(data):
    """
    Analyze distribution of works by date
    
    Args:
        data (list): List of dictionary entries
    
    Returns:
        pandas.DataFrame: DataFrame with date distribution
    """
    # Collect issued dates
    dates = []
    for entry in data:
        if 'date_issued' in entry:
            dates.extend(entry['date_issued'])
    
    # Convert to DataFrame
    df = pd.DataFrame({'date': dates})
    
    # Clean and convert dates
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    return df

def visualize_temporal_yearly(df):
    """Create visualization for works by year"""
    plt.figure(figsize=(12, 6))
    
    yearly_counts = df['date'].dt.year.value_counts().sort_index()
    
    # Create bar plot
    yearly_counts.plot(kind='bar', color='steelblue', alpha=0.8)
    plt.title('Number of Works by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Works')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add trend line
    years = yearly_counts.index
    counts = yearly_counts.values
    z = np.polyfit(years, counts, 1)
    p = np.poly1d(z)
    plt.plot(years, p(years), "r--", alpha=0.8, linewidth=2, label=f'Trend (slope: {z[0]:.2f})')
    plt.legend()
    
    plt.savefig('temporal_yearly_distribution.png')
    plt.close()

def visualize_temporal_cumulative(df):
    """Create visualization for cumulative works over time"""
    plt.figure(figsize=(12, 6))
    
    yearly_counts = df['date'].dt.year.value_counts().sort_index()
    cumulative_counts = yearly_counts.cumsum()
    
    # Create line plot
    cumulative_counts.plot(color='darkgreen', linewidth=2, marker='o', markersize=4)
    plt.title('Cumulative Number of Works Over Time')
    plt.xlabel('Year')
    plt.ylabel('Cumulative Number of Works')
    plt.grid(True, alpha=0.3)
    
    # Add annotations for key milestones
    total_works = cumulative_counts.iloc[-1]
    quarter_mark = total_works * 0.25
    half_mark = total_works * 0.5
    three_quarter_mark = total_works * 0.75
    
    for threshold, label in [(quarter_mark, '25%'), (half_mark, '50%'), (three_quarter_mark, '75%')]:
        year_idx = (cumulative_counts >= threshold).idxmax()
        plt.annotate(f'{label} reached in {year_idx}', 
                    xy=(year_idx, threshold), 
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.savefig('temporal_cumulative_distribution.png')
    plt.close()

def analyze_faculty_distribution(data):
    """
    Analyze distribution of works by faculty
    
    Args:
        data (list): List of dictionary entries
    
    Returns:
        dict: Dictionary with faculty distribution
    """
    faculty_counts = {}
    for entry in data:
        if 'faculty' in entry:
            for faculty in entry['faculty']:
                faculty_counts[faculty] = faculty_counts.get(faculty, 0) + 1
    
    return faculty_counts

def visualize_faculty_counts(faculty_counts):
    """Create visualization for faculty distribution"""
    plt.figure(figsize=(12, 8))
    
    # Sort faculties by count (descending)
    sorted_faculty = dict(sorted(faculty_counts.items(), key=lambda item: item[1], reverse=True))
    
    # Create abbreviations for better readability
    faculty_abbrevs = {}
    for faculty in sorted_faculty.keys():
        # Create abbreviation using first letter of each significant word
        words = faculty.split()
        abbrev = ''.join([word[0].upper() for word in words if len(word) > 2])
        if len(abbrev) < 2:  # Fallback for short names
            abbrev = faculty[:4].upper()
        faculty_abbrevs[faculty] = abbrev
    
    # Create horizontal bar chart for better label readability
    faculties = list(sorted_faculty.keys())
    counts = list(sorted_faculty.values())
    abbrevs = [faculty_abbrevs[faculty] for faculty in faculties]
    
    bars = plt.barh(abbrevs, counts, color='lightblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                str(count), va='center', fontsize=10)
    
    plt.title('Number of Works by Faculty')
    plt.xlabel('Number of Works')
    plt.ylabel('Faculty (Abbreviated)')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add legend with full faculty names
    legend_text = []
    for i in range(0, len(faculties), 2):  # Show every other faculty to avoid crowding
        legend_text.append(f"{faculty_abbrevs[faculties[i]]}: {faculties[i]}")
        if i + 1 < len(faculties):
            legend_text.append(f"{faculty_abbrevs[faculties[i+1]]}: {faculties[i+1]}")
    
    plt.figtext(0.02, 0.02, '\n'.join(legend_text), fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    plt.savefig('faculty_counts_distribution.png')
    plt.close()

def visualize_faculty_deviations(faculty_counts):
    """Create visualization for faculty deviations from average"""
    plt.figure(figsize=(12, 8))
    
    # Calculate average and deviations
    avg_count = np.mean(list(faculty_counts.values()))
    deviations = {faculty: (count - avg_count) / avg_count * 100 
                 for faculty, count in faculty_counts.items()}
    
    # Sort by deviation
    sorted_deviations = dict(sorted(deviations.items(), key=lambda item: item[1]))
    
    # Create abbreviations
    faculty_abbrevs = {}
    for faculty in sorted_deviations.keys():
        words = faculty.split()
        abbrev = ''.join([word[0].upper() for word in words if len(word) > 2])
        if len(abbrev) < 2:
            abbrev = faculty[:4].upper()
        faculty_abbrevs[faculty] = abbrev
    
    # Create horizontal bar chart
    faculties = list(sorted_deviations.keys())
    deviations_list = list(sorted_deviations.values())
    abbrevs = [faculty_abbrevs[faculty] for faculty in faculties]
    
    # Color code: negative deviations in red, positive in blue
    colors = ['darkred' if d < 0 else 'darkblue' for d in deviations_list]
    
    bars = plt.barh(abbrevs, deviations_list, color=colors, alpha=0.7)
    
    # Add zero line
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
    
    # Add value labels
    for bar, deviation in zip(bars, deviations_list):
        plt.text(deviation + (1 if deviation >= 0 else -1), 
                bar.get_y() + bar.get_height()/2,
                f'{deviation:+.1f}%', va='center', ha='left' if deviation >= 0 else 'right',
                fontsize=10)
    
    plt.title(f'Faculty Work Distribution: Deviation from Average ({avg_count:.1f} works)')
    plt.xlabel('Percentage Deviation from Average (%)')
    plt.ylabel('Faculty (Abbreviated)')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add legend with full faculty names
    legend_text = []
    for faculty in faculties:
        legend_text.append(f"{faculty_abbrevs[faculty]}: {faculty}")
    
    plt.figtext(0.02, 0.02, '\n'.join(legend_text), fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    plt.savefig('faculty_deviations_distribution.png')
    plt.close()

def main():
    # Load the full dataset
    input_file_path = 'data_split_v4/full_dataset.json'
    data = load_data(input_file_path)
    
    print("Creating individual visualizations...")
    
    # Length distributions (4 separate images)
    length_data = analyze_length_distribution(data)
    
    print("1. Creating title lengths visualization...")
    visualize_title_lengths(length_data['title_lengths'])
    
    print("2. Creating abstract lengths visualization...")
    visualize_abstract_lengths(length_data['abstract_lengths'])
    
    print("3. Creating subject tags visualization...")
    visualize_subject_tags(length_data['yso_tag_counts'])
    
    print("4. Creating additional tags visualization...")
    visualize_additional_tags(length_data['other_tag_counts'])
    
    # Temporal distributions (2 separate images)
    print("5. Creating temporal distributions...")
    temporal_df = analyze_temporal_distribution(data)
    visualize_temporal_yearly(temporal_df)
    visualize_temporal_cumulative(temporal_df)
    
    # Faculty distributions (2 separate images)
    print("6. Creating faculty distributions...")
    faculty_counts = analyze_faculty_distribution(data)
    visualize_faculty_counts(faculty_counts)
    visualize_faculty_deviations(faculty_counts)
    
    print("All visualizations created successfully!")
    print("Generated files:")
    print("- title_lengths_distribution.png")
    print("- abstract_lengths_distribution.png")
    print("- subject_tags_distribution.png")
    print("- additional_tags_distribution.png")
    print("- temporal_yearly_distribution.png")
    print("- temporal_cumulative_distribution.png")
    print("- faculty_counts_distribution.png")
    print("- faculty_deviations_distribution.png")

if __name__ == "__main__":
    main()