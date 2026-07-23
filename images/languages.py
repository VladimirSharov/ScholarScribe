import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_data(file_path):
    """Load JSON data from file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Faculty name standardization dictionary (Finnish to English)
faculty_mapping = {
    "Humanistis-yhteiskuntatieteellinen tiedekunta": "Faculty of Humanities and Social Sciences",
    "Humanistinen tiedekunta": "Faculty of Humanities",
    "Yhteiskuntatieteellinen tiedekunta": "Faculty of Social Sciences",
    "Yhteiskuntatieteellinen  tiedekunta": "Faculty of Social Sciences",  # Note the double space
    "Kasvatustieteiden ja psykologian tiedekunta": "Faculty of Education and Psychology",
    "Faculty of Education and Psychology": "Faculty of Education and Psychology",  # Already English
    "Kasvatustieteiden tiedekunta": "Faculty of Education",
    "Liikuntatieteellinen tiedekunta": "Faculty of Sport and Health Sciences",
    "Matemaattis-luonnontieteellinen tiedekunta": "Faculty of Sciences",
    "Kauppakorkeakoulu": "School of Business and Economics",
    "Informaatioteknologian tiedekunta": "Faculty of Information Technology"
}

def analyze_language_distribution(data):
    """
    Analyze language distribution of abstracts
    
    Args:
        data (list): List of dictionary entries
    
    Returns:
        dict: Dictionary with language counts
    """
    language_counts = {}
    for entry in data:
        if 'abstract_language' in entry:
            lang = entry['abstract_language']
            language_counts[lang] = language_counts.get(lang, 0) + 1
    
    return language_counts

def visualize_language_distribution(language_counts):
    """
    Create improved visualization for language distribution
    
    Args:
        language_counts (dict): Dictionary with language counts
    """
    # Get total count for percentage calculation
    total_count = sum(language_counts.values())
    
    # Create a list of main languages and group the rest
    main_languages = ['fin', 'eng']
    others = {k: v for k, v in language_counts.items() if k not in main_languages}
    
    # Prepare data for main pie chart
    main_data = {lang: language_counts.get(lang, 0) for lang in main_languages}
    main_data['others'] = sum(others.values())
    
    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Main pie chart
    wedges, texts, autotexts = ax1.pie(
        main_data.values(), 
        labels=main_data.keys(),
        autopct='%1.1f%%',
        textprops={'fontsize': 12},
        wedgeprops={'edgecolor': 'w'}
    )
    ax1.set_title(f'Abstract Language Distribution (Total: {total_count})', fontsize=14)
    
    # Others pie chart (only if there are others)
    if others:
        if len(others) > 0:
            wedges, texts, autotexts = ax2.pie(
                others.values(), 
                labels=others.keys(),
                autopct='%1.1f%%',
                textprops={'fontsize': 10},
                wedgeprops={'edgecolor': 'w'}
            )
            ax2.set_title('Distribution of Other Languages', fontsize=14)
    else:
        ax2.axis('off')  # Turn off the axis if no 'others'
        
    plt.tight_layout()
    plt.savefig('language_distribution_improved.png')
    plt.close()

def analyze_faculty_distribution(data):
    """
    Analyze distribution of works by faculty, standardizing to English names
    
    Args:
        data (list): List of dictionary entries
    
    Returns:
        dict: Dictionary with standardized faculty distribution
    """
    faculty_counts = {}
    
    for entry in data:
        if 'faculty' in entry:
            faculties = entry['faculty']
            for faculty in faculties:
                # Standardize to English using mapping
                english_faculty = faculty_mapping.get(faculty, faculty)
                faculty_counts[english_faculty] = faculty_counts.get(english_faculty, 0) + 1
    
    return faculty_counts

def visualize_faculty_distribution(faculty_counts):
    """
    Create visualization for faculty distribution with English names only
    
    Args:
        faculty_counts (dict): Dictionary with faculty counts
    """
    # Sort faculty counts from highest to lowest
    sorted_faculties = dict(sorted(faculty_counts.items(), key=lambda item: item[1], reverse=True))
    
    # Get total for percentage calculation
    total_theses = sum(faculty_counts.values())
    
    # Create the figure
    plt.figure(figsize=(12, 10))
    
    # Create a pie chart
    wedges, texts, autotexts = plt.pie(
        sorted_faculties.values(), 
        labels=None,  # We'll add a legend for better readability
        autopct='%1.1f%%',
        textprops={'fontsize': 12},
        wedgeprops={'edgecolor': 'w'}
    )
    
    # Add a title with total count
    plt.title(f'Distribution of Works by Faculty (Total: {total_theses})', fontsize=16)
    
    # Add a legend outside the pie
    plt.legend(
        wedges, 
        sorted_faculties.keys(), 
        title="Faculties", 
        loc="center left", 
        bbox_to_anchor=(1, 0, 0.5, 1)
    )
    
    plt.tight_layout()
    plt.savefig('faculty_distribution_english.png')
    plt.close()

def main():
    # Load the dataset
    file_path = 'data_split_v4/full_dataset.json'
    data = load_data(file_path)
    
    # Print basic statistics
    print(f"Total entries in dataset: {len(data)}")
    
    # Visualize language distribution
    language_counts = analyze_language_distribution(data)
    print(f"Language distribution: {language_counts}")
    visualize_language_distribution(language_counts)
    
    # Visualize faculty distribution
    faculty_counts = analyze_faculty_distribution(data)
    print(f"Number of unique faculties: {len(faculty_counts)}")
    print(f"Faculty distribution: {faculty_counts}")
    visualize_faculty_distribution(faculty_counts)

if __name__ == "__main__":
    main()