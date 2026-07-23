import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def load_visualization_data(file_path):
    """Load the visualization data file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_visualizations(data_file, output_dir='visualizations'):
    """Create visualizations from the evaluation data"""
    # Load data
    viz_data = load_visualization_data(data_file)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Create summary bar chart
    create_summary_chart(viz_data, output_dir)
    
    # Create distribution charts
    create_distribution_charts(viz_data, output_dir)
    
    # Create comparison chart
    create_comparison_chart(viz_data, output_dir)
    
    print(f"Visualizations saved to {output_dir} directory")

def create_summary_chart(viz_data, output_dir):
    """Create a summary bar chart of key metrics"""
    metrics = viz_data["metrics_summary"]
    
    # Prepare data
    categories = ["Subject Tags", "Additional Tags"]
    exact_match = [metrics["subject_tags"]["exact_match_average"], 
                   metrics["additional_tags"]["exact_match_average"]]
    bleu_score = [metrics["subject_tags"]["bleu_score_average"], 
                  metrics["additional_tags"]["bleu_score_average"]]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create grouped bar chart
    x = range(len(categories))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], exact_match, width, label='Exact Match Ratio')
    ax.bar([i + width/2 for i in x], bleu_score, width, label='BLEU Score')
    
    # Add labels and title
    ax.set_ylabel('Score (0-1)')
    ax.set_title(f'Tag Generation Performance Metrics\n{viz_data["model_name"]}')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    
    # Add value labels on bars
    for i, v in enumerate(exact_match):
        ax.text(i - width/2, v + 0.02, f'{v:.2f}', ha='center')
    
    for i, v in enumerate(bleu_score):
        ax.text(i + width/2, v + 0.02, f'{v:.2f}', ha='center')
    
    # Set y-axis limits
    ax.set_ylim(0, 1.1)
    
    # Add total samples info
    plt.figtext(0.5, 0.01, f'Total Samples: {viz_data["total_samples"]}', 
                ha='center', fontsize=10)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_metrics.png'), dpi=300)
    plt.close()

def create_distribution_charts(viz_data, output_dir):
    """Create distribution charts for exact match ratios"""
    # Subject tags distribution
    subject_dist = viz_data["distribution"]["subject_exact_match"]
    additional_dist = viz_data["distribution"]["additional_exact_match"]
    
    # Convert to DataFrames
    subject_df = pd.DataFrame(list(subject_dist.items()), columns=['Range', 'Count'])
    additional_df = pd.DataFrame(list(additional_dist.items()), columns=['Range', 'Count'])
    
    # Create figures
    create_single_distribution(subject_df, "Subject Tags", output_dir)
    create_single_distribution(additional_df, "Additional Tags", output_dir)

def create_single_distribution(df, tag_type, output_dir):
    """Create a single distribution chart"""
    plt.figure(figsize=(12, 6))
    
    # Create bar chart
    ax = sns.barplot(x='Range', y='Count', data=df)
    
    # Add labels and title
    plt.title(f'Distribution of Exact Match Ratios - {tag_type}')
    plt.xlabel('Exact Match Ratio Range')
    plt.ylabel('Number of Samples')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(df['Count']):
        ax.text(i, v + 0.5, str(v), ha='center')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{tag_type.lower().replace(" ", "_")}_distribution.png'), dpi=300)
    plt.close()

def create_comparison_chart(viz_data, output_dir):
    """Create a chart comparing perfect/zero matches"""
    metrics = viz_data["metrics_summary"]
    
    # Prepare data
    categories = ["Subject Tags", "Additional Tags"]
    perfect = [metrics["subject_tags"]["perfect_matches_percent"], 
               metrics["additional_tags"]["perfect_matches_percent"]]
    zero = [metrics["subject_tags"]["zero_matches_percent"], 
            metrics["additional_tags"]["zero_matches_percent"]]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create grouped bar chart
    x = range(len(categories))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], perfect, width, label='Perfect Matches (%)', color='green')
    ax.bar([i + width/2 for i in x], zero, width, label='Zero Matches (%)', color='red')
    
    # Add labels and title
    ax.set_ylabel('Percentage')
    ax.set_title('Perfect vs. Zero Matches')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    
    # Add value labels on bars
    for i, v in enumerate(perfect):
        ax.text(i - width/2, v + 1, f'{v:.1f}%', ha='center')
    
    for i, v in enumerate(zero):
        ax.text(i + width/2, v + 1, f'{v:.1f}%', ha='center')
    
    # Set y-axis limits
    ax.set_ylim(0, 100)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'perfect_zero_comparison.png'), dpi=300)
    plt.close()

def main():
    # File path
    data_file = "tag_visualization_data.json"
    output_dir = "tag_visualizations"
    
    # Check if input file exists
    if not os.path.exists(data_file):
        print(f"Error: Visualization data file '{data_file}' not found.")
        print("Please run the evaluation script first to generate this file.")
        return
    
    # Create visualizations
    print(f"Creating visualizations from '{data_file}'...")
    create_visualizations(data_file, output_dir)

if __name__ == "__main__":
    main()