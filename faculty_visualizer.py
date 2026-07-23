"""
Faculty Visualization Module
This module creates visualizations based on processed faculty data.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os
from datetime import datetime

class FacultyVisualizer:
    def __init__(self, output_dir="images"):
        """
        Initialize the visualizer
        
        Args:
            output_dir: Directory where images will be saved
        """
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = os.path.join(output_dir, f"faculties_{self.timestamp}")
        self.color_palette = None
        self.abbreviations = None
        self.created_files = []
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def setup_colors(self, faculty_counts):
        """
        Set up color palette for faculties
        
        Args:
            faculty_counts: Dictionary of faculty counts
            
        Returns:
            Dictionary mapping faculties to colors
        """
        # Use a good color palette that works well with many categories
        n_faculties = len(faculty_counts)
        
        if n_faculties <= 10:
            # Use qualitative palette for few faculties
            palette = sns.color_palette("tab10", n_faculties)
        else:
            # Use a custom colormap for many faculties
            cmap = LinearSegmentedColormap.from_list("custom_cmap", 
                                                   ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", 
                                                    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", 
                                                    "#bcbd22", "#17becf"], N=n_faculties)
            palette = [cmap(i/n_faculties) for i in range(n_faculties)]
        
        # Create color dictionary
        faculty_colors = {}
        for i, faculty in enumerate(faculty_counts.keys()):
            faculty_colors[faculty] = palette[i]
        
        self.color_palette = faculty_colors
        return faculty_colors
    
    def create_legend_with_counts(self, faculty_counts, faculty_pairs, abbreviations, metadata):
        """
        Create a legend image showing faculty names, abbreviations, and counts
        
        Args:
            faculty_counts: Dictionary of faculty counts
            faculty_pairs: Dictionary mapping English faculty names to Finnish names
            abbreviations: Dictionary of faculty abbreviations
            metadata: Dictionary of metadata about the analysis
            
        Returns:
            Path to the saved image
        """
        self.abbreviations = abbreviations
        
        # Sort faculties by count (descending)
        sorted_faculties = sorted(faculty_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Set up colors if not already done
        if not self.color_palette:
            self.setup_colors(faculty_counts)
            
        # Create figure
        fig, ax = plt.subplots(figsize=(14, len(sorted_faculties)*0.6 + 4))
        
        # Add a guide to explain the table structure
        guide_text = (
            "Faculty Legend Guide:\n"
            "- Abbr: Abbreviation used in visualizations\n"
            "- Faculty (EN): Faculty name in English\n"
            "- Faculty (FI): Faculty name in Finnish\n"
            "- Count: Number of works associated with this faculty\n\n"
            "Note: The current university structure has 6 main faculties, but historical names are included."
        )
        plt.figtext(0.1, 0.95, guide_text, ha='left', va='top', fontsize=9, 
                    bbox=dict(facecolor='lightyellow', alpha=0.5))
        
        # Table data
        table_data = []
        for faculty, count in sorted_faculties:
            # Find Finnish counterpart
            fi_name = "N/A"
            for en, fi in faculty_pairs.items():
                if en.lower() in faculty.lower() or faculty.lower() in en.lower():
                    fi_name = fi
                    break
                    
            if fi_name == "N/A" and "(Unknown Faculty)" in faculty:
                # This is a Finnish-only faculty
                fi_name = faculty.replace(" (Unknown Faculty)", "")
                
            abbr = abbreviations.get(faculty, "")
            table_data.append([abbr, faculty, fi_name, count])
        
        # Create table
        col_labels = ["Abbr.", "Faculty (EN)", "Faculty (FI)", "Count"]
        table = ax.table(cellText=table_data, colLabels=col_labels, 
                        loc='center', cellLoc='left')
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)  # Increased row height for better readability
        
        # Adjust column widths
        table.auto_set_column_width([0, 3])  # Auto-width for abbreviation and count
        for i in range(len(table_data) + 1):  # +1 for header
            table[(i, 1)].set_width(0.45)  # English name
            table[(i, 2)].set_width(0.45)  # Finnish name
        
        # Color the abbreviation cells
        for i, (faculty, _) in enumerate(sorted_faculties):
            color = self.color_palette[faculty]
            table[(i+1, 0)].set_facecolor(color)
            # Make text white for dark backgrounds
            if sum(color[:3]) < 1.5:  # Rough check for dark colors
                table[(i+1, 0)].get_text().set_color('white')
        
        # Add metadata as title/footer
        title_text = f"Faculty Distribution Legend\n"
        subtitle_text = f"Database: {os.path.basename(metadata['database_path'])}\n"
        date_text = f"Analysis timestamp: {metadata['analysis_timestamp']}"
        
        plt.suptitle(title_text, fontsize=16, y=0.98)
        plt.title(subtitle_text + date_text, fontsize=10)
        
        # Add note about data quality
        note_text = (
            "Note: The faculty names in the dataset show some inconsistency. "
            "This analysis attempted to normalize names to the 6 current faculty structure."
        )
        plt.figtext(0.5, 0.01, note_text, ha='center', fontsize=8, 
                    bbox=dict(facecolor='lightgray', alpha=0.3))
        
        # Remove axis
        ax.axis('off')
        
        # Save figure
        output_path = os.path.join(self.output_dir, f"faculty_legend_{self.timestamp}.png")
        plt.tight_layout(rect=[0, 0.03, 1, 0.92])  # Leave space for title and footer
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.created_files.append(output_path)
        return output_path
    
    def create_deviation_chart(self, faculty_counts, metadata):
        """
        Create a chart showing deviation from average count for each faculty
        
        Args:
            faculty_counts: Dictionary of faculty counts
            metadata: Dictionary of metadata about the analysis
            
        Returns:
            Path to the saved image
        """
        # Calculate average
        avg_count = sum(faculty_counts.values()) / len(faculty_counts)
        
        # Calculate deviations
        deviations = {k: v - avg_count for k, v in faculty_counts.items()}
        
        # Sort by deviation
        sorted_deviations = sorted(deviations.items(), key=lambda x: x[1])
        
        # Set up colors if not already done
        if not self.color_palette:
            self.setup_colors(faculty_counts)
            
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Use abbreviations if available
        if self.abbreviations:
            labels = [self.abbreviations.get(faculty, faculty[:5]) for faculty, _ in sorted_deviations]
        else:
            # Use shortened faculty names
            labels = [faculty[:15] + "..." if len(faculty) > 15 else faculty for faculty, _ in sorted_deviations]
            
        y_pos = np.arange(len(labels))
        values = [dev for _, dev in sorted_deviations]
        
        # Bar colors
        colors = [self.color_palette.get(faculty, 'blue') for faculty, _ in sorted_deviations]
        
        # Create horizontal bar chart
        ax.barh(y_pos, values, align='center', color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        
        # Add a line at zero
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(values):
            ax.text(v + (1 if v >= 0 else -1), i, f"{v:.1f}", 
                   va='center', fontsize=8,
                   ha='left' if v >= 0 else 'right')
        
        # Add annotation showing average
        ax.text(0.02, 0.02, f"Average: {avg_count:.1f} per faculty", 
               transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        
        # Set labels and title
        ax.set_xlabel('Deviation from Average')
        ax.set_title(f"Faculty Count Deviation from Average\n{metadata['database_path']}\n"
                    f"Analysis timestamp: {metadata['analysis_timestamp']}")
        
        # Adjust layout and save
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f"faculty_deviation_{self.timestamp}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.created_files.append(output_path)
        return output_path
    
    def create_pie_chart(self, faculty_counts, metadata):
        """
        Create a pie chart showing distribution of faculties
        
        Args:
            faculty_counts: Dictionary of faculty counts
            metadata: Dictionary of metadata about the analysis
            
        Returns:
            Path to the saved image
        """
        # Set up colors if not already done
        if not self.color_palette:
            self.setup_colors(faculty_counts)
            
        # Sort faculties by count (descending)
        sorted_faculties = sorted(faculty_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Use abbreviations if available
        if self.abbreviations:
            labels = [f"{self.abbreviations.get(faculty, faculty[:3])}: {count}" 
                     for faculty, count in sorted_faculties]
        else:
            labels = [f"{faculty[:10]}...: {count}" if len(faculty) > 10 else f"{faculty}: {count}" 
                     for faculty, count in sorted_faculties]
        
        sizes = [count for _, count in sorted_faculties]
        colors = [self.color_palette.get(faculty, 'blue') for faculty, _ in sorted_faculties]
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            sizes, 
            labels=None,  # We'll use legend instead of direct labels
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            pctdistance=0.85
        )
        
        # Make percentage text smaller and white for better contrast
        for autotext in autotexts:
            autotext.set_size(8)
            autotext.set_color('white')
        
        # Add legend with custom sorting
        ax.legend(
            wedges, 
            labels, 
            title="Faculties",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1)
        )
        
        # Add title with metadata
        ax.set_title(f"Faculty Distribution\n{metadata['database_path']}\n"
                    f"Analysis timestamp: {metadata['analysis_timestamp']}")
        
        # Add circle in the middle to make it a donut chart
        centre_circle = plt.Circle((0, 0), 0.5, fc='white')
        ax.add_patch(centre_circle)
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.set_aspect('equal')
        
        # Save figure
        output_path = os.path.join(self.output_dir, f"faculty_pie_{self.timestamp}.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.created_files.append(output_path)
        return output_path
    
    def create_all_visualizations(self, results):
        """
        Create all visualizations based on the processed data
        
        Args:
            results: Dictionary with processed data and metadata
            
        Returns:
            List of paths to saved images
        """
        faculty_counts = results["faculty_counts"]
        faculty_pairs = results["faculty_pairs"]
        abbreviations = results["abbreviations"]
        metadata = results["metadata"]
        
        # Create visualizations
        legend_path = self.create_legend_with_counts(faculty_counts, faculty_pairs, abbreviations, metadata)
        deviation_path = self.create_deviation_chart(faculty_counts, metadata)
        pie_path = self.create_pie_chart(faculty_counts, metadata)
        
        # Create metadata file
        metadata_path = os.path.join(self.output_dir, f"metadata_{self.timestamp}.txt")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write("# Faculty Analysis Metadata\n\n")
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        
        self.created_files.append(metadata_path)
        
        return self.created_files
