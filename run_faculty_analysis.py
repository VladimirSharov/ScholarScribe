"""
Main script to process faculty data and create visualizations
"""
import os
import sys
from datetime import datetime
from faculty_data_processor import FacultyDataProcessor
from faculty_visualizer import FacultyVisualizer

def main():
    print(f"Starting faculty analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set the data path
    data_path = "data_split_v4/full_dataset.json"
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)
    
    # Process the data
    print(f"Processing data from {data_path}...")
    processor = FacultyDataProcessor(data_path)
    
    if not processor.load_data():
        print("Failed to load data. Exiting.")
        sys.exit(1)
    
    print("Pairing faculty names...")
    processor.pair_faculty_names()
    
    print("Processing faculty counts...")
    processor.process_faculties()
    
    print("Processing faculties by year...")
    processor.process_faculties_by_year()
    
    # Get results
    results = processor.get_results()
    
    # Create visualizations
    print("Creating visualizations...")
    visualizer = FacultyVisualizer(output_dir="images")
    created_files = visualizer.create_all_visualizations(results)
    
    # Print results
    print("\nAnalysis complete!")
    print(f"Found {len(results['faculty_counts'])} unique faculties")
    print(f"Total entries processed: {results['metadata']['total_entries']}")
    print(f"Analysis timestamp: {results['metadata']['analysis_timestamp']}")
    print("\nCreated the following files:")
    for file_path in created_files:
        print(f"- {file_path}")
    
    print(f"\nAll visualizations saved to: {visualizer.output_dir}")

if __name__ == "__main__":
    main()
