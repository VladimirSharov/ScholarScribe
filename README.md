# Project Title
Description of the project, objective and problems.

## Overview
This project consists of a series of Python scripts designed to fetch, analyze, and process data from a designated website. The scripts handle JSON data extraction, ensure data integrity, and provide insights into the data's field structures and value distributions.

## Getting Started
These instructions will guide you on setting up and running the project on your local machine for development and testing purposes.

### Prerequisites
What you need to install the software:

python -m pip install -r requirements-frozen.txt
Installation
Follow these steps to set up your development environment:

Clone the repository:
git clone https://a.git

Install the required dependencies:
pip install -r requirements-frozen.txt

### Usage
Here’s how to use the scripts included in the project:

#### api_data_collector.py
Description: Initiates data collection from a specified website, organizes the data in JSON format, and handles file outputs.
Outputs:
output.json: Contains parsed entities. Note: [ is added at the beginning and ] at the end for JSON formatting.
state.json: Tracks the state of data fetching to enable continuation from the last fetched point.
Usage:
python api_data_collector.py

#### json_data_integrity_checker.py
Description: Evaluates the integrity of JSON data by counting fields and flagging data or format inconsistencies.
Output: Generates output_FieldStructure.json, summarizing field occurrences and potential issues.
Usage:
python json_data_integrity_checker.py

#### json_value_occurrence_counter.py
Description: Delves deeper into the data structure by counting each unique value occurrence, providing a detailed view of data variations.
Output: Produces output_ValueSortedByField.json, which details occurrences of unique values sorted by field.
Usage:
python json_value_occurrence_counter.py

#### v2_data_preparation.py
Description: Extracts specified fields from output.json, aligns field names according to predefined mappings, and organizes the data into structured formats.
Output: Data is saved in the prepared_datasets directory, structured according to specified field requirements.
Usage:
python v2_data_preparation.py

License
*Some*

### Development Log
See project_journal.txt for detailed logs of development progress, issues, and notes.

### Scheme
project/
│
├── data_collection/
│   ├── api_data_collector.py
│   └── output.json
│
├── data_preparation/
│   ├── v2_data_preparation.py
│   ├── prepared_datasets/
│   ├── field_translation/
│   ├── field_featureExtraction/
│   └── field_labelClassify/  # Optional for tagging or label creation
│
├── data_validation/
│   ├── json_value_occurrence_counter.py
│   └── json_data_integrity_checker.py
│
├── data_split/
│   └── stratify_split.py
│
├── model_embedding/
│   ├── specter2_embedding.py
│   └── embeddings/
│
├── model_training/
│   ├── specter2_model.py
│   └── trained_models/
│
├── model_validation/
│   └── model_evaluation.py
│
├── model_usage/
│   └── predict_tags.py
│
└── model_comparison/
    └── comparison.py