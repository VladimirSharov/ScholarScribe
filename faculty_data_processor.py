"""
Faculty Data Processor Module
This module handles loading and processing faculty data from the JSON dataset.
"""
import json
import os
from datetime import datetime
from collections import Counter, defaultdict
import langdetect

class FacultyDataProcessor:
    def __init__(self, data_path):
        """
        Initialize the processor with the path to the JSON data file
        
        Args:
            data_path: Path to the JSON data file
        """
        self.data_path = data_path
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.data = None
        self.faculty_counts = None
        self.faculty_pairs = None
        self.faculty_by_year = None
        self.metadata = {
            "database_path": data_path,
            "analysis_timestamp": self.timestamp,
            "script": os.path.basename(__file__)
        }
    
    def load_data(self):
        """Load the JSON data from the file"""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            self.metadata["total_entries"] = len(self.data)
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def detect_language(self, text):
        """
        Detect the language of a given text
        
        Args:
            text: The text to detect language for
            
        Returns:
            Language code ('en' for English, 'fi' for Finnish)
        """
        # Simple keyword-based detection for faculty names (more reliable than langdetect for this case)
        finnish_indicators = ['tiedekunta', 'laitos', 'yksikkö', 'Humanistinen', 'Matemaattis', 'Yhteiskuntatieteellinen']
        english_indicators = ['Faculty', 'School', 'of', 'and', 'Sciences', 'Humanities']
        
        # Count indicators
        fi_count = sum(1 for word in finnish_indicators if word.lower() in text.lower())
        en_count = sum(1 for word in english_indicators if word in text)
        
        # Decide based on indicator count
        if fi_count > en_count:
            return 'fi'
        elif en_count > fi_count:
            return 'en'
        else:
            # Fallback to general detection
            try:
                return langdetect.detect(text)
            except:
                # Last resort default
                return 'en' if 'Faculty' in text else 'fi'
    
    def pair_faculty_names(self):
        """
        Pair English and Finnish faculty names
        
        Returns:
            Dictionary of faculty pairs with standardized names
        """
        # First collect all faculty entries
        all_faculties = []
        for item in self.data:
            if "faculty" in item and isinstance(item["faculty"], list):
                all_faculties.extend(item["faculty"])
        
        # Standardize the known faculty structure based on official information
        # Based on Wikipedia and university website data
        standard_faculties = {
            'en': {
                'Faculty of Humanities and Social Sciences': set(),
                'Faculty of Information Technology': set(),
                'Faculty of Education and Psychology': set(),
                'Faculty of Sport and Health Sciences': set(),
                'Faculty of Mathematics and Science': set(),
                'School of Business and Economics': set(),
                # Historical or alternative names
                'Faculty of Humanities': set(),
                'Faculty of Social Sciences': set(),
                'Faculty of Education': set(),
                'Faculty of Sciences': set()
            },
            'fi': {
                'Humanistis-yhteiskuntatieteellinen tiedekunta': set(),
                'Informaatioteknologian tiedekunta': set(),
                'Kasvatustieteiden ja psykologian tiedekunta': set(),
                'Liikuntatieteellinen tiedekunta': set(),
                'Matemaattis-luonnontieteellinen tiedekunta': set(),
                'Jyväskylän yliopiston kauppakorkeakoulu': set(),
                # Historical or alternative names
                'Humanistinen tiedekunta': set(),
                'Yhteiskuntatieteellinen tiedekunta': set(),
                'Kasvatustieteiden tiedekunta': set(),
                'Kauppakorkeakoulu': set()
            }
        }
        
        # Group similar faculties that appear together
        faculty_pairs = defaultdict(set)
        
        for item in self.data:
            if "faculty" in item and isinstance(item["faculty"], list) and len(item["faculty"]) >= 2:
                # Sort by language to make pairs consistent
                en_faculties = []
                fi_faculties = []
                
                for faculty in item["faculty"]:
                    lang = self.detect_language(faculty)
                    if lang == 'en':
                        en_faculties.append(faculty)
                    else:
                        fi_faculties.append(faculty)
                
                # Create pairs from the sorted lists
                for en_faculty in en_faculties:
                    for fi_faculty in fi_faculties:
                        # Add to both directions for better matching
                        faculty_pairs[en_faculty].add(fi_faculty)
                        # Also track English to English (for handling typos/variants)
                        for other_en in en_faculties:
                            if en_faculty != other_en:
                                faculty_pairs[en_faculty].add(other_en)
        
        # Normalize faculty names based on frequency and official names
        # Count occurrences of each faculty name
        faculty_counts = Counter(all_faculties)
        
        # Now create the final faculty pairs
        final_pairs = {}
        
        # First, handle English names
        for standard_en, variants in standard_faculties['en'].items():
            best_match = standard_en
            # Find the best matching official name
            for faculty in faculty_pairs:
                if self.detect_language(faculty) == 'en':
                    # Check if this faculty name is close to our standard name
                    if standard_en.lower() in faculty.lower() or faculty.lower() in standard_en.lower():
                        if faculty_counts.get(faculty, 0) > faculty_counts.get(best_match, 0):
                            best_match = faculty
            
            # Find its Finnish pair
            best_fi_match = None
            highest_count = 0
            
            for fi_candidate in faculty_pairs.get(best_match, set()):
                if self.detect_language(fi_candidate) == 'fi':
                    count = faculty_counts.get(fi_candidate, 0)
                    if count > highest_count:
                        highest_count = count
                        best_fi_match = fi_candidate
            
            if best_fi_match:
                final_pairs[best_match] = best_fi_match
                
        # Handle potential missing pairs by looking at frequency of co-occurrence
        for en_faculty in (f for f in faculty_counts if self.detect_language(f) == 'en'):
            if en_faculty not in final_pairs:
                # Look for its Finnish pair among co-occurrences
                best_fi = None
                highest_score = 0
                
                for item in self.data:
                    if "faculty" in item and isinstance(item["faculty"], list) and en_faculty in item["faculty"]:
                        # Find Finnish faculties in this item
                        for faculty in item["faculty"]:
                            if self.detect_language(faculty) == 'fi':
                                score = faculty_counts.get(faculty, 0)
                                if score > highest_score:
                                    highest_score = score
                                    best_fi = faculty
                
                if best_fi:
                    final_pairs[en_faculty] = best_fi
        
        self.faculty_pairs = final_pairs
        return final_pairs
    
    def process_faculties(self):
        """
        Process faculties from the data and count occurrences
        
        Returns:
            Dictionary with faculty counts
        """
        if self.data is None:
            if not self.load_data():
                return {}
        
        if self.faculty_pairs is None:
            self.pair_faculty_names()
        
        # Define canonical faculty mapping based on university structure
        # This maps various faculty names to their official current name
        canonical_faculties = {
            'en': {
                'Faculty of Humanities': 'Faculty of Humanities and Social Sciences',
                'Faculty of Social Sciences': 'Faculty of Humanities and Social Sciences',
                'Faculty of Sciences': 'Faculty of Mathematics and Science',
                'School of Business and Economics': 'School of Business and Economics',
                'Faculty of Education': 'Faculty of Education and Psychology',
                'Faculty of Education and Psychology': 'Faculty of Education and Psychology',
                'Faculty of Information Technology': 'Faculty of Information Technology',
                'Faculty of Sport and Health Sciences': 'Faculty of Sport and Health Sciences',
                'Faculty of Mathematics and Science': 'Faculty of Mathematics and Science',
                'Faculty of Humanities and Social Sciences': 'Faculty of Humanities and Social Sciences',
            }
        }
        
        # Reverse map from faculty pairs
        canonical_faculties['fi'] = {}
        for en_name, fi_name in self.faculty_pairs.items():
            for canonical_en, mapped_en in canonical_faculties['en'].items():
                if canonical_en.lower() in en_name.lower() or en_name.lower() in canonical_en.lower():
                    canonical_faculties['fi'][fi_name] = mapped_en
                    break
        
        # Collect all unique faculty names to help with fuzzy matching
        all_faculties = []
        for item in self.data:
            if "faculty" in item and isinstance(item["faculty"], list):
                all_faculties.extend(item["faculty"])
        
        # Count the faculties using English names as standard
        faculty_counts = Counter()
        processed_items = 0
        
        for item in self.data:
            if "faculty" not in item or not isinstance(item["faculty"], list):
                continue
                
            processed_items += 1
            
            # Process this item's faculties
            counted_faculties = set()  # To avoid counting duplicates within one item
            
            for faculty in item["faculty"]:
                canonical_name = None
                lang = self.detect_language(faculty)
                
                # Try to find canonical name
                if lang == 'en':
                    # For English names, use canonical mapping
                    for canonical_en, mapped_en in canonical_faculties['en'].items():
                        if canonical_en.lower() in faculty.lower() or faculty.lower() in canonical_en.lower():
                            canonical_name = mapped_en
                            break
                    
                    # If not found, use as is
                    if not canonical_name:
                        canonical_name = faculty
                
                else:  # Finnish
                    # For Finnish names, try to find its English canonical pair
                    english_name = None
                    
                    # First check in our canonical Finnish mapping
                    for canon_fi, mapped_en in canonical_faculties['fi'].items():
                        if canon_fi.lower() in faculty.lower() or faculty.lower() in canon_fi.lower():
                            canonical_name = mapped_en
                            break
                    
                    # If not found, try faculty pairs
                    if not canonical_name:
                        for en, fi in self.faculty_pairs.items():
                            if fi.lower() in faculty.lower() or faculty.lower() in fi.lower():
                                # Found a match, now map to canonical
                                for canonical_en, mapped_en in canonical_faculties['en'].items():
                                    if canonical_en.lower() in en.lower() or en.lower() in canonical_en.lower():
                                        canonical_name = mapped_en
                                        break
                                if not canonical_name:
                                    canonical_name = en
                                break
                    
                    # If still not found, add as unknown
                    if not canonical_name:
                        canonical_name = f"{faculty} (Unknown Faculty)"
                
                # Count each canonical faculty once per item
                if canonical_name and canonical_name not in counted_faculties:
                    faculty_counts[canonical_name] += 1
                    counted_faculties.add(canonical_name)
        
        self.faculty_counts = faculty_counts
        self.metadata["total_faculties_counted"] = sum(faculty_counts.values())
        self.metadata["unique_faculties"] = len(faculty_counts)
        self.metadata["processed_items"] = processed_items
        
        return faculty_counts
    
    def process_faculties_by_year(self):
        """
        Process faculties by year to track distribution over time
        
        Returns:
            Dictionary with faculty counts by year
        """
        if self.data is None:
            if not self.load_data():
                return {}
                
        if self.faculty_pairs is None:
            self.pair_faculty_names()
        
        faculty_by_year = defaultdict(lambda: defaultdict(int))
        years_count = Counter()
        
        for item in self.data:
            if "faculty" not in item or not isinstance(item["faculty"], list):
                continue
                
            # Get the year
            year = None
            if "date_issued" in item and isinstance(item["date_issued"], list) and item["date_issued"]:
                # Extract just the year part (first 4 digits)
                year_str = str(item["date_issued"][0])
                if year_str.isdigit() and len(year_str) >= 4:
                    year = year_str[:4]
            
            if not year:
                continue
                
            years_count[year] += 1
            
            # Process this item's faculties
            en_faculties = set()
            
            for faculty in item["faculty"]:
                lang = self.detect_language(faculty)
                
                if lang == 'en':
                    en_faculties.add(faculty)
                else:
                    # Try to find its English pair
                    english_name = None
                    for en, fi in self.faculty_pairs.items():
                        if faculty == fi:
                            english_name = en
                            break
                    
                    if english_name:
                        en_faculties.add(english_name)
                    else:
                        # If no English pair found, use as is with a note
                        en_faculties.add(f"{faculty} (FI)")
            
            # Count each faculty once per item for this year
            for faculty in en_faculties:
                faculty_by_year[year][faculty] += 1
        
        self.faculty_by_year = dict(faculty_by_year)
        self.metadata["years_analyzed"] = sorted(years_count.keys())
        self.metadata["entries_with_year"] = sum(years_count.values())
        
        return self.faculty_by_year
    
    def get_abbreviations(self):
        """
        Create abbreviations for faculty names to avoid text overlap in visualizations
        
        Returns:
            Dictionary mapping full faculty names to abbreviations
        """
        if not self.faculty_counts:
            self.process_faculties()
            
        # Define standard abbreviations for known faculties
        standard_abbrs = {
            'Faculty of Humanities and Social Sciences': 'FHSS',
            'Faculty of Information Technology': 'FIT',
            'Faculty of Education and Psychology': 'FEP',
            'Faculty of Sport and Health Sciences': 'FSHS',
            'Faculty of Mathematics and Science': 'FMS',
            'School of Business and Economics': 'SBE',
            'Faculty of Humanities': 'FH',
            'Faculty of Social Sciences': 'FSS',
            'Faculty of Education': 'FE',
            'Faculty of Sciences': 'FS'
        }
        
        abbreviations = {}
        
        for faculty in self.faculty_counts.keys():
            # Use standard abbreviation if available
            for std_name, abbr in standard_abbrs.items():
                if std_name.lower() in faculty.lower() or faculty.lower() in std_name.lower():
                    abbreviations[faculty] = abbr
                    break
            
            # If no standard abbreviation found, create one
            if faculty not in abbreviations:
                # Create abbreviation based on first letters of main words
                words = faculty.split()
                if len(words) <= 2:
                    # For short names, use first 3 letters
                    abbr = faculty[:3].upper()
                else:
                    # For longer names, use first letter of each significant word
                    abbr = ''.join(word[0] for word in words if len(word) > 3 or word.isupper())
                    if len(abbr) < 2:
                        # If too short, use first 2 letters of first and last words
                        abbr = words[0][:2] + words[-1][:2]
                
                # Ensure uniqueness
                base_abbr = abbr.upper()
                abbr = base_abbr
                i = 1
                while abbr in abbreviations.values():
                    abbr = f"{base_abbr}{i}"
                    i += 1
                    
                abbreviations[faculty] = abbr
            
        return abbreviations
        
    def get_results(self):
        """
        Get all processed results
        
        Returns:
            Dictionary with all processed data and metadata
        """
        if not self.faculty_counts:
            self.process_faculties()
            
        if not self.faculty_by_year:
            self.process_faculties_by_year()
            
        abbreviations = self.get_abbreviations()
        
        # Calculate average count
        if self.faculty_counts:
            avg_count = sum(self.faculty_counts.values()) / len(self.faculty_counts)
            self.metadata["average_count_per_faculty"] = avg_count
        
        results = {
            "faculty_counts": self.faculty_counts,
            "faculty_pairs": self.faculty_pairs,
            "faculty_by_year": self.faculty_by_year,
            "abbreviations": abbreviations,
            "metadata": self.metadata
        }
        
        return results
