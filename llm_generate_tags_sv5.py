import json
import time
import os
import signal
import sys
import logging
from datetime import datetime
from openai import OpenAI
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tag_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("tag_generator")

class LlamaTagGenerator:
    def __init__(
        self,
        dataset_path: str,
        output_path: str,
        llm_url: str = "http://localhost:8080/v1",
        max_title_length: int = 500,
        max_abstract_length: int = 1500,
        batch_save_interval: int = 10
    ):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.llm_url = llm_url
        self.max_title_length = max_title_length
        self.max_abstract_length = max_abstract_length
        self.batch_save_interval = batch_save_interval
        self.client = None
        self.processed_count = 0
        self.results = []
        self.start_time = None
        
        # Metadata for the output file
        self.metadata = {
            "source_dataset": dataset_path,
            "model": "Meta-Llama-3.1-8B-Instruct.Q8_0",
            "generation_started": datetime.now().isoformat(),
            "generation_completed": None,
            "total_items": 0,
            "successful_items": 0,
            "failed_items": 0,
            "average_processing_time": 0,
            "max_processing_time": 0,
            "min_processing_time": float('inf'),
            "prompt_settings": {
                "max_title_length": max_title_length,
                "max_abstract_length": max_abstract_length
            }
        }
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)
    
    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals by saving current progress."""
        logger.warning(f"Received signal {signum}. Saving progress before shutdown...")
        self.save_results(final=True)
        sys.exit(0)
    
    def initialize_client(self):
        """Initialize the OpenAI client for llamafile."""
        try:
            self.client = OpenAI(
                base_url=self.llm_url,
                api_key="sk-no-key-required"
            )
            # Test the connection
            response = self.client.chat.completions.create(
                model="",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            logger.info("LLM client initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {str(e)}")
            return False
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load the dataset from file."""
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Dataset loaded successfully with {len(data)} entries")
            self.metadata["total_items"] = len(data)
            return data
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            raise
    
    def truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to the specified maximum length."""
        if isinstance(text, list) and len(text) > 0:
            # If text is a list, use the first item
            text = text[0]
        
        if not isinstance(text, str):
            return ""
        
        return text[:max_length] if len(text) > max_length else text
    
    def generate_tags(self, item: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """Generate tags for a single dataset item."""
        start_time = time.time()
        
        # Extract and prepare fields for the prompt
        thesis_title = self.truncate_text(item.get("thesis_title", [""])[0] if item.get("thesis_title") else "", self.max_title_length)
        abstract = self.truncate_text(item.get("abstract", [""])[0] if item.get("abstract") else "", self.max_abstract_length)
        language = self.truncate_text(item.get("language", [""])[0] if item.get("language") else "", 10)
        original_subject_tags = item.get("subject_tags", [])
        original_additional_tags = item.get("additional_tags", [])
        tag_count = len(original_subject_tags) + len(original_additional_tags)
        
        # Prepare the prompt

        prompt = ""
        
        # Prepare the prompt
        if language.lower().startswith("fi"):  # Check if language is Finnish
            prompt = f"""Luo {tag_count} avainsanaa seuraavan opinnäytetyön perusteella:
        Opinnäytetyön nimi: {thesis_title}
        Tiivistelmä: {abstract}
        Kieli: {language}
        Avainsanat voivat olla yleisiä akateemisia tai tutkimusaiheita tai tarkempia kuvailevia termejä.
        Vastaa seuraavassa JSON-muodossa:
        {{
            "tags": ["tagi1", "tagi2", ...]
        }}
        """
        else:  # Default to English
            prompt = f"""Generate {tag_count} tags based on the following thesis:
        Thesis Title: {thesis_title}
        Abstract: {abstract}
        Language: {language}
        Tags should include broad academic/research categories and specific descriptive terms.
        Format your response as a JSON object:
        {{
            "tags": ["tag1", "tag2", ...]
        }}
        """


        
        try:
            response = self.client.chat.completions.create(
                model="",
                messages=[
                    {"role": "system", "content": "You are an academic tagging assistant. Your task is to analyze academic texts and generate appropriate subject and additional tags."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            response_text = response.choices[0].message.content
            
            # Try to extract JSON from the response
            try:
                # Find JSON object in the response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    generated_tags = json.loads(json_str)
                else:
                    generated_tags = {"tags": []}
                    logger.warning(f"No valid JSON found in response: {response_text}")
            except json.JSONDecodeError:
                generated_tags = {"tags": []}
                logger.warning(f"Failed to parse JSON from response: {response_text}")
            
            processing_time = time.time() - start_time
            
            # Update metadata statistics
            self.metadata["successful_items"] += 1
            self.metadata["average_processing_time"] = ((self.metadata["average_processing_time"] * 
                                                       (self.metadata["successful_items"] - 1)) + 
                                                       processing_time) / self.metadata["successful_items"]
            self.metadata["max_processing_time"] = max(self.metadata["max_processing_time"], processing_time)
            self.metadata["min_processing_time"] = min(self.metadata["min_processing_time"], processing_time)
            
            return {
                "identifier": item.get("identifier", ""),
                "original_identifier": item.get("original_identifier", ""),
                "thesis_title": item.get("thesis_title", []),
                "original_subject_tags": original_subject_tags,
                "original_additional_tags": original_additional_tags,
                "generated_tags": generated_tags.get("tags", []),
                "processing_time_seconds": processing_time,
                "timestamp": datetime.now().isoformat()
            }, processing_time
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error generating tags for item {item.get('identifier', '')}: {str(e)}")
            self.metadata["failed_items"] += 1
            
            return {
                "identifier": item.get("identifier", ""),
                "original_identifier": item.get("original_identifier", ""),
                "error": str(e),
                "processing_time_seconds": processing_time,
                "timestamp": datetime.now().isoformat()
            }, processing_time
    
    def save_results(self, final: bool = False):
        """Save current results to the output file."""
        if not self.results:
            return
        
        if final:
            self.metadata["generation_completed"] = datetime.now().isoformat()
        
        output_data = {
            "metadata": self.metadata,
            "results": self.results
        }
        
        temp_output_path = f"{self.output_path}.temp"
        try:
            with open(temp_output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            # Rename the temp file to the final file to ensure atomic write
            os.replace(temp_output_path, self.output_path)
            logger.info(f"Results saved to {self.output_path} ({len(self.results)} entries)")
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
    
    def load_existing_results(self) -> bool:
        """Load existing results file if it exists to continue processing."""
        if not os.path.exists(self.output_path):
            return False
        
        try:
            with open(self.output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.metadata = data.get("metadata", self.metadata)
            self.results = data.get("results", [])
            self.processed_count = len(self.results)
            
            logger.info(f"Loaded {len(self.results)} existing results. Continuing from item {self.processed_count + 1}")
            return True
        except Exception as e:
            logger.error(f"Failed to load existing results: {str(e)}")
            return False
    
    def get_already_processed_ids(self) -> set:
        """Get set of already processed item identifiers."""
        return {item.get("identifier") for item in self.results if "identifier" in item}
    
    def run(self):
        """Run the tag generation process."""
        if not self.initialize_client():
            logger.error("Failed to initialize LLM client. Exiting.")
            return
        
        # Load existing results if available
        self.load_existing_results()
        processed_ids = self.get_already_processed_ids()
        
        # Load the dataset
        dataset = self.load_dataset()
        
        # Set start time
        self.start_time = time.time()
        
        # Process each item
        total_items = len(dataset)
        
        for i, item in enumerate(dataset):
            item_id = item.get("identifier", "")
            
            # Skip already processed items
            if item_id in processed_ids:
                logger.info(f"Skipping already processed item {i+1}/{total_items}: {item_id}")
                continue
            
            logger.info(f"Processing item {i+1}/{total_items}: {item_id}")
            
            result, processing_time = self.generate_tags(item)
            self.results.append(result)
            self.processed_count += 1
            
            # Log progress
            logger.info(f"Item {i+1}/{total_items} processed in {processing_time:.2f} seconds")
            
            # Save results at intervals
            if self.processed_count % self.batch_save_interval == 0:
                self.save_results()
            
            # Add small delay to prevent overwhelming the LLM
            time.sleep(0.1)
        
        # Save final results
        self.save_results(final=True)
        
        # Log completion
        total_time = time.time() - self.start_time
        logger.info(f"Processing completed. Total time: {total_time:.2f} seconds")
        logger.info(f"Processed {self.processed_count} items")
        logger.info(f"Results saved to {self.output_path}")


def run_tag_generation():
    """Main function to run the tag generation."""
    dataset_path = "data_split_v4/full_dataset_v3_test.json"
    output_path = "generated_tags_results.json"
    
    # Check if the dataset exists
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found at {dataset_path}")
        return
    
    # Start the tag generation
    generator = LlamaTagGenerator(
        dataset_path=dataset_path,
        output_path=output_path,
        batch_save_interval=10
    )
    
    generator.run()

if __name__ == "__main__":
    run_tag_generation()