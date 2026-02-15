import os
import csv
import json
import concurrent.futures
import threading
import pandas as pd
from pathlib import Path
import classifier
from google import genai
from tqdm import tqdm

# Configuration
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "../Data/CSVBaseData"
PROMPT_FILE = BASE_DIR / "prompts.csv"
OUTPUT_DIR = BASE_DIR / "Results"
AGGREGATED_FILE = OUTPUT_DIR / "aggregated_results.csv"
MODEL_NAME = "gemini-2.5-flash" # Way worse
# MODEL_NAME = "gemini-2.5-pro" # Decent, but limited to 1k per day
# MODEL_NAME = "gemini-3-pro-preview" # Currently not working
# MODEL_NAME = "gemini-3-flash" # Probably worse than 2.5 pro
MAX_WORKERS = 10
SAMPLE_SIZE = None  # Set to None to process all rows

# Classification Categories
CATEGORIES = [
    "Metal Textiles",
    "Natural Textiles", 
    "Synthetic Textiles",
    "Mineral Textiles",
    "Other"
]

# CSV file mapping to categories
CSV_FILES = {
    "Set-Metal-Textiles.csv": "Metal Textiles",
    "Set-Natural-Textiles.csv": "Natural Textiles",
    "Set-Synthetic-Textiles.csv": "Synthetic Textiles",
    "Set-Mineral-Textile-Without-Metal.csv": "Mineral Textiles",
    "Anti-Seed-3k.csv": "Other"
}

# Lock for writing to the single CSV file safely
write_lock = threading.Lock()



def get_latest_prompt(prompt_file):
    try:
        df = pd.read_csv(prompt_file)
        return df.iloc[-1]['prompt_template']
    except Exception as e:
        print(f"Error reading prompt file: {e}")
        return None

def extract_top_two(scores_dict):
    """
    Extracts the top 2 categories and their scores from a scores dictionary.
    Returns: predicted_one, confidence_one, predicted_two, confidence_two
    """
    if not scores_dict:
        return None, 0.0, None, 0.0
    
    # Sort items by score descending
    sorted_scores = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)
    
    predicted_one = sorted_scores[0][0] if len(sorted_scores) > 0 else None
    confidence_one = sorted_scores[0][1] if len(sorted_scores) > 0 else 0.0
    
    predicted_two = sorted_scores[1][0] if len(sorted_scores) > 1 else None
    confidence_two = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
    
    return predicted_one, confidence_one, predicted_two, confidence_two

def process_row(row, prompt_template, categories, model_name, true_category):
    try:
        title = row.get('Title', '') or ''
        abstract = row.get('Abstract', '') or ''
        # Convert to string to handle NaN values (which are floats in pandas)
        title = str(title) if not pd.isna(title) else ''
        abstract = str(abstract) if not pd.isna(abstract) else ''
        text = "Title: " + title + "\n" + "Abstract: " + abstract
        
        categories_str = ", ".join(categories)
        prompt = prompt_template.format(categories=categories_str, text=text)
        
        response_text = classifier.call_llm(prompt, model_name)
        
        # Clean potential markdown code blocks
        clean_response = response_text.replace("```json", "").replace("```", "").strip()
        
        result_json = json.loads(clean_response)
        scores = result_json.get('scores', {})
        
        pred_one, conf_one, pred_two, conf_two = extract_top_two(scores)
        
        return {
            'LensID': row.get('Lens ID', ''),
            'Title': row.get('Title', ''),
            'true_category': true_category,
            'predicted_one': pred_one,
            'confidence_one': conf_one,
            'predicted_two': pred_two,
            'confidence_two': conf_two
        }

    except Exception as e:
        print(f"Error processing row: {e}")
        return {
            'LensID': row.get('Lens ID', ''),
            'Title': row.get('Title', ''),
            'true_category': true_category,
            'predicted_one': 'ERROR',
            'confidence_one': 0.0,
            'predicted_two': 'ERROR',
            'confidence_two': 0.0
        }

def process_file_content(file_path, prompt_template, categories, writer):
    try:
        print(f"Processing file: {file_path.name}")
        
        # Get true_category from CSV_FILES mapping
        true_category = CSV_FILES.get(file_path.name, file_path.stem)
        
        df = pd.read_csv(file_path)
        
        if SAMPLE_SIZE:
             df = df.head(SAMPLE_SIZE)

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_row = {
                executor.submit(process_row, row.to_dict(), prompt_template, categories, MODEL_NAME, true_category): row 
                for _, row in df.iterrows()
            }
            
            for future in tqdm(concurrent.futures.as_completed(future_to_row), total=len(future_to_row), desc=f"Processing {file_path.name}"):
                try:
                    result = future.result()
                    with write_lock:
                        writer.writerow(result)
                        # Ensure flush happens eventually, but locking every row write is safe enough for CSV
                except Exception as e:
                    print(f"Error getting result: {e}")

        print(f"Finished {file_path.name}")

    except Exception as e:
        print(f"Error processing file {file_path.name}: {e}")

def main():
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)

    print(f"Categories: {CATEGORIES}")

    prompt_template = get_latest_prompt(PROMPT_FILE)
    if not prompt_template:
        print("Could not load prompt.")
        return

    # Prepare detailed output file
    target_columns = ['LensID', 'Title', 'true_category', 'predicted_one', 'confidence_one', 'predicted_two', 'confidence_two']
    
    # Open the single aggregated file
    # We use 'w' to overwrite clean on each run, or 'a' if we wanted to append. 
    # Usually for a fresh run we want to overwrite.
    with open(AGGREGATED_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=target_columns)
        writer.writeheader()
        
        for csv_file in CSV_FILES.keys():
            file_path = INPUT_DIR / csv_file
            if file_path.exists():
                process_file_content(file_path, prompt_template, CATEGORIES, writer)
            else:
                print(f"Warning: File not found: {file_path}")
            f.flush() # Flush after each file to be safe

if __name__ == "__main__":
    main()
