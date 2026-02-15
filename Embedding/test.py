import pickle
from pathlib import Path
from collections import Counter
import os

# Define the cache directory
CACHE_DIR = Path(__file__).parent / "cache"

def analyze_label_distribution():
    """
    Iterates through all .pkl files in the cache directory and prints the distribution of labels.
    """
    if not CACHE_DIR.exists():
        print(f"Cache directory not found: {CACHE_DIR}")
        return

    print(f"Scanning cache directory: {CACHE_DIR}\n")

    pkl_files = list(CACHE_DIR.glob("*.pkl"))
    
    if not pkl_files:
        print("No .pkl files found in the cache directory.")
        return

    for pkl_file in pkl_files:
        print(f"Processing file: {pkl_file.name}")
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            # Check if data is a dictionary and has 'labels' key
            if isinstance(data, dict) and 'labels' in data:
                labels = data['labels']
                if labels is None:
                     print(f"  - No labels found in this file (labels is None).")
                     print("-" * 40)
                     continue
                
                label_counts = Counter(labels)
                print(f"  - Total samples: {len(labels)}")
                print(f"  - Label Distribution:")
                for label, count in label_counts.most_common():
                    print(f"    - {label}: {count} ({count/len(labels):.2%})")
            else:
                print(f"  - No 'labels' key found in this file or data is not a dictionary.")
                if isinstance(data, dict):
                     print(f"    Available keys: {list(data.keys())}")
                else:
                     print(f"    Data type: {type(data)}")

        except Exception as e:
            print(f"  - Error reading file: {e}")
        
        print("-" * 40)

if __name__ == "__main__":
    analyze_label_distribution()
