import os
import csv
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import embedder
import classifier

# Configuration
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "../Data/CSVBaseData"
OUTPUT_DIR = BASE_DIR / "Results"

# Embedder type: "gemini" or "bert"
EMBEDDER_TYPE = "gemini"

MODEL_PATH = OUTPUT_DIR / f"svm_classifier_{EMBEDDER_TYPE}.pkl"

# Test size for train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Sample size for balancing (None = use calculated average of textile categories)
BALANCE_SAMPLE_SIZE = None

# Hyperparameter tuning (slower but may improve results)
# Hyperparameter tuning (slower but may improve results)
TUNE_HYPERPARAMETERS = False

# Force use of cached embeddings (skip generation even if data changed)
FORCE_USE_CACHE = True

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


def load_data() -> pd.DataFrame:
    """Load all CSV files and combine into a single DataFrame with labels."""
    all_data = []
    
    for csv_file, category in CSV_FILES.items():
        file_path = INPUT_DIR / csv_file
        if file_path.exists():
            df = pd.read_csv(file_path)
            df['category'] = category
            df['source_file'] = csv_file
            all_data.append(df)
            print(f"Loaded {csv_file}: {len(df)} rows -> {category}")
        else:
            print(f"Warning: File not found: {file_path}")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal rows loaded: {len(combined_df)}")
    
    return combined_df


def balance_data(df: pd.DataFrame, target_size: int = None) -> pd.DataFrame:
    """
    Balance the dataset by undersampling the 'Other' category.
    
    Args:
        df: Input DataFrame
        target_size: Target size for 'Other' category. If None, uses average of textile categories.
        
    Returns:
        Balanced DataFrame
    """
    # Calculate sizes
    textile_categories = [c for c in CATEGORIES if c != "Other"]
    textile_sizes = [len(df[df['category'] == c]) for c in textile_categories]
    avg_textile_size = int(np.mean(textile_sizes))
    
    if target_size is None:
        target_size = avg_textile_size
    
    print(f"\nBalancing data:")
    print(f"  Textile category average: {avg_textile_size}")
    print(f"  Target 'Other' size: {target_size}")
    
    # Separate data
    textile_df = df[df['category'] != "Other"]
    other_df = df[df['category'] == "Other"]
    
    # Undersample 'Other'
    other_sampled = other_df.sample(n=min(target_size, len(other_df)), random_state=RANDOM_STATE)
    
    # Combine
    balanced_df = pd.concat([textile_df, other_sampled], ignore_index=True)
    
    print(f"\nBalanced dataset:")
    for cat in CATEGORIES:
        count = len(balanced_df[balanced_df['category'] == cat])
        print(f"  {cat}: {count}")
    print(f"  Total: {len(balanced_df)}")
    
    return balanced_df


def prepare_texts(df: pd.DataFrame) -> list[str]:
    """Prepare text inputs for embedding."""
    texts = []
    for _, row in df.iterrows():
        text = embedder.create_text_from_row(row.to_dict())
        texts.append(text)
    return texts


def main():
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("=" * 60)
    print(f"Embedding ({EMBEDDER_TYPE}) + SVM Classification")
    print("=" * 60)
    
    # Set embedder
    embedder.CURRENT_EMBEDDER = EMBEDDER_TYPE
    
    # Load data
    print("\n1. Loading data...")
    df = load_data()
    
    # Balance data
    print("\n2. Balancing data...")
    df_balanced = balance_data(df, BALANCE_SAMPLE_SIZE)
    
    # Prepare texts
    print("\n3. Preparing texts...")
    texts = prepare_texts(df_balanced)
    labels = df_balanced['category'].values
    lens_ids = df_balanced.get('Lens ID', pd.Series([''] * len(df_balanced))).values
    titles = df_balanced.get('Title', pd.Series([''] * len(df_balanced))).values
    
    # Train/test split
    print("\n4. Splitting data...")
    (X_texts_train, X_texts_test, 
     y_train, y_test,
     ids_train, ids_test,
     titles_train, titles_test) = train_test_split(
        texts, labels, lens_ids, titles,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )
    print(f"  Train size: {len(X_texts_train)}")
    print(f"  Test size: {len(X_texts_test)}")
    
    # Generate embeddings
    print("\n5. Generating embeddings...")
    X_train = embedder.get_embeddings_batch(X_texts_train, labels=y_train, ids=ids_train, cache_name="train_embeddings", force_use_cache=FORCE_USE_CACHE)
    X_test = embedder.get_embeddings_batch(X_texts_test, labels=y_test, ids=ids_test, cache_name="test_embeddings", force_use_cache=FORCE_USE_CACHE)
    
    # Train classifier
    print("\n6. Training classifier...")
    clf = classifier.train_classifier(X_train, y_train, tune_hyperparameters=TUNE_HYPERPARAMETERS)
    
    # Save classifier
    classifier.save_classifier(clf, MODEL_PATH)
    
    # Evaluate
    print("\n7. Evaluating classifier...")
    predictions, probabilities = classifier.predict(clf, X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, predictions, labels=CATEGORIES)
    print(pd.DataFrame(cm, index=CATEGORIES, columns=CATEGORIES))
    

    # Cross-validation on FULL dataset for more robust estimate
    # Just leave it here for reference, but TLDR: seems robust enough

    # print("\n7b. Running 5-fold cross-validation for robust estimate...")
    # X_all = embedder.get_embeddings_batch(texts, labels=labels, ids=lens_ids, cache_name="all_embeddings")
    
    # cv_pipeline = classifier.create_svm_classifier()
    # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    # cv_scores = cross_val_score(cv_pipeline, X_all, labels, cv=cv, scoring='accuracy')
    
    # print(f"\nCross-validation results (5-fold):")
    # print(f"  Scores: {cv_scores}")
    # print(f"  Mean accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # # Get top-2 predictions for test set
    # print("\n8. Generating detailed results...")
    # top_two_results = classifier.get_top_two_predictions(clf, X_test)
    
    # # Save detailed results
    # results_file = OUTPUT_DIR / "aggregated_results.csv"
    # with open(results_file, 'w', newline='', encoding='utf-8') as f:
    #     writer = csv.DictWriter(f, fieldnames=[
    #         'LensID', 'Title', 'true_category', 
    #         'predicted_one', 'confidence_one', 
    #         'predicted_two', 'confidence_two'
    #     ])
    #     writer.writeheader()
        
    #     for i, result in enumerate(top_two_results):
    #         writer.writerow({
    #             'LensID': ids_test[i],
    #             'Title': titles_test[i],
    #             'true_category': y_test[i],
    #             'predicted_one': result['predicted_one'],
    #             'confidence_one': result['confidence_one'],
    #             'predicted_two': result['predicted_two'],
    #             'confidence_two': result['confidence_two']
    #         })
    
    # print(f"\nResults saved to: {results_file}")
    # print("\nDone!")


if __name__ == "__main__":
    main()
