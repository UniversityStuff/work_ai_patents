# Embedding + SVM Classification

Patent classification using Google Gemini embeddings and SVM classifier.

## Setup

```bash
pip install -r requirements.txt
```

Ensure your `.env` file in the project root contains:
```
GEMINI_API_KEY=your_api_key_here
```

## Usage

```bash
python main.py
```

## How It Works

1. **Data Loading**: Reads patent CSVs from `Data/CSVBaseData/`
2. **Class Balancing**: Undersamples "Other" category to match textile category sizes
3. **Embeddings**: Uses Gemini `text-embedding-004` model (cached in `cache/`) or bart for patents
4. **Training**: SVM with RBF kernel and balanced class weights
5. **Evaluation**: Train/test split (80/20), outputs accuracy, F1, and confusion matrix

## Configuration

Edit `main.py` to adjust:
- `TEST_SIZE`: Train/test split ratio (default: 0.2)
- `BALANCE_SAMPLE_SIZE`: Override automatic balancing size
- `TUNE_HYPERPARAMETERS`: Enable grid search (slower)
- `EMBEDDER_TYPE`: "gemini" or "bert"
- `FORCE_USE_CACHE`: "True" or "False"

## Output

Results are saved to `Results/`:
- see in console output
- `svm_classifier.pkl`: Trained model
