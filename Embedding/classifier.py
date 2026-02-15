import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
from pathlib import Path


def create_svm_classifier(class_weight: str = 'balanced') -> Pipeline:
    """
    Create an SVM classifier pipeline with standardization.
    
    Args:
        class_weight: Class weighting strategy ('balanced' recommended for imbalanced data)
        
    Returns:
        sklearn Pipeline with StandardScaler and SVC
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            kernel='rbf',
            class_weight=class_weight,
            probability=True,  # Enable probability estimates
            random_state=42
        ))
    ])
    return pipeline


def train_classifier(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    tune_hyperparameters: bool = False
) -> Pipeline:
    """
    Train an SVM classifier on the given data.
    
    Args:
        X_train: Training embeddings
        y_train: Training labels
        tune_hyperparameters: Whether to perform grid search for optimal parameters
        
    Returns:
        Trained classifier pipeline
    """
    if tune_hyperparameters:
        print("Performing hyperparameter tuning...")
        pipeline = create_svm_classifier()
        
        param_grid = {
            'svm__C': [0.1, 1.0, 10.0],
            'svm__gamma': ['scale', 'auto', 0.01, 0.1]
        }
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    else:
        print("Training SVM classifier...")
        pipeline = create_svm_classifier()
        pipeline.fit(X_train, y_train)
        return pipeline


def predict(classifier: Pipeline, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Make predictions using the trained classifier.
    
    Args:
        classifier: Trained classifier pipeline
        X: Embeddings to classify
        
    Returns:
        Tuple of (predictions, probabilities)
    """
    predictions = classifier.predict(X)
    probabilities = classifier.predict_proba(X)
    return predictions, probabilities


def get_top_two_predictions(
    classifier: Pipeline, 
    X: np.ndarray
) -> list[dict]:
    """
    Get top 2 predictions with confidence scores for each sample.
    
    Args:
        classifier: Trained classifier pipeline
        X: Embeddings to classify
        
    Returns:
        List of dicts with predicted_one, confidence_one, predicted_two, confidence_two
    """
    probabilities = classifier.predict_proba(X)
    classes = classifier.classes_
    
    results = []
    for probs in probabilities:
        # Get indices of top 2 probabilities
        top_indices = np.argsort(probs)[::-1][:2]
        
        result = {
            'predicted_one': classes[top_indices[0]],
            'confidence_one': probs[top_indices[0]],
            'predicted_two': classes[top_indices[1]] if len(top_indices) > 1 else None,
            'confidence_two': probs[top_indices[1]] if len(top_indices) > 1 else 0.0
        }
        results.append(result)
    
    return results


def save_classifier(classifier: Pipeline, path: Path) -> None:
    """Save trained classifier to disk."""
    with open(path, 'wb') as f:
        pickle.dump(classifier, f)
    print(f"Classifier saved to: {path}")


def load_classifier(path: Path) -> Pipeline:
    """Load trained classifier from disk."""
    with open(path, 'rb') as f:
        return pickle.load(f)
