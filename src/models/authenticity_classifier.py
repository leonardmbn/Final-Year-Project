import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, make_scorer)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json


# Feature columns (our 25 engineered features)
FEATURE_COLS = [
    'char_count', 'word_count', 'sentence_count', 'avg_word_length',
    'avg_sentence_length', 'unique_words', 'lexical_diversity',
    'first_person_singular', 'first_person_plural', 'first_person_total',
    'first_person_ratio', 'exclamation_count', 'question_count',
    'capital_ratio', 'superlative_count', 'superlative_ratio',
    'vader_compound', 'vader_positive', 'vader_negative', 'vader_neutral',
    'number_count', 'number_ratio', 'stopword_ratio',
    'max_word_freq', 'repeated_words_ratio'
]


def load_data():
    """Load featured dataset with TF-IDF + engineered features."""
    df = pd.read_csv(os.path.join("data", "processed", "ott_reviews_featured.csv"))
    
    # Engineered features
    X_eng = df[FEATURE_COLS].values
    
    # TF-IDF features (unigrams + bigrams)
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                            stop_words='english', min_df=2, max_df=0.95)
    X_tfidf = tfidf.fit_transform(df['text'])
    
    # Combine: TF-IDF (sparse) + engineered features (dense)
    X_combined = hstack([X_tfidf, csr_matrix(X_eng)])
    
    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    
    print(f"Dataset: {X_combined.shape[0]} samples")
    print(f"  TF-IDF features: {X_tfidf.shape[1]}")
    print(f"  Engineered features: {X_eng.shape[1]}")
    print(f"  Total features: {X_combined.shape[1]}")
    print(f"Classes: {dict(zip(le.classes_, np.bincount(y)))}")
    
    return X_combined, y, le, df, tfidf

def train_and_evaluate():
    """Train multiple classifiers and compare with 5-fold CV."""
    
    X, y, le, df, tfidf = load_data()
    
    # Define classifiers (no scaler needed — TF-IDF is already normalised)
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'Naive Bayes': GaussianNB(),
    }
    
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc',
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {}
    
    print("\n" + "=" * 70)
    print("MODEL COMPARISON (5-Fold Stratified Cross-Validation)")
    print("=" * 70)
    
    for name, clf in classifiers.items():
        print(f"\nTraining: {name}...")
        
        # Naive Bayes needs dense arrays
        if name == 'Naive Bayes':
            X_input = X.toarray()
        else:
            X_input = X
        
        cv_results = cross_validate(clf, X_input, y, cv=cv, scoring=scoring,
                                     return_train_score=False)
        
        results[name] = {
            'accuracy': cv_results['test_accuracy'].mean(),
            'accuracy_std': cv_results['test_accuracy'].std(),
            'precision': cv_results['test_precision'].mean(),
            'precision_std': cv_results['test_precision'].std(),
            'recall': cv_results['test_recall'].mean(),
            'recall_std': cv_results['test_recall'].std(),
            'f1': cv_results['test_f1'].mean(),
            'f1_std': cv_results['test_f1'].std(),
            'roc_auc': cv_results['test_roc_auc'].mean(),
            'roc_auc_std': cv_results['test_roc_auc'].std(),
        }
        
        print(f"  Accuracy:  {results[name]['accuracy']:.4f} (+/- {results[name]['accuracy_std']:.4f})")
        print(f"  Precision: {results[name]['precision']:.4f} (+/- {results[name]['precision_std']:.4f})")
        print(f"  Recall:    {results[name]['recall']:.4f} (+/- {results[name]['recall_std']:.4f})")
        print(f"  F1-Score:  {results[name]['f1']:.4f} (+/- {results[name]['f1_std']:.4f})")
        print(f"  ROC-AUC:   {results[name]['roc_auc']:.4f} (+/- {results[name]['roc_auc_std']:.4f})")
    
    # Best model
    best_name = max(results, key=lambda k: results[k]['f1'])
    print(f"\n{'=' * 70}")
    print(f"BEST MODEL: {best_name} (F1: {results[best_name]['f1']:.4f})")
    print(f"{'=' * 70}")
    
    # Confusion matrix and ROC using cross_val_predict
    from sklearn.model_selection import cross_val_predict
    best_clf = classifiers[best_name]
    
    if best_name == 'Naive Bayes':
        X_best = X.toarray()
    else:
        X_best = X
    
    y_pred = cross_val_predict(best_clf, X_best, y, cv=cv)
    y_prob = cross_val_predict(best_clf, X_best, y, cv=cv, method='predict_proba')[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=le.classes_))
    
    # Visualisations
    os.makedirs("results", exist_ok=True)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'Confusion Matrix - {best_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join("results", "confusion_matrix.png"), dpi=150)
    plt.close()
    print("Saved: results/confusion_matrix.png")
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = roc_auc_score(y, y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'{best_name} (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Baseline')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join("results", "roc_curve.png"), dpi=150)
    plt.close()
    print("Saved: results/roc_curve.png")
    
    # 3. Model Comparison
    model_names = list(results.keys())
    f1_scores = [results[m]['f1'] for m in model_names]
    f1_stds = [results[m]['f1_std'] for m in model_names]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2196F3', '#9C27B0', '#F44336']
    bars = ax.bar(model_names, f1_scores, yerr=f1_stds, capsize=5,
                  color=colors[:len(model_names)], edgecolor='black', linewidth=0.5)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('Model Comparison (5-Fold CV F1-Score)', fontsize=14)
    ax.set_ylim(0.5, 1.0)
    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join("results", "model_comparison.png"), dpi=150)
    plt.close()
    print("Saved: results/model_comparison.png")
    
    # 4. Feature Importance (Random Forest on engineered features only)
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    X_eng = df[FEATURE_COLS].values
    rf.fit(X_eng, y)
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    top_n = 15
    ax.barh(range(top_n), importances[sorted_idx[:top_n]][::-1],
            color='#2196F3', edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([FEATURE_COLS[i] for i in sorted_idx[:top_n]][::-1])
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title('Top 15 Features (Random Forest)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join("results", "feature_importance.png"), dpi=150)
    plt.close()
    print("Saved: results/feature_importance.png")
    
    # Save results
    with open(os.path.join("results", "model_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Saved: results/model_results.json")
    
    print(f"\n{'=' * 70}")
    print("ALL DONE!")
    print(f"{'=' * 70}")
    
    return results, best_name
if __name__ == "__main__":
    results, best = train_and_evaluate()