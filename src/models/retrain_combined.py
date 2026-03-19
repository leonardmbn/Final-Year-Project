import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from scipy.sparse import hstack, csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
    'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
    'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
    'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now',
}

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

sid = SentimentIntensityAnalyzer()


def extract_features(text):
    words = re.findall(r'[a-zA-Z]+', text.lower())
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    words_no_stop = [w for w in words if w not in STOP_WORDS]
    word_count = len(words)
    sentence_count = len(sentences)
    
    superlatives = ['best', 'worst', 'amazing', 'terrible', 'perfect', 'horrible',
                    'excellent', 'awful', 'fantastic', 'disgusting', 'wonderful',
                    'dreadful', 'outstanding', 'pathetic', 'incredible', 'unbelievable',
                    'fabulous', 'atrocious', 'superb', 'abysmal', 'magnificent',
                    'absolutely', 'totally', 'completely', 'utterly', 'extremely',
                    'never', 'always', 'every', 'nothing', 'everything']
    
    sentiment = sid.polarity_scores(text)
    fp_singular = sum(1 for w in words if w in ['i', 'me', 'my', 'mine', 'myself'])
    fp_plural = sum(1 for w in words if w in ['we', 'us', 'our', 'ours', 'ourselves'])
    sup_count = sum(1 for w in words if w in superlatives)
    
    word_freq = {}
    for w in words:
        word_freq[w] = word_freq.get(w, 0) + 1
    
    if word_count == 0:
        return None
    
    return {
        'char_count': len(text),
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_word_length': np.mean([len(w) for w in words]),
        'avg_sentence_length': word_count / sentence_count if sentence_count > 0 else 0,
        'unique_words': len(set(words)),
        'lexical_diversity': len(set(words)) / word_count,
        'first_person_singular': fp_singular,
        'first_person_plural': fp_plural,
        'first_person_total': fp_singular + fp_plural,
        'first_person_ratio': (fp_singular + fp_plural) / word_count,
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'capital_ratio': sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0,
        'superlative_count': sup_count,
        'superlative_ratio': sup_count / word_count,
        'vader_compound': sentiment['compound'],
        'vader_positive': sentiment['pos'],
        'vader_negative': sentiment['neg'],
        'vader_neutral': sentiment['neu'],
        'number_count': len(re.findall(r'\d+', text)),
        'number_ratio': len(re.findall(r'\d+', text)) / word_count,
        'stopword_ratio': (word_count - len(words_no_stop)) / word_count,
        'max_word_freq': max(word_freq.values()),
        'repeated_words_ratio': sum(1 for v in word_freq.values() if v > 1) / len(word_freq),
    }


def build_combined_dataset():
    """Combine Ott et al. with Amazon likely fake/genuine."""
    
    # Load Ott et al.
    print("Loading Ott et al. dataset...")
    ott = pd.read_csv(os.path.join("data", "processed", "ott_reviews_featured.csv"))
    ott_texts = ott['text'].tolist()
    ott_labels = ott['label'].tolist()
    print(f"  Ott: {len(ott)} reviews")
    
    # Load Amazon likely fake
    print("Loading Amazon likely fake reviews...")
    fake = pd.read_csv(os.path.join("data", "processed", "amazon_likely_fake.csv"))
    # Sample 2000 to balance with Ott
    fake_sample = fake.sample(n=min(2000, len(fake)), random_state=42)
    fake_texts = fake_sample['text'].tolist()
    fake_labels = ['fake'] * len(fake_texts)
    print(f"  Amazon fake: {len(fake_texts)} reviews")
    
    # Load Amazon likely genuine
    print("Loading Amazon likely genuine reviews...")
    genuine = pd.read_csv(os.path.join("data", "processed", "amazon_likely_genuine.csv"))
    genuine_sample = genuine.sample(n=min(2000, len(genuine)), random_state=42)
    genuine_texts = genuine_sample['text'].tolist()
    genuine_labels = ['genuine'] * len(genuine_texts)
    print(f"  Amazon genuine: {len(genuine_texts)} reviews")
    
    # Combine
    all_texts = ott_texts + fake_texts + genuine_texts
    all_labels = ott_labels + fake_labels + genuine_labels
    
    print(f"\nTotal combined: {len(all_texts)} reviews")
    print(f"  Fake: {all_labels.count('fake')}")
    print(f"  Genuine: {all_labels.count('genuine')}")
    
    # Extract features for Amazon reviews
    print("\nExtracting features for Amazon reviews...")
    all_features = []
    
    # Ott already has features
    for _, row in ott.iterrows():
        all_features.append({col: row[col] for col in FEATURE_COLS})
    
    # Extract for Amazon
    amazon_texts = fake_texts + genuine_texts
    for idx, text in enumerate(amazon_texts):
        feat = extract_features(str(text))
        if feat is None:
            feat = {col: 0 for col in FEATURE_COLS}
        all_features.append(feat)
        
        if (idx + 1) % 1000 == 0:
            print(f"  {idx + 1}/{len(amazon_texts)} processed...")
    
    return all_texts, all_labels, all_features


def train_combined():
    all_texts, all_labels, all_features = build_combined_dataset()
    
    # Build feature matrices
    print("\nBuilding TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                            stop_words='english', min_df=2, max_df=0.95)
    X_tfidf = tfidf.fit_transform(all_texts)
    
    X_eng = np.array([[f[col] for col in FEATURE_COLS] for f in all_features])
    X_combined = hstack([X_tfidf, csr_matrix(X_eng)])
    
    le = LabelEncoder()
    y = le.fit_transform(all_labels)
    
    print(f"\nFinal dataset: {X_combined.shape[0]} samples, {X_combined.shape[1]} features")
    
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'Naive Bayes': GaussianNB(),
    }
    
    scoring = {'accuracy': 'accuracy', 'precision': 'precision',
               'recall': 'recall', 'f1': 'f1', 'roc_auc': 'roc_auc'}
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {}
    
    print("\n" + "=" * 70)
    print("COMBINED MODEL COMPARISON (5-Fold CV)")
    print("=" * 70)
    
    for name, clf in classifiers.items():
        print(f"\nTraining: {name}...")
        
        if name == 'Naive Bayes':
            X_input = X_combined.toarray()
        else:
            X_input = X_combined
        
        cv_results = cross_validate(clf, X_input, y, cv=cv, scoring=scoring,
                                     return_train_score=False)
        
        results[name] = {
            'accuracy': cv_results['test_accuracy'].mean(),
            'accuracy_std': cv_results['test_accuracy'].std(),
            'f1': cv_results['test_f1'].mean(),
            'f1_std': cv_results['test_f1'].std(),
            'precision': cv_results['test_precision'].mean(),
            'recall': cv_results['test_recall'].mean(),
            'roc_auc': cv_results['test_roc_auc'].mean(),
        }
        
        print(f"  Accuracy:  {results[name]['accuracy']:.4f} (+/- {results[name]['accuracy_std']:.4f})")
        print(f"  F1-Score:  {results[name]['f1']:.4f} (+/- {results[name]['f1_std']:.4f})")
        print(f"  Precision: {results[name]['precision']:.4f}")
        print(f"  Recall:    {results[name]['recall']:.4f}")
        print(f"  ROC-AUC:   {results[name]['roc_auc']:.4f}")
    
    best_name = max(results, key=lambda k: results[k]['f1'])
    print(f"\n{'=' * 70}")
    print(f"BEST COMBINED MODEL: {best_name} (F1: {results[best_name]['f1']:.4f})")
    print(f"{'=' * 70}")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "combined_model_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Saved: results/combined_model_results.json")
    
    # Train best model on full combined data and save
    print(f"\nTraining final {best_name} on full combined data...")
    import pickle
    
    best_clf = classifiers[best_name]
    if best_name == 'Naive Bayes':
        best_clf.fit(X_combined.toarray(), y)
    else:
        best_clf.fit(X_combined, y)
    
    os.makedirs("models", exist_ok=True)
    with open(os.path.join("models", "combined_classifier.pkl"), "wb") as f:
        pickle.dump(best_clf, f)
    with open(os.path.join("models", "combined_tfidf.pkl"), "wb") as f:
        pickle.dump(tfidf, f)
    with open(os.path.join("models", "combined_label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    
    print("Saved combined model to models/")
    print(f"\n{'=' * 70}")
    print("DONE!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    train_combined()