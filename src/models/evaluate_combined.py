import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
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
    if word_count == 0:
        return {col: 0 for col in FEATURE_COLS}
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
    return {
        'char_count': len(text), 'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_word_length': np.mean([len(w) for w in words]),
        'avg_sentence_length': word_count / sentence_count if sentence_count > 0 else 0,
        'unique_words': len(set(words)),
        'lexical_diversity': len(set(words)) / word_count,
        'first_person_singular': fp_singular, 'first_person_plural': fp_plural,
        'first_person_total': fp_singular + fp_plural,
        'first_person_ratio': (fp_singular + fp_plural) / word_count,
        'exclamation_count': text.count('!'), 'question_count': text.count('?'),
        'capital_ratio': sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0,
        'superlative_count': sup_count,
        'superlative_ratio': sup_count / word_count,
        'vader_compound': sentiment['compound'], 'vader_positive': sentiment['pos'],
        'vader_negative': sentiment['neg'], 'vader_neutral': sentiment['neu'],
        'number_count': len(re.findall(r'\d+', text)),
        'number_ratio': len(re.findall(r'\d+', text)) / word_count,
        'stopword_ratio': (word_count - len(words_no_stop)) / word_count,
        'max_word_freq': max(word_freq.values()),
        'repeated_words_ratio': sum(1 for v in word_freq.values() if v > 1) / len(word_freq),
    }


def evaluate():
    print("Loading Ott et al. dataset...")
    ott = pd.read_csv(os.path.join("data", "processed", "ott_reviews_featured.csv"))

    print("Loading Amazon likely fake...")
    fake = pd.read_csv(os.path.join("data", "processed", "amazon_likely_fake.csv"))
    fake_sample = fake.sample(n=min(2000, len(fake)), random_state=42)

    print("Loading Amazon likely genuine...")
    genuine = pd.read_csv(os.path.join("data", "processed", "amazon_likely_genuine.csv"))
    genuine_sample = genuine.sample(n=min(2000, len(genuine)), random_state=42)

    all_texts = ott['text'].tolist() + fake_sample['text'].tolist() + genuine_sample['text'].tolist()
    all_labels = ott['label'].tolist() + ['fake'] * len(fake_sample) + ['genuine'] * len(genuine_sample)

    print(f"Total: {len(all_texts)} reviews ({all_labels.count('fake')} fake, {all_labels.count('genuine')} genuine)")

    all_features = []
    for _, row in ott.iterrows():
        all_features.append({col: row[col] for col in FEATURE_COLS})

    print("Extracting features for Amazon reviews...")
    amazon_texts = fake_sample['text'].tolist() + genuine_sample['text'].tolist()
    for idx, text in enumerate(amazon_texts):
        all_features.append(extract_features(str(text)))
        if (idx + 1) % 1000 == 0:
            print(f"  {idx + 1}/{len(amazon_texts)}...")

    print("\nBuilding TF-IDF...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                            stop_words='english', min_df=2, max_df=0.95)
    X_tfidf = tfidf.fit_transform(all_texts)
    X_eng = np.array([[f[col] for col in FEATURE_COLS] for f in all_features])
    X = hstack([X_tfidf, csr_matrix(X_eng)])

    le = LabelEncoder()
    y = le.fit_transform(all_labels)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\nGenerating cross-validated predictions...")
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    y_pred = cross_val_predict(clf, X, y, cv=cv)
    y_prob = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')[:, 1]

    print("\nClassification Report (Combined Model):")
    print(classification_report(y, y_pred, target_names=le.classes_))

    os.makedirs("results", exist_ok=True)

    # 1. Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax,
                annot_kws={'size': 16})
    ax.set_xlabel('Predicted', fontsize=13)
    ax.set_ylabel('Actual', fontsize=13)
    ax.set_title('Confusion Matrix - Combined Model (Random Forest)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join("results", "combined_confusion_matrix.png"), dpi=150)
    plt.close()
    print("Saved: results/combined_confusion_matrix.png")

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = roc_auc_score(y, y_prob)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'Combined Model (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Baseline (AUC = 0.5)')
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('ROC Curve - Combined Model', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join("results", "combined_roc_curve.png"), dpi=150)
    plt.close()
    print("Saved: results/combined_roc_curve.png")

    # 3. Feature Importance (engineered features)
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_eng, y)
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(12, 8))
    top_n = 20
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))[::-1]
    ax.barh(range(top_n), importances[sorted_idx[:top_n]][::-1],
            color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([FEATURE_COLS[i] for i in sorted_idx[:top_n]][::-1], fontsize=10)
    ax.set_xlabel('Feature Importance', fontsize=13)
    ax.set_title('Top 20 Features - Combined Model (Random Forest)', fontsize=14)
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join("results", "combined_feature_importance.png"), dpi=150)
    plt.close()
    print("Saved: results/combined_feature_importance.png")

    # 4. Ott-only vs Combined comparison
    ott_results = json.load(open(os.path.join("results", "model_results.json")))
    combined_results = json.load(open(os.path.join("results", "combined_model_results.json")))
    metrics = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
    ott_scores = [ott_results['Random Forest'][m] for m in metrics]
    combined_scores = [combined_results['Random Forest'][m] for m in metrics]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35
    bars1 = ax.bar(x - width/2, ott_scores, width, label='Ott et al. Only', color='#2196F3', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, combined_scores, width, label='Combined (Ott + Amazon)', color='#FF9800', edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Score', fontsize=13)
    ax.set_title('Model Performance: Ott-Only vs Combined Dataset', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper().replace('_', '-') for m in metrics], fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0.7, 1.0)
    ax.grid(True, axis='y', alpha=0.3)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join("results", "ott_vs_combined_comparison.png"), dpi=150)
    plt.close()
    print("Saved: results/ott_vs_combined_comparison.png")

    print(f"\n{'=' * 60}")
    print("ALL EVALUATION CHARTS GENERATED")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    evaluate()