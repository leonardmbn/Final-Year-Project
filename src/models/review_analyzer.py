import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack, csr_matrix
import re

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.models.sentiment_analyzer import SentimentAnalyzer
from src.models.linguistic_detector import LinguisticPatternDetector


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


class ReviewAnalyzer:
    """Main pipeline combining all NLP components."""
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.linguistic_detector = LinguisticPatternDetector()
        self.classifier = None
        self.tfidf = None
        self.label_encoder = None
        self.is_trained = False
    
    def extract_features_single(self, text):
        """Extract 25 engineered features from a single review."""
        words = re.findall(r'[a-zA-Z]+', text.lower())
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        words_no_stop = [w for w in words if w not in STOP_WORDS]
        
        word_count = len(words)
        sentence_count = len(sentences)
        
        superlatives = [
            'best', 'worst', 'amazing', 'terrible', 'perfect', 'horrible',
            'excellent', 'awful', 'fantastic', 'disgusting', 'wonderful',
            'dreadful', 'outstanding', 'pathetic', 'incredible', 'unbelievable',
            'fabulous', 'atrocious', 'superb', 'abysmal', 'magnificent',
            'absolutely', 'totally', 'completely', 'utterly', 'extremely',
            'never', 'always', 'every', 'nothing', 'everything',
        ]
        
        sentiment = self.sentiment_analyzer.analyze(text)
        
        fp_singular = sum(1 for w in words if w in ['i', 'me', 'my', 'mine', 'myself'])
        fp_plural = sum(1 for w in words if w in ['we', 'us', 'our', 'ours', 'ourselves'])
        sup_count = sum(1 for w in words if w in superlatives)
        
        word_freq = {}
        for w in words:
            word_freq[w] = word_freq.get(w, 0) + 1
        
        features = {
            'char_count': len(text),
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'avg_sentence_length': word_count / sentence_count if sentence_count > 0 else 0,
            'unique_words': len(set(words)),
            'lexical_diversity': len(set(words)) / word_count if word_count > 0 else 0,
            'first_person_singular': fp_singular,
            'first_person_plural': fp_plural,
            'first_person_total': fp_singular + fp_plural,
            'first_person_ratio': (fp_singular + fp_plural) / word_count if word_count > 0 else 0,
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'capital_ratio': sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0,
            'superlative_count': sup_count,
            'superlative_ratio': sup_count / word_count if word_count > 0 else 0,
            'vader_compound': sentiment['compound'],
            'vader_positive': sentiment['positive'],
            'vader_negative': sentiment['negative'],
            'vader_neutral': sentiment['neutral'],
            'number_count': len(re.findall(r'\d+', text)),
            'number_ratio': len(re.findall(r'\d+', text)) / word_count if word_count > 0 else 0,
            'stopword_ratio': (word_count - len(words_no_stop)) / word_count if word_count > 0 else 0,
            'max_word_freq': max(word_freq.values()) if word_freq else 0,
            'repeated_words_ratio': sum(1 for v in word_freq.values() if v > 1) / len(word_freq) if word_freq else 0,
        }
        
        return features
    
    def train(self):
        """Train the classifier on the Ott et al. dataset."""
        print("Loading training data...")
        df = pd.read_csv(os.path.join("data", "processed", "ott_reviews_featured.csv"))
        
        X_eng = df[FEATURE_COLS].values
        
        print("Building TF-IDF features...")
        self.tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                                     stop_words='english', min_df=2, max_df=0.95)
        X_tfidf = self.tfidf.fit_transform(df['text'])
        X_combined = hstack([X_tfidf, csr_matrix(X_eng)])
        
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df['label'])
        
        print("Training Random Forest classifier...")
        self.classifier = RandomForestClassifier(n_estimators=200, random_state=42)
        self.classifier.fit(X_combined, y)
        
        self.is_trained = True
        print(f"Model trained on {len(df)} reviews. Ready for analysis.")
    
    def save_model(self, path="models"):
        """Save trained model to disk."""
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "classifier.pkl"), "wb") as f:
            pickle.dump(self.classifier, f)
        with open(os.path.join(path, "tfidf.pkl"), "wb") as f:
            pickle.dump(self.tfidf, f)
        with open(os.path.join(path, "label_encoder.pkl"), "wb") as f:
            pickle.dump(self.label_encoder, f)
        print(f"Model saved to {path}/")
    
    def load_model(self, path="models"):
        """Load trained model from disk."""
        with open(os.path.join(path, "classifier.pkl"), "rb") as f:
            self.classifier = pickle.load(f)
        with open(os.path.join(path, "tfidf.pkl"), "rb") as f:
            self.tfidf = pickle.load(f)
        with open(os.path.join(path, "label_encoder.pkl"), "rb") as f:
            self.label_encoder = pickle.load(f)
        self.is_trained = True
        print("Model loaded successfully.")
    
    def analyze(self, text, rating=None):
        """Full analysis pipeline for a single review."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() or load_model() first.")
        
        # 1. Authenticity classification
        features = self.extract_features_single(text)
        X_eng = np.array([[features[col] for col in FEATURE_COLS]])
        X_tfidf = self.tfidf.transform([text])
        X_combined = hstack([X_tfidf, csr_matrix(X_eng)])
        
        proba = self.classifier.predict_proba(X_combined)[0]
        class_labels = self.label_encoder.classes_
        fake_idx = list(class_labels).index('fake')
        genuine_idx = list(class_labels).index('genuine')
        
        fake_prob = proba[fake_idx]
        genuine_prob = proba[genuine_idx]
        authenticity_score = round(genuine_prob * 100, 1)
        
        if authenticity_score >= 70:
            verdict = "LIKELY GENUINE"
        elif authenticity_score >= 40:
            verdict = "UNCERTAIN"
        else:
            verdict = "LIKELY FAKE"
        
        # 2. Sentiment analysis
        sentiment = self.sentiment_analyzer.analyze(text)
        
        # 3. Rating consistency
        consistency = None
        if rating is not None:
            consistency = self.sentiment_analyzer.check_rating_consistency(text, rating)
        
        # 4. Linguistic pattern detection
        patterns = self.linguistic_detector.detect(text)
        
        return {
            'authenticity_score': authenticity_score,
            'verdict': verdict,
            'fake_probability': round(fake_prob * 100, 1),
            'genuine_probability': round(genuine_prob * 100, 1),
            'sentiment': sentiment,
            'consistency': consistency,
            'linguistic_patterns': patterns,
        }
    
    def format_full_report(self, text, rating=None):
        """Generate a complete formatted report."""
        result = self.analyze(text, rating)
        
        lines = []
        lines.append("=" * 50)
        lines.append("   REVIEW AUTHENTICITY ANALYSIS REPORT")
        lines.append("=" * 50)
        
        # Verdict
        lines.append("")
        lines.append(f"VERDICT: {result['verdict']}")
        lines.append(f"Authenticity Score: {result['authenticity_score']}%")
        lines.append(f"  Genuine: {result['genuine_probability']}%  |  Fake: {result['fake_probability']}%")
        
        # Sentiment
        s = result['sentiment']
        lines.append("")
        lines.append(f"SENTIMENT: {s['label'].upper()} ({s['intensity']})")
        lines.append(f"  Compound: {s['compound']}  |  Pos: {s['positive']}  |  Neg: {s['negative']}  |  Neu: {s['neutral']}")
        
        # Rating consistency
        if result['consistency']:
            c = result['consistency']
            lines.append("")
            lines.append(f"RATING CHECK: {c['message']}")
        
        # Linguistic flags
        p = result['linguistic_patterns']
        lines.append("")
        lines.append(f"LINGUISTIC FLAGS: {p['flag_count']} pattern(s) detected (Risk: {p['risk_level']})")
        for flag in p['flags']:
            lines.append(f"  [{flag['severity']}] {flag['pattern']}: {flag['detail']}")
        
        lines.append("")
        lines.append("=" * 50)
        
        return "\n".join(lines)


if __name__ == "__main__":
    analyzer = ReviewAnalyzer()
    analyzer.train()
    analyzer.save_model()
    
    print("\n\n")
    
    tests = [
        {
            'text': "I absolutely loved this product! It is the best thing I have ever bought. Amazing quality, perfect design. I would highly recommend to everyone! My life has changed completely!",
            'rating': 5,
        },
        {
            'text': "Ordered the 64GB model in black on March 3rd. Battery lasts about 6 hours with heavy use. Camera is decent in daylight but struggles in low light. The 6.1 inch screen has good colour accuracy. Paid $749 which feels fair. One issue: the charging port is slightly loose.",
            'rating': 4,
        },
        {
            'text': "worst product ever do not buy total waste of money i want refund terrible quality broke after one day",
            'rating': 5,
        },
    ]
    
    for i, test in enumerate(tests):
        print(f"\nTEST {i+1}:")
        print(f"Review: {test['text'][:80]}...")
        print(f"Rating: {test['rating']} stars")
        print()
        print(analyzer.format_full_report(test['text'], test['rating']))