import os
import re
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def simple_tokenize(text):
    """Basic tokenizer - no NLTK needed."""
    return re.findall(r'[a-zA-Z]+', text.lower())


def simple_sent_tokenize(text):
    """Basic sentence splitter - no NLTK needed."""
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]


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
    'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd',
    'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn',
    'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn',
    'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'
}


def extract_features(df):
    """Extract linguistic and statistical features from review text."""
    
    sid = SentimentIntensityAnalyzer()
    
    features = []
    
    for idx, row in df.iterrows():
        text = row['text']
        
        words = simple_tokenize(text)
        sentences = simple_sent_tokenize(text)
        words_no_stop = [w for w in words if w not in STOP_WORDS]
        
        # --- Length features ---
        char_count = len(text)
        word_count = len(words)
        sentence_count = len(sentences)
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # --- Lexical diversity ---
        unique_words = len(set(words))
        lexical_diversity = unique_words / word_count if word_count > 0 else 0
        
        # --- Pronoun features ---
        first_person_singular = sum(1 for w in words if w in ['i', 'me', 'my', 'mine', 'myself'])
        first_person_plural = sum(1 for w in words if w in ['we', 'us', 'our', 'ours', 'ourselves'])
        first_person_total = first_person_singular + first_person_plural
        first_person_ratio = first_person_total / word_count if word_count > 0 else 0
        
        # --- Punctuation features ---
        exclamation_count = text.count('!')
        question_count = text.count('?')
        capital_ratio = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
        
        # --- Superlative/extreme words ---
        superlatives = ['best', 'worst', 'amazing', 'terrible', 'perfect', 'horrible',
                       'excellent', 'awful', 'fantastic', 'disgusting', 'wonderful',
                       'dreadful', 'outstanding', 'pathetic', 'incredible', 'unbelievable',
                       'fabulous', 'atrocious', 'superb', 'abysmal', 'magnificent',
                       'absolutely', 'totally', 'completely', 'utterly', 'extremely',
                       'never', 'always', 'every', 'nothing', 'everything']
        superlative_count = sum(1 for w in words if w in superlatives)
        superlative_ratio = superlative_count / word_count if word_count > 0 else 0
        
        # --- Sentiment (VADER standalone) ---
        sentiment_scores = sid.polarity_scores(text)
        vader_compound = sentiment_scores['compound']
        vader_positive = sentiment_scores['pos']
        vader_negative = sentiment_scores['neg']
        vader_neutral = sentiment_scores['neu']
        
        # --- Detail/specificity indicators ---
        number_count = len(re.findall(r'\d+', text))
        number_ratio = number_count / word_count if word_count > 0 else 0
        
        # --- Stopword ratio ---
        stopword_ratio = (word_count - len(words_no_stop)) / word_count if word_count > 0 else 0
        
        # --- Repetition ---
        word_freq = {}
        for w in words:
            word_freq[w] = word_freq.get(w, 0) + 1
        max_word_freq = max(word_freq.values()) if word_freq else 0
        repeated_words_ratio = sum(1 for v in word_freq.values() if v > 1) / len(word_freq) if word_freq else 0
        
        features.append({
            'char_count': char_count,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_word_length': round(avg_word_length, 4),
            'avg_sentence_length': round(avg_sentence_length, 4),
            'unique_words': unique_words,
            'lexical_diversity': round(lexical_diversity, 4),
            'first_person_singular': first_person_singular,
            'first_person_plural': first_person_plural,
            'first_person_total': first_person_total,
            'first_person_ratio': round(first_person_ratio, 4),
            'exclamation_count': exclamation_count,
            'question_count': question_count,
            'capital_ratio': round(capital_ratio, 4),
            'superlative_count': superlative_count,
            'superlative_ratio': round(superlative_ratio, 4),
            'vader_compound': round(vader_compound, 4),
            'vader_positive': round(vader_positive, 4),
            'vader_negative': round(vader_negative, 4),
            'vader_neutral': round(vader_neutral, 4),
            'number_count': number_count,
            'number_ratio': round(number_ratio, 4),
            'stopword_ratio': round(stopword_ratio, 4),
            'max_word_freq': max_word_freq,
            'repeated_words_ratio': round(repeated_words_ratio, 4),
        })
        
        if (idx + 1) % 200 == 0:
            print(f"  Processed {idx + 1}/{len(df)} reviews...")
    
    features_df = pd.DataFrame(features)
    return features_df


if __name__ == "__main__":
    print("Loading dataset...")
    df = pd.read_csv(os.path.join("data", "processed", "ott_reviews.csv"))
    
    print(f"Extracting features from {len(df)} reviews...")
    features_df = extract_features(df)
    
    df_combined = pd.concat([df, features_df], axis=1)
    
    output_path = os.path.join("data", "processed", "ott_reviews_featured.csv")
    df_combined.to_csv(output_path, index=False)
    
    print(f"\nFeature extraction complete!")
    print(f"Total features extracted: {len(features_df.columns)}")
    print(f"Features: {list(features_df.columns)}")
    
    print("\n--- Feature Statistics (Fake vs Genuine) ---")
    for col in features_df.columns:
        fake_mean = df_combined[df_combined['label'] == 'fake'][col].mean()
        genuine_mean = df_combined[df_combined['label'] == 'genuine'][col].mean()
        print(f"{col:30s} | Fake: {fake_mean:8.4f} | Genuine: {genuine_mean:8.4f}")
