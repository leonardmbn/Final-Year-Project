import os
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

def extract_quick_features(text):
    """Extract key features from a single review (fast version)."""
    words = re.findall(r'[a-zA-Z]+', text.lower())
    word_count = len(words)
    if word_count == 0:
        return None
    
    fp_words = ['i', 'me', 'my', 'mine', 'myself', 'we', 'our', 'ours']
    superlatives = ['best', 'worst', 'amazing', 'terrible', 'perfect', 'horrible',
                    'excellent', 'awful', 'fantastic', 'disgusting', 'wonderful',
                    'absolutely', 'totally', 'completely', 'never', 'always']
    
    return {
        'word_count': word_count,
        'lexical_diversity': len(set(words)) / word_count,
        'first_person_ratio': sum(1 for w in words if w in fp_words) / word_count,
        'superlative_ratio': sum(1 for w in words if w in superlatives) / word_count,
        'exclamation_count': text.count('!'),
        'number_count': len(re.findall(r'\d+', text)),
    }


def process_amazon_data():
    print("Loading Amazon reviews...")
    df = pd.read_csv(os.path.join("data", "raw", "scraped", "amazon_academic_reviews.csv"))
    print(f"Loaded {len(df)} reviews")
    
    # Sample 50,000 for processing speed
    print("Sampling 50,000 reviews for analysis...")
    df_sample = df.sample(n=50000, random_state=42).reset_index(drop=True)
    
    # Extract features
    print("Extracting features...")
    sid = SentimentIntensityAnalyzer()
    
    features = []
    for idx, row in df_sample.iterrows():
        text = str(row['text'])
        feat = extract_quick_features(text)
        if feat is None:
            continue
        
        sentiment = sid.polarity_scores(text)
        feat['vader_compound'] = sentiment['compound']
        feat['rating'] = row['rating']
        feat['verified_purchase'] = row['verified_purchase']
        feat['category'] = row['category']
        feat['text'] = text
        features.append(feat)
        
        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx + 1}/50000...")
    
    df_feat = pd.DataFrame(features)
    print(f"\nProcessed {len(df_feat)} reviews with features")
    
    # Sentiment-rating inconsistency analysis
    print("\n" + "=" * 60)
    print("SENTIMENT-RATING INCONSISTENCY ANALYSIS")
    print("=" * 60)
    
    def check_inconsistency(row):
        compound = row['vader_compound']
        rating = row['rating']
        if rating >= 4 and compound < -0.25:
            return 'severe_mismatch'
        elif rating <= 2 and compound > 0.25:
            return 'severe_mismatch'
        elif rating >= 4 and compound < 0.05:
            return 'mild_mismatch'
        elif rating <= 2 and compound > -0.05:
            return 'mild_mismatch'
        else:
            return 'consistent'
    
    df_feat['consistency'] = df_feat.apply(check_inconsistency, axis=1)
    print(df_feat['consistency'].value_counts())
    
    # Suspicious review indicators
    print("\n" + "=" * 60)
    print("SUSPICIOUS REVIEW INDICATORS")
    print("=" * 60)
    
    df_feat['suspicious_score'] = 0
    
    # High first-person pronoun usage
    df_feat.loc[df_feat['first_person_ratio'] > 0.08, 'suspicious_score'] += 1
    # High superlative usage
    df_feat.loc[df_feat['superlative_ratio'] > 0.02, 'suspicious_score'] += 1
    # Low lexical diversity
    df_feat.loc[df_feat['lexical_diversity'] < 0.55, 'suspicious_score'] += 1
    # No specific numbers
    df_feat.loc[df_feat['number_count'] == 0, 'suspicious_score'] += 1
    # Excessive exclamation marks
    df_feat.loc[df_feat['exclamation_count'] >= 3, 'suspicious_score'] += 1
    # Sentiment-rating mismatch
    df_feat.loc[df_feat['consistency'] == 'severe_mismatch', 'suspicious_score'] += 2
    # Very short reviews
    df_feat.loc[df_feat['word_count'] < 15, 'suspicious_score'] += 1
    # Unverified purchase
    df_feat.loc[df_feat['verified_purchase'] == False, 'suspicious_score'] += 1
    
    print(f"\nSuspicious Score Distribution:")
    print(df_feat['suspicious_score'].value_counts().sort_index())
    
    # Flag reviews with score >= 4 as likely fake
    likely_fake = df_feat[df_feat['suspicious_score'] >= 4]
    likely_genuine = df_feat[df_feat['suspicious_score'] <= 1]
    
    print(f"\nLikely fake (score >= 4): {len(likely_fake)} ({len(likely_fake)/len(df_feat)*100:.1f}%)")
    print(f"Likely genuine (score <= 1): {len(likely_genuine)} ({len(likely_genuine)/len(df_feat)*100:.1f}%)")
    
    # Compare verified vs unverified
    print("\n" + "=" * 60)
    print("VERIFIED vs UNVERIFIED COMPARISON")
    print("=" * 60)
    
    for col in ['first_person_ratio', 'superlative_ratio', 'lexical_diversity', 'vader_compound']:
        verified_mean = df_feat[df_feat['verified_purchase'] == True][col].mean()
        unverified_mean = df_feat[df_feat['verified_purchase'] == False][col].mean()
        print(f"{col:25s} | Verified: {verified_mean:.4f} | Unverified: {unverified_mean:.4f}")
    
    # Save processed sample
    output = os.path.join("data", "processed", "amazon_analyzed.csv")
    df_feat.to_csv(output, index=False)
    print(f"\nSaved analyzed data to {output}")
    
    # Save likely fake and genuine subsets for potential retraining
    likely_fake[['text', 'rating', 'category']].to_csv(
        os.path.join("data", "processed", "amazon_likely_fake.csv"), index=False)
    likely_genuine[['text', 'rating', 'category']].to_csv(
        os.path.join("data", "processed", "amazon_likely_genuine.csv"), index=False)
    print(f"Saved {len(likely_fake)} likely fake reviews")
    print(f"Saved {len(likely_genuine)} likely genuine reviews")
    
    print(f"\n{'='*60}")
    print("DONE!")
    print(f"{'='*60}")


if __name__ == "__main__":
    process_amazon_data()