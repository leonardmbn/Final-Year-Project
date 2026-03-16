import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentAnalyzer:
    """Sentiment analysis module using VADER."""
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze(self, text):
        """Analyze sentiment of a single review text."""
        scores = self.analyzer.polarity_scores(text)
        
        # Determine overall sentiment label
        compound = scores['compound']
        if compound >= 0.05:
            label = 'positive'
        elif compound <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'
        
        # Determine intensity
        abs_compound = abs(compound)
        if abs_compound >= 0.75:
            intensity = 'strong'
        elif abs_compound >= 0.5:
            intensity = 'moderate'
        elif abs_compound >= 0.25:
            intensity = 'mild'
        else:
            intensity = 'weak'
        
        return {
            'compound': round(compound, 4),
            'positive': round(scores['pos'], 4),
            'negative': round(scores['neg'], 4),
            'neutral': round(scores['neu'], 4),
            'label': label,
            'intensity': intensity,
        }
    
    def check_rating_consistency(self, text, rating):
        """Check if sentiment matches the star rating."""
        sentiment = self.analyze(text)
        compound = sentiment['compound']
        
        # Expected sentiment ranges for each star rating
        expected = {
            1: (-1.0, -0.25),
            2: (-0.6, 0.0),
            3: (-0.25, 0.25),
            4: (0.0, 0.6),
            5: (0.25, 1.0),
        }
        
        if rating not in expected:
            return {
                'consistent': None,
                'message': 'Invalid rating provided.',
                'sentiment': sentiment,
            }
        
        low, high = expected[rating]
        is_consistent = low <= compound <= high
        
        # Calculate mismatch severity
        if is_consistent:
            mismatch_score = 0.0
            message = f"Sentiment is consistent with {rating}-star rating."
        else:
            # How far outside the expected range
            if compound < low:
                mismatch_score = round(abs(compound - low), 4)
            else:
                mismatch_score = round(abs(compound - high), 4)
            
            if mismatch_score > 0.5:
                severity = "SEVERE"
            elif mismatch_score > 0.25:
                severity = "MODERATE"
            else:
                severity = "MILD"
            
            message = (f"{severity} inconsistency detected. "
                      f"Text sentiment is {sentiment['label']} ({compound:.2f}) "
                      f"but rating is {rating} stars.")
        
        return {
            'consistent': is_consistent,
            'mismatch_score': mismatch_score,
            'message': message,
            'sentiment': sentiment,
            'rating': rating,
        }
    
    def format_report(self, text, rating=None):
        """Generate a formatted sentiment report."""
        sentiment = self.analyze(text)
        
        report = []
        report.append("--- SENTIMENT ANALYSIS ---")
        report.append(f"Overall: {sentiment['label'].upper()} ({sentiment['intensity']})")
        report.append(f"Compound Score: {sentiment['compound']}")
        report.append(f"Positive: {sentiment['positive']}  |  Negative: {sentiment['negative']}  |  Neutral: {sentiment['neutral']}")
        
        if rating is not None:
            consistency = self.check_rating_consistency(text, rating)
            report.append("")
            report.append("--- RATING CONSISTENCY ---")
            report.append(consistency['message'])
            if not consistency['consistent'] and consistency['consistent'] is not None:
                report.append(f"Mismatch Score: {consistency['mismatch_score']}")
        
        return "\n".join(report)


if __name__ == "__main__":
    sa = SentimentAnalyzer()
    
    # Test with example reviews
    tests = [
        {
            'text': "This hotel was absolutely amazing! The staff were incredibly friendly and the room was spotless. Best vacation ever!",
            'rating': 5,
        },
        {
            'text': "Terrible experience. The room was dirty, staff were rude, and the noise kept me up all night. Would never return.",
            'rating': 1,
        },
        {
            'text': "The room was okay. Nothing special but nothing terrible either. Average hotel for the price.",
            'rating': 3,
        },
        {
            'text': "This was the worst hotel I have ever stayed in. Absolutely disgusting. Avoid at all costs!",
            'rating': 5,  # Inconsistent on purpose
        },
        {
            'text': "Amazing location, wonderful service, perfect in every way! Could not recommend more highly!",
            'rating': 1,  # Inconsistent on purpose
        },
    ]
    
    for i, test in enumerate(tests):
        print(f"\n{'=' * 60}")
        print(f"TEST {i+1}: Rating = {test['rating']} stars")
        print(f"Text: {test['text'][:80]}...")
        print()
        print(sa.format_report(test['text'], test['rating']))