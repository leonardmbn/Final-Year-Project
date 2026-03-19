import re
from collections import Counter


class FakeReviewTypeDetector:
    """Classifies suspected fake reviews by type based on linguistic patterns."""
    
    def detect_type(self, text, authenticity_score=None, sentiment=None, patterns=None):
        """
        Classify a review into fake review types:
        - incentivised: paid/rewarded for positive review
        - bot_generated: automated/template text
        - competitor_sabotage: fake negative to harm competitor
        - paid_promotion: professional fake positive review
        """
        
        text_lower = text.lower()
        words = re.findall(r'[a-zA-Z]+', text_lower)
        word_count = len(words)
        
        if word_count == 0:
            return {'type': 'unknown', 'confidence': 0, 'reasoning': []}
        
        scores = {
            'incentivised': 0,
            'bot_generated': 0,
            'competitor_sabotage': 0,
            'paid_promotion': 0,
        }
        reasoning = {k: [] for k in scores}
        
        # ============================================
        # INCENTIVISED REVIEW SIGNALS
        # ============================================
        # Disclosure-like language (sometimes they slip)
        incentive_phrases = ['free product', 'free sample', 'in exchange', 'gifted',
                            'received this', 'was given', 'sent me', 'provided by',
                            'discount code', 'coupon', 'promo', 'complimentary',
                            'in return for', 'honest review', 'unbiased review',
                            'was sent', 'received for']
        found_incentive = [p for p in incentive_phrases if p in text_lower]
        if found_incentive:
            scores['incentivised'] += 30
            reasoning['incentivised'].append(f"Incentive language detected: {', '.join(found_incentive[:3])}")
        
        # Overly positive but vague
        if sentiment and sentiment.get('compound', 0) > 0.7:
            unique_ratio = len(set(words)) / word_count if word_count > 0 else 0
            if unique_ratio < 0.6:
                scores['incentivised'] += 15
                reasoning['incentivised'].append("Highly positive with limited vocabulary")
        
        # Short, generic positive
        if word_count < 40 and sentiment and sentiment.get('compound', 0) > 0.5:
            scores['incentivised'] += 10
            reasoning['incentivised'].append("Short and generically positive")
        
        # ============================================
        # BOT-GENERATED REVIEW SIGNALS
        # ============================================
        # Repetitive structure
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if len(sentences) >= 3:
            lengths = [len(s.split()) for s in sentences]
            if lengths:
                length_std = (sum((l - sum(lengths)/len(lengths))**2 for l in lengths) / len(lengths)) ** 0.5
                if length_std < 2.0:
                    scores['bot_generated'] += 20
                    reasoning['bot_generated'].append("Uniform sentence lengths (templated structure)")
        
        # Generic template phrases
        template_phrases = ['highly recommend', 'would recommend', 'five stars',
                           '5 stars', 'must buy', 'must have', 'best product',
                           'great product', 'love this', 'love it', 'works great',
                           'works well', 'good quality', 'great quality',
                           'fast shipping', 'fast delivery', 'as described',
                           'as expected', 'arrived on time']
        found_templates = [p for p in template_phrases if p in text_lower]
        if len(found_templates) >= 3:
            scores['bot_generated'] += 25
            reasoning['bot_generated'].append(f"Multiple template phrases: {', '.join(found_templates[:4])}")
        elif len(found_templates) >= 2:
            scores['bot_generated'] += 15
            reasoning['bot_generated'].append(f"Template phrases: {', '.join(found_templates)}")
        
        # Very low lexical diversity
        unique_ratio = len(set(words)) / word_count if word_count > 0 else 0
        if unique_ratio < 0.45:
            scores['bot_generated'] += 20
            reasoning['bot_generated'].append(f"Very low vocabulary diversity ({unique_ratio:.0%})")
        
        # No personal details or specifics
        numbers = re.findall(r'\d+', text)
        personal_pronouns = sum(1 for w in words if w in ['i', 'me', 'my', 'mine', 'myself'])
        if len(numbers) == 0 and personal_pronouns == 0 and word_count > 20:
            scores['bot_generated'] += 15
            reasoning['bot_generated'].append("No personal pronouns or specific details")
        
        # Very short review
        if word_count < 15:
            scores['bot_generated'] += 15
            reasoning['bot_generated'].append(f"Suspiciously short ({word_count} words)")
        
        # ============================================
        # COMPETITOR SABOTAGE SIGNALS
        # ============================================
        # Strongly negative with no specific complaints
        if sentiment and sentiment.get('compound', 0) < -0.5:
            scores['competitor_sabotage'] += 10
            reasoning['competitor_sabotage'].append("Strongly negative sentiment")
            
            # Negative but no specific product details
            if len(numbers) == 0:
                scores['competitor_sabotage'] += 15
                reasoning['competitor_sabotage'].append("No specific details to support complaints")
            
            # Uses extreme/absolute negative language
            extreme_neg = ['worst', 'terrible', 'horrible', 'disgusting', 'awful',
                          'pathetic', 'atrocious', 'abysmal', 'garbage', 'trash',
                          'scam', 'fraud', 'rip off', 'ripoff', 'waste of money',
                          'stay away', 'do not buy', 'dont buy', 'avoid']
            found_extreme = [w for w in extreme_neg if w in text_lower]
            if len(found_extreme) >= 2:
                scores['competitor_sabotage'] += 20
                reasoning['competitor_sabotage'].append(f"Multiple extreme negative terms: {', '.join(found_extreme[:3])}")
        
        # Mentions competitor products
        competitor_signals = ['buy this instead', 'better alternative', 'switch to',
                             'went with', 'got the', 'bought the', 'try the',
                             'go with', 'chose instead', 'much better option']
        found_competitor = [p for p in competitor_signals if p in text_lower]
        if found_competitor:
            scores['competitor_sabotage'] += 20
            reasoning['competitor_sabotage'].append(f"References alternatives: {', '.join(found_competitor[:2])}")
        
        # ============================================
        # PAID PROMOTION SIGNALS
        # ============================================
        # Extremely positive with marketing-like language
        promo_phrases = ['game changer', 'life changing', 'changed my life',
                        'cant live without', 'everyone needs', 'must have',
                        'best investment', 'worth every penny', 'exceeded expectations',
                        'above and beyond', 'absolutely love', 'blown away',
                        'could not be happier', 'beyond impressed', 'top notch',
                        'world class', 'premium quality', 'outstanding quality']
        found_promo = [p for p in promo_phrases if p in text_lower]
        if len(found_promo) >= 2:
            scores['paid_promotion'] += 25
            reasoning['paid_promotion'].append(f"Marketing-style language: {', '.join(found_promo[:3])}")
        elif len(found_promo) >= 1:
            scores['paid_promotion'] += 10
            reasoning['paid_promotion'].append(f"Promotional phrasing: {found_promo[0]}")
        
        # Reads like an advertisement
        if sentiment and sentiment.get('compound', 0) > 0.8:
            superlatives = sum(1 for w in words if w in 
                             ['best', 'amazing', 'perfect', 'incredible', 'fantastic',
                              'wonderful', 'excellent', 'outstanding', 'superb', 'fabulous'])
            if superlatives >= 3:
                scores['paid_promotion'] += 20
                reasoning['paid_promotion'].append(f"Excessive superlatives ({superlatives} found)")
        
        # Long, detailed positive (professional reviewer style)
        if word_count > 150 and sentiment and sentiment.get('compound', 0) > 0.6:
            if personal_pronouns / word_count > 0.08:
                scores['paid_promotion'] += 10
                reasoning['paid_promotion'].append("Lengthy, positive, pronoun-heavy (professional reviewer pattern)")
        
        # ============================================
        # DETERMINE TYPE
        # ============================================
        max_type = max(scores, key=scores.get)
        max_score = scores[max_type]
        
        if max_score < 15:
            return {
                'type': 'none_detected',
                'confidence': 0,
                'reasoning': ['No strong fake review type indicators found.'],
                'all_scores': scores,
            }
        
        # Confidence: normalise to 0-100
        confidence = min(max_score, 100)
        
        type_labels = {
            'incentivised': 'Incentivised Review',
            'bot_generated': 'Bot-Generated/Templated',
            'competitor_sabotage': 'Competitor Sabotage',
            'paid_promotion': 'Paid Promotion',
        }
        
        return {
            'type': max_type,
            'type_label': type_labels[max_type],
            'confidence': confidence,
            'reasoning': reasoning[max_type],
            'all_scores': scores,
        }
    
    def format_report(self, text, sentiment=None):
        result = self.detect_type(text, sentiment=sentiment)
        
        lines = []
        lines.append("--- FAKE REVIEW TYPE ANALYSIS ---")
        
        if result['type'] == 'none_detected':
            lines.append("No strong fake review type indicators detected.")
        else:
            lines.append(f"Likely Type: {result['type_label']}")
            lines.append(f"Confidence: {result['confidence']}%")
            lines.append("Reasoning:")
            for r in result['reasoning']:
                lines.append(f"  - {r}")
        
        lines.append("")
        lines.append("Type Scores:")
        for t, s in result['all_scores'].items():
            bar = "█" * (s // 5) + "░" * (20 - s // 5)
            lines.append(f"  {t:25s} {bar} {s}")
        
        return "\n".join(lines)


if __name__ == "__main__":
    from sentiment_analyzer import SentimentAnalyzer
    
    sa = SentimentAnalyzer()
    td = FakeReviewTypeDetector()
    
    tests = [
        # Incentivised
        "I received this product for free in exchange for my honest review. It works great and I love it! The quality is good and fast shipping. Would recommend!",
        
        # Bot-generated
        "Great product. Good quality. Fast shipping. Highly recommend. Five stars. Works great. Love it. As described. Must buy.",
        
        # Competitor sabotage
        "This is the worst product I have ever bought. Terrible quality, absolute garbage. Total scam. Stay away. Do not buy this trash. Buy the Samsung version instead, much better option.",
        
        # Paid promotion
        "This product is an absolute game changer! It has exceeded expectations in every way. The premium quality is outstanding and I am beyond impressed. Best investment I have ever made. Everyone needs this in their life!",
        
        # Genuine
        "Bought this phone case 3 weeks ago for my iPhone 14. Fits well, the buttons are easy to press. One corner has a small gap but nothing major. Dropped my phone once on concrete from about 3 feet and no damage. For $12.99 it does the job.",
    ]
    
    for i, text in enumerate(tests):
        sentiment = sa.analyze(text)
        print(f"\n{'=' * 60}")
        print(f"TEST {i+1}: {text[:70]}...")
        print()
        print(td.format_report(text, sentiment=sentiment))