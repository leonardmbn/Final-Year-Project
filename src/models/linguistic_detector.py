import re
from collections import Counter


class LinguisticPatternDetector:
    """Detects linguistic patterns associated with fake reviews."""
    
    SUPERLATIVES = [
        'best', 'worst', 'amazing', 'terrible', 'perfect', 'horrible',
        'excellent', 'awful', 'fantastic', 'disgusting', 'wonderful',
        'dreadful', 'outstanding', 'pathetic', 'incredible', 'unbelievable',
        'fabulous', 'atrocious', 'superb', 'abysmal', 'magnificent',
    ]
    
    ABSOLUTES = [
        'absolutely', 'totally', 'completely', 'utterly', 'extremely',
        'never', 'always', 'every', 'nothing', 'everything', 'definitely',
        'certainly', 'undoubtedly', 'without a doubt', 'hands down',
    ]
    
    GENERIC_PHRASES = [
        'highly recommend', 'would recommend', 'do not recommend',
        'stay away', 'waste of money', 'bang for your buck',
        'value for money', 'top notch', 'first class', 'world class',
        'above and beyond', 'exceeded expectations', 'fell short',
        'left much to be desired', 'could not be happier',
        'could not be more disappointed', 'five stars', '5 stars',
        'one star', '1 star', 'must buy', 'must have', 'game changer',
        'life changing', 'best purchase ever', 'worst purchase ever',
    ]
    
    FIRST_PERSON = ['i', 'me', 'my', 'mine', 'myself', 'we', 'our', 'ours']
    
    def __init__(self):
        pass
    
    def detect(self, text):
        """Run all pattern detections on a review text."""
        words = re.findall(r'[a-zA-Z]+', text.lower())
        word_count = len(words)
        
        if word_count == 0:
            return {'flags': [], 'risk_score': 0, 'details': {}}
        
        flags = []
        details = {}
        
        # 1. Excessive first-person pronouns
        fp_count = sum(1 for w in words if w in self.FIRST_PERSON)
        fp_ratio = fp_count / word_count
        details['first_person_ratio'] = round(fp_ratio, 4)
        if fp_ratio > 0.08:
            flags.append({
                'pattern': 'Excessive first-person pronouns',
                'severity': 'HIGH' if fp_ratio > 0.12 else 'MEDIUM',
                'detail': f"First-person pronoun ratio: {fp_ratio:.2%} (threshold: 8%)",
            })
        
        # 2. Superlative overuse
        sup_count = sum(1 for w in words if w in self.SUPERLATIVES)
        sup_ratio = sup_count / word_count
        details['superlative_ratio'] = round(sup_ratio, 4)
        if sup_ratio > 0.02:
            flags.append({
                'pattern': 'Superlative language overuse',
                'severity': 'HIGH' if sup_ratio > 0.04 else 'MEDIUM',
                'detail': f"Found {sup_count} superlative words ({sup_ratio:.2%} of text)",
            })
        
        # 3. Absolute language
        abs_count = sum(1 for w in words if w in self.ABSOLUTES)
        details['absolute_count'] = abs_count
        if abs_count >= 3:
            flags.append({
                'pattern': 'Excessive absolute language',
                'severity': 'HIGH' if abs_count >= 5 else 'MEDIUM',
                'detail': f"Found {abs_count} absolute terms (e.g., always, never, completely)",
            })
        
        # 4. Generic/templated phrases
        text_lower = text.lower()
        found_generic = [p for p in self.GENERIC_PHRASES if p in text_lower]
        details['generic_phrases'] = found_generic
        if len(found_generic) >= 2:
            flags.append({
                'pattern': 'Generic/templated phrasing',
                'severity': 'HIGH' if len(found_generic) >= 3 else 'MEDIUM',
                'detail': f"Found {len(found_generic)} generic phrases: {', '.join(found_generic[:3])}",
            })
        
        # 5. Low lexical diversity
        unique = len(set(words))
        diversity = unique / word_count
        details['lexical_diversity'] = round(diversity, 4)
        if diversity < 0.55:
            flags.append({
                'pattern': 'Low vocabulary diversity',
                'severity': 'HIGH' if diversity < 0.45 else 'MEDIUM',
                'detail': f"Lexical diversity: {diversity:.2%} (threshold: 55%)",
            })
        
        # 6. Lack of specific details
        numbers = re.findall(r'\d+', text)
        details['number_count'] = len(numbers)
        if len(numbers) == 0 and word_count > 50:
            flags.append({
                'pattern': 'No specific details or numbers',
                'severity': 'MEDIUM',
                'detail': "Review contains no numerical references (dates, prices, quantities)",
            })
        
        # 7. Excessive exclamation marks
        excl_count = text.count('!')
        details['exclamation_count'] = excl_count
        if excl_count >= 3:
            flags.append({
                'pattern': 'Excessive exclamation marks',
                'severity': 'HIGH' if excl_count >= 5 else 'MEDIUM',
                'detail': f"Found {excl_count} exclamation marks",
            })
        
        # 8. Very short or very long review
        details['word_count'] = word_count
        if word_count < 20:
            flags.append({
                'pattern': 'Suspiciously short review',
                'severity': 'MEDIUM',
                'detail': f"Only {word_count} words — genuine reviews tend to be more detailed",
            })
        elif word_count > 300:
            flags.append({
                'pattern': 'Unusually long review',
                'severity': 'LOW',
                'detail': f"{word_count} words — may indicate copied or generated content",
            })
        
        # 9. Repetitive word usage
        word_freq = Counter(words)
        # Exclude common words
        content_words = {w: c for w, c in word_freq.items() 
                        if w not in self.FIRST_PERSON and len(w) > 3 and c > 2}
        details['repeated_content_words'] = dict(list(content_words.items())[:5])
        if len(content_words) >= 3:
            flags.append({
                'pattern': 'Repetitive vocabulary',
                'severity': 'MEDIUM',
                'detail': f"Multiple words repeated 3+ times: {', '.join(list(content_words.keys())[:3])}",
            })
        
        # Calculate overall risk score (0-100)
        severity_weights = {'HIGH': 15, 'MEDIUM': 10, 'LOW': 5}
        risk_score = sum(severity_weights.get(f['severity'], 0) for f in flags)
        risk_score = min(risk_score, 100)
        
        return {
            'flags': flags,
            'flag_count': len(flags),
            'risk_score': risk_score,
            'risk_level': 'HIGH' if risk_score >= 40 else 'MEDIUM' if risk_score >= 20 else 'LOW',
            'details': details,
        }
    
    def format_report(self, text):
        """Generate a formatted pattern detection report."""
        result = self.detect(text)
        
        report = []
        report.append("--- LINGUISTIC PATTERN ANALYSIS ---")
        report.append(f"Risk Level: {result['risk_level']} (Score: {result['risk_score']}/100)")
        report.append(f"Patterns Flagged: {result['flag_count']}")
        
        if result['flags']:
            report.append("")
            for i, flag in enumerate(result['flags'], 1):
                report.append(f"  [{flag['severity']}] {flag['pattern']}")
                report.append(f"         {flag['detail']}")
        else:
            report.append("  No suspicious patterns detected.")
        
        return "\n".join(report)


if __name__ == "__main__":
    ld = LinguisticPatternDetector()
    
    tests = [
        # Likely fake - vague, superlative, no specifics
        "This is absolutely the best product I have ever purchased! Amazing quality, perfect in every way. I would highly recommend this to everyone. Totally worth it! Best purchase ever!",
        
        # Likely genuine - specific details, balanced
        "Stayed 3 nights in room 412 during the week of March 15th. The bathroom had a small leak under the sink which maintenance fixed within 2 hours. Breakfast buffet was decent, especially the omelette station. WiFi was spotty on the 4th floor. Room rate was $189/night which felt fair for downtown location.",
        
        # Borderline - some flags
        "I really enjoyed my stay here. The staff were friendly and the room was clean. I would definitely come back. My family loved the pool area.",
        
        # Bot-like - generic, templated
        "Highly recommend this product. Five stars. Exceeded expectations. Must buy. Best purchase ever. Would recommend to everyone. Top notch quality.",
    ]
    
    for i, text in enumerate(tests):
        print(f"\n{'=' * 60}")
        print(f"TEST {i+1}: {text[:70]}...")
        print()
        print(ld.format_report(text))