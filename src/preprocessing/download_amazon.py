import requests
import json
import pandas as pd
import os

categories = {
    'Electronics': 'https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/Electronics.jsonl',
    'Beauty_Personal_Care': 'https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/Beauty_and_Personal_Care.jsonl',
    'Cell_Phones': 'https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/Cell_Phones_and_Accessories.jsonl',
    'Home_Kitchen': 'https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/Home_and_Kitchen.jsonl',
    'Health_Household': 'https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/Health_and_Household.jsonl',
}

REVIEWS_PER_CATEGORY = 200000

os.makedirs(os.path.join("data", "raw", "scraped"), exist_ok=True)

for category, url in categories.items():
    output_file = os.path.join("data", "raw", "scraped", f"amazon_{category}.csv")
    
    # Skip if already downloaded
    if os.path.exists(output_file):
        existing = pd.read_csv(output_file)
        print(f"\n{category}: Already have {len(existing)} reviews, skipping.")
        continue
    
    print(f"\nStreaming {category}...")
    reviews = []
    count = 0
    
    try:
        response = requests.get(url, stream=True, timeout=60)
        
        for line in response.iter_lines():
            if count >= REVIEWS_PER_CATEGORY:
                break
            
            if line:
                try:
                    review = json.loads(line.decode('utf-8'))
                    text = review.get('text', '').strip()
                    if len(text) > 30:
                        reviews.append({
                            'text': text,
                            'rating': review.get('rating', None),
                            'title': review.get('title', ''),
                            'verified_purchase': review.get('verified_purchase', None),
                            'category': category,
                            'source': 'amazon_academic',
                        })
                        count += 1
                        
                        if count % 5000 == 0:
                            print(f"  {count}/{REVIEWS_PER_CATEGORY} reviews collected")
                
                except json.JSONDecodeError:
                    continue
        
        # Save this category immediately
        if reviews:
            df = pd.DataFrame(reviews)
            df.to_csv(output_file, index=False)
            print(f"  Done: {count} reviews saved to {output_file}")
    
    except Exception as e:
        # Save whatever we have so far
        if reviews:
            df = pd.DataFrame(reviews)
            df.to_csv(output_file, index=False)
            print(f"  Error: {e}")
            print(f"  Saved {len(reviews)} reviews before error")
        else:
            print(f"  Error with {category}: {e}")

# Combine all files
print(f"\n{'='*50}")
print("Combining all categories...")
all_files = [os.path.join("data", "raw", "scraped", f) for f in os.listdir(os.path.join("data", "raw", "scraped")) if f.startswith("amazon_") and f.endswith(".csv")]

if all_files:
    combined = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
    combined.to_csv(os.path.join("data", "raw", "scraped", "amazon_academic_reviews.csv"), index=False)
    print(f"Total: {len(combined)} reviews")
    print(f"\nCategory breakdown:")
    print(combined['category'].value_counts())
    print(f"\nRating distribution:")
    print(combined['rating'].value_counts().sort_index())