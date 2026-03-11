import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import os
from fake_useragent import UserAgent

ua = UserAgent()

def get_headers():
    return {
        'User-Agent': ua.random,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
    }


def scrape_amazon_reviews(product_url, max_pages=5):
    """Scrape reviews from an Amazon product page."""
    
    reviews = []
    
    # Extract ASIN from URL
    asin = None
    if '/dp/' in product_url:
        asin = product_url.split('/dp/')[1].split('/')[0].split('?')[0]
    elif '/product-reviews/' in product_url:
        asin = product_url.split('/product-reviews/')[1].split('/')[0].split('?')[0]
    
    if not asin:
        print(f"Could not extract ASIN from URL: {product_url}")
        return reviews
    
    print(f"Scraping ASIN: {asin}")
    
    for page in range(1, max_pages + 1):
        url = f"https://www.amazon.com/product-reviews/{asin}/ref=cm_cr_getr_d_paging_btm_next_{page}?pageNumber={page}"
        
        try:
            time.sleep(random.uniform(2, 5))  # Be polite
            response = requests.get(url, headers=get_headers(), timeout=15)
            
            if response.status_code != 200:
                print(f"  Page {page}: Status {response.status_code}, skipping...")
                continue
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            review_divs = soup.find_all('div', {'data-hook': 'review'})
            
            if not review_divs:
                print(f"  Page {page}: No reviews found, stopping.")
                break
            
            for review_div in review_divs:
                try:
                    # Review text
                    text_elem = review_div.find('span', {'data-hook': 'review-body'})
                    text = text_elem.get_text(strip=True) if text_elem else ""
                    
                    # Star rating
                    rating_elem = review_div.find('i', {'data-hook': 'review-star-rating'})
                    if not rating_elem:
                        rating_elem = review_div.find('i', {'data-hook': 'cmps-review-star-rating'})
                    rating_text = rating_elem.get_text(strip=True) if rating_elem else ""
                    rating = float(rating_text.split(' ')[0]) if rating_text else None
                    
                    # Review title
                    title_elem = review_div.find('a', {'data-hook': 'review-title'})
                    if not title_elem:
                        title_elem = review_div.find('span', {'data-hook': 'review-title'})
                    title = title_elem.get_text(strip=True) if title_elem else ""
                    
                    # Verified purchase
                    verified_elem = review_div.find('span', {'data-hook': 'avp-badge'})
                    verified = True if verified_elem else False
                    
                    # Helpful votes
                    helpful_elem = review_div.find('span', {'data-hook': 'helpful-vote-statement'})
                    helpful_text = helpful_elem.get_text(strip=True) if helpful_elem else "0"
                    helpful_votes = 0
                    if 'one' in helpful_text.lower():
                        helpful_votes = 1
                    elif helpful_text:
                        nums = [int(s) for s in helpful_text.split() if s.isdigit()]
                        helpful_votes = nums[0] if nums else 0
                    
                    # Date
                    date_elem = review_div.find('span', {'data-hook': 'review-date'})
                    date = date_elem.get_text(strip=True) if date_elem else ""
                    
                    # Reviewer name
                    name_elem = review_div.find('span', class_='a-profile-name')
                    reviewer = name_elem.get_text(strip=True) if name_elem else ""
                    
                    if text:  # Only add if we got review text
                        reviews.append({
                            'text': text,
                            'title': title,
                            'rating': rating,
                            'verified_purchase': verified,
                            'helpful_votes': helpful_votes,
                            'date': date,
                            'reviewer': reviewer,
                            'asin': asin,
                            'source': 'amazon',
                            'page': page
                        })
                
                except Exception as e:
                    print(f"  Error parsing review: {e}")
                    continue
            
            print(f"  Page {page}: Scraped {len(review_divs)} reviews")
        
        except Exception as e:
            print(f"  Page {page}: Error - {e}")
            continue
    
    return reviews


def scrape_multiple_products(product_urls, category, max_pages=5):
    """Scrape reviews from multiple product URLs."""
    
    all_reviews = []
    
    for i, url in enumerate(product_urls):
        print(f"\nProduct {i+1}/{len(product_urls)}: {url[:80]}...")
        product_reviews = scrape_amazon_reviews(url, max_pages=max_pages)
        
        for review in product_reviews:
            review['category'] = category
        
        all_reviews.extend(product_reviews)
        print(f"  Total reviews so far: {len(all_reviews)}")
        
        # Longer pause between products
        if i < len(product_urls) - 1:
            wait = random.uniform(5, 10)
            print(f"  Waiting {wait:.1f}s before next product...")
            time.sleep(wait)
    
    return all_reviews


if __name__ == "__main__":
    # INSTRUCTIONS:
    # 1. Go to Amazon.com
    # 2. Search for products in each category
    # 3. Copy the product page URLs
    # 4. Paste them in the lists below
    
    # Example format - REPLACE THESE with real URLs you find
    products = {
        'electronics': [
            "https://www.amazon.com/dp/B0C2C9NHZW",
        ],
        'beauty': [
            # Paste 3-4 Amazon product URLs for beauty/skincare here
        ],
        'supplements': [
            # Paste 3-4 Amazon product URLs for supplements here
        ],
        'home_kitchen': [
            # Paste 3-4 Amazon product URLs for home/kitchen here
        ],
    }
    
    all_reviews = []
    
    for category, urls in products.items():
        if not urls:
            print(f"\nSkipping {category} - no URLs provided")
            continue
        print(f"\n{'='*60}")
        print(f"SCRAPING CATEGORY: {category.upper()}")
        print(f"{'='*60}")
        reviews = scrape_multiple_products(urls, category, max_pages=5)
        all_reviews.extend(reviews)
    
    if all_reviews:
        df = pd.DataFrame(all_reviews)
        os.makedirs(os.path.join("data", "raw", "scraped"), exist_ok=True)
        output_path = os.path.join("data", "raw", "scraped", "amazon_reviews.csv")
        df.to_csv(output_path, index=False)
        print(f"\n{'='*60}")
        print(f"DONE! Scraped {len(df)} total reviews")
        print(f"Saved to {output_path}")
        print(f"Category breakdown:\n{df['category'].value_counts()}")
    else:
        print("\nNo reviews scraped. Add product URLs to the script first!")