import os
import time
import random
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--disable-notifications')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver


def scrape_ebay_product_reviews(driver, item_url, max_reviews=100):
    """Scrape reviews from an eBay product listing."""
    
    reviews = []
    
    # Extract item ID
    item_id = None
    if '/itm/' in item_url:
        item_id = item_url.split('/itm/')[1].split('?')[0].split('/')[0]
    
    if not item_id:
        print("Could not extract item ID")
        return reviews
    
    print(f"  Item ID: {item_id}")
    
    # Go to the product page first to get product title
    driver.get(item_url)
    time.sleep(random.uniform(3, 5))
    
    product_title = ""
    try:
        title_elem = driver.find_element(By.CSS_SELECTOR, 'h1.x-item-title__mainTitle span, h1 span.ux-textspans--BOLD')
        product_title = title_elem.text.strip()
        print(f"  Product: {product_title[:60]}...")
    except:
        print("  Could not get product title")
    
    # Now go to the reviews/feedback page for this item
    # eBay product reviews URL pattern
    review_url = f"https://www.ebay.com/fdbk/mweb_profile?item_id={item_id}&filter=feedback_page%3ARECEIVED_AS_SELLER&q={item_id}"
    
    driver.get(review_url)
    time.sleep(random.uniform(3, 5))
    
    # Try to get reviews from feedback page
    page = 1
    while len(reviews) < max_reviews:
        try:
            # Look for feedback entries
            feedback_cards = driver.find_elements(By.CSS_SELECTOR, 'div.card, div[class*="feedback"], div[class*="review"]')
            
            if not feedback_cards:
                # Try alternative selectors
                feedback_cards = driver.find_elements(By.XPATH, '//div[contains(@class, "fdbk")]//div[contains(@class, "card")]')
            
            if not feedback_cards:
                print(f"  Page {page}: No feedback found")
                break
            
            new_reviews = 0
            for card in feedback_cards:
                try:
                    text = card.text.strip()
                    if text and len(text) > 10:
                        # Try to extract rating
                        rating = None
                        try:
                            stars = card.find_elements(By.CSS_SELECTOR, '[class*="star"], [aria-label*="star"]')
                            if stars:
                                label = stars[0].get_attribute('aria-label') or ""
                                nums = [float(s) for s in label.split() if s.replace('.','').isdigit()]
                                rating = nums[0] if nums else None
                        except:
                            pass
                        
                        review_data = {
                            'text': text,
                            'rating': rating,
                            'product_title': product_title,
                            'item_id': item_id,
                            'source': 'ebay',
                            'page': page
                        }
                        
                        # Avoid duplicates
                        if text not in [r['text'] for r in reviews]:
                            reviews.append(review_data)
                            new_reviews += 1
                
                except Exception as e:
                    continue
            
            print(f"  Page {page}: Found {new_reviews} new reviews (total: {len(reviews)})")
            
            if new_reviews == 0:
                break
            
            # Try to click next page
            try:
                next_btn = driver.find_element(By.CSS_SELECTOR, 'a[rel="next"], button[aria-label="Next"], [class*="next"]')
                next_btn.click()
                time.sleep(random.uniform(2, 4))
                page += 1
            except:
                print("  No more pages")
                break
        
        except Exception as e:
            print(f"  Error on page {page}: {e}")
            break
    
    return reviews


def scrape_ebay_search_reviews(driver, search_query, category, max_products=4, max_reviews_per=50):
    """Search eBay for products and scrape their reviews."""
    
    all_reviews = []
    
    search_url = f"https://www.ebay.com/sch/i.html?_nkw={search_query.replace(' ', '+')}&_sop=12"
    driver.get(search_url)
    time.sleep(random.uniform(3, 5))
    
    # Get product links
    product_links = []
    try:
        items = driver.find_elements(By.CSS_SELECTOR, 'a.s-item__link')
        for item in items[:max_products * 2]:  # Get extra in case some fail
            href = item.get_attribute('href')
            if href and '/itm/' in href:
                product_links.append(href)
    except:
        print("Could not find product links")
        return all_reviews
    
    product_links = list(set(product_links))[:max_products]
    print(f"Found {len(product_links)} products for '{search_query}'")
    
    for i, link in enumerate(product_links):
        print(f"\nProduct {i+1}/{len(product_links)}")
        product_reviews = scrape_ebay_product_reviews(driver, link, max_reviews=max_reviews_per)
        
        for review in product_reviews:
            review['category'] = category
            review['search_query'] = search_query
        
        all_reviews.extend(product_reviews)
        time.sleep(random.uniform(3, 6))
    
    return all_reviews


if __name__ == "__main__":
    
    searches = {
        'electronics': ['wireless earbuds', 'phone case', 'usb charger'],
        'beauty': ['face serum', 'moisturizer cream', 'vitamin c serum'],
        'supplements': ['protein powder', 'multivitamin', 'fish oil'],
        'home_kitchen': ['air fryer', 'blender', 'kitchen gadget'],
    }
    
    driver = setup_driver()
    all_reviews = []
    
    try:
        for category, queries in searches.items():
            print(f"\n{'='*60}")
            print(f"CATEGORY: {category.upper()}")
            print(f"{'='*60}")
            
            for query in queries:
                print(f"\nSearching: {query}")
                reviews = scrape_ebay_search_reviews(driver, query, category, max_products=3, max_reviews_per=50)
                all_reviews.extend(reviews)
                print(f"Total reviews collected: {len(all_reviews)}")
                time.sleep(random.uniform(5, 8))
    
    finally:
        driver.quit()
    
    if all_reviews:
        df = pd.DataFrame(all_reviews)
        os.makedirs(os.path.join("data", "raw", "scraped"), exist_ok=True)
        output_path = os.path.join("data", "raw", "scraped", "ebay_reviews.csv")
        df.to_csv(output_path, index=False)
        print(f"\n{'='*60}")
        print(f"DONE! Scraped {len(df)} total reviews")
        print(f"Saved to {output_path}")
        if 'category' in df.columns:
            print(f"Category breakdown:\n{df['category'].value_counts()}")
    else:
        print("\nNo reviews scraped.")