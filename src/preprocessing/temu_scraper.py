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


def scrape_temu_reviews(driver, product_url, max_reviews=200):
    """Scrape reviews from a Temu product page."""
    
    reviews = []
    
    driver.get(product_url)
    time.sleep(random.uniform(5, 8))
    
    # Get product title
    product_title = ""
    try:
        title_elem = driver.find_element(By.CSS_SELECTOR, 'h1')
        product_title = title_elem.text.strip()
        print(f"  Product: {product_title[:60]}...")
    except:
        print("  Could not get product title")
    
    # Scroll down to reviews section
    try:
        review_section = driver.find_element(By.CSS_SELECTOR, 'div#reviewContent, div[id*="review"]')
        driver.execute_script("arguments[0].scrollIntoView(true);", review_section)
        time.sleep(2)
    except:
        print("  Could not find review section, scrolling down...")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 0.5);")
        time.sleep(3)
    
    page = 1
    last_count = 0
    stale_attempts = 0
    
    while len(reviews) < max_reviews and stale_attempts < 3:
        
        # Find all review blocks
        review_blocks = driver.find_elements(By.CSS_SELECTOR, 'div._244ldJXl')
        
        if not review_blocks:
            # Try alternative selectors
            review_blocks = driver.find_elements(By.CSS_SELECTOR, 'div[class*="244ldJXl"], div._46rO8bke > div')
        
        for block in review_blocks:
            try:
                # Review text
                text = ""
                try:
                    text_elem = block.find_element(By.CSS_SELECTOR, 'section.YxrbHrgw span, div._2EOoyd2j span')
                    text = text_elem.text.strip()
                except:
                    try:
                        text_elem = block.find_element(By.CSS_SELECTOR, 'section span')
                        text = text_elem.text.strip()
                    except:
                        pass
                
                if not text or len(text) < 5:
                    continue
                
                # Skip duplicates
                if text in [r['text'] for r in reviews]:
                    continue
                
                # Reviewer name
                reviewer = ""
                try:
                    name_elem = block.find_element(By.CSS_SELECTOR, 'div._XTEkYd1M, div[class*="XTEkYd1M"]')
                    reviewer = name_elem.text.strip()
                except:
                    pass
                
                # Date and location
                date_location = ""
                try:
                    date_elem = block.find_element(By.CSS_SELECTOR, 'div._1tSRIohB, div[class*="1tSRIohB"]')
                    date_location = date_elem.get_attribute('aria-label') or date_elem.text.strip()
                except:
                    pass
                
                # Star rating
                rating = None
                try:
                    stars_container = block.find_element(By.CSS_SELECTOR, 'div._21wXPU_9, div[class*="21wXPU_9"]')
                    filled_stars = stars_container.find_elements(By.CSS_SELECTOR, 'img[src*="star"], svg[class*="fill"], div[style*="color: rgb(255"]')
                    if filled_stars:
                        rating = len(filled_stars)
                except:
                    try:
                        # Try aria-label approach
                        rating_elem = block.find_element(By.CSS_SELECTOR, '[aria-label*="star"], [aria-label*="rating"]')
                        label = rating_elem.get_attribute('aria-label')
                        nums = [float(s) for s in label.split() if s.replace('.','').isdigit()]
                        rating = nums[0] if nums else None
                    except:
                        pass
                
                reviews.append({
                    'text': text,
                    'reviewer': reviewer,
                    'date_location': date_location,
                    'rating': rating,
                    'product_title': product_title,
                    'source': 'temu',
                })
            
            except Exception as e:
                continue
        
        print(f"  Scroll {page}: Total unique reviews: {len(reviews)}")
        
        # Check if we got new reviews
        if len(reviews) == last_count:
            stale_attempts += 1
        else:
            stale_attempts = 0
        last_count = len(reviews)
        
        # Try to load more reviews - click "next" or scroll
        try:
            # Look for pagination or "more" button
            next_btns = driver.find_elements(By.CSS_SELECTOR, 
                'button[aria-label="Next"], a[aria-label="Next"], '
                'div[class*="next"], button[class*="next"], '
                'div._1AFVb_qh button, nav button:last-child')
            
            clicked = False
            for btn in next_btns:
                try:
                    if btn.is_displayed() and btn.is_enabled():
                        driver.execute_script("arguments[0].click();", btn)
                        clicked = True
                        time.sleep(random.uniform(2, 4))
                        break
                except:
                    continue
            
            if not clicked:
                # Try scrolling within review section
                driver.execute_script("""
                    var reviewSection = document.querySelector('#reviewContent');
                    if (reviewSection) {
                        reviewSection.scrollTop += 500;
                    } else {
                        window.scrollBy(0, 500);
                    }
                """)
                time.sleep(random.uniform(2, 3))
        
        except:
            pass
        
        page += 1
        
        if page > 20:
            break
    
    return reviews


if __name__ == "__main__":
    
    # ADD TEMU PRODUCT URLs HERE
    # Find products with lots of reviews in each category
    products = {
        'electronics': [
            # Paste Temu URLs for electronics products here
        ],
        'beauty': [
            # Paste Temu URLs for beauty products here
        ],
        'supplements': [
            # Paste Temu URLs for supplements/health here
        ],
        'home_kitchen': [
            # Paste Temu URLs for home/kitchen here
        ],
    }
    
    # For testing with your URL
    test_url = "https://www.temu.com/ng/-travel-backpack-suitable-for--men-and-women-designed-for-16-inch-laptops--with-airline-carry-on-regulations-featuring-a-luggage-sleeve-sturdy-college-backpack-ideal-for--school-use-or-as-a-gift-g-601099519621915.html"
    
    driver = setup_driver()
    all_reviews = []
    
    try:
        # TEST RUN with single product
        print("="*60)
        print("TEST SCRAPE - Single product")
        print("="*60)
        reviews = scrape_temu_reviews(driver, test_url, max_reviews=50)
        
        for r in reviews:
            r['category'] = 'test'
        all_reviews.extend(reviews)
        
        print(f"\nTest complete: {len(reviews)} reviews scraped")
        if reviews:
            print(f"Sample review: {reviews[0]['text'][:100]}...")
    
    finally:
        driver.quit()
    
    if all_reviews:
        df = pd.DataFrame(all_reviews)
        os.makedirs(os.path.join("data", "raw", "scraped"), exist_ok=True)
        output_path = os.path.join("data", "raw", "scraped", "temu_reviews_test.csv")
        df.to_csv(output_path, index=False)
        print(f"\nSaved to {output_path}")
    else:
        print("\nNo reviews scraped.")