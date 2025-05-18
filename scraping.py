import csv
import time
import random
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Configuration des User-Agents
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/118.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1'
]

def get_random_user_agent():
    return random.choice(USER_AGENTS)

# Initialisation du driver Chrome
def init_driver():
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--window-size=1920x1080')
    options.add_argument(f'user-agent={get_random_user_agent()}')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    driver = webdriver.Chrome(options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver

# URL de base et configuration
BASE_URL = "https://www.amazon.com/s?k=All-New+Fire+HD+8+Tablet%2C+8+HD+Display%2C+Wi-Fi&crid=B7GZQYLJP641&sprefix=all-new+fire+hd+8+tablet%2C+8+hd+display%2C+wi-fi%2Caps%2C181&ref=nb_sb_noss"
MAX_PAGES = 1000
MAX_REVIEW_PAGES = 1000
driver = init_driver()

def get_all_page_urls(base_url, max_pages):
    return [base_url + f"&page={page}" for page in range(1, max_pages + 1)]

def clean_text(text):
    return ' '.join(text.strip().split()) if text else ''

def extract_product_card_info(card):
    try:
        return {
            'name': clean_text(card.h2.text) if card.h2 else 'N/A',
            'url': "https://www.amazon.com" + card.h2.a.get('href') if card.h2 and card.h2.a else 'N/A',
            'price': clean_text(card.find('span', 'a-offscreen').text) if card.find('span', 'a-offscreen') else 'N/A',
            'rating': clean_text(card.i.text) if card.i else 'N/A',
            'review_count': clean_text(card.find('span', {'class': 'a-size-base s-underline-text'}).text) 
                         if card.find('span', {'class': 'a-size-base s-underline-text'}) else 'N/A'
        }
    except Exception as e:
        print(f"Error extracting card info: {e}")
        return None

def extract_technical_details(soup):
    details = {
        'asins': '',
        'brand': '',
        'manufacturer': '',
        'categories': []
    }

    # Méthode 1: Tableau des spécifications
    for table in soup.find_all('table', class_='prodDetTable'):
        for row in table.find_all('tr'):
            try:
                th = row.find('th').text.strip()
                td = row.find('td').text.strip()
                
                if 'ASIN' in th:
                    details['asins'] = clean_text(td)
                elif 'Manufacturer' in th:
                    details['manufacturer'] = clean_text(td)
                elif 'Brand' in th:
                    details['brand'] = clean_text(td)
            except:
                continue

    # Méthode 2: Bullet points
    bullet_div = soup.find('div', {'id': 'detailBullets_feature_div'})
    if bullet_div:
        for item in bullet_div.find_all('span', {'class': 'a-list-item'}):
            text = clean_text(item.text)
            if not details['asins'] and 'ASIN' in text:
                details['asins'] = text.split(':')[-1].strip()
            if not details['manufacturer'] and 'Manufacturer' in text:
                details['manufacturer'] = text.split(':')[-1].strip()
            if not details['brand'] and 'Brand' in text:
                details['brand'] = text.split(':')[-1].strip()

    return details

def scrape_reviews_page(reviews_url):
    driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": get_random_user_agent()})
    driver.get(reviews_url)
    time.sleep(random.uniform(3, 6))
    
    reviews_data = []
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    for card in soup.find_all('div', {'data-hook': 'review'}):
        try:
            review = {
                'date': clean_text(card.find('span', {'data-hook': 'review-date'}).text) if card.find('span', {'data-hook': 'review-date'}) else '',
                'dateAdded': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                'dateSeen': [time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())],
                'doRecommend': True,
                'id': card.get('id', f"rev_{random.randint(100000,999999)}"),
                'numHelpful': 0,
                'rating': 0.0,
                'sourceURLs': reviews_url,
                'text': '',
                'title': '',
                'userCity': '',
                'userProvince': '',
                'username': 'Anonymous'
            }

            # Review content
            if card.find('span', {'data-hook': 'review-body'}):
                review['text'] = clean_text(card.find('span', {'data-hook': 'review-body'}).text)
            if card.find('a', {'data-hook': 'review-title'}):
                review['title'] = clean_text(card.find('a', {'data-hook': 'review-title'}).text)

            # Rating
            rating_element = card.find('i', {'data-hook': 'review-star-rating'}) or card.find('i', {'data-hook': 'cmps-review-star-rating'})
            if rating_element:
                review['rating'] = float(rating_element.text.split()[0])
                review['doRecommend'] = review['rating'] >= 4

            # User info
            if card.find('span', {'class': 'a-profile-name'}):
                review['username'] = clean_text(card.find('span', {'class': 'a-profile-name'}).text)

            # Helpful votes
            if card.find('span', {'data-hook': 'helpful-vote-statement'}):
                helpful_text = clean_text(card.find('span', {'data-hook': 'helpful-vote-statement'}).text)
                if 'person' in helpful_text:
                    review['numHelpful'] = int(helpful_text.split()[0].replace(',', ''))

            reviews_data.append(review)
        except Exception as e:
            print(f"Error parsing review: {e}")
            continue

    return reviews_data

def get_product_details(product_url):
    try:
        # Change user agent before loading product page
        driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": get_random_user_agent()})
        driver.get(product_url)
        time.sleep(random.uniform(2, 5))
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # Get basic product info
        product_data = {
            'id': f"prod_{int(time.time())}_{random.randint(1000,9999)}",
            'name': clean_text(soup.find('span', {'id': 'productTitle'}).text) if soup.find('span', {'id': 'productTitle'}) else '',
            'price': '',
            'rating': '',
            'review_count': ''
        }
        
        # Get technical details
        tech_details = extract_technical_details(soup)
        product_data.update(tech_details)
        
        # Get reviews
        reviews_url = product_url.replace('dp', 'product-reviews') + "?reviewerType=all_reviews"
        all_reviews = []
        
        for page in range(1, MAX_REVIEW_PAGES + 1):
            if page > 1:
                reviews_url = f"{reviews_url}&pageNumber={page}"
            
            page_reviews = scrape_reviews_page(reviews_url)
            all_reviews.extend(page_reviews)
            
            if len(page_reviews) < 10:  # Less than 10 reviews on page, likely last page
                break
            
            time.sleep(random.uniform(4, 8))  # Longer delay between review pages
        
        # Structure final data
        result = product_data.copy()
        
        if all_reviews:
            for key in all_reviews[0].keys():
                result[f'reviews.{key}'] = [review[key] for review in all_reviews]
        
        return result

    except Exception as e:
        print(f"Error scraping product {product_url}: {e}")
        return None

    except Exception as e:
        print(f"Error scraping product {product_url}: {e}")
        return None

def main():
    print("Starting Amazon scraper...")
    page_urls = get_all_page_urls(BASE_URL, MAX_PAGES)
    all_data = []
    
    for page_num, page_url in enumerate(page_urls, 1):
        print(f"\nProcessing page {page_num}/{len(page_urls)}")
        
        # Change user agent before loading page
        driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": get_random_user_agent()})
        driver.get(page_url)
        time.sleep(random.uniform(3, 7))
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        product_cards = soup.find_all('div', {'data-component-type': 's-search-result'})
        
        for card in product_cards:
            basic_info = extract_product_card_info(card)
            if not basic_info or not basic_info.get('url'):
                continue
                
            print(f"Scraping product: {basic_info['name'][:50]}...")
            product_details = get_product_details(basic_info['url'])
            
            if product_details:
                product_details.update({
                    'price': basic_info['price'],
                    'rating': basic_info['rating'],
                    'review_count': basic_info['review_count']
                })
                all_data.append(product_details)
                
                # Save periodically
                if len(all_data) % 5 == 0:
                    save_to_csv(all_data, 'amazon_data_partial.csv')
                
                time.sleep(random.uniform(2, 5))  # Delay between products
    
    # Final save
    save_to_csv(all_data, 'dataset_projet_web_mining.csv')
    print("\nScraping completed successfully!")
    driver.quit()

def save_to_csv(data, filename):
    fieldnames = [
        'id', 'name', 'asins', 'brand', 'categories', 'manufacturer',
        'price', 'rating', 'review_count',
        'reviews.date', 'reviews.dateAdded', 'reviews.dateSeen',
        'reviews.doRecommend', 'reviews.id', 'reviews.numHelpful',
        'reviews.rating', 'reviews.sourceURLs', 'reviews.text',
        'reviews.title', 'reviews.userCity', 'reviews.userProvince',
        'reviews.username'
    ]
    
    with open(filename, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for product in data:
            if not product.get('reviews.id'):
                # Product with no reviews
                row = {k: v for k, v in product.items() if not k.startswith('reviews.')}
                row.update({k: '' for k in fieldnames if k.startswith('reviews.')})
                writer.writerow(row)
            else:
                # Product with reviews
                num_reviews = len(product['reviews.id'])
                for i in range(num_reviews):
                    row = {k: v for k, v in product.items() if not k.startswith('reviews.')}
                    row.update({
                        'reviews.date': product['reviews.date'][i] if i < len(product['reviews.date']) else '',
                        'reviews.dateAdded': product['reviews.dateAdded'][i] if i < len(product['reviews.dateAdded']) else '',
                        'reviews.dateSeen': ','.join(product['reviews.dateSeen'][i]) if i < len(product['reviews.dateSeen']) else '',
                        'reviews.doRecommend': product['reviews.doRecommend'][i] if i < len(product['reviews.doRecommend']) else '',
                        'reviews.id': product['reviews.id'][i] if i < len(product['reviews.id']) else '',
                        'reviews.numHelpful': product['reviews.numHelpful'][i] if i < len(product['reviews.numHelpful']) else 0,
                        'reviews.rating': product['reviews.rating'][i] if i < len(product['reviews.rating']) else 0.0,
                        'reviews.sourceURLs': product['reviews.sourceURLs'][i] if i < len(product['reviews.sourceURLs']) else '',
                        'reviews.text': product['reviews.text'][i] if i < len(product['reviews.text']) else '',
                        'reviews.title': product['reviews.title'][i] if i < len(product['reviews.title']) else '',
                        'reviews.userCity': product['reviews.userCity'][i] if i < len(product['reviews.userCity']) else '',
                        'reviews.userProvince': product['reviews.userProvince'][i] if i < len(product['reviews.userProvince']) else '',
                        'reviews.username': product['reviews.username'][i] if i < len(product['reviews.username']) else ''
                    })
                    writer.writerow(row)

if __name__ == "__main__":
    main()