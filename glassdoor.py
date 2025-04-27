from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import json
import requests
from urllib.parse import urljoin

# Configuration
BASE_URL = "https://www.careerjet.com.eg/jobs?s=cyber+security+engineer&l="
MAX_PAGES = 5  # Set to None to scrape all pages
OUTPUT_FILE = "careerjet_jobs.json"
HEADLESS = True  # Set to False to see the browser window

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def init_driver():
    """Initialize Selenium WebDriver in fully headless mode"""
    options = Options()
    
    # Essential headless configurations
    options.add_argument("--headless=new")  # New headless mode in Chrome 109+
    options.add_argument("--disable-gpu")  # GPU acceleration can cause issues in headless
    options.add_argument("--no-sandbox")  # Bypass OS security model
    options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems
    
    # Stealth configurations
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    # Disable logging and unnecessary features
    options.add_argument("--log-level=3")
    options.add_experimental_option("excludeSwitches", ["enable-logging"])
    options.add_experimental_option('useAutomationExtension', False)
    
    # Memory/performance optimizations
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-application-cache")
    
    # Initialize driver
    try:
        driver = webdriver.Chrome(options=options)
        # Additional stealth settings
        driver.execute_cdp_cmd('Network.setUserAgentOverride', {
            "userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })
        driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
            'source': '''
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                })
            '''
        })
        return driver
    except Exception as e:
        raise RuntimeError(f"Failed to initialize WebDriver: {str(e)}")

def scrape_job_links(driver, base_url, max_pages=3):
    """Scrape job links from multiple pages using Selenium"""
    all_links = []
    
    try:
        for page in range(1, max_pages + 1):
            url = f"{base_url}&p={page}" if "?" in base_url else f"{base_url}?p={page}"
            print(f"Scraping page {page}/{max_pages}...")
            
            driver.get(url)
            
            # Wait for jobs to load
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "article.job"))
            )
            
            # Scroll to load dynamic content (if needed)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
            
            # Get job links
            jobs = driver.find_elements(By.CSS_SELECTOR, "article.job")
            if not jobs:
                print("No jobs found on page, stopping pagination")
                break
                
            for job in jobs:
                try:
                    link = job.find_element(By.CSS_SELECTOR, "h2 a").get_attribute("href")
                    if link and link not in all_links:
                        all_links.append(link)
                except Exception as e:
                    print(f"Error extracting link: {e}")
                    continue
            
            time.sleep(2)  # Be polite
            
    except Exception as e:
        print(f"Error during pagination: {e}")
    
    return all_links

def extract_section(soup, section_title):
    """Extracts a specific section from job description"""
    section = []
    
    # Try to find the section by heading
    for heading in ['h2', 'h3', 'h4', 'strong', 'b']:
        elements = soup.find_all(heading, string=lambda text: section_title.lower() in str(text).lower())
        for elem in elements:
            next_node = elem.next_sibling
            while next_node:
                if next_node.name == 'ul':
                    section.extend([li.get_text(strip=True) for li in next_node.find_all('li')])
                elif next_node.name == 'p':
                    section.append(next_node.get_text(strip=True))
                elif next_node.name in ['h2', 'h3', 'h4']:
                    break
                next_node = next_node.next_sibling
    
    return section if section else []

def scrape_job_details(driver, url):
    """Scrape detailed job information from a single job page"""
    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "body")))
        time.sleep(1)  # Allow JavaScript to execute
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        job = {
            "job_url": url,
            "title": soup.select_one('h1').get_text(strip=True) if soup.select_one('h1') else "N/A",
            "company": soup.select_one('.company').get_text(strip=True) if soup.select_one('.company') else "N/A",
            "location": soup.select_one('.location').get_text(strip=True) if soup.select_one('.location') else "N/A",
            "description": "",
            "key_responsibilities": extract_section(soup, "Key Responsibilities") or extract_section(soup, "Responsibilities"),
            "requirements": extract_section(soup, "Requirements") or extract_section(soup, "Qualifications"),
            "benefits": extract_section(soup, "Benefits"),
            "full_text": soup.get_text(strip=True, separator='\n')
        }
        
        # Get main description
        description_section = soup.select_one('.job-description, .content, #job_body, .desc')
        if description_section:
            job["description"] = description_section.get_text(strip=True, separator='\n')
        
        return job
    
    except Exception as e:
        print(f"Error scraping job {url}: {e}")
        return None

def main():
    driver = init_driver()
    try:
        print("Starting Careerjet Egypt scraper...")
        
        # Step 1: Get all job links
        print("Collecting job links...")
        job_links = scrape_job_links(driver, BASE_URL, MAX_PAGES)
        print(f"Found {len(job_links)} job links")
        
        # Step 2: Scrape detailed job information
        jobs_data = []
        for i, link in enumerate(job_links):
            print(f"Processing job {i+1}/{len(job_links)}: {link}")
            job_data = scrape_job_details(driver, link)
            if job_data:
                jobs_data.append(job_data)
            time.sleep(1)  # Be polite between requests
        
        # Step 3: Save results
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(jobs_data, f, ensure_ascii=False, indent=2)
        
        print(f"Successfully saved {len(jobs_data)} jobs to {OUTPUT_FILE}")
    
    finally:
        driver.quit()
        print("Browser closed")

if __name__ == "__main__":
    main()