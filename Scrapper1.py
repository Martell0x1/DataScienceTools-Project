from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import json
import time

def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    service = Service(executable_path='/usr/bin/chromedriver')
    return webdriver.Chrome(service=service, options=chrome_options)

def get_job_links(driver, page_num):
    """Get all job links from a search results page"""
    search_url = f"https://wuzzuf.net/search/jobs/?a=navbg%7Cspbg&q=Cyber%20Security%20Engineer"
    driver.get(search_url)
    
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div.e1v1l3u10"))
    )
    
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    job_cards = soup.find_all("div", class_="e1v1l3u10")
    
    links = []
    for card in job_cards:
        link = card.find("a", class_="css-o171kl")
        if link and link.get("href"):
            links.append(link["href"])
    
    return links

def scrape_job_page(driver, url):
    """Scrape a single job page"""
    if not url.startswith("http"):
        url = "https://wuzzuf.net" + url
        
    driver.get(url)
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located(
            (By.CSS_SELECTOR, "section.css-3kx5e2, section.css-ghicub, div.css-3qn4oq")
        )
    )
    
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    job_data = {
        "title": extract_with_fallback(soup, ["h1.css-f9uh36", "h2.css-lu59jur"]),
        "company": extract_with_fallback(soup, ["a.css-1cxc9zk"]),
        "location": extract_with_fallback(soup, ["span.css-1ve4b75"]),
        "details": extract_job_details_section(soup),
        "description_info": extract_description_and_requirements(soup),
        "posted_date": extract_with_fallback(soup, ["span.css-182mrdn"]),
        "job_url": url
    }
    
    cleaned_data = {k: v for k, v in job_data.items() if v is not None}
    
    if "details" in cleaned_data:
        cleaned_data.update(cleaned_data.pop("details"))
    if "description_info" in cleaned_data:
        cleaned_data.update(cleaned_data.pop("description_info"))
    
    return cleaned_data

def extract_job_details_section(soup):
    """Extracts key details and skills from the job details section"""
    details_section = soup.find("section", class_="css-3kx5e2")
    if not details_section:
        return None
    
    details = {}
    
    for row in details_section.find_all("div", class_="css-rcl8e5"):
        label_span = row.find("span", class_=lambda c: c and ("wn0avc" in c or "wm@ave" in c))
        value_span = row.find("span", class_="css-4xky9y")
        
        if not (label_span and value_span):
            continue
            
        label = label_span.get_text(strip=True).replace(":", "").lower()
        value = value_span.get_text(strip=True)
        
        if "experience" in label:
            details["experience"] = value
        elif "education" in label:
            details["education"] = value
        elif "salary" in label:
            details["salary"] = value
    
    skills_header = details_section.find("h4", class_="css-1ott775", string="Skills And Tools:")
    if skills_header:
        skills_div = skills_header.find_next("div", class_="css-qe7mba")
        if skills_div:
            skills = [skill.get_text(strip=True) 
                     for skill in skills_div.find_all("a", class_="css-g65o95")]
            if skills:
                details["skills"] = ", ".join(skills)
    
    return details

def extract_description_and_requirements(soup):
    """Extract both description and requirements from their sections"""
    result = {}
    
    sections = soup.find_all("section", class_="css-ghicub")
    
    for section in sections:
        description_header = section.find("h2", class_="css-fwj1k5", string=lambda t: t and "Job Description" in t)
        if description_header:
            content_div = section.find("div", class_="css-1uobp1k")
            if content_div:
                result["description"] = content_div.get_text(separator="\n").strip()
        
        requirements_header = section.find("h2", class_="css-fwj1k5", string=lambda t: t and "Job Requirements" in t)
        if requirements_header:
            content_div = section.find("div", class_=lambda c: c and "css-1t5f0fr" in c)
            if content_div:
                result["requirements"] = content_div.get_text(separator="\n").strip()
    
    return result if result else None

def extract_with_fallback(soup, selectors):
    for selector in selectors:
        element = soup.select_one(selector)
        if element:
            return element.get_text(strip=True)
    return None

def main():
    driver = setup_driver()
    all_jobs = []
    
    try:
        for page_num in range(13):
            print(f"Scraping page {page_num + 1}...")
            job_links = get_job_links(driver, page_num)
            
            for link in job_links:
                try:
                    job_data = scrape_job_page(driver, link)
                    if job_data:
                        all_jobs.append(job_data)
                        print(f"Scraped: {job_data['title']}")
                    time.sleep(2)  # to prevent ip blocking
                except Exception as e:
                    print(f"Error scraping {link}: {str(e)}")
                    continue
    
    finally:
        driver.quit()
    
    # Save all jobs to file
    if all_jobs:
        with open("dataScient.json", "w", encoding="utf-8") as f:
            json.dump(all_jobs, f, ensure_ascii=False, indent=2)
        print(f"\nSuccessfully saved {len(all_jobs)} jobs to wuzzuf_jobs_2pages.json")
    else:
        print("\nNo jobs were scraped")

if __name__ == "__main__":
    main()
