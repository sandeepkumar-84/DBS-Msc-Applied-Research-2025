import requests
from bs4 import BeautifulSoup
import os

urls_dbs = [
    "https://www.dbs.ie/dbs-staff",
    "https://www.dbs.ie/about-dbs/academic-departments",
    "https://www.dbs.ie/about-dbs/our-academic-team"
]

headers = {'User-Agent': 'Mozilla/5.0'}

#in colab the file will be created in the default location. It will be changed in python code file to store it onto disk.
scraped_data_dir = "C:\content\Research-Chatbot"
scraped_data_file_path = "C:\content\Research-Chatbot\scraped_dbs.txt"

os.makedirs(scraped_data_dir, exist_ok=True)
# file in which the scraped data will be loaded
# data from 'h1', 'h2', 'h3', 'p', 'li' html tags will be extracted below.
with open(scraped_data_file_path, "w", encoding="utf-8") as f:
    for url in urls_dbs:
        print(f"Scraping: {url}")
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            for tag in soup.find_all(['h1', 'h2', 'h3', 'p', 'li']):
                text = tag.get_text(strip=True)
                if text:
                    f.write(f"{text}\n")
        else:
            print(f"Failed to fetch {url}: {response.status_code}")