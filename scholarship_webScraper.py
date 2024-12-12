
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

class ScholarshipScraper:
    def __init__(self):
        """
        Initializes the ScholarshipScraper with base URL and headers.
        """
        self.base_url = "https://www.careeronestop.org/toolkit/training/find-scholarships.aspx?curPage={page}&studyplacefilter={state}"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }
        self.scholarships = []

    def scrape_scholarships(self, states):
        """
        Scrapes scholarship data for each state across all pages.

        Args:
            states (list): List of state names to scrape data for.

        Returns:
            None: Data is stored in `self.scholarships`.
        """
        for state in states:
            print(f"Fetching data for {state}...")
            page = 1
            while True:
                url = self.base_url.format(page=page, state=state)
                print(f"  Fetching page {page} for {state}...")
                response = requests.get(url, headers=self.headers)
                if response.status_code != 200:
                    print(f"  Failed to fetch page {page} for {state}. Status code: {response.status_code}")
                    break
                soup = BeautifulSoup(response.content, "html.parser")
                rows = soup.select("table.cos-table-responsive tbody tr")
                if not rows:
                    print(f"  No more scholarships found for {state}.")
                    break
                for row in rows:
                    scholarship = {
                        'State': state,
                        'Page': page,
                        'Award Name': row.select_one('td[headers="thAN"] a').get_text(strip=True) if row.select_one('td[headers="thAN"] a') else None,
                        'Level of Study': row.select_one('td[headers="thLOS"]').get_text(strip=True) if row.select_one('td[headers="thLOS"]') else None,
                        'Award Type': row.select_one('td[headers="thAT"]').get_text(strip=True) if row.select_one('td[headers="thAT"]') else None,
                        'Award Amount': row.select_one('td[headers="thAA"]').get_text(strip=True) if row.select_one('td[headers="thAA"]') else None,
                    }
                    self.scholarships.append(scholarship)
                page += 1
                time.sleep(1)

    def save_scholarships(self, output_file):
        """
        Saves the scraped scholarships data to a CSV file.

        Args:
            output_file (str): Path to the output file.

        Returns:
            None
        """
        df = pd.DataFrame(self.scholarships)
        df.to_csv(output_file, index=False)
        print(f"Scholarship data saved to {output_file}")
