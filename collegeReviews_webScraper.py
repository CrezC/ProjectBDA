
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import pandas as pd
import time

class WebScraper_CR:
    def __init__(self, driver_path):
        """
        Initializes the WebScraper with the path to the Selenium WebDriver.

        Args:
            driver_path (str): Path to the Chrome WebDriver executable.
        """
        self.driver_path = driver_path
        self.service = Service(self.driver_path)
        self.driver = webdriver.Chrome(service=self.service)
        self.all_reviews = []

    def scrape_reviews(self, universities_df):
        """
        Scrapes reviews for a list of universities from a predefined website.

        Args:
            universities_df (pd.DataFrame): DataFrame containing a column 'University' with university names.

        Returns:
            None: Reviews are stored in the instance variable `self.all_reviews`.
        """
        for index, row in universities_df.iterrows():
            university_name = row['University']
            formatted_university_name = university_name.lower().replace(" ", "-")
            url = f"https://www.gradreports.com/colleges/{formatted_university_name}"
            print(f"Scraping reviews for: {university_name} ({url})")

            try:
                self.driver.get(url)
                time.sleep(3)
                reviews = self.driver.find_elements(By.CLASS_NAME, "reviews__item")
                for review in reviews:
                    try:
                        review_text = review.find_element(By.CLASS_NAME, "reviews__text").text
                        self.all_reviews.append({'University': university_name, 'Review': review_text})
                    except Exception as e:
                        print(f"Error extracting review for {university_name}: {e}")
                        continue
            except Exception as e:
                print(f"Error accessing URL for {university_name}: {e}")
                continue

    def save_reviews(self, output_file):
        """
        Saves the scraped reviews to an Excel file.

        Args:
            output_file (str): Path to the output Excel file.

        Returns:
            None
        """
        reviews_df = pd.DataFrame(data=self.all_reviews)
        reviews_df.to_excel(output_file, index=False)
        print(f"Reviews saved to {output_file}")

    def close(self):
        """
        Closes the Selenium WebDriver.

        Args:
            None

        Returns:
            None
        """
        self.driver.quit()
