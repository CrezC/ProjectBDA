
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class FreelancerAnalysis:
    def __init__(self, csv_file):
        """
        Initialize the FreelancerAnalysis class by loading the dataset and parsing dates.

        Args:
            csv_file (str): Path to the CSV file containing contract data.
        """
        self.df = pd.read_csv(csv_file)
        self.df['start_date'] = self.df['start_date'].apply(self.parse_mixed_dates)
        self.df['end_date'] = self.df['end_date'].apply(self.parse_mixed_dates)
        print("Data loaded and dates parsed successfully.")

    @staticmethod
    def parse_mixed_dates(date_str):
        """
        Parse dates that might be in different formats.

        Args:
            date_str (str): Date string to parse.

        Returns:
            pd.Timestamp: Parsed date or NaT if parsing fails.
        """
        if pd.isna(date_str):
            return pd.NaT

        # Try different date formats
        formats_to_try = [
            '%Y-%m-%d %H:%M:%S.%f',  # Format with milliseconds
            '%Y-%m-%d %H:%M:%S',     # Standard format
            '%Y-%m-%d'               # Just date
        ]
        
        for date_format in formats_to_try:
            try:
                return pd.to_datetime(date_str, format=date_format)
            except ValueError:
                continue
        
        # If none of the formats work, try pandas' flexible parser
        try:
            return pd.to_datetime(date_str)
        except:
            return pd.NaT

    def filter_data_by_year(self, start_year, end_year):
        """
        Filters the dataset for contracts within a specified year range.

        Args:
            start_year (int): The start year for filtering.
            end_year (int): The end year for filtering.

        Returns:
            pd.DataFrame: Filtered dataset containing only contracts within the specified year range.
        """
        mask = (self.df['start_date'].dt.year >= start_year) & (self.df['start_date'].dt.year <= end_year)
        return self.df[mask]

    @staticmethod
    def compute_yearly_growth(filtered_df):
        """
        Computes the year-over-year growth for contract counts.

        Args:
            filtered_df (pd.DataFrame): The filtered dataset for analysis.

        Returns:
            pd.Series: Yearly contract counts.
            pd.Series: Year-over-year growth percentages.
        """
        yearly_contracts = filtered_df.groupby(filtered_df['start_date'].dt.year).size()
        yearly_growth = yearly_contracts.pct_change() * 100
        return yearly_contracts, yearly_growth

    @staticmethod
    def plot_monthly_timeline(filtered_df):
        """
        Plots the monthly contract volume timeline.

        Args:
            filtered_df (pd.DataFrame): The filtered dataset for analysis.

        Returns:
            None
        """
        monthly_contracts = filtered_df.groupby(filtered_df['start_date'].dt.to_period('M')).size()
        plt.figure(figsize=(15, 6))
        monthly_contracts.plot(kind='line', marker='o', color='blue', linewidth=2)
        plt.title('Contract Volume Timeline (Monthly)', pad=20, fontsize=14)
        plt.xlabel('Time Period', fontsize=12)
        plt.ylabel('Number of Contracts', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

    @staticmethod
    def plot_yearly_summary(yearly_contracts, yearly_growth):
        """
        Plots the yearly contract summary with growth rates.

        Args:
            yearly_contracts (pd.Series): Yearly contract counts.
            yearly_growth (pd.Series): Year-over-year growth percentages.

        Returns:
            None
        """
        plt.figure(figsize=(15, 6))
        yearly_contracts.plot(kind='bar', color='skyblue')
        plt.title('Yearly Contract Volume', pad=20, fontsize=14)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Total Contracts', fontsize=12)

        # Add growth rates on top of bars
        for i, v in enumerate(yearly_contracts):
            if i > 0:  # Only for years after the first
                growth = yearly_growth[yearly_growth.index[i-1]]
                plt.text(i, v, f'{growth:+.1f}%', 
                         horizontalalignment='center', 
                         verticalalignment='bottom')
        plt.tight_layout()
        plt.show()
