"""
Google Trends Weekly Data Scraper (Fixed)

Google Trends returns:
- Daily data for ranges < 90 days
- Weekly data for ranges 90 days to ~5 months
- Monthly data for ranges > 5 months

To get WEEKLY data for 2008-2016, we need to scrape in ~4-5 month chunks
and stitch them together.

Install: pip install pytrends python-dateutil
"""

import pandas as pd
import numpy as np
import time
from pytrends.request import TrendReq
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')


def scrape_weekly_in_chunks(keywords: list,
                            start_date: str = '2008-01-01',
                            end_date: str = '2016-01-01',
                            geo: str = '',
                            chunk_months: int = 4,
                            sleep_time: int = 10) -> pd.DataFrame:
    """
    Scrape Google Trends in small chunks to get WEEKLY data.
    
    Parameters:
    -----------
    keywords : list
        List of search terms (max 5)
    start_date : str
        Start date 'YYYY-MM-DD'
    end_date : str
        End date 'YYYY-MM-DD'
    geo : str
        Geographic region ('' for worldwide)
    chunk_months : int
        Months per chunk (4-5 months gives weekly data)
    sleep_time : int
        Seconds between requests (avoid rate limits)
    
    Returns:
    --------
    DataFrame with weekly search interest
    """
    print("="*60)
    print("GOOGLE TRENDS WEEKLY SCRAPER")
    print("="*60)
    print(f"\nKeywords: {keywords}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Chunk size: {chunk_months} months")
    print(f"Sleep time: {sleep_time} seconds between requests")
    
    pytrends = TrendReq(hl='en-US', tz=360, retries=3, backoff_factor=0.5)
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    all_chunks = []
    current = start
    chunk_num = 0
    
    # Calculate total chunks
    total_months = (end.year - start.year) * 12 + (end.month - start.month)
    total_chunks = (total_months // chunk_months) + 1
    
    print(f"\nTotal chunks to fetch: ~{total_chunks}")
    print("-"*60)
    
    while current < end:
        chunk_num += 1
        
        # Create chunk (4-5 months to get weekly data)
        chunk_end = current + relativedelta(months=chunk_months)
        if chunk_end > end:
            chunk_end = end
        
        chunk_start_str = current.strftime('%Y-%m-%d')
        chunk_end_str = chunk_end.strftime('%Y-%m-%d')
        timeframe = f'{chunk_start_str} {chunk_end_str}'
        
        print(f"\n[{chunk_num}/{total_chunks}] Fetching: {timeframe}")
        
        success = False
        retries = 3
        
        for attempt in range(retries):
            try:
                pytrends.build_payload(
                    kw_list=keywords,
                    cat=0,
                    timeframe=timeframe,
                    geo=geo,
                    gprop=''
                )
                
                time.sleep(sleep_time)
                df_chunk = pytrends.interest_over_time()
                
                if not df_chunk.empty:
                    if 'isPartial' in df_chunk.columns:
                        df_chunk = df_chunk.drop('isPartial', axis=1)
                    
                    # Check if we got weekly data (more than 1 point per month)
                    points_per_month = len(df_chunk) / chunk_months
                    
                    if points_per_month >= 3:  # Weekly = ~4 points per month
                        print(f"  Got {len(df_chunk)} weekly data points")
                        all_chunks.append(df_chunk)
                        success = True
                        break
                    else:
                        print(f"  Got {len(df_chunk)} points (~{points_per_month:.1f}/month) - may be monthly")
                        # Still save it, better than nothing
                        all_chunks.append(df_chunk)
                        success = True
                        break
                else:
                    print(f"  No data returned")
                    
            except Exception as e:
                print(f"  Attempt {attempt+1} failed: {e}")
                time.sleep(sleep_time * 2)
        
        if not success:
            print(f"  WARNING: Could not get data for {timeframe}")
        
        if chunk_end >= end:
            break
        
        current = chunk_end - timedelta(days=7)
    
    if not all_chunks:
        print("\nERROR: No data retrieved!")
        return pd.DataFrame()
    
    # Combine all chunks
    print("\n" + "-"*60)
    print("Combining chunks...")
    
    df_combined = pd.concat(all_chunks)
    
    # Remove duplicates (from overlapping chunks)
    df_combined = df_combined[~df_combined.index.duplicated(keep='first')]
    
    # Sort by date
    df_combined = df_combined.sort_index()
    
    # Verify we have weekly data
    if len(df_combined) > 0:
        date_diffs = df_combined.index.to_series().diff().dropna()
        median_diff = date_diffs.median().days
        print(f"\nMedian days between data points: {median_diff}")
        
        if median_diff <= 8:
            print("Confirmed: WEEKLY data")
        else:
            print(f"WARNING: Data appears to be {median_diff}-day intervals")
    
    print(f"\nFinal combined data:")
    print(f"  Total data points: {len(df_combined)}")
    print(f"  Date range: {df_combined.index.min()} to {df_combined.index.max()}")
    
    return df_combined


def main():
    """Main function to scrape and save weekly Google Trends data."""
    
    import os
    os.makedirs('data', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Keywords
    keywords = ['Zara Dress', 'Chanel Bag']
    
    # Date range matching SFS data
    start_date = '2008-01-01'
    end_date = '2016-01-01'
    
    # Scrape weekly data
    df = scrape_weekly_in_chunks(
        keywords=keywords,
        start_date=start_date,
        end_date=end_date,
        geo='',  # Worldwide
        chunk_months=4,  # 4 months per chunk to ensure weekly data
        sleep_time=10  # 10 seconds between requests
    )
    
    if df.empty:
        print("\nFailed to retrieve data.")
        print("Google Trends has rate limits - wait a few minutes and retry.")
        return None
    
    # Rename columns
    df = df.rename(columns={
        'Zara Dress': 'zara_search_interest',
        'Chanel Bag': 'chanel_search_interest'
    })
    
    # Ensure datetime index
    df.index = pd.to_datetime(df.index)
    df.index.name = 'date'
    
    # Print summary
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"\nTotal weeks: {len(df)}")
    print(f"Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    
    # Calculate expected weeks
    expected_weeks = (datetime.strptime(end_date, '%Y-%m-%d') - 
                      datetime.strptime(start_date, '%Y-%m-%d')).days // 7
    print(f"Expected weeks: ~{expected_weeks}")
    print(f"Coverage: {len(df) / expected_weeks * 100:.1f}%")
    
    print(f"\nZara Dress search interest:")
    print(df['zara_search_interest'].describe())
    
    print(f"\nChanel Bag search interest:")
    print(df['chanel_search_interest'].describe())
    
    # Sample data
    print("\n" + "="*60)
    print("SAMPLE DATA (first 20 rows)")
    print("="*60)
    print(df.head(20))
    
    # Save to CSV
    output_path = 'data/google_trends_weekly.csv'
    df.to_csv(output_path)
    print(f"\nSaved to: {output_path}")
    
    # Plot
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        axes[0].plot(df.index, df['zara_search_interest'], 
                     color='#2A9D8F', linewidth=1)
        axes[0].set_title('Zara Dress - Weekly Google Search Interest (2008-2016)', 
                          fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Search Interest (0-100)')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(df.index, df['chanel_search_interest'], 
                     color='#E76F51', linewidth=1)
        axes[1].set_title('Chanel Bag - Weekly Google Search Interest (2008-2016)', 
                          fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Search Interest (0-100)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/google_trends_weekly.png', dpi=150)
        print(f"Plot saved to: plots/google_trends_weekly.png")
        plt.show()
        
    except Exception as e:
        print(f"Could not create plot: {e}")
    
    return df


if __name__ == "__main__":
    df = main()