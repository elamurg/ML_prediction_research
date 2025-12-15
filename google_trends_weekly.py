"""
Google Trends Weekly Data Scraper

This script scrapes weekly Google Trends data for Zara Dress and Chanel Bag
from 2008 to 2016 using pytrends.

NOTE: Google Trends API has rate limits. This script includes:
- Chunked requests to avoid timeouts
- Sleep delays between requests
- Retry logic for failed requests

Install: pip install pytrends
"""

import pandas as pd
import time
from pytrends.request import TrendReq
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def scrape_google_trends_weekly(keywords: list, 
                                 start_date: str = '2008-01-01',
                                 end_date: str = '2016-01-01',
                                 geo: str = '',
                                 sleep_time: int = 5) -> pd.DataFrame:
    """
    Scrape weekly Google Trends data for given keywords.
    
    Parameters:
    -----------
    keywords : list
        List of search terms (max 5)
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    geo : str
        Geographic region ('' for worldwide, 'US' for United States)
    sleep_time : int
        Seconds to sleep between requests
    
    Returns:
    --------
    DataFrame with weekly search interest
    """
    print("="*60)
    print("GOOGLE TRENDS WEEKLY SCRAPER")
    print("="*60)
    print(f"\nKeywords: {keywords}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Region: {'Worldwide' if geo == '' else geo}")
    
    pytrends = TrendReq(hl='en-US', tz=360)
    
    timeframe = f'{start_date} {end_date}'
    
    print(f"\nRequesting data for timeframe: {timeframe}")
    print("This may take a moment...")
    
    try:
        pytrends.build_payload(
            kw_list=keywords,
            cat=0,  # All categories
            timeframe=timeframe,
            geo=geo,
            gprop=''  # Web search
        )
        
        time.sleep(sleep_time)
        df = pytrends.interest_over_time()
        
        if df.empty:
            print("WARNING: No data returned. Trying chunked approach...")
            return scrape_google_trends_chunked(keywords, start_date, end_date, geo, sleep_time)
      
        if 'isPartial' in df.columns:
            df = df.drop('isPartial', axis=1)
        
        print(f"\nData retrieved successfully!")
        print(f"Shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        
        return df
        
    except Exception as e:
        print(f"Error: {e}")
        print("Trying chunked approach...")
        return scrape_google_trends_chunked(keywords, start_date, end_date, geo, sleep_time)


def scrape_google_trends_chunked(keywords: list,
                                  start_date: str = '2008-01-01',
                                  end_date: str = '2016-01-01',
                                  geo: str = '',
                                  sleep_time: int = 10) -> pd.DataFrame:
    """
    Scrape Google Trends in chunks to avoid rate limits.
    
    Splits the date range into yearly chunks and combines results.
    """
    print("\n" + "="*60)
    print("CHUNKED SCRAPING (by year)")
    print("="*60)
    
    pytrends = TrendReq(hl='en-US', tz=360)
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    all_data = []
    current = start
    
    while current < end:
        chunk_end = min(current + timedelta(days=365), end)
        
        chunk_start_str = current.strftime('%Y-%m-%d')
        chunk_end_str = chunk_end.strftime('%Y-%m-%d')
        timeframe = f'{chunk_start_str} {chunk_end_str}'
        
        print(f"\nFetching: {timeframe}")
        
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
                all_data.append(df_chunk)
                print(f"  Retrieved {len(df_chunk)} data points")
            else:
                print(f"  No data for this period")
                
        except Exception as e:
            print(f"  Error: {e}")
            print(f"  Retrying after longer sleep...")
            time.sleep(sleep_time * 2)
            
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
                    all_data.append(df_chunk)
                    print(f"  Retry successful: {len(df_chunk)} data points")
            except Exception as e2:
                print(f"  Retry failed: {e2}")
        
        current = chunk_end
    
    if not all_data:
        print("\nERROR: No data retrieved!")
        return pd.DataFrame()
    
    df_combined = pd.concat(all_data)

    df_combined = df_combined[~df_combined.index.duplicated(keep='first')]

    df_combined = df_combined.sort_index()
    
    print(f"\nCombined data:")
    print(f"  Shape: {df_combined.shape}")
    print(f"  Date range: {df_combined.index.min()} to {df_combined.index.max()}")
    
    return df_combined


def normalize_trends_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Google Trends data to 0-100 scale.
    
    Google Trends data from different time chunks may have different scales.
    This normalizes everything to a consistent 0-100 scale.
    """
    df_normalized = df.copy()
    
    for col in df_normalized.columns:
        max_val = df_normalized[col].max()
        if max_val > 0:
            df_normalized[col] = (df_normalized[col] / max_val) * 100
    
    return df_normalized


def main():
    """Main function to scrape and save Google Trends data."""

    keywords = ['Zara Dress', 'Chanel Bag']
 
    start_date = '2008-01-01'
    end_date = '2016-01-01'
    
    print("\n" + "="*60)
    print("SCRAPING GOOGLE TRENDS DATA")
    print("="*60)
    
    df = scrape_google_trends_weekly(
        keywords=keywords,
        start_date=start_date,
        end_date=end_date,
        geo='',  # Worldwide
        sleep_time=5
    )
    
    if df.empty:
        print("\nFailed to retrieve data. Please try again later.")
        print("Google Trends has rate limits - wait a few minutes and retry.")
        return
    
    df = df.rename(columns={
        'Zara Dress': 'zara_search_interest',
        'Chanel Bag': 'chanel_search_interest'
    })
  
    df.index = pd.to_datetime(df.index)
    df.index.name = 'date'
   
    print("\n" + "="*60)
    print("SAMPLE DATA")
    print("="*60)
    print(df.head(20))
    
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"\nTotal weeks: {len(df)}")
    print(f"Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    print(f"\nZara Dress search interest:")
    print(f"  Min: {df['zara_search_interest'].min()}")
    print(f"  Max: {df['zara_search_interest'].max()}")
    print(f"  Mean: {df['zara_search_interest'].mean():.2f}")
    print(f"\nChanel Bag search interest:")
    print(f"  Min: {df['chanel_search_interest'].min()}")
    print(f"  Max: {df['chanel_search_interest'].max()}")
    print(f"  Mean: {df['chanel_search_interest'].mean():.2f}")
    
    output_path = 'data/google_trends_weekly.csv'
    df.to_csv(output_path)
    print(f"\nSaved to: {output_path}")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f'data/google_trends_weekly_{timestamp}.csv'
    df.to_csv(backup_path)
    print(f"Backup saved to: {backup_path}")
    
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        axes[0].plot(df.index, df['zara_search_interest'], 
                     color='#2A9D8F', linewidth=1.5)
        axes[0].set_title('Zara Dress - Weekly Google Search Interest', 
                          fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Search Interest (0-100)')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(df.index, df['chanel_search_interest'], 
                     color='#E76F51', linewidth=1.5)
        axes[1].set_title('Chanel Bag - Weekly Google Search Interest', 
                          fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Search Interest (0-100)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/google_trends_weekly.png', dpi=150)
        print(f"\nPlot saved to: plots/google_trends_weekly.png")
        plt.show()
        
    except Exception as e:
        print(f"\nCould not create plot: {e}")
    
    return df


if __name__ == "__main__":
    import os
    os.makedirs('data', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    df = main()