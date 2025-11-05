"""
Helper script to download the household power consumption dataset
"""

import urllib.request
import zipfile
import os

def download_dataset():
    """Download and extract the household power consumption dataset"""
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
    zip_filename = "household_power_consumption.zip"
    
    print("Downloading dataset from UCI ML Repository...")
    print(f"URL: {url}")
    
    try:
        # Download the file
        urllib.request.urlretrieve(url, zip_filename)
        print(f"✅ Downloaded: {zip_filename}")
        
        # Extract the zip file
        print("Extracting...")
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall('.')
        
        print("✅ Extracted: household_power_consumption.txt")
        
        # Rename .txt to .csv for consistency
        if os.path.exists('household_power_consumption.txt'):
            os.rename('household_power_consumption.txt', 'household_power_consumption.csv')
            print("✅ Renamed to: household_power_consumption.csv")
        
        # Clean up zip file
        os.remove(zip_filename)
        print("✅ Cleaned up zip file")
        
        print("\n" + "="*60)
        print("Dataset downloaded successfully!")
        print("You can now run: python run_pipeline.py")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\nManual download instructions:")
        print("1. Visit: https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption")
        print("2. Download the dataset")
        print("3. Extract and place 'household_power_consumption.txt' in this directory")
        print("4. Rename it to 'household_power_consumption.csv'")


if __name__ == "__main__":
    download_dataset()
