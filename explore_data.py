import pandas as pd
import glob
import os

# 1. Path to your new files
path = 'data/raw/training/*.psv'
all_files = glob.glob(path)

if not all_files:
    print("❌ Still can't find the files. Check if the folder name is exactly 'training'.")
else:
    print(f"✅ Found {len(all_files)} patient records!")

    # 2. Load the very first patient file
    # Use sep='|' because PSV stands for Pipe Separated Values
    df = pd.read_csv(all_files[0], sep='|')

    print(f"\n--- Data Preview for Patient: {os.path.basename(all_files[0])} ---")
    print(df[['HR', 'Temp', 'O2Sat', 'SepsisLabel']].head(10))
    
    # 3. Check for missing data (important for your project report!)
    print("\n--- Missing Values Count ---")
    print(df.isnull().sum())