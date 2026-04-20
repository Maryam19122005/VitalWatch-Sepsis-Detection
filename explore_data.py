import pandas as pd
import glob
import os

path = 'data/raw/training/*.psv'
all_files = glob.glob(path)

if not all_files:
    print("Error: Still can't find the files. Check if the folder name is exactly 'training'.")
else:
    print(f"Success: Found {len(all_files)} patient records!")
    df = pd.read_csv(all_files[0], sep='|')

    print(f"\n--- Data Preview for Patient: {os.path.basename(all_files[0])} ---")
    print(df[['HR', 'Temp', 'O2Sat', 'SepsisLabel']].head(10))

    print("\n--- Missing Values Count ---")
    print(df.isnull().sum())