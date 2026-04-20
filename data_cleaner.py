import pandas as pd
import glob
import os
if not os.path.exists('data/processed'):
    os.makedirs('data/processed')

all_files = glob.glob('data/raw/training/*.psv')

for i in range(len(all_files)):
    file_path = all_files[i]
    file_name = os.path.basename(file_path) 
    
    df = pd.read_csv(file_path, sep='|')
    
    df_cleaned = df.ffill().bfill()
    df_cleaned = df_cleaned.fillna(0)
    avg_hr = df_cleaned['HR'].mean()
        
    print(f"Patient {i+1} ({file_name}): Avg HR = {avg_hr:.2f} | Missing Values = {df_cleaned.isnull().sum().sum()}")
    output_path = f'data/processed/{file_name}'
    df_cleaned.to_csv(output_path, index=False)
        
    print(f"Cleaned and saved: {file_name}")