import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('postgresql://postgres:1234@localhost:5432/vitalwatch_db')

print("Loading data...")
df = pd.read_sql("SELECT * FROM patient_vitals", engine)

# Check what column names actually exist in your DB
print("Your columns are:", list(df.columns))
print(f"Loaded {len(df):,} rows")

# Standardize: strip spaces, lowercase everything consistently
df.columns = [col.strip().lower() for col in df.columns]
print("Columns after fix:", list(df.columns[:5]), "...")

all_patients = []
total = df['patient_id'].nunique()
print(f"Total unique patients: {total}")

for i, (patient_id, group) in enumerate(df.groupby('patient_id')):
    g = group.copy().sort_values('iculos')  # lowercase now

    # Rolling averages
    g['hr_mean_3h']   = g['hr'].rolling(window=3,  min_periods=1).mean()
    g['hr_mean_6h']   = g['hr'].rolling(window=6,  min_periods=1).mean()
    g['sbp_mean_3h']  = g['sbp'].rolling(window=3, min_periods=1).mean()
    g['map_mean_3h']  = g['map'].rolling(window=3, min_periods=1).mean()
    g['resp_mean_3h'] = g['resp'].rolling(window=3, min_periods=1).mean()

    # Rate of change (trends)
    g['hr_trend']   = g['hr'].diff()
    g['sbp_trend']  = g['sbp'].diff()
    g['temp_trend'] = g['temp'].diff()
    g['resp_trend'] = g['resp'].diff()

    # Volatility
    g['hr_std_3h']  = g['hr'].rolling(window=3,  min_periods=1).std().fillna(0)
    g['sbp_std_3h'] = g['sbp'].rolling(window=3, min_periods=1).std().fillna(0)

    # Clinical alarm flags
    g['flag_tachy']     = (g['hr']    > 100).astype(int)
    g['flag_hypoxia']   = (g['o2sat'] < 92).astype(int)
    g['flag_fever']     = (g['temp']  > 38.3).astype(int)
    g['flag_hypotemp']  = (g['temp']  < 36.0).astype(int)
    g['flag_low_bp']    = (g['map']   < 65).astype(int)
    g['flag_tachypnea'] = (g['resp']  > 22).astype(int)

    # qSOFA score
    g['qsofa'] = g['flag_tachypnea'] + g['flag_low_bp']
    g['hours_in_icu'] = g['iculos']

    all_patients.append(g)

    if (i + 1) % 2000 == 0:
        print(f"  Processed {i+1} / {total} patients...")

print("Combining all patients...")
df_features = pd.concat(all_patients, ignore_index=True)
df_features = df_features.fillna(0)

print(f"Final shape: {df_features.shape}")
print(f"Columns now: {df_features.shape[1]}")

print("Saving to database as 'patient_features'...")
df_features.to_sql('patient_features', engine, if_exists='replace', index=False)
print("SUCCESS! Table 'patient_features' is ready in pgAdmin.")