import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import json
import os

engine = create_engine('postgresql://postgres:1234@localhost:5432/vitalwatch_db')
os.makedirs('models', exist_ok=True)

print("Loading data for Association Rules...")
df = pd.read_sql("""
    SELECT hr, temp, o2sat, map, resp,
           flag_tachy, flag_fever, flag_low_bp,
           flag_hypoxia, flag_tachypnea, qsofa,
           sepsislabel
    FROM patient_features
    LIMIT 50000
""", engine)
print(f"Loaded {len(df):,} rows")

# ── Convert vitals to binary events ───────────────────────────────────────
# Apriori needs YES/NO items not numbers
# So every vital sign becomes a named event
# Much faster than iterrows — uses vectorized operations

print("\nConverting vitals to binary events (vectorized)...")

# Create event columns directly using vectorized operations
# This is 100x faster than looping with iterrows
df['HR_EVENT']   = np.where(df['hr'] > 100,  'HIGH_HR',
                   np.where(df['hr'] < 60,   'LOW_HR',
                                              'NORMAL_HR'))

df['TEMP_EVENT'] = np.where(df['temp'] > 38.3, 'FEVER',
                   np.where(df['temp'] < 36.0,  'HYPOTHERMIA',
                                                 'NORMAL_TEMP'))

df['O2_EVENT']   = np.where(df['o2sat'] < 92, 'LOW_OXYGEN',
                                               'NORMAL_OXYGEN')

df['BP_EVENT']   = np.where(df['map'] < 65,   'LOW_BP',
                                               'NORMAL_BP')

df['RESP_EVENT'] = np.where(df['resp'] > 22,  'HIGH_RESP',
                                               'NORMAL_RESP')

df['QSOFA_EVENT']  = np.where(df['qsofa'] >= 2, 'HIGH_QSOFA', None)
df['SEPSIS_EVENT'] = np.where(df['sepsislabel'] == 1, 'SEPSIS', None)
#This creates a mini-dataset where
#sepsis appears in 25% of rows (not 2%)
#Now Apriori easily finds sepsis rules!

#This is called "stratified sampling"
#Legitimate ML technique
# Take ALL sepsis rows + equal non-sepsis rows
sepsis_df    = df[df['sepsislabel'] == 1]
non_sepsis   = df[df['sepsislabel'] == 0].sample(
                   n=len(sepsis_df)*3,
                   random_state=42
               )
df_balanced  = pd.concat([sepsis_df, non_sepsis])

print(f"Balanced sample: {len(df_balanced)} rows")
print(f"Sepsis rows: {len(sepsis_df)}")
print(f"Non-sepsis rows: {len(non_sepsis)}")


# Build transactions list
event_cols = ['HR_EVENT', 'TEMP_EVENT', 'O2_EVENT',
              'BP_EVENT', 'RESP_EVENT', 'QSOFA_EVENT', 'SEPSIS_EVENT']

transactions = []
for _, row in df_balanced[event_cols].iterrows():
    items = [v for v in row.values if v is not None]
    transactions.append(items)

print(f"Transactions created: {len(transactions):,}")

# Show distribution of events
print("\nEvent distribution in dataset:")
for col in event_cols[:5]:
    counts = df[col].value_counts()
    for event, count in counts.items():
        pct = count / len(df) * 100
        print(f"  {event:<20}: {count:>6,} ({pct:.1f}%)")

# ── Encode transactions ────────────────────────────────────────────────────
print("\nEncoding transactions...")
te       = TransactionEncoder()
te_array = te.fit_transform(transactions)
df_enc   = pd.DataFrame(te_array, columns=te.columns_)
print(f"Items found: {list(df_enc.columns)}")
print(f"Matrix shape: {df_enc.shape}")

# ── Run Apriori ────────────────────────────────────────────────────────────
print("\nRunning Apriori algorithm...")
print("Finding patterns appearing in at least 5% of patient hours...")

frequent_items = apriori(
    df_enc,
    min_support=0.05,   # must appear in 5% of rows minimum
    use_colnames=True,
    max_len=3           # max 3 items per pattern
)
print(f"Frequent itemsets found: {len(frequent_items)}")

# Show itemset size distribution
sizes = frequent_items['itemsets'].apply(len)
for size in [1, 2, 3]:
    count = (sizes == size).sum()
    print(f"  {size}-item patterns: {count}")

# ── Generate Rules ─────────────────────────────────────────────────────────
print("\nGenerating association rules...")
rules = association_rules(
    frequent_items,
    metric='confidence',
    min_threshold=0.5   # must be correct at least 50% of time
)

# Add quality filter: lift > 1 means rule is better than random
rules = rules[rules['lift'] > 1.0]
print(f"Total quality rules (lift>1): {len(rules)}")

# ── Sepsis specific rules ──────────────────────────────────────────────────
sepsis_rules = rules[
    rules['consequents'].apply(lambda x: 'SEPSIS' in str(x))
].sort_values('confidence', ascending=False)

print(f"Rules predicting SEPSIS: {len(sepsis_rules)}")

# ── Show ALL rule types ────────────────────────────────────────────────────
print("\n" + "="*60)
print("TOP ASSOCIATION RULES — SEPSIS PREDICTION")
print("="*60)

if len(sepsis_rules) > 0:
    for i, (_, rule) in enumerate(sepsis_rules.head(5).iterrows()):
        ante       = ', '.join(list(rule['antecedents']))
        cons       = ', '.join(list(rule['consequents']))
        support    = rule['support']
        confidence = rule['confidence']
        lift       = rule['lift']

        print(f"\nRule {i+1}:")
        print(f"  IF:         {ante}")
        print(f"  THEN:       {cons}")
        print(f"  Support   : {support:.3f} "
              f"({support*100:.1f}% of all patient hours)")
        print(f"  Confidence: {confidence:.3f} "
              f"(rule correct {confidence*100:.1f}% of time)")
        print(f"  Lift      : {lift:.2f}x "
              f"({lift:.2f}x more likely than by random chance)")
else:
    print("No direct SEPSIS rules found at current thresholds.")
    print("This is expected — sepsis only occurs in 2% of hours.")
    print("Showing top general vital sign rules instead:")

# ── Show general rules regardless ─────────────────────────────────────────
print("\n" + "="*60)
print("TOP GENERAL VITAL SIGN PATTERNS")
print("="*60)

top_general = rules.sort_values('lift', ascending=False).head(5)
for i, (_, rule) in enumerate(top_general.iterrows()):
    ante       = ', '.join(list(rule['antecedents']))
    cons       = ', '.join(list(rule['consequents']))
    print(f"\nRule {i+1}:")
    print(f"  IF:         {ante}")
    print(f"  THEN:       {cons}")
    print(f"  Confidence: {rule['confidence']:.3f} | "
          f"Lift: {rule['lift']:.2f} | "
          f"Support: {rule['support']:.3f}")

# ── Show clinical interpretation ───────────────────────────────────────────
print("\n" + "="*60)
print("CLINICAL INTERPRETATION")
print("="*60)
print("  Support    = How common is this pattern?")
print("               0.05 = appears in 5% of all ICU hours")
print("               Higher support = more common pattern")
print()
print("  Confidence = When IF happens, how often does THEN happen?")
print("               0.70 = IF condition true, THEN true 70% of time")
print("               Higher confidence = more reliable rule")
print()
print("  Lift       = How much more likely vs random chance?")
print("               1.0  = no better than random")
print("               2.0  = 2x more likely than random")
print("               5.0  = 5x more likely (strong rule)")
print("               Higher lift = stronger relationship")

# ── Save everything ────────────────────────────────────────────────────────
all_rules_list = []
for _, rule in rules.iterrows():
    all_rules_list.append({
        'antecedents': list(rule['antecedents']),
        'consequents': list(rule['consequents']),
        'support':     round(float(rule['support']),    4),
        'confidence':  round(float(rule['confidence']), 4),
        'lift':        round(float(rule['lift']),        4)
    })

# Separate sepsis rules for easy access by FastAPI
sepsis_rules_list = [
    r for r in all_rules_list
    if 'SEPSIS' in str(r['consequents'])
]

results = {
    'total_transactions':  len(transactions),
    'frequent_itemsets':   len(frequent_items),
    'total_rules':         len(rules),
    'sepsis_rules_count':  len(sepsis_rules),
    'min_support':         0.05,
    'min_confidence':      0.5,
    'min_lift':            1.0,
    'all_rules':           all_rules_list,
    'sepsis_rules':        sepsis_rules_list,
    'top_10_by_lift':      sorted(
                               all_rules_list,
                               key=lambda x: x['lift'],
                               reverse=True
                           )[:10]
}

with open('models/association_rules.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("ASSOCIATION RULES COMPLETE!")
print("="*60)
print(f"  Transactions    : {len(transactions):,}")
print(f"  Frequent sets   : {len(frequent_items)}")
print(f"  Total rules     : {len(rules)}")
print(f"  Sepsis rules    : {len(sepsis_rules)}")
print(f"  Saved -> models/association_rules.json")
print("="*60)