import pandas as pd
from AutoClean import AutoClean

dataset_path = '../dataset/interactions/Appliances.jsonl'
output_clean = '../dataset/cleaned_interactions.parquet'

RECENT_YEARS = 11
MIN_INTERACTIONS = 5
MAX_DROP_RATIO = 0.35


df = pd.read_json(dataset_path, lines=True)[['user_id', 'parent_asin', 'rating', 'timestamp']].sort_values('timestamp', ascending=True)


df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
cutoff = pd.Timestamp.now() - pd.DateOffset(years=RECENT_YEARS)
df = df[df['timestamp'] >= cutoff]

df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df = df[df['rating'] >= 3]

user_freq = df['user_id'].value_counts()
item_freq = df['parent_asin'].value_counts()

low_u = user_freq[user_freq < MIN_INTERACTIONS].index
low_i = item_freq[item_freq < MIN_INTERACTIONS].index

u_ratio = df['user_id'].isin(low_u).mean()
i_ratio = df['parent_asin'].isin(low_i).mean()


if u_ratio < MAX_DROP_RATIO:
    df = df[~df['user_id'].isin(low_u)]
if i_ratio < MAX_DROP_RATIO:
    df = df[~df['parent_asin'].isin(low_i)]
else:
    df_u = df[~df['user_id'].isin(low_u)]
    df_i = df[~df['parent_asin'].isin(low_i)]
    df_u.to_parquet("user.parquet", index = False)
    df_i.to_parquet("item.parquet", index = False)

n_users, n_items = df['user_id'].nunique(), df['parent_asin'].nunique()
if n_users > n_items:
    print(f'Dataset has more users({n_users}) than items({n_items}): item-based CF is recommended.')
else:
    print(f'Dataset has less users({n_users}) than items({n_items}): user-based CF is recommended.')
train_end = cutoff + pd.DateOffset(years=5)
test_end  = train_end + pd.DateOffset(months = 2)

df_train = df[df['timestamp'] < train_end]
df_test = df[(df['timestamp'] >= train_end) & (df['timestamp'] < test_end)]

df_train = AutoClean(df_train).output
df_test = AutoClean(df_test).output

print(f"Train: {df_train.shape[0]} rows, {df_train['user_id'].nunique()} users, "
      f"{df_train['parent_asin'].nunique()} items")
print(f"Test:  {df_test.shape[0]} rows, {df_test['user_id'].nunique()} users, "
      f"{df_test['parent_asin'].nunique()} items")

df_train.to_parquet('../dataset/train.parquet', index=False)
df_test.to_parquet( '../dataset/test.parquet',  index=False)