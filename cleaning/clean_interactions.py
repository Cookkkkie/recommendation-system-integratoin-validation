import pandas as pd

dataset_path = '/home/oleksii/PycharmProjects/PythonProject/dataset/interactions/Appliances.jsonl'
output_clean = '/home/oleksii/PycharmProjects/PythonProject/dataset/cleaned_interactions.parquet'

RECENT_YEARS = 6
MIN_INTERACTIONS = 5
MAX_DROP_RATIO = 0.35


df = pd.read_json(dataset_path, lines=True)[['user_id', 'parent_asin', 'rating', 'timestamp']].sort_values('timestamp', ascending=True)


df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')

seven_years_ago = pd.Timestamp.now() - pd.DateOffset(years=RECENT_YEARS)
df = df[df['timestamp'] >= seven_years_ago]


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

n_users, n_items = df['user_id'].nunique(), df['parent_asin'].nunique()
if n_users > n_items:
    print('Dataset has more users than items: item-based CF is recommended.')
else:
    print('Dataset has more items than users: user-based CF is recommended.')

train_end = seven_years_ago + pd.DateOffset(years=3)
test_end  = train_end + pd.DateOffset(years=2)

df_train = df[df['timestamp'] < train_end]
df_test = df[(df['timestamp'] >= train_end) & (df['timestamp'] < test_end)]

print(f"Train: {df_train.shape[0]} rows, {df_train['user_id'].nunique()} users, "
      f"{df_train['parent_asin'].nunique()} items")
print(f"Test:  {df_test.shape[0]} rows, {df_test['user_id'].nunique()} users, "
      f"{df_test['parent_asin'].nunique()} items")

df_train.to_parquet('../dataset/train.parquet', index=False)
df_test.to_parquet( '../dataset/test.parquet',  index=False)