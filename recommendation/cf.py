import pandas as pd
from surprise import Dataset, Reader, KNNBasic

df = pd.read_parquet('../dataset/train.parquet')
test_df = pd.read_parquet("../dataset/test.parquet")

common_users = pd.Index(df['user_id']).intersection(test_df['user_id'])

df = df[ df['user_id'].isin(common_users) ].copy()
test_df  = test_df [ test_df ['user_id'].isin(common_users) ].copy()
test_df.to_parquet("../dataset/test_aug.parquet", index=False)
print("success")
user_ids = df['user_id'].unique()
print(len(user_ids))

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'parent_asin', 'rating']], reader)

trainset = data.build_full_trainset()

sim_options = {
    'name': 'cosine',
    'user_based': False
}
algo = KNNBasic(sim_options=sim_options)
algo.fit(trainset)


def get_top_n(user_id, n=5):
    try:
        inner_uid = trainset.to_inner_uid(user_id)
    except ValueError:
        raise ValueError(f"User {user_id} not found in the training data.")

    all_inner_iids = set(trainset.all_items())
    rated_inner_iids = set(iid for (iid, _) in trainset.ur[inner_uid])

    candidates = all_inner_iids - rated_inner_iids

    predictions = []
    for iid in candidates:
        raw_iid = trainset.to_raw_iid(iid)
        pred = algo.predict(user_id, raw_iid)
        predictions.append((raw_iid, pred.est))

    top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
    return [iid for (iid, _) in top_n]

rows = []
for user_id in user_ids:
    try:
        top5 = get_top_n(user_id, n=10)
    except ValueError:
        continue
    for asin in top5:
        rows.append({
            'user_id': user_id,
            'recommended_asin': asin
        })
recs_df = pd.DataFrame(rows).to_parquet('../dataset/all_user_recs_cf.parquet', index=False)

