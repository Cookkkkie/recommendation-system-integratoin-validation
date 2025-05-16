import pandas as pd
from surprise import Dataset, Reader, KNNBaseline

df = pd.read_parquet("../dataset/train.parquet")
test_df = pd.read_parquet("../dataset/test.parquet")

common_users = pd.Index(df["user_id"]).intersection(test_df["user_id"])

df = df[df["user_id"].isin(common_users)].copy()
test_df = test_df[test_df["user_id"].isin(common_users)].copy()
test_df.to_parquet("../dataset/test_aug.parquet", index=False)

user_ids = df["user_id"].unique()


reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[["user_id", "parent_asin", "rating"]], reader)

trainset = data.build_full_trainset()

# To set user-based CF, set "user_based": True
sim_options = {"name": "cosine", "user_based": False}
algo = KNNBaseline(sim_options=sim_options, k=25)
algo.fit(trainset)


def get_top_n(user_id, n=5):
    inner_uid = trainset.to_inner_uid(user_id)

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
    top10 = get_top_n(user_id, n=10)
    for asin in top10:
        rows.append({"user_id": user_id, "recommended_asin": asin})
#For future evaluation keep name all_user_recs_cf_ib for Item-Based and all_user_recs_cf_ub for User-Based
recs_df = pd.DataFrame(rows).to_parquet(
    "../dataset/all_user_recs_cf_ib.parquet", index=False
)
