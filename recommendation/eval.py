import pandas as pd
import numpy as np

df_recs_cbf = pd.read_parquet('../dataset/all_user_recs_cbf.parquet')
df_recs_cf = pd.read_parquet('../dataset/all_user_recs_cf.parquet')
df_recs_hybrid = pd.read_parquet('../dataset/all_user_recs_hybrid.parquet')
df_test = pd.read_parquet('../dataset/test_aug.parquet')

# Aggregate actual and predicted items per user
actuals = df_test.groupby('user_id')['parent_asin'].apply(list).to_dict()
predicteds_cf = df_recs_cf.groupby('user_id')['recommended_asin'].apply(list).to_dict()
predicteds_cbf = df_recs_cbf.groupby('user_id')['recommended_asin'].apply(list).to_dict()
predicteds_hybrid = df_recs_hybrid.groupby('user_id')['recommended_asin'].apply(list).to_dict()

def apk(actual, predicted, k):
    if not actual:
        return 0.0
    score = 0.0
    hits = 0
    for i, p in enumerate(predicted[:k]):
        if p in actual and p not in predicted[:i]:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(actual), k)

def mapk(actuals, predicteds, k):
    scores = []
    for user, act in actuals.items():
        pred = predicteds.get(user, [])
        scores.append(apk(act, pred, k))
    return np.mean(scores)

K = 10
map = mapk(actuals, predicteds_cf, K)
print(f"CF MAP@{K}: {map:.4}")

map = mapk(actuals, predicteds_cbf, K)
print(f"CBF MAP@{K}: {map:.4}")

map = mapk(actuals, predicteds_hybrid, K)
print(f"Hybrid MAP@{K}: {map:.4}")

