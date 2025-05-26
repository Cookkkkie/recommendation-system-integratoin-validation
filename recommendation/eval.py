import pandas as pd
import numpy as np

def apk(actual, predicted, k):
    if not actual:
        return 0.0

    predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)

def mapk(actuals, predicteds, k):
    scores = []
    for u, act in actuals.items():
        pred = predicteds.get(u, [])
        scores.append(apk(act, pred, k))
    return np.mean(scores)

df_test = pd.read_parquet('../dataset/test_aug.parquet')
actuals = df_test.groupby('user_id')['parent_asin'].apply(list).to_dict()

K = 5

df_recs_cf_ub = pd.read_parquet('../dataset/all_user_recs_cf_ub.parquet')
predicteds_cf_ub = df_recs_cf_ub.groupby('user_id')['recommended_asin'].apply(list).to_dict()
print(f"CF User-Based MAP@{K}:      {mapk(actuals, predicteds_cf_ub, K):.4f}")

df_recs_cf_ib = pd.read_parquet('../dataset/all_user_recs_cf_ib.parquet')
predicteds_cf_ib = df_recs_cf_ib.groupby('user_id')['recommended_asin'].apply(list).to_dict()
print(f"CF Item-Based MAP@{K}:      {mapk(actuals, predicteds_cf_ib, K):.4f}")

df_recs_cbf = pd.read_parquet('../dataset/all_user_recs_cbf.parquet')
predicteds_cbf = df_recs_cbf.groupby('user_id')['recommended_asin'].apply(list).to_dict()
print(f"CBF MAP@{K}:    {mapk(actuals, predicteds_cbf, K):.4f}")

df_recs_hybrid = pd.read_parquet('../dataset/all_user_recs_hybrid.parquet')
predicteds_hybrid = df_recs_hybrid.groupby('user_id')['recommended_asin'].apply(list).to_dict()
print(f"Hybrid MAP@{K}: {mapk(actuals, predicteds_hybrid, K):.4f}")



