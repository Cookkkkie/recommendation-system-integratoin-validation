import pandas as pd

N = 10
w_cf = 0.6
w_cbf = 1- w_cf

cf_df = pd.read_parquet('../dataset/all_user_recs_cf.parquet')
cbf_df = pd.read_parquet('../dataset/all_user_recs_cbf.parquet')

cf_lists = cf_df.groupby('user_id')['recommended_asin'].apply(list)
cbf_lists = cbf_df.groupby('user_id')['recommended_asin'].apply(list)


def weighted_hybrid(cf_list, cbf_list, w_cf, w_cbf, N):
    cf_scores = {item: 1.0 / (rank + 1) for rank, item in enumerate(cf_list)}
    cbf_scores = {item: 1.0 / (rank + 1) for rank, item in enumerate(cbf_list)}

    all_items = set(cf_list) | set(cbf_list)

    combined = {}
    for item in all_items:
        score_cf = cf_scores.get(item, 0.0)
        score_cbf = cbf_scores.get(item, 0.0)
        combined[item] = w_cf * score_cf + w_cbf * score_cbf

    topn = sorted(combined, key=combined.get, reverse=True)[:N]
    return topn


all_users = cf_lists.index.union(cbf_lists.index)
hybrid_series = pd.Series(
    {u: weighted_hybrid(
        cf_lists.get(u, []),
        cbf_lists.get(u, []),
        w_cf, w_cbf, N)
        for u in all_users},
    name='recommended_asin'
)

hybrid_series.index.name = 'user_id'
hybrid_df = hybrid_series.explode().to_frame()

hybrid_df.to_parquet(
    '../dataset/all_user_recs_hybrid.parquet',
    index=True
)

print(f"Hybrid recommendations (N={N}, w_cf={w_cf}, w_cbf={w_cbf}) written to ../dataset/all_user_recs_hybrid.parquet")
