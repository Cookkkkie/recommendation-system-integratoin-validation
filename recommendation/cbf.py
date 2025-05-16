import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TEXT_COLS = ["title"]
LIST_COLS = ["features", "description", "details"]


def preprocess_metadata(df, text_cols, list_cols):
    df = df.copy()
    for col in list_cols:
        df[col] = df[col].apply(
            lambda x: " ".join(x) if isinstance(x, list) else ""
        )
    df["content"] = (
        df[text_cols].astype(str).agg(" ".join, axis=1)
        + " "
        + df[list_cols].agg(" ".join, axis=1)
        + " "
        + df["price"].astype(str)
    )
    return df


def build_tfidf_matrix(df):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["content"])
    return vectorizer, tfidf_matrix


def recommend_items(user_id, user_hist, meta_df, tfidf_matrix, top_n: int = 10):
    items = user_hist.get(user_id, [])
    if not items:
        return []
    last_asin = items[-1]
    idxs = meta_df.index[meta_df["parent_asin"] == last_asin].tolist()
    if not idxs:
        return []
    idx = idxs[0]

    sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sims[idx] = -1

    top_idx = sims.argsort()[-top_n:][::-1]

    recs = []
    for i in top_idx:
        asin = meta_df.iloc[i]["parent_asin"]
        recs.append(asin)
    return recs


if __name__ == "__main__":
    meta_df = pd.read_parquet("../dataset/meta.parquet")
    train_df = pd.read_parquet("../dataset/train.parquet")
    test_df = pd.read_parquet("../dataset/test.parquet")

    common_users = pd.Index(train_df["user_id"]).intersection(test_df["user_id"])
    train_df = train_df[train_df["user_id"].isin(common_users)].copy()
    test_df = test_df[test_df["user_id"].isin(common_users)].copy()

    user_hist = (
        train_df.sort_values("timestamp")
        .groupby("user_id")["parent_asin"]
        .apply(list)
        .to_dict()
    )

    meta_df = preprocess_metadata(meta_df, TEXT_COLS, LIST_COLS)
    _, tfidf_matrix = build_tfidf_matrix(meta_df)

    all_rows = []
    for user_id in user_hist.keys():
        recs = recommend_items(user_id, user_hist, meta_df, tfidf_matrix, top_n=10)
        for asin in recs:
            all_rows.append(
                {"user_id": user_id, "recommended_asin": asin}
            )

    recs_df = pd.DataFrame(all_rows)
    print(recs_df.head())
    recs_df.to_parquet("../dataset/all_user_recs_cbf.parquet", index=False)
    print("Recommendations with price & details saved to all_user_recs_cbf.parquet")
