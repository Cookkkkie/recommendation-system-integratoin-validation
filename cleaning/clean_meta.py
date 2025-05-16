import pandas as pd
import re
import string
from AutoClean import AutoClean
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

INPUT_FILE = '../dataset/meta/meta_Appliances.jsonl'
OUTPUT_FILE = '../dataset/meta.parquet'
COLS_TO_KEEP = [
    'parent_asin',
    'title',
    'features',
    'description',
    'details',
    'price'
]
TEXT_COLS       = ['title']
LIST_COLS       = ['features', 'description', 'details']
LEMMATIZE_COLS  = ['title', 'features', 'description', 'details']

df_raw = pd.read_json(INPUT_FILE, lines=True)

for col in COLS_TO_KEEP:
    if col not in df_raw.columns:
        if col in LIST_COLS:
            df_raw[col] = [[] for _ in range(len(df_raw))]
        else:
            df_raw[col] = ''

df = df_raw[COLS_TO_KEEP].copy()

df_lists   = df[LIST_COLS].reset_index(drop=True)
df_price   = df['price'].reset_index(drop=True)
df_text    = df.drop(columns=LIST_COLS + ['price'])

pipeline = AutoClean(df_text, mode='auto')
df_clean  = pipeline.output

df_clean = pd.concat([df_clean.reset_index(drop=True), df_lists, df_price], axis=1)
df_clean['price'] = pd.to_numeric(df_clean['price'], errors='coerce').fillna(0.0)

lemmatizer  = WordNetLemmatizer()
stop_words  = set(stopwords.words('english'))
punct_re    = re.compile(f"[{re.escape(string.punctuation)}]")

def lemmatize_text(text: str) -> str:
    tokens = nltk.word_tokenize(str(text).lower())
    lemmas = []
    for tok in tokens:
        tok = punct_re.sub('', tok)
        if not tok or tok in stop_words:
            continue
        lemmas.append(lemmatizer.lemmatize(tok))
    return ' '.join(lemmas)

for col in LEMMATIZE_COLS:
    if col in TEXT_COLS:
        df_clean[col] = df_clean[col].fillna('').apply(lemmatize_text)
    else:
        df_clean[col] = df_clean[col].apply(
            lambda lst: [lemmatize_text(item) for item in (lst or [])]
        )

df_clean.to_parquet(OUTPUT_FILE, index=False)
print(f"Cleaned data saved to {OUTPUT_FILE}.")
