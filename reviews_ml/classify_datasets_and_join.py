import pandas as pd

df = pd.read_json(r"reviews_ml\set_human.json")
df["classe"] = "humano"
df_ia = pd.read_json(r"reviews_ml\set_ai.json")
df_ia["classe"] = "ia"

df_combined = pd.concat([df, df_ia], ignore_index=True)
df_combined.rename(columns={"review_text": "texto"}, inplace=True)
df_combined.to_json(
    "reviews_ml/set_reviews.json", orient="records", lines=False, force_ascii=False
)
print(df_combined.head())
