import pandas as pd

df_reviews = pd.read_json(r"reviews_ml\set_reviews.json")
df_redacao = pd.read_json(r"redacao\set_redacoes.json")
df_wiki = pd.read_json(r"saida_wiki\set_wiki.json")
df_combined = pd.concat([df_reviews, df_redacao, df_wiki], ignore_index=True)
df_combined = df_combined[["texto", "classe"]]
df_combined.to_json(
    "dataset_final.json", orient="records", lines=False, force_ascii=False
)
