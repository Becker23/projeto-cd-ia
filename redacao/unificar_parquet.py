import pandas as pd
import glob
import re
import os

# Detecta automaticamente o diretório onde o script está sendo executado
caminho = os.path.dirname(os.path.abspath(__file__))

# --- 1. Carregar textos de IA ---
ia_df = pd.read_parquet(os.path.join(caminho, "ai_texts.parquet"))
ia_df = ia_df.rename(columns={"ano_enem": "ano", "response": "texto"})
ia_df = ia_df[["ano", "texto", "label"]]

# --- 2. Carregar textos humanos ---
arquivos_humanos = glob.glob(os.path.join(caminho, "redacao_*.parquet"))
dfs_humanos = []

for arquivo in arquivos_humanos:
    # Extrair o ano do nome do arquivo (ex: redacao_2022.parquet → 2022)
    match = re.search(r"redacao_(\d+)\.parquet", os.path.basename(arquivo))
    if not match:
        continue
    ano = int(match.group(1))

    df = pd.read_parquet(arquivo)
    df = df.rename(columns={"text": "texto"})
    df["ano"] = ano
    df = df[["ano", "texto", "label"]]
    dfs_humanos.append(df)

# --- 3. Unificar tudo ---
dataset_unificado = pd.concat([ia_df] + dfs_humanos, ignore_index=True)

# --- 4. Renomear 'label' -> 'classe' e ajustar valores ---
dataset_unificado = dataset_unificado.rename(columns={"label": "classe"})
dataset_unificado["classe"] = dataset_unificado["classe"].replace(
    {"ai": "ia", "human": "humano"}
)

# --- 5. Salvar resultado ---
output_path = os.path.join(caminho, "set_redacoes.parquet")
dataset_unificado.to_parquet(output_path, index=False)
dataset_unificado.to_json(
    output_path.replace(".parquet", ".json"),
    orient="records",
    lines=False,
    force_ascii=False,
)

print("Dataset unificado salvo como set_redacoes.parquet")
print("Linhas totais:", len(dataset_unificado))
print(dataset_unificado["classe"].value_counts())
print(dataset_unificado.head())
