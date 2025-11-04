import os
import re
import glob
import pandas as pd
from pathlib import Path


BASE_DIR = Path(
    r"C:\Users\Enzo\Documents\projects\projeto-cd-ia\saida_wiki\textos_wiki"
)


# ------------- Cleaning helpers -------------
def clean_text(text):
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"={2,}.*?={2,}", " ", text)  # remove títulos wiki
    text = re.sub(r"##.*?\n", " ", text)  # remove títulos do Gemini
    text = re.sub(r"\[\d+\]", " ", text)  # remove [1], [2]
    text = re.sub(r"http\S+|www.\S+", " ", text)  # remove URLs
    text = re.sub(
        r"\b(Neste artigo|Vamos explorar|Desmistificando|Uma introdução clara|Entenda|Explicando)\b.*?:",
        " ",
        text,
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ------------- Load all pairs -------------
orig_paths = sorted(glob.glob(str(BASE_DIR / "*__original.txt")))
ia_paths = sorted(glob.glob(str(BASE_DIR / "*__ia.txt")))

pairs = []


# Match originals with corresponding IA by common prefix
def base_key(p: str) -> str:
    return os.path.basename(p).replace("__original.txt", "").replace("__ia.txt", "")


orig_map = {base_key(p): p for p in orig_paths}
ia_map = {base_key(p): p for p in ia_paths}

common_keys = sorted(set(orig_map.keys()).intersection(set(ia_map.keys())))

records = []
for key in common_keys:
    with open(orig_map[key], "r", encoding="utf-8") as f:
        text_h = f.read()
    with open(ia_map[key], "r", encoding="utf-8") as f:
        text_ai = f.read()
    records.append(
        {
            "titulo": key,
            "texto": clean_text(text_h),
            "classe": "humano",
            "fonte_path": orig_map[key],
        }
    )
    records.append(
        {
            "titulo": key,
            "texto": clean_text(text_ai),
            "classe": "ia",
            "fonte_path": ia_map[key],
        }
    )

df = pd.DataFrame(records)
dataset_path = str("saida_wiki/set_wiki.json")
df.to_json(dataset_path, orient="records", lines=False, force_ascii=False)
