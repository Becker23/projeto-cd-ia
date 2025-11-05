# routes/datasets_routes.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import os

router = APIRouter(prefix="/dataset", tags=["Datasets"])


class SampleResponse(BaseModel):
    samples: list[dict]


@router.get("/random25", response_model=SampleResponse)
def get_random_25(max_chars: int = 200):
    file_path = os.path.join(
        os.path.dirname(__file__), os.pardir, "models", "dataset_final.json"
    )
    print(file_path)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    try:
        df = pd.read_json(file_path, orient="records", lines=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading JSON: {e}")

    # Sample 25 random rows
    try:
        sample_df = df.sample(n=25)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sampling: {e}")

    # Apply snippet logic
    def snippet(text: str) -> str:
        if not isinstance(text, str):
            return text
        if len(text) > max_chars:
            return text[:max_chars] + "…"
        return text

    # Assume the column is named 'text' — adjust if yours is different
    sample_df["snippet_text"] = sample_df["texto"].apply(snippet)

    # Optionally you may choose to **replace** the original text, or include both.
    # For example you can drop the full text from output and only send snippet_text:
    records = sample_df.drop(columns=["texto"]).to_dict(orient="records")

    return SampleResponse(samples=records)
