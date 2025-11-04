from fastapi import FastAPI
from contextlib import asynccontextmanager

from fastapi.middleware.cors import CORSMiddleware
from routes.predictions_routes import router as prediction_router
from routes.dataset_routes import router as dataset_router
from models import load_models

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()

    yield

app = FastAPI(lifespan=lifespan, title="Text Classification API")
app.include_router(prediction_router, tags=["Prediction"])
app.include_router(dataset_router, tags=["Dataset"])

origins = [
    "http://localhost:5173",  # Vite default dev URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # Frontend URL(s)
    allow_credentials=True,
    allow_methods=["*"],         # Allow all methods (POST, GET, etc.)
    allow_headers=["*"],         # Allow all headers
)

@app.get("/")
def root():
    return {"message": "Text Classification API is running"}
