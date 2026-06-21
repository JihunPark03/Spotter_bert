from fastapi import FastAPI
from pydantic import BaseModel

from inference import predict_prob, load_model

app = FastAPI()


@app.on_event("startup")
def preload_assets_startup():
    print("[ML] Loading ModernBERT ad detector...")
    load_model()
    print("[ML] Server Ready.")


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    prob_ad: float


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    """
    Receive a JSON body {"text": "..."} and return {"prob_ad": <float>}.
    This aligns with backend_server.ml_client.request_inference.
    """
    prob = predict_prob(payload.text)
    return {"prob_ad": prob}

@app.post("/reload-model")
def reload_model():
    print("Reloading model...")
    load_model()
    return {"status": "reloaded"}
