from fastapi import FastAPI, HTTPException
from app.schemas import TextPair
from app.predictor import load_model_and_tokenizer, predict_similarity

app = FastAPI(title="Similarity Prediction API", version="1.0")

# Cargar modelo y tokenizer una vez
model, tokenizer = load_model_and_tokenizer()

@app.get("/")
def read_root():
    return {"message": "Servicio de similitud de frases activo."}

@app.post("/predict")
def predict_endpoint(payload: TextPair):
    try:
        score = predict_similarity(payload.sentence1, payload.sentence2, model, tokenizer)
        return {
            "sentence1": payload.sentence1,
            "sentence2": payload.sentence2,
            "similarity_score": score,
            "rounded_score": round(score, 1)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
