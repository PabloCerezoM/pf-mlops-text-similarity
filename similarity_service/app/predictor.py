import os
import torch
import wandb
from transformers import AutoTokenizer
from app.model import RegressionModel_v3
from dotenv import load_dotenv

load_dotenv()

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
ARTIFACT_NAME = os.getenv("WANDB_ARTIFACT")
MODEL_NAME = os.getenv("MODEL_NAME")

def log(msg):
    print(f"[🟢 SIMILARITY SERVICE] {msg}")

def load_model_and_tokenizer():
    log("Inicializando sesión wandb...")
    wandb.login()

    log(f"Usando proyecto: {WANDB_PROJECT}")
    run = wandb.init(project=WANDB_PROJECT, job_type="inference", reinit=True)

    log(f"Descargando artefacto: {ARTIFACT_NAME}")
    artifact = run.use_artifact(ARTIFACT_NAME, type="model")
    artifact_dir = artifact.download()
    log(f"Artefacto descargado en: {artifact_dir}")

    # Como el tokenizer se subió sin carpeta, se carga desde la raíz del artefacto
    tokenizer_path = artifact_dir
    model_path = os.path.join(artifact_dir, "modelo_final_all_unfrozen.pt")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en {model_path}")

    log(f"Cargando tokenizer desde: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

    log(f"Cargando modelo desde: {model_path}")
    model = RegressionModel_v3(model_name=MODEL_NAME)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    log("✅ Modelo y tokenizer cargados correctamente")
    return model, tokenizer


def predict_similarity(text1: str, text2: str, model, tokenizer):
    log(f"🔍 Predicción solicitada:")
    log(f" - Sentence 1: {text1}")
    log(f" - Sentence 2: {text2}")

    inputs = tokenizer(text1, text2, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    score = output.item()

    log(f"📊 Similitud calculada: {score:.4f}")
    return score
