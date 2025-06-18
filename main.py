from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import os
import requests

from sklearn.linear_model import LogisticRegression
import mlflow.sklearn

app = FastAPI()

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

DATABASE_URL = "sqlite:///./data.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(Integer, index=True)
    x1 = Column(Float)
    x2 = Column(Float)
    y = Column(Integer)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class PredictInput(BaseModel):
    x1: float
    x2: float

def send_discord_alert(message: str):
    """I'm not late captain."""
    payload = {"content": message}
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload)
        response.raise_for_status()
    except Exception as e:
        print("❌ Error:", e)


def send_discord_embed(message):
    data = {"embeds": [{
        "title": "Résultats du pipeline",
        "description": message,
        "color": 5814783,
        "fields": [{
            "name": "Status",
            "value": "Succès",
            "inline": True
        }]
    }]}
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=data)
        response.raise_for_status()
        print("Embed envoyé avec succès !")
    except Exception as e:
        print("Erreur lors de l'envoi de l'embed :", e)


@app.on_event("startup")
def notify_startup():
    send_discord_alert("Embark for our journey !")
    print("Startup notification envoyée.")

@app.get("/health")
def health_check():
    send_discord_alert("There is no death... YET")
    #send_discord_embed("test")
    return {"status": "ok"}

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    send_discord_alert(f"Save me captain: {str(exc)}\nURL: {request.url}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"}
    )

@app.get("/boom")
def cause_error():
    raise ValueError("Something exploded")

@app.get("/notify/test")
def test_notify():
    send_discord_embed("Ceci est un test manuel de webhook.")
    return {"message": "Notification envoyée"}

@app.post("/generate")
def generate_dataset(db: Session = Depends(get_db)):
    np.random.seed()
    hour = datetime.now().hour
    shift = -0.5 if hour % 2 == 1 else 0.5

    x1 = np.random.rand(100)
    x2 = np.random.rand(100) + shift
    y = (x1 + x2 > 1).astype(int)

    batch_id = int(datetime.now().timestamp())
    for i in range(len(x1)):
        row = Dataset(batch_id=batch_id, x1=float(x1[i]), x2=float(x2[i]), y=int(y[i]))
        db.add(row)
    db.commit()
    return {"message": "Dataset generated", "batch_id": batch_id}

@app.post("/retrain")
def retrain_model(db: Session = Depends(get_db)):
    last_batch_id = db.query(Dataset.batch_id).order_by(Dataset.batch_id.desc()).first()
    if not last_batch_id:
        raise HTTPException(status_code=404, detail="No dataset found")

    rows = db.query(Dataset).filter(Dataset.batch_id == last_batch_id[0]).all()
    if not rows:
        raise HTTPException(status_code=404, detail="No data in last batch")

    df = pd.DataFrame([{"x1": r.x1, "x2": r.x2, "y": r.y} for r in rows])
    X, y = df[["x1", "x2"]], df["y"]

    model = LogisticRegression()
    model.fit(X, y)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    with mlflow.start_run():
        mlflow.log_param("n_samples", len(df))
        mlflow.sklearn.log_model(model, "logreg_model")

    return {"message": "Model retrained", "batch_id": last_batch_id[0]}

@app.post("/predict")
def predict(input: PredictInput):
    if not os.path.exists("model.pkl"):
        raise HTTPException(status_code=404, detail="Model not found. Please retrain.")

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    X = np.array([[input.x1, input.x2]])
    y_pred = model.predict(X)[0]
    return {"prediction": int(y_pred)}