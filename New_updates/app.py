from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import predict_attention_state
import uvicorn

app = FastAPI(title="EEG Attention State Predictor", description="API to predict attention state from EEG data", version="1.0.0")

class EEGData(BaseModel):
    eeg_data: list[float]  # List of 3584 floats (256 samples * 14 channels)

@app.post("/predict")
async def predict(data: EEGData):
    try:
        if len(data.eeg_data) != 3584:
            raise HTTPException(status_code=400, detail="EEG data must be exactly 3584 values (256 samples * 14 channels)")
        prediction = predict_attention_state(data.eeg_data)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "EEG Attention State Predictor API", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
