from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import shutil
import os
import json
from parking_spot_detection import ParkingSpotDetection
import uvicorn

app = FastAPI()

@app.post("/process")
async def process(
        video: UploadFile = File(...),
        conf_threshold_parkings: float = Form(0.8),
        conf_threshold_vehicles: float = Form(0.5)
):
    # Guardar el video subido
    video_filename = "input_video.mp4"
    with open(video_filename, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # Crear una instancia de la clase de detección
    detector = ParkingSpotDetection(
        parking_model_path="models/yolo11n-detect-parking.pt",
        vehicle_model_path="models/yolov8n-visdrone.pt",
        confidence_threshold=conf_threshold_parkings
    )

    # Procesar el video
    output_video_path, txt_yolo, json_yolo = detector.process_video_gradio(
        video_filename, 'ffmpeg', conf_threshold_vehicles
    )

    # Devolver los resultados
    response = {
        "output_video": output_video_path,
        "txt_yolo": txt_yolo,
        "json_yolo": json.loads(json_yolo)
    }

    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)