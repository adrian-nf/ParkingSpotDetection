from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
import shutil
import os
import json
from parking_spot_detection import ParkingSpotDetection
import uvicorn
import zipfile
from io import BytesIO

app = FastAPI()

@app.post("/process")
async def process(
        video: UploadFile = File(...),
        conf_threshold_parkings: float = Form(0.8),
        conf_threshold_vehicles: float = Form(0.5)
):
    # Save the uploaded video to disk
    video_filename = "input_video.mp4"
    with open(video_filename, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # Initialize the ParkingSpotDetection class with the appropriate model paths and thresholds
    detector = ParkingSpotDetection(
        parking_model_path="models/yolo11n-detect-parking.pt",
        vehicle_model_path="models/yolov8n-visdrone.pt",
        parkings_confidence_threshold=conf_threshold_parkings,
        vehicles_confidence_threshold=conf_threshold_vehicles
    )

    # Process the video
    output_video_path, txt_yolo, json_yolo = detector.process_video_gradio(
        video_filename, 'ffmpeg'
    )

    # Check if the video was processed successfully
    if not os.path.exists(output_video_path):
        return JSONResponse(content={"error": "Error processing video"}, status_code=500)

    # Create a ZIP file in memory to bundle the processed video and metadata
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        # Add the processed video
        zip_file.write(output_video_path, "processed_video.mp4")
        # Add the metadata
        zip_file.writestr("txt_yolo.txt", txt_yolo)
        zip_file.writestr("json_yolo.json", json_yolo)

    zip_buffer.seek(0)

    # Return the ZIP file as a response
    return StreamingResponse(
        zip_buffer,
        media_type="application/x-zip-compressed",
        headers={"Content-Disposition": "attachment; filename=results.zip"}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)