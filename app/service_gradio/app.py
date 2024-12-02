import gradio as gr
import requests
import os
from tempfile import NamedTemporaryFile
import zipfile
from io import BytesIO
import shutil
import json

def stream_object_detection(video, conf_threshold_parkings, conf_threshold_vehicles):
    if video is None:
        return None, "No video provided", None

    # Service model Url
    url = "http://model_service:8000/process"

    # Read the video file and send it to the server
    with open(video, "rb") as f:
        files = {'video': (os.path.basename(video), f, 'video/mp4')}
        data = {
            'conf_threshold_parkings': str(conf_threshold_parkings),
            'conf_threshold_vehicles': str(conf_threshold_vehicles)
        }
        response = requests.post(url, files=files, data=data, stream=True)

    if response.status_code == 200:
        # Read ZIP file contents into memory
        zip_content = BytesIO(response.content)
        with zipfile.ZipFile(zip_content) as zip_file:
            # Extract the processed video
            with zip_file.open("processed_video.mp4") as video_file:
                with NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                    shutil.copyfileobj(video_file, temp_video)
                    processed_video_path = temp_video.name

            # Extract txt_yolo
            with zip_file.open("txt_yolo.txt") as txt_file:
                txt_yolo = txt_file.read().decode('utf-8')

            # Extract json_yolo
            with zip_file.open("json_yolo.json") as json_file:
                json_yolo = json.load(json_file)

        return processed_video_path, txt_yolo, json_yolo
    else:
        return None, "Error processing video", None

def create_interface():
    video = gr.Video(label="Video Source")
    conf_threshold_parkings = gr.Slider(
        label="Parking Spots Confidence Threshold",
        minimum=0.0,
        maximum=1.0,
        step=0.05,
        value=0.80,
    )
    conf_threshold_vehicles = gr.Slider(
        label="Vehicles Confidence Threshold",
        minimum=0.0,
        maximum=1.0,
        step=0.05,
        value=0.50,
    )

    output_video = gr.Video(label="Processed Video")
    output_text = gr.Textbox(label="Parking Boxes in YOLO Format", lines=10)
    output_json = gr.JSON(label="Parking Boxes in JSON")

    interface = gr.Interface(
        fn=stream_object_detection,
        inputs=[video, conf_threshold_parkings, conf_threshold_vehicles],
        outputs=[output_video, output_text, output_json],
    )

    interface.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    create_interface()