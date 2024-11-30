import gradio as gr
import requests
import os

def stream_object_detection(video, conf_threshold_parkings, conf_threshold_vehicles):
    if video is None:
        return None, "No video provided", None

    # URL del servicio de modelos
    url = "http://model_service:8000/process"

    # Leer el video y enviarlo al servidor
    with open(video, "rb") as f:
        files = {'video': (os.path.basename(video), f, 'video/mp4')}
        data = {
            'conf_threshold_parkings': str(conf_threshold_parkings),
            'conf_threshold_vehicles': str(conf_threshold_vehicles)
        }
        response = requests.post(url, files=files, data=data)

    if response.status_code == 200:
        result = response.json()
        output_video_path = result['output_video']
        txt_yolo = result['txt_yolo']
        json_yolo = result['json_yolo']

        # Descargar el video procesado
        processed_video = output_video_path

        return processed_video, txt_yolo, json_yolo
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