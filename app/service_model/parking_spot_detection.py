import cv2
import numpy as np
from ultralytics import YOLO
import subprocess
import json


class ParkingSpotDetection:
    def __init__(self, parking_model_path, vehicle_model_path, confidence_threshold=0.5):
        self.parking_model = YOLO(parking_model_path)
        self.vehicle_model = YOLO(vehicle_model_path)
        self.available_spots = set()
        self.occupied_spots = set()
        self.confidence_threshold = confidence_threshold

    def txt_yolo(self, frame, parking_spots):
        """
        Converts the parking spot bounding box coordinates to YOLO format.

        :param frame: Image or video frame where the parking spots are detected.
        :param parking_spots: List of bounding box coordinates for the parking spots.
        :return: A string containing the bounding boxes in YOLO format.
        """
        frame_height, frame_width = frame.shape[:2]
        new_parking_spots = []

        for bbox in parking_spots:
            x1, y1, x2, y2 = bbox
            cx, cy = self.get_centroid(bbox)

            width = x2 - x1
            height = y2 - y1

            cx /= frame_width
            cy /= frame_height
            width /= frame_width
            height /= frame_height

            new_parking_spots.append(f"0 {cx:.4f} {cy:.4f} {width:.4f} {height:.4f}")

        return "\n".join(new_parking_spots)

    def json_yolo(self, parking_spots):
        """
        Converts the parking spot bounding box coordinates to a json format.

        :param parking_spots: List of bounding box coordinates for the parking spots.
        :return: A json containing the bounding boxes in json format.
                The format is compatible with the `ParkingManagement` module from `ultralytics.solutions`.
        """
        new_parking_spots = []
        for bbox in parking_spots:
            x1, y1, x2, y2 = bbox
            polygon = [[int(x1), int(y1)], [int(x2), int(y1)], [int(x2), int(y2)], [int(x1), int(y2)]]
            new_parking_spots.append({"points": polygon})

        return json.dumps(new_parking_spots, indent=4)

    def process_video_with_ffmpeg(self, ffmpeg_path, input_file, output_file="video_result.mp4"):
        """
        Processes the input video using ffmpeg to apply transformations or other processing.
        **Note**: This method requires ffmpeg to be installed on your system. 
        You can download and install ffmpeg from https://ffmpeg.org/.

        :param input_file: str
            The path to the input video file that needs to be processed.
        :param output_file: str, optional
            The path where the processed video will be saved, including the filename. 
            Defaults to "video_result.mp4" if not provided.
        :return: str
            The path to the output video file after processing.
            This path points to the location where the processed video is saved.
        """
        try:
            print(f"Processing video from {input_file} to {output_file}...")
            result = subprocess.run(
                [ffmpeg_path, '-y', '-i', input_file, output_file],
                check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True
            )
            print(result.stdout)
            return output_file
        except subprocess.CalledProcessError as e:
            print(f"Error executing ffmpeg: {e.stderr}")
            return False

    def detect_objects(self, frame, model, threshold):
        detections = model(frame)
        boxes = detections[0].boxes.xyxy.cpu().numpy()
        confidences = detections[0].boxes.conf.cpu().numpy()
        class_indices = detections[0].boxes.cls.cpu().numpy()
        class_names = detections[0].names

        filtered_boxes = []
        filtered_class_indices = []
        filtered_confidences = []

        for box, confidence, class_idx in zip(boxes, confidences, class_indices):
            if confidence >= threshold:
                filtered_boxes.append(box)
                filtered_class_indices.append(class_idx)
                filtered_confidences.append(confidence)

        return filtered_boxes, filtered_class_indices, filtered_confidences, class_names

    def get_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    def is_car_in_parking_spot(self, car_bbox, parking_bbox):
        cx, cy = self.get_centroid(car_bbox)
        px1, py1, px2, py2 = parking_bbox
        px1, px2 = min(px1, px2), max(px1, px2)
        py1, py2 = min(py1, py2), max(py1, py2)
        return px1 <= cx <= px2 and py1 <= cy <= py2

    def update_parking_status(self, cars, parkings):
        for idx, parking in enumerate(parkings):
            parking_occupied = False
            for car in cars:
                if self.is_car_in_parking_spot(car, parking):
                    parking_occupied = True
                    break
            if parking_occupied:
                if idx not in self.occupied_spots:
                    self.occupied_spots.add(idx)
                if idx in self.available_spots:
                    self.available_spots.remove(idx)
            else:
                if idx not in self.available_spots:
                    self.available_spots.add(idx)
                if idx in self.occupied_spots:
                    self.occupied_spots.remove(idx)

    def process_frame(self, frame, parkings_predictions, confidence_threshold_vehicles):
        boxes_parkings, class_indices_parkings, confidences_parkings, class_names_parkings = parkings_predictions

        parkings = []
        for box, class_idx in zip(boxes_parkings, class_indices_parkings):
            if class_names_parkings[int(class_idx)] in ['parking-spot', 'parking-spot-disabled']:
                parkings.append(box)

        boxes_vehicles, class_indices_vehicles, confidences_vehicles, class_names_vehicles = self.detect_objects(frame, self.vehicle_model, confidence_threshold_vehicles)
        cars = []
        for box, class_idx in zip(boxes_vehicles, class_indices_vehicles):
            if class_names_vehicles[int(class_idx)] in ['car', 'van']:
                cars.append(box)

        parking_bboxes = []
        for d in parkings:
            parking_bboxes.append((d[0], d[1], d[2], d[3]))

        self.update_parking_status(cars, parking_bboxes)

        return boxes_vehicles, class_indices_vehicles, confidences_vehicles, class_names_vehicles

    def draw_bboxes(self, frame, boxes, class_indices, confidences, class_names, color):
        for box, class_idx, confidence in zip(boxes, class_indices, confidences):
            x1, y1, x2, y2 = box
            label = class_names[int(class_idx)]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    def draw_bboxes_parking_spots(self, frame, boxes, class_indices, class_names):
        for i, (box, class_idx) in enumerate(zip(boxes, class_indices)):
            x1, y1, x2, y2 = box
            label = class_names[int(class_idx)]

            if i in self.occupied_spots:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            #cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        available_count = len(self.available_spots)
        occupied_count = len(self.occupied_spots)

        cv2.rectangle(frame, (540, 10), (760, 60), (0, 0, 0), -1)
        cv2.rectangle(frame, (540, 70), (760, 120), (0, 0, 0), -1)
        cv2.putText(frame, f"Available: {available_count}", (550, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Occupied: {occupied_count}", (550, 105), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return frame


    def draw_bboxes_vehicles(self, frame, boxes, class_indices, confidences, class_names):
        for box, class_idx, confidence in zip(boxes, class_indices, confidences):
            x1, y1, x2, y2 = box
            label = class_names[int(class_idx)]
            if label == 'car' or 'van':
                color = (255, 0, 0)
                cx, cy = self.get_centroid(box)
                cv2.circle(frame, (int(cx), int(cy)), 5, (255, 0, 0), -1)

        return frame


    def process_video_gradio(self, video_path, ffmpeg_path, vehicles_confidence_threshold=0.5):
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), "Error opening video file"

        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        ret, frame_parking = cap.read()
        assert ret, "Error reading parking frame"

        # Detect parking spots in the first frame
        parkings_predictions = self.detect_objects(frame_parking, self.parking_model, self.confidence_threshold)
        class_names_parkings = parkings_predictions[3]
        boxes_parkings = []
        class_indices_parkings = []
        confidences_parkings = []
        for box, class_idx, confidence in zip(parkings_predictions[0], parkings_predictions[1], parkings_predictions[2]):
            if parkings_predictions[3][int(class_idx)] in ['parking-spot', 'parking-spot-disabled']:
                boxes_parkings.append(box)
                class_indices_parkings.append(class_idx)
                confidences_parkings.append(confidence)

        # Prepare video writer
        output_path = "processed_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        # Process video frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detect vehicles in the current frame
            vehicle_detections = self.process_frame(frame, parkings_predictions, vehicles_confidence_threshold)

            # Draw bounding boxes for vehicles and parking spots
            frame = self.draw_bboxes_vehicles(frame, *vehicle_detections)
            frame = self.draw_bboxes_parking_spots(frame, boxes_parkings, class_indices_parkings, class_names_parkings)


            video_writer.write(frame)

        cap.release()
        video_writer.release()

        # Return processed video path, txt yolo format and parking spot JSON
        return self.process_video_with_ffmpeg(ffmpeg_path, output_path), self.txt_yolo(frame_parking, boxes_parkings),self.json_yolo(boxes_parkings)