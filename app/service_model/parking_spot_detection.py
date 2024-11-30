import cv2
import numpy as np
from ultralytics import YOLO
import subprocess
import json


class ParkingSpotDetection:
    """
    A class for detecting parking spots and vehicle occupancy in video frames using YOLO models.

    This class provides functionalities for:
    - Detecting parking spots in a given video or image frame.
    - Detect whether the detected parking spots are available or occupied.
    - Generating bounding box data for parking spots in both YOLO and JSON formats.

    Attributes:
    ----------
    parking_model : YOLO
        The YOLO model used to detect parking spots.
    vehicle_model : YOLO
        The YOLO model used to detect vehicles.
    available_spots : set
        A set of indices representing currently available parking spots.
    occupied_spots : set
        A set of indices representing currently occupied parking spots.
    confidence_threshold : float
        Minimum confidence score to consider a detection as valid.
    """
    def __init__(self, parking_model_path, vehicle_model_path, parkings_confidence_threshold=0.5, vehicles_confidence_threshold=0.5):
        """
        Initializes the ParkingSpotDetection object.

        :param parking_model_path: str
            Path to the YOLO model file for detecting parking spots.
        :param vehicle_model_path: str
            Path to the YOLO model file for detecting vehicles.
        :param parkings_confidence_threshold: float, optional
            The confidence threshold for parking spot detections. Detections below this threshold are ignored.
            Default is 0.5.
        :param vehicles_confidence_threshold: float, optional
            The confidence threshold for vehicle detections. Detections below this threshold are ignored.
            Default is 0.5.
        """
        self.parking_model = YOLO(parking_model_path)
        self.vehicle_model = YOLO(vehicle_model_path)
        self.available_spots = set()
        self.occupied_spots = set()
        self.parkings_confidence_threshold=parkings_confidence_threshold
        self.vehicles_confidence_threshold= vehicles_confidence_threshold

    def txt_yolo(self, frame, parking_spots):
        """
        Converts parking spot bounding boxes to YOLO format.

        YOLO format includes:
        - Class ID (fixed to 0 as it represents parking spots here).
        - Relative x-center and y-center of the bounding box.
        - Relative width and height of the bounding box.

        :param frame: np.array
            The frame containing the parking spots, used to calculate relative dimensions.
        :param parking_spots: list
            List of bounding box coordinates [(x1, y1, x2, y2), ...].
        :return: str
            A string where each line represents a bounding box in YOLO format.
            Example: 0 0.2117 0.1593 0.0731 0.0634
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
        Converts parking spot bounding boxes to JSON format.

        JSON format includes:
        - "points": A list of four [x, y] coordinates representing the bounding box polygon.

        :param parking_spots: list
            List of bounding box coordinates [(x1, y1, x2, y2), ...].
        :return: str
            A JSON-formatted string where each parking spot is represented as a polygon.
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
        """
        Detects objects in a frame using a specified YOLO model.

        :param frame: np.array
            The image or video frame to process.
        :param model: YOLO
            The YOLO model used for object detection.
        :param threshold: float
            Confidence threshold for filtering detections.
        :return: tuple
            A tuple containing:
            - List of bounding boxes (x1, y1, x2, y2) for detected objects.
            - List of class indices for detected objects.
            - List of confidence scores for detected objects.
            - List of class names corresponding to the detected objects.
        """
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
        """
        Calculates the centroid of a bounding box.

        :param bbox: tuple or list
            Bounding box coordinates (x1, y1, x2, y2).
        :return: tuple
            Centroid coordinates (cx, cy).
        """
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    def is_car_in_parking_spot(self, car_bbox, parking_bbox):
        """
        Checks if the centroid of a car is inside a parking spot.

        :param car_bbox: tuple or list
            Bounding box coordinates (x1, y1, x2, y2) for the car.
        :param parking_bbox: tuple or list
            Bounding box coordinates (x1, y1, x2, y2) for the parking spot.
        :return: bool
            True if the car's centroid is within the parking spot, False otherwise.
        """
        cx, cy = self.get_centroid(car_bbox)
        px1, py1, px2, py2 = parking_bbox
        px1, px2 = min(px1, px2), max(px1, px2)
        py1, py2 = min(py1, py2), max(py1, py2)
        return px1 <= cx <= px2 and py1 <= cy <= py2

    def update_parking_status(self, cars, parkings):
        """
        Updates the status of parking spots as either available or occupied.

        :param cars: list
            List of bounding boxes (x1, y1, x2, y2) for detected cars.
        :param parkings: list
            List of bounding boxes (x1, y1, x2, y2) for parking spots.
        """
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

    def process_frame(self, frame, parkings_predictions):
        """
        Processes a single video frame to detect vehicles and update parking statuses.

        :param frame: np.array
            The current video frame being processed.
        :param parkings_predictions: tuple
            Parking spot detection results from `detect_objects`.
        :return: tuple
            A tuple containing:
            - List of bounding boxes for detected vehicles.
            - List of class indices for detected vehicles.
            - List of confidence scores for detected vehicles.
            - List of class names for detected vehicles.
        """
        boxes_parkings, class_indices_parkings, confidences_parkings, class_names_parkings = parkings_predictions
    
        parkings = []
        for box, class_idx in zip(boxes_parkings, class_indices_parkings):
            if class_names_parkings[int(class_idx)] in ['parking-spot', 'parking-spot-disabled']:
                parkings.append(box)
    
        boxes_vehicles, class_indices_vehicles, confidences_vehicles, class_names_vehicles = self.detect_objects(frame, self.vehicle_model, self.vehicles_confidence_threshold)
        cars = []
        for box, class_idx in zip(boxes_vehicles, class_indices_vehicles):
            if class_names_vehicles[int(class_idx)] in ['car', 'van']:
                cars.append(box)

        parking_bboxes = []
        for d in parkings:
            parking_bboxes.append((d[0], d[1], d[2], d[3]))
        
        self.update_parking_status(cars, parking_bboxes)
        
        return boxes_vehicles, class_indices_vehicles, confidences_vehicles, class_names_vehicles

    def draw_bboxes_parking_spots(self, frame, boxes, class_indices, class_names):
        """
        Draws bounding boxes around parking spots and adds status information.

        :param frame: np.array
            The frame where the bounding boxes are drawn.
        :param boxes: list
            List of bounding boxes for parking spots.
        :param class_indices: list
            List of class indices for the parking spots.
        :param class_names: list
            List of class names corresponding to the class indices.
        :return: np.array
            The frame with drawn bounding boxes and status information.
        """
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
        """
        Draws bounding boxes around detected vehicles and marks their centroids.

        :param frame: np.array
            The frame where the bounding boxes are drawn.
        :param boxes: list
            List of bounding boxes for vehicles.
        :param class_indices: list
            List of class indices for the vehicles.
        :param confidences: list
            List of confidence scores for the vehicles.
        :param class_names: list
            List of class names corresponding to the class indices.
        :return: np.array
            The frame with drawn bounding boxes and centroids for vehicles.
        """
        for box, class_idx, confidence in zip(boxes, class_indices, confidences):
            x1, y1, x2, y2 = box
            label = class_names[int(class_idx)]
            if label == 'car' or 'van':
                color = (255, 0, 0)
                cx, cy = self.get_centroid(box)
                cv2.circle(frame, (int(cx), int(cy)), 5, (255, 0, 0), -1)
    
        return frame

    def process_video_gradio(self, video_path, ffmpeg_path):
        """
        Processes a video to detect parking spots and vehicles, applies bounding boxes,
        and generates output in YOLO and JSON formats.

        This method is primarily designed for use in Gradio applications, 
        but it can also be used in other contexts.

        :param video_path: str
            Path to the input video file.
        :param ffmpeg_path: str
            Path to the ffmpeg executable for additional processing.
        :return: tuple
            A tuple containing:
            - The path to the processed video file.
            - A string in YOLO format with parking spot bounding boxes.
            - A JSON string with parking spot bounding box coordinates.
        """
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), "Error opening video file"
    
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        ret, frame_parking = cap.read()
        assert ret, "Error reading parking frame"
    
        # Detect parking spots in the first frame
        parkings_predictions = self.detect_objects(frame_parking, self.parking_model, self.parkings_confidence_threshold)
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
            vehicle_detections = self.process_frame(frame, parkings_predictions)
    
            # Draw bounding boxes for vehicles and parking spots
            frame = self.draw_bboxes_vehicles(frame, *vehicle_detections)
            frame = self.draw_bboxes_parking_spots(frame, boxes_parkings, class_indices_parkings, class_names_parkings)
            
    
            video_writer.write(frame)
    
        cap.release()
        video_writer.release()
    
        # Return processed video path, txt yolo format and parking spot JSON
        return self.process_video_with_ffmpeg(ffmpeg_path, output_path), self.txt_yolo(frame_parking, boxes_parkings),self.json_yolo(boxes_parkings)