# vehicle.py

import cv2
import os
import json
from datetime import datetime
import numpy as np
import csv
import threading

# Global lock for thread-safe CSV writing
CSV_WRITING_LOCK = threading.Lock()

class Vehicle:
    def __init__(self, track_id, stream_id):
        self.track_id = track_id
        self.stream_id = stream_id
        
        self.best_vehicle_img = None
        self.best_plate_img = None
        self.best_ocr_text = None
        
        # Quality tracking attributes
        self.best_plate_confidence = 0.0
        self.best_ocr_confidence = 0.0
        self.best_plate_area = 0
        self.best_plate_sharpness = 0.0
        
        self.frame_count = 0
        self.last_seen = datetime.now()
        
        # FIX: Add a flag to prevent multiple saves for the same object
        self.has_been_saved = False

    def _calculate_sharpness(self, image):
        """Calculate image sharpness using Laplacian variance"""
        if image is None: return 0.0
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _calculate_plate_area(self, box):
        """Calculate the area of the license plate bounding box"""
        if box is None: return 0
        width = abs(box[2] - box[0])
        height = abs(box[3] - box[1])
        return width * height

    def _is_better_plate(self, new_confidence, new_box, new_sharpness):
        """Determine if a new plate detection is better than the current best using a weighted score."""
        if self.best_plate_img is None:
            return True
        
        new_area = self._calculate_plate_area(new_box)
        
        # Normalize area and sharpness for consistent scoring
        norm_area = min(new_area / 30000.0, 1.0) 
        norm_sharpness = min(new_sharpness / 2000.0, 1.0)
        
        current_norm_area = min(self.best_plate_area / 30000.0, 1.0)
        current_norm_sharpness = min(self.best_plate_sharpness / 2000.0, 1.0)

        current_score = (self.best_plate_confidence * 0.5) + (current_norm_area * 0.25) + (current_norm_sharpness * 0.25)
        new_score = (new_confidence * 0.5) + (norm_area * 0.25) + (norm_sharpness * 0.25)
        
        return new_score > current_score

    def _is_better_ocr(self, new_text, new_confidence):
        """Determine if a new OCR result is better."""
        if not new_text or not new_text.strip(): return False
        if self.best_ocr_text is None: return True

        current_score = self.best_ocr_confidence * 0.7 + (len(self.best_ocr_text) * 0.1) + (0.2 if self.best_ocr_text.isalnum() else 0)
        new_score = new_confidence * 0.7 + (len(new_text) * 0.1) + (0.2 if new_text.isalnum() else 0)

        return new_score > new_score

    def update(self, vehicle_crop, plate_crop=None, plate_box=None, plate_confidence=0.0, ocr_text=None, ocr_confidence=0.0):
        """Update vehicle's state with new data, keeping only the best quality."""
        self.frame_count += 1
        self.last_seen = datetime.now()

        if plate_crop is not None and plate_box is not None:
            plate_sharpness = self._calculate_sharpness(plate_crop)
            if self._is_better_plate(plate_confidence, plate_box, plate_sharpness):
                print(f"[Track ID {self.track_id}] New best plate found. Sharpness: {plate_sharpness:.2f}, Conf: {plate_confidence:.2f}")
                self.best_plate_img = plate_crop
                self.best_vehicle_img = vehicle_crop
                self.best_plate_confidence = plate_confidence
                self.best_plate_sharpness = plate_sharpness
                self.best_plate_area = self._calculate_plate_area(plate_box)

        if ocr_text and self._is_better_ocr(ocr_text, ocr_confidence):
            print(f"[Track ID {self.track_id}] New best OCR found: '{ocr_text}' (Conf: {ocr_confidence:.2f})")
            self.best_ocr_text = ocr_text
            self.best_ocr_confidence = ocr_confidence
            
    def get_final_data(self):
        """Compile the best data for this vehicle."""
        if self.best_ocr_text is None or self.best_plate_img is None:
            return None
            
        return {
            "track_id": int(self.track_id),
            "plate_text": self.best_ocr_text,
            "ocr_confidence": float(round(self.best_ocr_confidence, 2)),
            "plate_confidence": float(round(self.best_plate_confidence, 2)),
            "timestamp": self.last_seen.strftime('%Y-%m-%d %H:%M:%S')
        }

    def save_final_results(self):
        """Saves images and appends a record to a CSV file, ensuring it only runs once."""
        # FIX: Check if this vehicle's data has already been saved.
        if self.has_been_saved:
            return None
            
        final_data = self.get_final_data()
        if final_data is None:
            print(f"[Track ID {self.track_id}] Discarded. Not enough quality data.")
            return None

        ocr_text = final_data.get('plate_text', 'UNKNOWN')
        sanitized_ocr = "".join(c for c in ocr_text if c.isalnum()).upper()
        if not sanitized_ocr:
            sanitized_ocr = "UNKNOWN_PLATE"

        print(f"[Track ID {self.track_id}] Saving final results to CSV. Plate: '{ocr_text}'")
        
        hourly_folder = self.last_seen.strftime('%Y-%m-%d-%H')
        save_path = os.path.join("output", self.stream_id, hourly_folder)
        os.makedirs(save_path, exist_ok=True)
        
        base_filename = f"{self.track_id}"
        plate_filename = f"{base_filename}_{sanitized_ocr}_plate.jpg"
        vehicle_filename = f"{base_filename}_vehicle.jpg"
        
        plate_filepath = os.path.join(save_path, plate_filename)
        vehicle_filepath = os.path.join(save_path, vehicle_filename)
        
        # if self.best_vehicle_img is not None:
        #     cv2.imwrite(vehicle_filepath, self.best_vehicle_img)
        # if self.best_plate_img is not None:
        #     cv2.imwrite(plate_filepath, self.best_plate_img)

        # Append results to a single CSV file for the hour
        csv_path = os.path.join(save_path, "results.csv")
        csv_headers = ["vehicle_id", "timestamp", "vehicle_image_path", "plate_image_path", "OCR_text"]
        csv_row = {
            "vehicle_id": self.track_id,
            "timestamp": final_data['timestamp'],
            # "vehicle_image_path": vehicle_filepath,
            # "plate_image_path": plate_filepath,
            "OCR_text": ocr_text
        }

        # Use a lock to ensure thread-safe writing to the CSV file
        with CSV_WRITING_LOCK:
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=csv_headers)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(csv_row)
        
        # FIX: Set the flag to true after a successful save.
        self.has_been_saved = True
            
        return final_data
