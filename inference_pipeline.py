# inference_pipeline.py (Updated with Vehicle Class)

import cv2
import time
import threading
import queue
import random
import os
import json
from collections import deque
from datetime import datetime
import numpy as np
from ultralytics import YOLO
import torch
# Import the new Vehicle class
from vehicle import Vehicle
from ocr import OcrEngine
# --- Global Results Storage ---
ANPR_RESULTS = {}
RESULTS_LOCK = threading.Lock()


# --- The Core Pipeline Processor ---
class StreamProcessor:
    def __init__(self, rtsp_url, stream_id,skip_frames = 3):
        self.rtsp_url = rtsp_url
        self.stream_id = stream_id
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        self.data_path = './output'
        os.makedirs(self.data_path, exist_ok=True)
        # --- Frame Skipping ---
        self.skip_frames = skip_frames
        self.frame_num = 0
        # --- FPS Calculation Attributes ---
        self.fps_start_time = time.time()
        self.frame_count_for_fps = 0
        self.fps = 0
        self.running = threading.Event(); self.running.set()
        
        # Dictionary to store active Vehicle objects, keyed by track_id
        self.tracked_vehicles = {}

        print("Initializing models for stream:", self.stream_id)
        self.vehicle_model = YOLO("./model_weights/yolo11n.pt")
        self.plate_model = YOLO("./model_weights/lp.pt")
        self.ocr_model = OcrEngine("./model_weights/ocr_yolo11.pt")
        print("All models initialized.")

        self.frame_queue = queue.Queue(maxsize=10)
        self.vehicle_queue = queue.Queue(maxsize=10)
        self.plate_queue = queue.Queue(maxsize=10)
        self.display_queue = queue.Queue(maxsize=2)

        with RESULTS_LOCK:
            ANPR_RESULTS[self.stream_id] = deque(maxlen=50)
        os.makedirs(f"output/{self.stream_id}", exist_ok=True)
        self.threads = []

    def start(self):
        self.threads = [
            threading.Thread(target=self._read_frames, daemon=True),
            threading.Thread(target=self._track_vehicles_and_manage_lifecycle, daemon=True),
            threading.Thread(target=self._detect_plates, daemon=True),
            threading.Thread(target=self._perform_ocr_and_update, daemon=True)
        ]
        for t in self.threads: t.start()
        print(f"Pipeline started for stream: {self.stream_id}")

    def stop(self):
        print(f"Stopping stream processor for {self.stream_id}...")
        self.running.clear()
        # Clean up any remaining vehicles
        for track_id, vehicle in list(self.tracked_vehicles.items()):
             final_data = vehicle.save_final_results()
             if final_data:
                 with RESULTS_LOCK: ANPR_RESULTS[self.stream_id].append(final_data)
        
        for q in [self.frame_queue, self.vehicle_queue, self.plate_queue, self.display_queue]:
            with q.mutex: q.queue.clear()
            try: q.put_nowait(None)
            except queue.Full: pass
        for t in self.threads: t.join(timeout=2.0)
        print(f"Stream {self.stream_id} stopped.")

    def is_running(self): return self.running.is_set()
    def get_display_frame(self):
        try: return self.display_queue.get_nowait()
        except queue.Empty: return None

    def _read_frames(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            print(f"Error: Could not open RTSP stream: {self.rtsp_url}")
            self.running.clear(); return
        while self.running.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Stream ended. Reconnecting..."); cap.release(); time.sleep(2)
                cap = cv2.VideoCapture(self.rtsp_url)
                if not cap.isOpened(): print("Reconnect failed. Stopping."); self.running.clear(); break
                continue
            # --- Frame Skipping Logic ---
            self.frame_num += 1
            if self.frame_num % self.skip_frames != 0:
                continue # Skip this frame
            try: self.frame_queue.put(frame, timeout=1)
            except queue.Full: continue
        cap.release()

    def _track_vehicles_and_manage_lifecycle(self):
        while self.running.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
                if frame is None: continue
            except queue.Empty: continue

            display_frame = frame.copy()
            tracked_results = self.vehicle_model.track(frame,persist=True,classes=[2,3,4,6,8], verbose=False, device=self.device,conf=0.5)
            
            current_track_ids = set()
            for r in tracked_results:
                track_ids = r.boxes.id.int().cpu().numpy() if r.boxes.id is not None else []
                current_track_ids.update(track_ids)
            # --- Vehicle Lifecycle Management ---
            # 1. Identify and save results for vehicles that have disappeared
            existing_track_ids = set(self.tracked_vehicles.keys())
            disappeared_ids = existing_track_ids - current_track_ids
            for track_id in disappeared_ids:
                vehicle = self.tracked_vehicles.pop(track_id)
                final_data = vehicle.save_final_results()
                if final_data:
                    with RESULTS_LOCK:
                        ANPR_RESULTS[self.stream_id].append(final_data)

            # 2. Process currently tracked vehicles
            for r in tracked_results:
                track_ids = r.boxes.id.int().cpu().numpy() if r.boxes.id is not None else [] 
                for box, track_id in zip(r.boxes,track_ids):
                    if track_id not in self.tracked_vehicles:
                        print(f"[Track ID {track_id}] New vehicle detected.")
                        self.tracked_vehicles[track_id] = Vehicle(track_id, self.stream_id)

                    # Push to the next pipeline stage for plate detection
                    try:
                        self.vehicle_queue.put((frame, track_id, box.xyxy[0]), timeout=1)
                    except queue.Full:
                        continue
                    
                    # Draw tracking info
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    #x1, y1, x2, y2 = [int(c) for c in box]
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # --- FPS Calculation and Display ---
            self.frame_count_for_fps += 1
            # Calculate FPS over a 30-frame window for stability
            if self.frame_count_for_fps >= 30:
                end_time = time.time()
                elapsed_time = end_time - self.fps_start_time
                self.fps = self.frame_count_for_fps / elapsed_time
                self.fps_start_time = end_time
                self.frame_count_for_fps = 0

            # Draw the FPS on the frame
            fps_text = f"FPS: {self.fps:.2f}"
            cv2.putText(display_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
                
            # Update the display queue
            try:
                self.display_queue.put_nowait(display_frame)
            except queue.Full:
                self.display_queue.get_nowait()
                self.display_queue.put_nowait(display_frame)

    def _detect_plates(self):
        while self.running.is_set():
            try:
                frame, track_id, vehicle_box = self.vehicle_queue.get(timeout=1)
                if frame is None: continue
            except queue.Empty: continue

            vehicle = self.tracked_vehicles.get(track_id)
            if not vehicle: continue

            x1, y1, x2, y2 = [int(c) for c in vehicle_box]
            vehicle_crop = frame[y1:y2, x1:x2]
            if vehicle_crop.size == 0: continue
            
            plate_results = self.plate_model.predict(vehicle_crop,verbose=False, device=self.device,conf=0.5)

            if not plate_results:
                # Still update the vehicle object to increment frame count, etc.
                vehicle.update(vehicle_crop=vehicle_crop)
            else:
                for res in plate_results[0]:
                    for pb in res.boxes : 
                        px1, py1, px2, py2 = map(int, pb.xyxy[0])
                        plate_crop = vehicle_crop[py1:py2, px1:px2]
                        if plate_crop.size > 0:
                            try:
                                self.plate_queue.put((track_id, vehicle_crop, plate_crop, pb.xyxy[0], round(float(pb.conf[0]),3)), timeout=1)
                            except queue.Full:
                                continue

    def _perform_ocr_and_update(self):
        while self.running.is_set():
            try:
                track_id, vehicle_crop, plate_crop, plate_box, plate_confidence = self.plate_queue.get(timeout=1)
                if track_id is None: continue
            except queue.Empty: continue

            vehicle = self.tracked_vehicles.get(track_id)
            if not vehicle: continue
            
            #ocr_text, ocr_confidence = self.ocr_model.process_lp(plate_crop) 
            ocr_text = self.ocr_model.process_lp(plate_crop) 
            # This is the core update step. The Vehicle object handles all the quality logic.
            vehicle.update(
                vehicle_crop=vehicle_crop,
                plate_crop=plate_crop,
                plate_box=plate_box,
                plate_confidence=plate_confidence,
                ocr_text=ocr_text,
                ocr_confidence= 0 #ocr_confidence
            )

