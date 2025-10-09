from ultralytics import YOLO
import cv2 
import numpy as np
#from utils.ocr_utils import straighten_by_top_edge

class OcrEngine:
    def __init__(self,weights_path='./ocr_yolo11.pt',device='cpu'):
        self.weights_path = weights_path
        self.model = YOLO(self.weights_path)
        self.device = device
        self.char_map = {
                            0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
                            10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
                            20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'P', 25: 'Q', 26: 'R', 27: 'S', 28: 'T', 29: 'U',
                            30: 'V', 31: 'W', 32: 'X', 33: 'Y', 34: 'Z'
                            }   

    def process_lp(self, frame):
        roi = frame 
        op = self.model.predict(source=roi,verbose=False, device=self.device,conf=0.4)
        
        if (len(op[0].boxes.data) > 5):
            lp = self.postprocess_lp(op)
        else :
            lp = None
        #print(lp)
        return lp
    def postprocess_lp(self, ocr_op):
        # sort the detction ,class and generate the final Licence plate 
        
        #op = self.sort_and_combine(ocr_op)
        licence_number = self.sort_and_create_vehicle_number(ocr_op)       
                
        return licence_number
    def predict_and_plot(self, frame):
        op = self.model.predict(source=frame,verbose=False, device=self.device)
        
        return op[0].plot(),op
    
    def sort_and_combine(self,detections):
        # Extract the bounding box and class information
        # Each row: [x1, y1, x2, y2, confidence, class]
        # Sort first by y1 (vertical axis), then by x1 (horizontal axis)
        #import pdb;pdb.set_trace()
        sorted_detections = sorted(detections[0].boxes.data, key=lambda x: (x[1], x[0]))
        
        # Combine the sorted detections into final number plate string
        number_plate = ''.join([str(detections[0].names[int(det[-1])]) for det in sorted_detections])
        
        return number_plate[::-1]

    def sort_and_create_vehicle_number(self,detections, char_map=None, rotation_threshold=15):
        """
        Sort detections from a YOLO model and create a vehicle number.
        
        Args:
            detections: List of detections in format [x1, y1, x2, y2, confidence, class]
            char_map: Dictionary mapping class IDs to characters or a function
            rotation_threshold: Threshold in degrees to consider a plate as rotated
            
        Returns:
            String representing the vehicle number
        """
        import numpy as np
        import math
        
        if not detections:
            return ""
        
        if char_map is None:
            char_map = self.char_map
        
        # Extract bounding box information
        detections  = detections[0].boxes.data.cpu().numpy()
        boxes = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            boxes.append({
                'center_x': center_x, 'center_y': center_y,
                'width': width, 'height': height,
                'class': int(cls)
            })
        
        
        # Group characters into lines based on y-coordinate
        # Calculate average height to determine line threshold
        avg_height = sum(box['height'] for box in boxes) / len(boxes)
        
        # Sort by y-coordinate
        boxes.sort(key=lambda box: box['center_y'])
        
        # Group into lines
        lines = []
        current_line = [boxes[0]]
        
        for i in range(1, len(boxes)):
            # If the y-coordinate difference is small, add to current line
            if abs(boxes[i]['center_y'] - current_line[0]['center_y']) < avg_height * 0.7:
                current_line.append(boxes[i])
            else:
                # Start a new line
                lines.append(current_line)
                current_line = [boxes[i]]
        
        # Add the last line
        if current_line:
            lines.append(current_line)
        
        # Sort each line by x-coordinate
        for i in range(len(lines)):
            lines[i].sort(key=lambda box: box['center_x'])
        
        # Combine to get the vehicle number
        vehicle_number = ""
        for i, line in enumerate(sorted(lines, key=lambda line: line[0]['center_y'])):
            if i > 0:
                vehicle_number += "_"  # Add a space between lines
            line_text = "".join(char_map[box['class']] for box in line)
            vehicle_number += line_text
        
        return vehicle_number.strip()
