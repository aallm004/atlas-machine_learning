#!/usr/bin/env python3
"""Module for class Yolo"""
from tensorflow import keras as K
import numpy as np


class Yolo:
    """Class Yolo that uses the Yolo v3 algorithm for object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize Yolo"""
        self.model = K.models.load_model(model_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
        
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]

    def process_outputs(self, outputs, image_size):
        """Process Darknet model outputs"""
        boxes = []
        box_confidences = []
        box_class_probs = []

        for idx, output in enumerate(outputs):
            grid_height, grid_width, anchors_count = output.shape[:3]
            
            # Extract confidence and class probabilities
            box_confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            box_confidences.append(box_confidence)
            box_class_probs.append(1 / (1 + np.exp(-output[..., 5:])))
            
            # Create grid coordinates
            grid_x = np.tile(np.arange(grid_width), grid_height).reshape(grid_height, grid_width, 1, 1)
            grid_y = np.tile(np.arange(grid_height).reshape(-1, 1), grid_width).reshape(grid_height, grid_width, 1, 1)

            # Get box coords and dimensions
            tx = output[..., 0:1]
            ty = output[..., 1:2]
            tw = output[..., 2:3]
            th = output[..., 3:4]

            # Apply sigmoid to tx, ty and get center coordinates
            bx = (1 / (1 + np.exp(-tx))) + grid_x
            by = (1 / (1 + np.exp(-ty))) + grid_y
            
            # Get width and height
            pw = self.anchors[idx, :, 0].reshape(1, 1, anchors_count, 1)
            ph = self.anchors[idx, :, 1].reshape(1, 1, anchors_count, 1)
            bw = pw * np.exp(tw)
            bh = ph * np.exp(th)

            # Convert to corner coordinates
            x1 = (bx - bw/2) * image_size[1] / grid_width
            y1 = (by - bh/2) * image_size[0] / grid_height
            x2 = (bx + bw/2) * image_size[1] / grid_width
            y2 = (by + bh/2) * image_size[0] / grid_height

            # Stack coordinates
            box = np.concatenate((x1, y1, x2, y2), axis=-1)
            boxes.append(box)

        return boxes, box_confidences, box_class_probs
