#!/usr/bin/env python3
"""Module for class Yolo"""
from tensorflow import keras as K
import numpy as np


class Yolo:
    """Class Yolo that uses the Yolo v3 algorithm for object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize Yolo with model and parameters"""
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
            
            # Box confidence sigmoid
            box_confidences.append(1 / (1 + np.exp(-output[..., 4:5])))
            
            # Box class probabilities sigmoid
            box_class_probs.append(1 / (1 + np.exp(-output[..., 5:])))
            
            # Create meshgrid
            grid_x = np.arange(grid_width).reshape(1, grid_width, 1)
            grid_y = np.arange(grid_height).reshape(grid_height, 1, 1)
            
            # Tile to match shape
            box_xy = 1 / (1 + np.exp(-output[..., :2]))
            box_wh = np.exp(output[..., 2:4]) * self.anchors[idx]
            
            # Add grid offsets
            box_xy[..., 0] = (box_xy[..., 0] + grid_x) / grid_width
            box_xy[..., 1] = (box_xy[..., 1] + grid_y) / grid_height
            
            # Normalize to image size
            box_wh[..., 0] /= grid_width
            box_wh[..., 1] /= grid_height
            
            # Convert to corner coordinates
            box_x1y1 = box_xy - box_wh / 2
            box_x2y2 = box_xy + box_wh / 2
            box = np.concatenate((box_x1y1, box_x2y2), axis=-1)
            
            # Scale to image size
            box = box * np.tile(image_size, 2)
            boxes.append(box)

        return boxes, box_confidences, box_class_probs
